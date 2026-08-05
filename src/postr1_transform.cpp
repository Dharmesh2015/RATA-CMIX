#include "postr1_transform.h"

#include "models/scr2_tokens.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

namespace postr1 {
namespace {

constexpr char kPlanMagic[4] = {'F', '4', 'T', 'X'};
constexpr char kStreamMagic[4] = {'F', '4', 'P', 'T'};
constexpr std::uint16_t kPlanVersion = 1;
constexpr std::uint16_t kStreamVersion = 1;
constexpr std::uint32_t kMaxBlocks = 1u << 20;
constexpr std::uint32_t kMaxBlockBytes = 64u << 20;
constexpr std::uint8_t kDictionaryMarker = 1;
constexpr std::uint8_t kPairMarker = 2;

struct EncodedBlock {
  BlockSpec spec;
  std::uint32_t checksum = 0;
  std::vector<std::uint8_t> payload;
};

bool ReadFile(const std::string& path, std::vector<std::uint8_t>* bytes) {
  FILE* input = std::fopen(path.c_str(), "rb");
  if (!input) return false;
#ifdef _WIN32
  if (_fseeki64(input, 0, SEEK_END) != 0) {
    std::fclose(input);
    return false;
  }
  const __int64 measured = _ftelli64(input);
  if (measured < 0 || _fseeki64(input, 0, SEEK_SET) != 0) {
#else
  if (fseeko(input, 0, SEEK_END) != 0) {
    std::fclose(input);
    return false;
  }
  const off_t measured = ftello(input);
  if (measured < 0 || fseeko(input, 0, SEEK_SET) != 0) {
#endif
    std::fclose(input);
    return false;
  }
  if (static_cast<std::uint64_t>(measured) >
      std::numeric_limits<std::size_t>::max()) {
    std::fclose(input);
    return false;
  }
  bytes->resize(static_cast<std::size_t>(measured));
  const std::size_t read = bytes->empty()
      ? 0 : std::fread(bytes->data(), 1, bytes->size(), input);
  const bool ok = read == bytes->size() && std::ferror(input) == 0;
  std::fclose(input);
  return ok;
}

bool ReplaceFile(const std::string& path,
    const std::vector<std::uint8_t>& bytes) {
  const std::string temporary = path + ".postr1.new";
  const std::string backup = path + ".postr1.old";
  std::remove(temporary.c_str());
  std::remove(backup.c_str());
  FILE* output = std::fopen(temporary.c_str(), "wb");
  if (!output) return false;
  const std::size_t written = bytes.empty()
      ? 0 : std::fwrite(bytes.data(), 1, bytes.size(), output);
  bool ok = written == bytes.size() && std::fflush(output) == 0;
  if (std::fclose(output) != 0) ok = false;
  if (!ok) {
    std::remove(temporary.c_str());
    return false;
  }
  if (std::rename(path.c_str(), backup.c_str()) != 0) {
    std::remove(temporary.c_str());
    return false;
  }
  if (std::rename(temporary.c_str(), path.c_str()) != 0) {
    std::rename(backup.c_str(), path.c_str());
    std::remove(temporary.c_str());
    return false;
  }
  std::remove(backup.c_str());
  return true;
}

template <typename T>
void PutLittle(std::vector<std::uint8_t>* output, T value) {
  for (unsigned int shift = 0; shift < sizeof(T) * 8; shift += 8) {
    output->push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

template <typename T>
bool GetLittle(const std::vector<std::uint8_t>& input,
    std::size_t* position, T* value) {
  if (*position > input.size() ||
      sizeof(T) > input.size() - *position) return false;
  *value = 0;
  for (unsigned int shift = 0; shift < sizeof(T) * 8; shift += 8) {
    *value |= static_cast<T>(input[(*position)++]) << shift;
  }
  return true;
}

void PutVarint(std::vector<std::uint8_t>* output, std::uint64_t value) {
  do {
    std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7f);
    value >>= 7;
    if (value != 0) byte |= 0x80;
    output->push_back(byte);
  } while (value != 0);
}

bool GetVarint(const std::vector<std::uint8_t>& input,
    std::size_t* position, std::uint64_t* value) {
  *value = 0;
  for (unsigned int shift = 0; shift < 64; shift += 7) {
    if (*position >= input.size()) return false;
    const std::uint8_t byte = input[(*position)++];
    *value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) return true;
  }
  return false;
}

std::uint32_t Checksum(const std::vector<std::uint8_t>& bytes) {
  std::uint32_t hash = 2166136261u;
  for (std::uint8_t byte : bytes) {
    hash = (hash ^ byte) * 16777619u;
  }
  return hash;
}

std::uint64_t Hash8(const std::uint8_t* data) {
  std::uint64_t value = 0xcbf29ce484222325ULL;
  for (unsigned int i = 0; i < 8; ++i) {
    value = (value ^ data[i]) * 0x100000001b3ULL;
  }
  return value;
}

bool ReadPlan(const std::string& path, std::uint64_t stream_size,
    std::vector<BlockSpec>* blocks) {
  std::vector<std::uint8_t> bytes;
  if (!ReadFile(path, &bytes)) return false;
  std::size_t position = 0;
  if (bytes.size() < 4 ||
      std::memcmp(bytes.data(), kPlanMagic, 4) != 0) return false;
  position = 4;
  std::uint16_t version = 0;
  std::uint16_t flags = 0;
  std::uint64_t recorded_size = 0;
  std::uint32_t count = 0;
  if (!GetLittle(bytes, &position, &version) ||
      !GetLittle(bytes, &position, &flags) ||
      !GetLittle(bytes, &position, &recorded_size) ||
      !GetLittle(bytes, &position, &count) ||
      version != kPlanVersion || flags != 0 ||
      recorded_size != stream_size || count == 0 || count > kMaxBlocks) {
    return false;
  }
  blocks->clear();
  blocks->reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    BlockSpec block;
    std::uint8_t reference_count = 0;
    std::uint16_t operation_count = 0;
    if (!GetLittle(bytes, &position, &block.offset) ||
        !GetLittle(bytes, &position, &block.length) ||
        !GetLittle(bytes, &position, &block.physical_order) ||
        !GetLittle(bytes, &position, &block.stage_mask) ||
        !GetLittle(bytes, &position, &block.stream_class) ||
        !GetLittle(bytes, &position, &reference_count) ||
        !GetLittle(bytes, &position, &operation_count) ||
        block.length == 0 || block.length > kMaxBlockBytes ||
        (block.stage_mask & ~kAllStages) != 0 || reference_count > 8 ||
        operation_count > 256) {
      return false;
    }
    block.references.resize(reference_count);
    for (std::uint64_t& reference : block.references) {
      if (!GetLittle(bytes, &position, &reference)) return false;
    }
    block.operations.resize(operation_count);
    for (Operation& operation : block.operations) {
      if (!GetLittle(bytes, &position, &operation.type) ||
          !GetLittle(bytes, &position, &operation.a) ||
          !GetLittle(bytes, &position, &operation.b) ||
          !GetLittle(bytes, &position, &operation.c) ||
          (operation.type != 1 && operation.type != 2)) {
        return false;
      }
    }
    blocks->push_back(std::move(block));
  }
  if (position != bytes.size()) return false;

  std::vector<const BlockSpec*> by_offset;
  by_offset.reserve(blocks->size());
  std::set<std::uint32_t> physical_orders;
  for (const BlockSpec& block : *blocks) {
    by_offset.push_back(&block);
    if (!physical_orders.insert(block.physical_order).second) return false;
  }
  std::sort(by_offset.begin(), by_offset.end(),
      [](const BlockSpec* left, const BlockSpec* right) {
        return left->offset < right->offset;
      });
  std::uint64_t expected = 0;
  for (const BlockSpec* block : by_offset) {
    if (block->offset != expected || block->offset > stream_size ||
        block->length > stream_size - block->offset) return false;
    expected += block->length;
  }
  return expected == stream_size;
}

void RotateLeft(std::vector<std::uint8_t>* bytes, std::uint32_t amount) {
  if (bytes->empty()) return;
  amount %= bytes->size();
  std::rotate(bytes->begin(), bytes->begin() + amount, bytes->end());
}

bool ApplyPmd1(std::vector<std::uint8_t>* bytes,
    const std::vector<Operation>& operations, bool inverse) {
  auto apply = [bytes, inverse](const Operation& operation) {
    if (operation.type == 1) {
      if (bytes->empty()) return true;
      const std::uint32_t amount = operation.a % bytes->size();
      RotateLeft(bytes, inverse
          ? static_cast<std::uint32_t>(bytes->size()) - amount : amount);
      return true;
    }
    if (operation.a > bytes->size() ||
        operation.b > bytes->size() - operation.a || operation.b == 0) {
      return false;
    }
    const std::uint32_t amount = operation.c % operation.b;
    auto begin = bytes->begin() + operation.a;
    std::rotate(begin,
        begin + (inverse ? operation.b - amount : amount),
        begin + operation.b);
    return true;
  };
  if (!inverse) {
    for (const Operation& operation : operations) {
      if (!apply(operation)) return false;
    }
  } else {
    for (auto it = operations.rbegin(); it != operations.rend(); ++it) {
      if (!apply(*it)) return false;
    }
  }
  return true;
}

unsigned int ByteLane(std::uint8_t byte) {
  if ((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
      byte >= 0x80) return 0;
  if (byte >= '0' && byte <= '9') return 1;
  if (byte == '<' || byte == '>' || byte == '[' || byte == ']' ||
      byte == '{' || byte == '}' || byte == '|' || byte == '=') return 2;
  if (byte == ' ' || byte == '\n' || byte == '\r' || byte == '\t') return 3;
  if (byte == '/' || byte == ':' || byte == '?' || byte == '&' ||
      byte == '#' || byte == '.') return 4;
  return 5;
}

std::vector<std::uint8_t> EncodeTyped(
    const std::vector<std::uint8_t>& input) {
  std::array<std::vector<std::uint8_t>, 6> lanes;
  std::vector<std::pair<std::uint8_t, std::uint32_t>> runs;
  for (std::uint8_t byte : input) {
    const std::uint8_t lane = static_cast<std::uint8_t>(ByteLane(byte));
    lanes[lane].push_back(byte);
    if (runs.empty() || runs.back().first != lane) {
      runs.emplace_back(lane, 1);
    } else {
      ++runs.back().second;
    }
  }
  std::vector<std::uint8_t> output;
  PutLittle(&output, static_cast<std::uint32_t>(runs.size()));
  for (const auto& run : runs) {
    output.push_back(run.first);
    PutVarint(&output, run.second);
  }
  for (const auto& lane : lanes) {
    PutLittle(&output, static_cast<std::uint32_t>(lane.size()));
  }
  for (const auto& lane : lanes) output.insert(output.end(), lane.begin(), lane.end());
  return output;
}

bool DecodeTyped(const std::vector<std::uint8_t>& input,
    std::vector<std::uint8_t>* output) {
  std::size_t position = 0;
  std::uint32_t run_count = 0;
  if (!GetLittle(input, &position, &run_count) || run_count > input.size()) {
    return false;
  }
  std::vector<std::pair<std::uint8_t, std::uint32_t>> runs;
  runs.reserve(run_count);
  std::uint64_t total = 0;
  for (std::uint32_t i = 0; i < run_count; ++i) {
    if (position >= input.size()) return false;
    const std::uint8_t lane = input[position++];
    std::uint64_t length = 0;
    if (lane >= 6 || !GetVarint(input, &position, &length) ||
        length == 0 || length > UINT32_MAX) return false;
    runs.emplace_back(lane, static_cast<std::uint32_t>(length));
    total += length;
  }
  std::array<std::uint32_t, 6> lengths{};
  std::array<std::size_t, 6> starts{};
  for (unsigned int lane = 0; lane < 6; ++lane) {
    if (!GetLittle(input, &position, &lengths[lane])) return false;
    starts[lane] = position;
  }
  std::size_t lane_start = position;
  for (unsigned int lane = 0; lane < 6; ++lane) {
    starts[lane] = lane_start;
    if (lengths[lane] > input.size() - lane_start) return false;
    lane_start += lengths[lane];
  }
  if (lane_start != input.size()) return false;
  std::array<std::uint32_t, 6> used{};
  output->clear();
  output->reserve(static_cast<std::size_t>(total));
  for (const auto& run : runs) {
    if (run.second > lengths[run.first] - used[run.first]) return false;
    const std::size_t begin = starts[run.first] + used[run.first];
    output->insert(output->end(), input.begin() + begin,
        input.begin() + begin + run.second);
    used[run.first] += run.second;
  }
  return output->size() == total;
}

using Dictionary = std::vector<std::vector<std::uint8_t>>;

Dictionary Scr2Dictionary() {
  Dictionary dictionary;
  dictionary.reserve(scr2::kTokenCount);
  for (unsigned int code = 1; code <= scr2::kTokenCount; ++code) {
    const scr2::TokenInfo& token = scr2::kTokens[code];
    const std::uint8_t* data = scr2::PatternData(code);
    dictionary.emplace_back(data, data + token.length);
  }
  return dictionary;
}

Dictionary WikiDictionary() {
  static const char* patterns[] = {
      "<page>", "</page>", "<title>", "</title>", "<text",
      "</text>", "[[", "]]", "{{", "}}", "<ref", "</ref>",
      "[[Category:", "==References==", "==External links==",
      "{{cite ", "{{Infobox", "|url=", "|title=", "|date=",
      "http://", "https://", "class=\"wikitable\"", "\n* ", "\n# "};
  Dictionary dictionary;
  for (const char* pattern : patterns) {
    dictionary.emplace_back(pattern, pattern + std::strlen(pattern));
  }
  return dictionary;
}

Dictionary MineWords(const std::vector<std::uint8_t>& input) {
  std::map<std::string, std::uint32_t> counts;
  for (std::size_t position = 0; position < input.size();) {
    if (!std::isalnum(static_cast<unsigned char>(input[position]))) {
      ++position;
      continue;
    }
    const std::size_t begin = position++;
    while (position < input.size() &&
        (std::isalnum(static_cast<unsigned char>(input[position])) ||
         input[position] == '_' || input[position] == '-')) ++position;
    const std::size_t length = position - begin;
    if (length >= 4 && length <= 63) {
      ++counts[std::string(reinterpret_cast<const char*>(input.data() + begin),
          length)];
    }
  }
  std::vector<std::pair<std::int64_t, std::string>> ranked;
  for (const auto& entry : counts) {
    const std::int64_t score = static_cast<std::int64_t>(entry.second) *
        (entry.first.size() - 2) - entry.first.size() - 2;
    if (entry.second >= 3 && score > 0) ranked.emplace_back(score, entry.first);
  }
  std::sort(ranked.begin(), ranked.end(),
      [](const auto& left, const auto& right) { return left.first > right.first; });
  Dictionary dictionary;
  for (std::size_t i = 0; i < ranked.size() && i < 254; ++i) {
    dictionary.emplace_back(ranked[i].second.begin(), ranked[i].second.end());
  }
  return dictionary;
}

Dictionary MinePhrases(const std::vector<std::uint8_t>& input) {
  std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> positions;
  for (std::uint32_t position = 0; position + 8 <= input.size(); position += 2) {
    auto& bucket = positions[Hash8(input.data() + position)];
    if (bucket.size() < 8) bucket.push_back(position);
  }
  std::vector<std::pair<std::int64_t, std::vector<std::uint8_t>>> ranked;
  for (const auto& entry : positions) {
    if (entry.second.size() < 3) continue;
    std::size_t length = 8;
    while (length < 32) {
      bool same = true;
      for (std::uint32_t position : entry.second) {
        if (position + length >= input.size() ||
            input[position + length] != input[entry.second[0] + length]) {
          same = false;
          break;
        }
      }
      if (!same) break;
      ++length;
    }
    const std::int64_t score = static_cast<std::int64_t>(entry.second.size()) *
        (length - 2) - length - 2;
    if (score > 0) ranked.push_back({score,
        std::vector<std::uint8_t>(input.begin() + entry.second[0],
            input.begin() + entry.second[0] + length)});
  }
  std::sort(ranked.begin(), ranked.end(),
      [](const auto& left, const auto& right) { return left.first > right.first; });
  Dictionary dictionary;
  std::set<std::vector<std::uint8_t>> seen;
  for (const auto& entry : ranked) {
    if (dictionary.size() >= 254) break;
    if (seen.insert(entry.second).second) dictionary.push_back(entry.second);
  }
  return dictionary;
}

std::vector<std::uint8_t> EncodeDictionary(
    const std::vector<std::uint8_t>& input, const Dictionary& source) {
  Dictionary dictionary = source;
  if (dictionary.size() > 254) dictionary.resize(254);
  std::vector<std::uint8_t> output;
  PutLittle(&output, static_cast<std::uint16_t>(dictionary.size()));
  for (const auto& pattern : dictionary) {
    output.push_back(static_cast<std::uint8_t>(pattern.size()));
    output.insert(output.end(), pattern.begin(), pattern.end());
  }
  for (std::size_t position = 0; position < input.size();) {
    unsigned int best = 0;
    std::size_t best_length = 0;
    for (unsigned int index = 0; index < dictionary.size(); ++index) {
      const auto& pattern = dictionary[index];
      if (pattern.size() <= best_length || pattern.size() > input.size() - position) continue;
      if (std::memcmp(input.data() + position, pattern.data(), pattern.size()) == 0) {
        best = index + 1;
        best_length = pattern.size();
      }
    }
    if (best != 0) {
      output.push_back(kDictionaryMarker);
      output.push_back(static_cast<std::uint8_t>(best));
      position += best_length;
    } else {
      const std::uint8_t byte = input[position++];
      output.push_back(byte);
      if (byte == kDictionaryMarker) output.push_back(0);
    }
  }
  return output;
}

bool DecodeDictionary(const std::vector<std::uint8_t>& input,
    std::vector<std::uint8_t>* output) {
  std::size_t position = 0;
  std::uint16_t count = 0;
  if (!GetLittle(input, &position, &count) || count > 254) return false;
  Dictionary dictionary;
  dictionary.reserve(count);
  for (unsigned int i = 0; i < count; ++i) {
    if (position >= input.size()) return false;
    const std::size_t length = input[position++];
    if (length == 0 || length > input.size() - position) return false;
    dictionary.emplace_back(input.begin() + position,
        input.begin() + position + length);
    position += length;
  }
  output->clear();
  while (position < input.size()) {
    const std::uint8_t byte = input[position++];
    if (byte != kDictionaryMarker) {
      output->push_back(byte);
      continue;
    }
    if (position >= input.size()) return false;
    const std::uint8_t code = input[position++];
    if (code == 0) output->push_back(kDictionaryMarker);
    else if (code <= dictionary.size()) {
      output->insert(output->end(), dictionary[code - 1].begin(),
          dictionary[code - 1].end());
    } else return false;
  }
  return true;
}

std::vector<std::uint8_t> EncodePairs(const std::vector<std::uint8_t>& input) {
  std::array<std::uint32_t, 65536> counts{};
  for (std::size_t i = 0; i + 1 < input.size(); ++i) {
    ++counts[(static_cast<unsigned int>(input[i]) << 8) | input[i + 1]];
  }
  std::vector<std::pair<std::uint32_t, std::uint16_t>> ranked;
  for (unsigned int pair = 0; pair < counts.size(); ++pair) {
    if (counts[pair] >= 4) ranked.emplace_back(counts[pair], pair);
  }
  std::sort(ranked.begin(), ranked.end(), std::greater<>());
  if (ranked.size() > 64) ranked.resize(64);
  std::unordered_map<std::uint16_t, std::uint8_t> code;
  std::vector<std::uint8_t> output;
  output.push_back(static_cast<std::uint8_t>(ranked.size()));
  for (unsigned int i = 0; i < ranked.size(); ++i) {
    PutLittle(&output, ranked[i].second);
    code[ranked[i].second] = static_cast<std::uint8_t>(i + 1);
  }
  for (std::size_t i = 0; i < input.size();) {
    if (i + 1 < input.size()) {
      const std::uint16_t pair =
          (static_cast<unsigned int>(input[i]) << 8) | input[i + 1];
      const auto found = code.find(pair);
      if (found != code.end()) {
        output.push_back(kPairMarker);
        output.push_back(found->second);
        i += 2;
        continue;
      }
    }
    const std::uint8_t byte = input[i++];
    output.push_back(byte);
    if (byte == kPairMarker) output.push_back(0);
  }
  return output;
}

bool DecodePairs(const std::vector<std::uint8_t>& input,
    std::vector<std::uint8_t>* output) {
  if (input.empty()) return false;
  std::size_t position = 0;
  const unsigned int count = input[position++];
  std::vector<std::uint16_t> pairs(count);
  for (std::uint16_t& pair : pairs) {
    if (!GetLittle(input, &position, &pair)) return false;
  }
  output->clear();
  while (position < input.size()) {
    const std::uint8_t byte = input[position++];
    if (byte != kPairMarker) {
      output->push_back(byte);
      continue;
    }
    if (position >= input.size()) return false;
    const std::uint8_t code = input[position++];
    if (code == 0) output->push_back(kPairMarker);
    else if (code <= pairs.size()) {
      output->push_back(static_cast<std::uint8_t>(pairs[code - 1] >> 8));
      output->push_back(static_cast<std::uint8_t>(pairs[code - 1]));
    } else return false;
  }
  return true;
}

std::vector<std::uint8_t> EncodeWct(const std::vector<std::uint8_t>& input) {
  std::array<std::uint32_t, 256> counts{};
  for (std::uint8_t byte : input) ++counts[byte];
  std::array<std::uint8_t, 256> order{};
  for (unsigned int i = 0; i < 256; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
      [&counts](std::uint8_t left, std::uint8_t right) {
        return counts[left] > counts[right];
      });
  std::array<std::uint8_t, 256> map{};
  for (unsigned int i = 0; i < 256; ++i) map[order[i]] = i;
  std::vector<std::uint8_t> output(order.begin(), order.end());
  output.reserve(256 + input.size());
  for (std::uint8_t byte : input) output.push_back(map[byte]);
  return output;
}

bool DecodeWct(const std::vector<std::uint8_t>& input,
    std::vector<std::uint8_t>* output) {
  if (input.size() < 256) return false;
  std::array<bool, 256> seen{};
  for (unsigned int i = 0; i < 256; ++i) {
    if (seen[input[i]]) return false;
    seen[input[i]] = true;
  }
  output->resize(input.size() - 256);
  for (std::size_t i = 256; i < input.size(); ++i) {
    (*output)[i - 256] = input[input[i]];
  }
  return true;
}

std::vector<std::uint8_t> EncodeScrr(const std::vector<std::uint8_t>& input) {
  std::vector<std::uint8_t> output;
  std::vector<std::uint8_t> previous;
  std::size_t position = 0;
  while (position < input.size()) {
    std::size_t end = position;
    while (end < input.size() && input[end] != '\n') ++end;
    if (end < input.size()) ++end;
    std::vector<std::uint8_t> line(input.begin() + position, input.begin() + end);
    std::vector<std::pair<std::uint32_t, std::uint8_t>> differences;
    if (line.size() == previous.size()) {
      for (std::uint32_t i = 0; i < line.size(); ++i) {
        if (line[i] != previous[i]) differences.emplace_back(i, line[i]);
      }
    }
    const bool delta = !previous.empty() && line.size() == previous.size() &&
        differences.size() * 3 + 4 < line.size();
    output.push_back(delta ? 1 : 0);
    PutVarint(&output, line.size());
    if (delta) {
      PutVarint(&output, differences.size());
      std::uint32_t last = 0;
      for (const auto& difference : differences) {
        PutVarint(&output, difference.first - last);
        output.push_back(difference.second);
        last = difference.first;
      }
    } else output.insert(output.end(), line.begin(), line.end());
    previous.swap(line);
    position = end;
  }
  return output;
}

bool DecodeScrr(const std::vector<std::uint8_t>& input,
    std::vector<std::uint8_t>* output) {
  std::size_t position = 0;
  std::vector<std::uint8_t> previous;
  output->clear();
  while (position < input.size()) {
    const std::uint8_t mode = input[position++];
    std::uint64_t length = 0;
    if (mode > 1 || !GetVarint(input, &position, &length) ||
        length > kMaxBlockBytes) return false;
    std::vector<std::uint8_t> line;
    if (mode == 0) {
      if (length > input.size() - position) return false;
      line.assign(input.begin() + position, input.begin() + position + length);
      position += length;
    } else {
      if (previous.size() != length) return false;
      line = previous;
      std::uint64_t count = 0;
      if (!GetVarint(input, &position, &count) || count > length) return false;
      std::uint64_t index = 0;
      for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t delta = 0;
        if (!GetVarint(input, &position, &delta) ||
            delta > length - index || position >= input.size()) return false;
        index += delta;
        if (index >= line.size()) return false;
        line[index] = input[position++];
      }
    }
    output->insert(output->end(), line.begin(), line.end());
    previous.swap(line);
  }
  return true;
}

struct RefPosition {
  std::uint8_t reference;
  std::uint32_t offset;
};

std::vector<std::uint8_t> EncodeRlz(const std::vector<std::uint8_t>& input,
    const std::vector<const std::vector<std::uint8_t>*>& references) {
  std::unordered_map<std::uint64_t, std::vector<RefPosition>> index;
  for (unsigned int r = 0; r < references.size(); ++r) {
    const auto& reference = *references[r];
    for (std::uint32_t p = 0; p + 8 <= reference.size(); p += 4) {
      auto& candidates = index[Hash8(reference.data() + p)];
      if (candidates.size() < 8) candidates.push_back(
          {static_cast<std::uint8_t>(r), p});
    }
  }
  std::vector<std::uint8_t> output;
  PutLittle(&output, static_cast<std::uint32_t>(input.size()));
  std::vector<std::uint8_t> literals;
  auto flush_literals = [&] {
    if (literals.empty()) return;
    output.push_back(0);
    PutVarint(&output, literals.size());
    output.insert(output.end(), literals.begin(), literals.end());
    literals.clear();
  };
  std::size_t position = 0;
  while (position < input.size()) {
    std::size_t best_length = 0;
    RefPosition best{};
    std::vector<std::pair<std::uint32_t, std::uint8_t>> best_mismatches;
    if (position + 8 <= input.size()) {
      const auto found = index.find(Hash8(input.data() + position));
      if (found != index.end()) {
        for (const RefPosition candidate : found->second) {
          const auto& reference = *references[candidate.reference];
          std::size_t length = 0;
          std::vector<std::pair<std::uint32_t, std::uint8_t>> mismatches;
          while (position + length < input.size() &&
              candidate.offset + length < reference.size() &&
              length < 65535) {
            if (input[position + length] != reference[candidate.offset + length]) {
              mismatches.emplace_back(static_cast<std::uint32_t>(length),
                  input[position + length]);
              if (mismatches.size() > 2 + length / 32) break;
            }
            ++length;
          }
          if (length >= 16 && length > best_length &&
              mismatches.size() * 3 + 8 < length) {
            best_length = length;
            best = candidate;
            best_mismatches.swap(mismatches);
          }
        }
      }
    }
    if (best_length < 16) {
      literals.push_back(input[position++]);
      if (literals.size() == 65535) flush_literals();
      continue;
    }
    flush_literals();
    output.push_back(1);
    output.push_back(best.reference);
    PutVarint(&output, best.offset);
    PutVarint(&output, best_length);
    PutVarint(&output, best_mismatches.size());
    std::uint32_t previous = 0;
    for (const auto& mismatch : best_mismatches) {
      PutVarint(&output, mismatch.first - previous);
      output.push_back(mismatch.second);
      previous = mismatch.first;
    }
    position += best_length;
  }
  flush_literals();
  return output;
}

bool DecodeRlz(const std::vector<std::uint8_t>& input,
    const std::vector<const std::vector<std::uint8_t>*>& references,
    std::vector<std::uint8_t>* output) {
  std::size_t position = 0;
  std::uint32_t expected = 0;
  if (!GetLittle(input, &position, &expected) || expected > kMaxBlockBytes) return false;
  output->clear();
  output->reserve(expected);
  while (position < input.size() && output->size() < expected) {
    const std::uint8_t mode = input[position++];
    if (mode == 0) {
      std::uint64_t length = 0;
      if (!GetVarint(input, &position, &length) ||
          length > input.size() - position || length > expected - output->size()) return false;
      output->insert(output->end(), input.begin() + position,
          input.begin() + position + length);
      position += length;
    } else if (mode == 1) {
      if (position >= input.size()) return false;
      const std::uint8_t reference_id = input[position++];
      std::uint64_t offset = 0, length = 0, mismatch_count = 0;
      if (reference_id >= references.size() ||
          !GetVarint(input, &position, &offset) ||
          !GetVarint(input, &position, &length) ||
          !GetVarint(input, &position, &mismatch_count) ||
          offset > references[reference_id]->size() ||
          length > references[reference_id]->size() - offset ||
          length > expected - output->size() || mismatch_count > length) return false;
      const std::size_t begin = output->size();
      output->insert(output->end(), references[reference_id]->begin() + offset,
          references[reference_id]->begin() + offset + length);
      std::uint64_t mismatch = 0;
      for (std::uint64_t i = 0; i < mismatch_count; ++i) {
        std::uint64_t delta = 0;
        if (!GetVarint(input, &position, &delta) || delta > length - mismatch ||
            position >= input.size()) return false;
        mismatch += delta;
        if (mismatch >= length) return false;
        (*output)[begin + mismatch] = input[position++];
      }
    } else return false;
  }
  return position == input.size() && output->size() == expected;
}

std::vector<std::uint8_t> EncodeXor(const std::vector<std::uint8_t>& input,
    const std::vector<const std::vector<std::uint8_t>*>& references) {
  std::size_t best = references.size();
  std::size_t equal = 0;
  for (std::size_t r = 0; r < references.size(); ++r) {
    if (references[r]->size() != input.size()) continue;
    std::size_t candidate = 0;
    for (std::size_t i = 0; i < input.size(); ++i) {
      candidate += input[i] == (*references[r])[i];
    }
    if (candidate > equal) {
      equal = candidate;
      best = r;
    }
  }
  if (best == references.size() ||
      equal * 100 < input.size() * 95) return {};
  std::vector<std::uint8_t> output;
  output.reserve(input.size() + 1);
  output.push_back(static_cast<std::uint8_t>(best));
  for (std::size_t i = 0; i < input.size(); ++i) {
    output.push_back(input[i] ^ (*references[best])[i]);
  }
  return output;
}

bool DecodeXor(const std::vector<std::uint8_t>& input,
    const std::vector<const std::vector<std::uint8_t>*>& references,
    std::vector<std::uint8_t>* output) {
  if (input.empty() || input[0] >= references.size() ||
      references[input[0]]->size() + 1 != input.size()) return false;
  output->resize(input.size() - 1);
  for (std::size_t i = 0; i < output->size(); ++i) {
    (*output)[i] = input[i + 1] ^ (*references[input[0]])[i];
  }
  return true;
}

bool ApplyStages(const BlockSpec& spec,
    const std::vector<std::uint8_t>& original,
    const std::vector<const std::vector<std::uint8_t>*>& references,
    std::vector<std::uint8_t>* output) {
  *output = original;
  if (spec.stage_mask & kApproximateRlz) {
    *output = EncodeRlz(*output, references);
  } else if (spec.stage_mask & kXorReference) {
    *output = EncodeXor(*output, references);
    if (output->empty()) return false;
  }
  if ((spec.stage_mask & kPmd1Permutation) &&
      !ApplyPmd1(output, spec.operations, false)) return false;
  if (spec.stage_mask & kWctRemap) *output = EncodeWct(*output);
  if (spec.stage_mask & kTypedStreams) *output = EncodeTyped(*output);
  if (spec.stage_mask & kScr2Shorthand) {
    *output = EncodeDictionary(*output, Scr2Dictionary());
  }
  if (spec.stage_mask & kCxWordSymbols) {
    *output = EncodeDictionary(*output, MineWords(*output));
  }
  if (spec.stage_mask & kSczBytePairs) *output = EncodePairs(*output);
  if (spec.stage_mask & kScrrRecordResidual) *output = EncodeScrr(*output);
  if (spec.stage_mask & kWikiNative2) {
    *output = EncodeDictionary(*output, WikiDictionary());
  }
  if (spec.stage_mask & kPhraseMacros) {
    *output = EncodeDictionary(*output, MinePhrases(*output));
  }
  return true;
}

bool UndoStages(const BlockSpec& spec,
    const std::vector<std::uint8_t>& encoded,
    const std::vector<const std::vector<std::uint8_t>*>& references,
    std::vector<std::uint8_t>* output) {
  *output = encoded;
  std::vector<std::uint8_t> next;
  if ((spec.stage_mask & kPhraseMacros) &&
      (!DecodeDictionary(*output, &next))) return false;
  if (spec.stage_mask & kPhraseMacros) output->swap(next);
  if ((spec.stage_mask & kWikiNative2) &&
      !DecodeDictionary(*output, &next)) return false;
  if (spec.stage_mask & kWikiNative2) output->swap(next);
  if ((spec.stage_mask & kScrrRecordResidual) &&
      !DecodeScrr(*output, &next)) return false;
  if (spec.stage_mask & kScrrRecordResidual) output->swap(next);
  if ((spec.stage_mask & kSczBytePairs) &&
      !DecodePairs(*output, &next)) return false;
  if (spec.stage_mask & kSczBytePairs) output->swap(next);
  if ((spec.stage_mask & kCxWordSymbols) &&
      !DecodeDictionary(*output, &next)) return false;
  if (spec.stage_mask & kCxWordSymbols) output->swap(next);
  if ((spec.stage_mask & kScr2Shorthand) &&
      !DecodeDictionary(*output, &next)) return false;
  if (spec.stage_mask & kScr2Shorthand) output->swap(next);
  if ((spec.stage_mask & kTypedStreams) &&
      !DecodeTyped(*output, &next)) return false;
  if (spec.stage_mask & kTypedStreams) output->swap(next);
  if ((spec.stage_mask & kWctRemap) && !DecodeWct(*output, &next)) return false;
  if (spec.stage_mask & kWctRemap) output->swap(next);
  if ((spec.stage_mask & kPmd1Permutation) &&
      !ApplyPmd1(output, spec.operations, true)) return false;
  if ((spec.stage_mask & kXorReference) &&
      !DecodeXor(*output, references, &next)) return false;
  if (spec.stage_mask & kXorReference) output->swap(next);
  if ((spec.stage_mask & kApproximateRlz) &&
      !DecodeRlz(*output, references, &next)) return false;
  if (spec.stage_mask & kApproximateRlz) output->swap(next);
  return true;
}

void WriteBlockSpec(std::vector<std::uint8_t>* output,
    const EncodedBlock& block) {
  PutLittle(output, block.spec.offset);
  PutLittle(output, block.spec.length);
  PutLittle(output, block.spec.physical_order);
  PutLittle(output, block.spec.stage_mask);
  output->push_back(block.spec.stream_class);
  output->push_back(static_cast<std::uint8_t>(block.spec.references.size()));
  PutLittle(output, static_cast<std::uint16_t>(block.spec.operations.size()));
  PutLittle(output, static_cast<std::uint32_t>(block.payload.size()));
  PutLittle(output, block.checksum);
  for (std::uint64_t reference : block.spec.references) PutLittle(output, reference);
  for (const Operation& operation : block.spec.operations) {
    output->push_back(operation.type);
    PutLittle(output, operation.a);
    PutLittle(output, operation.b);
    PutLittle(output, operation.c);
  }
}

bool ReadBlockSpec(const std::vector<std::uint8_t>& input,
    std::size_t* position, EncodedBlock* block,
    std::uint32_t* payload_size) {
  std::uint8_t reference_count = 0;
  std::uint16_t operation_count = 0;
  if (!GetLittle(input, position, &block->spec.offset) ||
      !GetLittle(input, position, &block->spec.length) ||
      !GetLittle(input, position, &block->spec.physical_order) ||
      !GetLittle(input, position, &block->spec.stage_mask) ||
      !GetLittle(input, position, &block->spec.stream_class) ||
      !GetLittle(input, position, &reference_count) ||
      !GetLittle(input, position, &operation_count) ||
      !GetLittle(input, position, payload_size) ||
      !GetLittle(input, position, &block->checksum) ||
      reference_count > 8 || operation_count > 256 ||
      (block->spec.stage_mask & ~kAllStages) != 0) return false;
  block->spec.references.resize(reference_count);
  for (std::uint64_t& reference : block->spec.references) {
    if (!GetLittle(input, position, &reference)) return false;
  }
  block->spec.operations.resize(operation_count);
  for (Operation& operation : block->spec.operations) {
    if (!GetLittle(input, position, &operation.type) ||
        !GetLittle(input, position, &operation.a) ||
        !GetLittle(input, position, &operation.b) ||
        !GetLittle(input, position, &operation.c)) return false;
  }
  return true;
}

}  // namespace

bool EncodeFile(const std::string& path, const std::string& plan_path,
    std::uint64_t* original_size, std::uint64_t* transformed_size) {
  std::vector<std::uint8_t> original;
  if (!ReadFile(path, &original)) return false;
  std::vector<BlockSpec> specs;
  if (!ReadPlan(plan_path, original.size(), &specs)) return false;
  std::sort(specs.begin(), specs.end(),
      [](const BlockSpec& left, const BlockSpec& right) {
        return left.physical_order < right.physical_order;
      });

  std::map<std::uint64_t, std::vector<std::uint8_t>> restored;
  std::vector<EncodedBlock> blocks;
  blocks.reserve(specs.size());
  for (const BlockSpec& spec : specs) {
    std::vector<std::uint8_t> bytes(original.begin() + spec.offset,
        original.begin() + spec.offset + spec.length);
    std::vector<const std::vector<std::uint8_t>*> references;
    for (std::uint64_t reference : spec.references) {
      const auto found = restored.find(reference);
      if (found == restored.end()) return false;
      references.push_back(&found->second);
    }
    EncodedBlock block;
    block.spec = spec;
    block.checksum = Checksum(bytes);
    if (!ApplyStages(spec, bytes, references, &block.payload)) return false;
    restored.emplace(spec.offset, std::move(bytes));
    blocks.push_back(std::move(block));
  }

  std::vector<std::uint8_t> output;
  output.insert(output.end(), kStreamMagic, kStreamMagic + 4);
  PutLittle(&output, kStreamVersion);
  PutLittle(&output, static_cast<std::uint16_t>(0));
  PutLittle(&output, static_cast<std::uint64_t>(original.size()));
  PutLittle(&output, static_cast<std::uint32_t>(blocks.size()));
  for (const EncodedBlock& block : blocks) WriteBlockSpec(&output, block);
  for (const EncodedBlock& block : blocks) {
    output.insert(output.end(), block.payload.begin(), block.payload.end());
  }
  *original_size = original.size();
  *transformed_size = output.size();
  return ReplaceFile(path, output);
}

bool DecodeFile(const std::string& path, std::uint64_t* restored_size) {
  std::vector<std::uint8_t> input;
  if (!ReadFile(path, &input) || input.size() < 20 ||
      std::memcmp(input.data(), kStreamMagic, 4) != 0) return false;
  std::size_t position = 4;
  std::uint16_t version = 0;
  std::uint16_t flags = 0;
  std::uint64_t original_size = 0;
  std::uint32_t count = 0;
  if (!GetLittle(input, &position, &version) ||
      !GetLittle(input, &position, &flags) ||
      !GetLittle(input, &position, &original_size) ||
      !GetLittle(input, &position, &count) ||
      version != kStreamVersion || flags != 0 || count == 0 ||
      count > kMaxBlocks || original_size >
          std::numeric_limits<std::size_t>::max()) return false;
  std::vector<EncodedBlock> blocks(count);
  std::vector<std::uint32_t> payload_sizes(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    if (!ReadBlockSpec(input, &position, &blocks[i], &payload_sizes[i])) {
      return false;
    }
  }
  for (std::uint32_t i = 0; i < count; ++i) {
    if (payload_sizes[i] > input.size() - position) return false;
    blocks[i].payload.assign(input.begin() + position,
        input.begin() + position + payload_sizes[i]);
    position += payload_sizes[i];
  }
  if (position != input.size()) return false;

  std::map<std::uint64_t, std::vector<std::uint8_t>> restored;
  std::vector<std::uint8_t> output(static_cast<std::size_t>(original_size));
  std::vector<bool> covered(output.size(), false);
  for (const EncodedBlock& block : blocks) {
    std::vector<const std::vector<std::uint8_t>*> references;
    for (std::uint64_t reference : block.spec.references) {
      const auto found = restored.find(reference);
      if (found == restored.end()) return false;
      references.push_back(&found->second);
    }
    std::vector<std::uint8_t> bytes;
    if (!UndoStages(block.spec, block.payload, references, &bytes) ||
        bytes.size() != block.spec.length || Checksum(bytes) != block.checksum ||
        block.spec.offset > original_size ||
        bytes.size() > original_size - block.spec.offset) return false;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      if (covered[block.spec.offset + i]) return false;
      covered[block.spec.offset + i] = true;
      output[block.spec.offset + i] = bytes[i];
    }
    restored.emplace(block.spec.offset, std::move(bytes));
  }
  if (std::find(covered.begin(), covered.end(), false) != covered.end()) return false;
  *restored_size = output.size();
  return ReplaceFile(path, output);
}

bool IsEncodedFile(const std::string& path) {
  FILE* input = std::fopen(path.c_str(), "rb");
  if (!input) return false;
  char magic[4] = {};
  const bool ok = std::fread(magic, 1, sizeof(magic), input) == sizeof(magic) &&
      std::memcmp(magic, kStreamMagic, sizeof(magic)) == 0;
  std::fclose(input);
  return ok;
}

}  // namespace postr1

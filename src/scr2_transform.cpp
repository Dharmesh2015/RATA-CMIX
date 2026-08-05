#include "scr2_transform.h"

#include "models/scr2_tokens.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/types.h>
#endif

namespace scr2 {
namespace {

struct TrieNode {
  TrieNode() { next.fill(-1); }
  std::array<std::int16_t, 256> next;
  std::uint8_t code = 0;
};

const std::vector<TrieNode>& TokenTrie() {
  static const std::vector<TrieNode> trie = [] {
    std::vector<TrieNode> nodes(1);
    nodes.reserve(1 + sizeof(kPatternBytes));
    for (unsigned int code = 1; code <= kTokenCount; ++code) {
      int node = 0;
      const TokenInfo& token = kTokens[code];
      const std::uint8_t* pattern = PatternData(code);
      for (unsigned int index = 0; index < token.length; ++index) {
        std::int16_t& child = nodes[node].next[pattern[index]];
        if (child < 0) {
          if (nodes.size() >=
              static_cast<size_t>(std::numeric_limits<std::int16_t>::max())) {
            std::abort();
          }
          child = static_cast<std::int16_t>(nodes.size());
          nodes.emplace_back();
        }
        node = child;
      }
      nodes[node].code = static_cast<std::uint8_t>(code);
    }
    return nodes;
  }();
  return trie;
}

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
      std::numeric_limits<size_t>::max()) {
    std::fclose(input);
    return false;
  }
  bytes->resize(static_cast<size_t>(measured));
  const size_t count = bytes->empty()
      ? 0 : std::fread(bytes->data(), 1, bytes->size(), input);
  const bool ok = count == bytes->size() && std::ferror(input) == 0;
  std::fclose(input);
  return ok;
}

bool WriteReplacement(const std::string& path,
    const std::vector<std::uint8_t>& bytes) {
  const std::string temporary = path + ".scr2.new";
  const std::string backup = path + ".scr2.old";
  std::remove(temporary.c_str());
  std::remove(backup.c_str());
  FILE* output = std::fopen(temporary.c_str(), "wb");
  if (!output) return false;
  const size_t count = bytes.empty()
      ? 0 : std::fwrite(bytes.data(), 1, bytes.size(), output);
  bool wrote = count == bytes.size() && std::fflush(output) == 0;
  if (std::fclose(output) != 0) wrote = false;
  if (!wrote) {
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

}  // namespace

bool EncodeFile(const std::string& path, std::uint64_t* input_size,
    std::uint64_t* output_size) {
  std::vector<std::uint8_t> input;
  if (!ReadFile(path, &input)) return false;
  std::vector<std::uint8_t> output;
  output.reserve(input.size());
  const std::vector<TrieNode>& trie = TokenTrie();

  size_t position = 0;
  while (position < input.size()) {
    int node = 0;
    std::uint8_t best_code = 0;
    size_t best_length = 0;
    const size_t available = input.size() - position;
    const size_t limit =
        std::min<size_t>(available, kMaxPatternLength);
    for (size_t length = 0; length < limit; ++length) {
      const std::int16_t child = trie[node].next[input[position + length]];
      if (child < 0) break;
      node = child;
      if (trie[node].code != 0) {
        best_code = trie[node].code;
        best_length = length + 1;
      }
    }
    if (best_code != 0) {
      output.push_back(kMarker);
      output.push_back(best_code);
      position += best_length;
    } else {
      const std::uint8_t byte = input[position++];
      output.push_back(byte);
      if (byte == kMarker) output.push_back(0);
    }
  }

  *input_size = input.size();
  *output_size = output.size();
  return WriteReplacement(path, output);
}

bool DecodeFile(const std::string& path, std::uint64_t* output_size) {
  std::vector<std::uint8_t> input;
  if (!ReadFile(path, &input)) return false;
  std::vector<std::uint8_t> output;
  output.reserve(input.size());

  for (size_t position = 0; position < input.size(); ++position) {
    const std::uint8_t byte = input[position];
    if (byte != kMarker) {
      output.push_back(byte);
      continue;
    }
    if (++position >= input.size()) return false;
    const std::uint8_t code = input[position];
    if (code == 0) {
      output.push_back(kMarker);
    } else if (code <= kTokenCount) {
      const TokenInfo& token = kTokens[code];
      const std::uint8_t* pattern = PatternData(code);
      output.insert(output.end(), pattern, pattern + token.length);
    } else {
      return false;
    }
  }

  *output_size = output.size();
  return WriteReplacement(path, output);
}

}  // namespace scr2

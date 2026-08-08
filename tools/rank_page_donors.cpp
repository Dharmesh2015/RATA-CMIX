// Fast research-only ranking of causal donor windows for complete-page packs.
//
// This scanner never changes the compressed format. It uses cheap post-R1
// phrase signatures to shortlist page/donor pairs; the normal warm FX4 search
// must still arithmetic-code every retained candidate before accepting it.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr std::array<std::uint32_t, 5> kDonorLengths{
    256, 1024, 4096, 16384, 65536};
constexpr std::uint32_t kSampleStride = 64;
constexpr std::size_t kSignatureCount = 8;
constexpr std::size_t kPostingCapacity = 12;
constexpr std::uint8_t kMixedClass = 10;

struct Pack {
  std::uint32_t id = 0;
  std::uint64_t offset = 0;
  std::uint32_t length = 0;
  std::uint8_t stream_class = kMixedClass;
  std::uint32_t feature_mask = 0;
};

struct Posting {
  std::array<std::uint64_t, kPostingCapacity> offsets{};
  std::uint32_t seen = 0;
  std::uint8_t count = 0;

  void Add(std::uint64_t offset) {
    for (std::uint8_t i = 0; i < count; ++i) {
      if (offsets[i] == offset) return;
    }
    if (count < offsets.size()) {
      offsets[count++] = offset;
    } else {
      // Preserve class-diverse old representatives while refreshing recency.
      offsets[2 + (seen % (offsets.size() - 2))] = offset;
    }
    ++seen;
  }
};

struct CandidateScore {
  std::uint32_t score = 0;
  std::uint32_t hits = 0;
};

struct ProfileScore {
  std::uint32_t mentions = 0;
  std::uint32_t score = 0;
  std::uint32_t hits = 0;
};

struct RecipientScore {
  std::uint32_t pack_id = 0;
  std::uint64_t offset = 0;
  std::uint32_t length = 0;
  std::uint32_t best_score = 0;
  std::uint32_t best_hits = 0;
  std::uint32_t candidates = 0;
};

std::vector<std::string> Split(const std::string& text, char delimiter) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find(delimiter, start);
    fields.push_back(text.substr(start,
        end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return fields;
}

bool ParseU64(const std::string& text, std::uint64_t* value) {
  if (text.empty()) return false;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0') return false;
  *value = parsed;
  return true;
}

int Column(const std::vector<std::string>& header, const char* name) {
  const auto found = std::find(header.begin(), header.end(), name);
  return found == header.end() ? -1 :
      static_cast<int>(found - header.begin());
}

bool LoadPacks(const std::string& path, std::uint64_t stream_size,
    std::vector<Pack>* packs) {
  std::ifstream input(path);
  std::string line;
  if (!input.is_open() || !std::getline(input, line)) return false;
  const auto header = Split(line, ',');
  const int id_column = Column(header, "pack_id");
  const int offset_column = Column(header, "post_r1_start");
  const int end_column = Column(header, "post_r1_end");
  const int length_column = Column(header, "post_r1_length");
  const int class_column = Column(header, "stream_class");
  const int feature_column = Column(header, "feature_mask");
  if (id_column < 0 || offset_column < 0 || end_column < 0 ||
      length_column < 0) {
    return false;
  }

  std::uint64_t previous_end = 0;
  while (std::getline(input, line)) {
    const auto fields = Split(line, ',');
    if (fields.size() != header.size()) return false;
    std::uint64_t id = 0;
    std::uint64_t offset = 0;
    std::uint64_t end = 0;
    std::uint64_t length = 0;
    std::uint64_t stream_class = kMixedClass;
    std::uint64_t feature_mask = 0;
    if (!ParseU64(fields[id_column], &id) ||
        !ParseU64(fields[offset_column], &offset) ||
        !ParseU64(fields[end_column], &end) ||
        !ParseU64(fields[length_column], &length) ||
        (class_column >= 0 &&
         !ParseU64(fields[class_column], &stream_class)) ||
        (feature_column >= 0 &&
         !ParseU64(fields[feature_column], &feature_mask)) ||
        id != packs->size() || id > std::numeric_limits<std::uint32_t>::max() ||
        length == 0 || length > std::numeric_limits<std::uint32_t>::max() ||
        end < offset || end - offset != length || end > stream_size ||
        offset < previous_end || stream_class > kMixedClass ||
        feature_mask > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    packs->push_back({static_cast<std::uint32_t>(id), offset,
        static_cast<std::uint32_t>(length),
        static_cast<std::uint8_t>(stream_class),
        static_cast<std::uint32_t>(feature_mask)});
    previous_end = end;
  }
  return !packs->empty();
}

std::uint64_t MixHash(std::uint64_t value) {
  value ^= value >> 30;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

std::uint64_t PhraseHash(const std::uint8_t* bytes) {
  std::uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
  for (unsigned int i = 0; i < 16; ++i) {
    hash = MixHash(hash ^
        (static_cast<std::uint64_t>(bytes[i]) + i * 257u));
  }
  return hash;
}

std::uint64_t TokenHash(const std::uint8_t* bytes, std::size_t length) {
  std::uint64_t hash = UINT64_C(0xd6e8feb86659fd93);
  unsigned int useful = 0;
  for (std::size_t i = 0; i < length; i += 4) {
    const std::uint8_t byte = bytes[i];
    std::uint8_t token = 0;
    if (byte >= 128) token = byte;
    else if (byte >= '0' && byte <= '9') token = '0';
    else if (byte == '<' || byte == '>' || byte == '[' || byte == ']' ||
        byte == '{' || byte == '}' || byte == '|' || byte == '=' ||
        byte == '&' || byte == ';') {
      token = byte;
    }
    if (token != 0) {
      hash = MixHash(hash ^
          (static_cast<std::uint64_t>(token) + useful * 65537u));
      ++useful;
    }
  }
  return useful >= 8 ? hash ^ UINT64_C(0xa5a5a5a5a5a5a5a5) : 0;
}

std::vector<std::uint64_t> Signatures(
    const std::uint8_t* bytes, std::size_t length) {
  std::array<std::uint64_t, kSignatureCount> minima;
  minima.fill(std::numeric_limits<std::uint64_t>::max());
  for (std::size_t i = 0; i + 16 <= length; i += kSampleStride) {
    const std::uint64_t hash = PhraseHash(bytes + i);
    auto found = std::lower_bound(minima.begin(), minima.end(), hash);
    if (found != minima.end() && *found != hash) {
      std::move_backward(found, minima.end() - 1, minima.end());
      *found = hash;
    }
  }
  std::vector<std::uint64_t> result;
  result.reserve(kSignatureCount + 1);
  for (const std::uint64_t hash : minima) {
    if (hash != std::numeric_limits<std::uint64_t>::max()) {
      result.push_back(hash);
    }
  }
  const std::uint64_t token = TokenHash(bytes, length);
  if (token != 0) result.push_back(token);
  return result;
}

std::uint64_t SignatureKey(std::uint64_t signature,
    std::uint8_t stream_class) {
  return MixHash(signature ^
      (UINT64_C(0x9e3779b97f4a7c15) * (stream_class + 1u)));
}

std::uint64_t WindowKey(std::uint32_t offset, unsigned int length_index,
    std::uint8_t stream_class) {
  return (static_cast<std::uint64_t>(offset) << 8) |
      (static_cast<std::uint64_t>(length_index) << 4) | stream_class;
}

std::uint32_t WindowOffset(std::uint64_t key) {
  return static_cast<std::uint32_t>(key >> 8);
}

std::uint32_t WindowLength(std::uint64_t key) {
  const unsigned int index = static_cast<unsigned int>((key >> 4) & 15u);
  return index < kDonorLengths.size() ? kDonorLengths[index] : 0;
}

std::uint8_t WindowClass(std::uint64_t key) {
  return static_cast<std::uint8_t>(key & 15u);
}

void Usage() {
  std::cerr << "usage: postr1_page_donor_ranker POST_R1 RECIPIENTS_CSV "
               "OUT_CSV [TOP_K] [STOP_OFFSET]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    Usage();
    return 2;
  }
  unsigned int top_k = 12;
  if (argc >= 5) {
    char* end = nullptr;
    const unsigned long value = std::strtoul(argv[4], &end, 10);
    if (!end || *end != '\0' || value == 0 || value > 64) {
      Usage();
      return 2;
    }
    top_k = static_cast<unsigned int>(value);
  }
  std::uint64_t stop_offset = std::numeric_limits<std::uint64_t>::max();
  if (argc == 6) {
    char* end = nullptr;
    stop_offset = std::strtoull(argv[5], &end, 10);
    if (!end || *end != '\0' || stop_offset == 0) {
      Usage();
      return 2;
    }
  }

  const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
  struct stat info {};
  if (fd < 0 || fstat(fd, &info) != 0 || info.st_size <= 0) {
    if (fd >= 0) close(fd);
    std::cerr << "cannot open post-R1 stream\n";
    return 1;
  }
  const std::size_t stream_size = static_cast<std::size_t>(info.st_size);
  const auto* stream = static_cast<const std::uint8_t*>(
      mmap(nullptr, stream_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (stream == MAP_FAILED) {
    close(fd);
    std::cerr << "cannot mmap post-R1 stream\n";
    return 1;
  }
  madvise(const_cast<std::uint8_t*>(stream), stream_size, MADV_SEQUENTIAL);

  std::vector<Pack> packs;
  if (!LoadPacks(argv[2], stream_size, &packs)) {
    munmap(const_cast<std::uint8_t*>(stream), stream_size);
    close(fd);
    std::cerr << "invalid page-pack CSV\n";
    return 1;
  }

  std::ofstream candidates(argv[3], std::ios::trunc);
  const std::string recipient_path = std::string(argv[3]) + ".recipients.csv";
  std::ofstream recipients(recipient_path, std::ios::trunc);
  if (!candidates.is_open() || !recipients.is_open()) {
    munmap(const_cast<std::uint8_t*>(stream), stream_size);
    close(fd);
    std::cerr << "cannot create output CSV\n";
    return 1;
  }
  std::array<std::unordered_map<std::uint64_t, ProfileScore>,
      kMixedClass + 1> class_profiles;
  candidates << "pack_id,recipient_offset,recipient_length,candidate_rank,"
                "donor_offset,donor_length,phrase_score,signature_hits,"
                "recipient_class,donor_class\n";

  std::unordered_map<std::uint64_t, Posting> postings;
  postings.reserve(2000000);
  std::size_t indexed_pack = 0;
  std::vector<RecipientScore> recipient_scores;
  recipient_scores.reserve(packs.size());

  auto index_window = [&](const Pack& donor_pack, std::uint32_t offset,
                          unsigned int length_index) {
    const std::uint32_t length = kDonorLengths[length_index];
    if (static_cast<std::uint64_t>(offset) + length >
        donor_pack.offset + donor_pack.length) {
      return;
    }
    const std::uint64_t key =
        WindowKey(offset, length_index, donor_pack.stream_class);
    for (const std::uint64_t signature :
         Signatures(stream + offset, length)) {
      postings[SignatureKey(signature, donor_pack.stream_class)].Add(key);
    }
  };

  for (const Pack& pack : packs) {
    if (pack.offset >= stop_offset ||
        pack.offset + pack.length > stop_offset) break;
    while (indexed_pack < pack.id) {
      const Pack& donor_pack = packs[indexed_pack++];
      if (donor_pack.offset + donor_pack.length > pack.offset) continue;
      const std::uint64_t aligned_start =
          (donor_pack.offset + 255u) & ~UINT64_C(255);
      const std::uint64_t donor_end = donor_pack.offset + donor_pack.length;
      for (unsigned int length_index = 0;
           length_index < kDonorLengths.size(); ++length_index) {
        const std::uint32_t length = kDonorLengths[length_index];
        if (aligned_start + length > donor_end) continue;
        std::array<std::uint64_t, 3> positions{{
            aligned_start,
            ((donor_end - length) & ~UINT64_C(255)),
            (((aligned_start + donor_end - length) / 2u) & ~UINT64_C(255))}};
        for (unsigned int position_index = 0;
             position_index < positions.size(); ++position_index) {
          const std::uint64_t position = positions[position_index];
          if (position < aligned_start || position + length > donor_end ||
              position > std::numeric_limits<std::uint32_t>::max()) {
            continue;
          }
          bool duplicate = false;
          for (unsigned int prior = 0; prior < position_index; ++prior) {
            duplicate |= positions[prior] == position;
          }
          if (!duplicate) {
            index_window(donor_pack, static_cast<std::uint32_t>(position),
                length_index);
          }
        }
      }
    }

    std::unordered_map<std::uint64_t, CandidateScore> scores;
    for (unsigned int length_index = 0;
         length_index < kDonorLengths.size(); ++length_index) {
      const std::uint32_t length = kDonorLengths[length_index];
      if (length > pack.length) continue;
      const std::uint64_t last = pack.offset + pack.length - length;
      const std::uint64_t stride = std::max<std::uint64_t>(length, 4096u);
      for (std::uint64_t position = pack.offset;;) {
        for (const std::uint64_t signature :
             Signatures(stream + position, length)) {
          const auto found = postings.find(
              SignatureKey(signature, pack.stream_class));
          if (found == postings.end()) continue;
          for (std::uint8_t i = 0; i < found->second.count; ++i) {
            const std::uint64_t donor_key = found->second.offsets[i];
            const std::uint32_t donor_offset = WindowOffset(donor_key);
            const std::uint32_t donor_length = WindowLength(donor_key);
            if (donor_length == 0 ||
                static_cast<std::uint64_t>(donor_offset) + donor_length >
                    pack.offset) {
              continue;
            }
            CandidateScore& score = scores[donor_key];
            score.score += 1u + length_index * 2u;
            ++score.hits;
          }
        }
        if (position == last) break;
        position = std::min(last, position + stride);
      }
    }

    std::vector<std::pair<std::uint64_t, CandidateScore>> all_ranked(
        scores.begin(), scores.end());
    std::sort(all_ranked.begin(), all_ranked.end(),
        [](const auto& left, const auto& right) {
          if (left.second.score != right.second.score) {
            return left.second.score > right.second.score;
          }
          if (left.second.hits != right.second.hits) {
            return left.second.hits > right.second.hits;
          }
          return left.first < right.first;
        });
    std::vector<std::pair<std::uint64_t, CandidateScore>> ranked;
    ranked.reserve(std::min<std::size_t>(top_k, all_ranked.size()));
    std::unordered_set<std::uint32_t> source_buckets;
    for (unsigned int pass = 0; pass < 2 && ranked.size() < top_k; ++pass) {
      for (const auto& candidate : all_ranked) {
        const std::uint32_t bucket = WindowOffset(candidate.first) >> 16;
        if (pass == 0 && !source_buckets.insert(bucket).second) continue;
        if (std::find_if(ranked.begin(), ranked.end(),
                [&](const auto& selected) {
                  return selected.first == candidate.first;
                }) == ranked.end()) {
          ranked.push_back(candidate);
          if (ranked.size() == top_k) break;
        }
      }
    }
    for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
      ProfileScore& profile =
          class_profiles[pack.stream_class][ranked[rank].first];
      ++profile.mentions;
      profile.score += ranked[rank].second.score;
      profile.hits += ranked[rank].second.hits;
      candidates << pack.id << ',' << pack.offset << ',' << pack.length
          << ',' << rank << ',' << WindowOffset(ranked[rank].first) << ','
          << WindowLength(ranked[rank].first) << ','
          << ranked[rank].second.score << ',' << ranked[rank].second.hits
          << ',' << static_cast<unsigned int>(pack.stream_class) << ','
          << static_cast<unsigned int>(WindowClass(ranked[rank].first))
          << '\n';
    }
    recipient_scores.push_back({pack.id, pack.offset, pack.length,
        ranked.empty() ? 0u : ranked.front().second.score,
        ranked.empty() ? 0u : ranked.front().second.hits,
        static_cast<std::uint32_t>(ranked.size())});
    if ((pack.id & 127u) == 0) {
      std::cerr << "ranked page recipients " << pack.id + 1 << '/'
                << packs.size() << '\r' << std::flush;
    }
  }

  std::sort(recipient_scores.begin(), recipient_scores.end(),
      [](const RecipientScore& left, const RecipientScore& right) {
        if (left.best_score != right.best_score) {
          return left.best_score > right.best_score;
        }
        if (left.best_hits != right.best_hits) {
          return left.best_hits > right.best_hits;
        }
        return left.pack_id < right.pack_id;
      });
  recipients << "rank,pack_id,recipient_offset,recipient_length,"
                "best_phrase_score,best_signature_hits,candidate_count\n";
  for (std::size_t rank = 0; rank < recipient_scores.size(); ++rank) {
    const RecipientScore& score = recipient_scores[rank];
    recipients << rank << ',' << score.pack_id << ',' << score.offset << ','
        << score.length << ',' << score.best_score << ',' << score.best_hits
        << ',' << score.candidates << '\n';
  }

  std::array<std::vector<std::pair<std::uint64_t, ProfileScore>>,
      kMixedClass + 1> selected_profiles;
  const std::string profile_path = std::string(argv[3]) + ".profiles.csv";
  const std::string shared_path = std::string(argv[3]) + ".shared.csv";
  std::ofstream profiles(profile_path, std::ios::trunc);
  std::ofstream shared(shared_path, std::ios::trunc);
  if (!profiles.is_open() || !shared.is_open()) {
    munmap(const_cast<std::uint8_t*>(stream), stream_size);
    close(fd);
    std::cerr << "cannot create shared-profile CSVs\n";
    return 1;
  }
  profiles << "stream_class,profile_rank,donor_offset,donor_length,"
              "recipient_mentions,phrase_score,signature_hits\n";
  shared << "pack_id,recipient_offset,recipient_length,candidate_rank,"
            "donor_offset,donor_length,phrase_score,signature_hits,"
            "recipient_class,donor_class\n";
  for (unsigned int stream_class = 0;
       stream_class < selected_profiles.size(); ++stream_class) {
    std::vector<std::pair<std::uint64_t, ProfileScore>> ranked_profiles(
        class_profiles[stream_class].begin(),
        class_profiles[stream_class].end());
    std::sort(ranked_profiles.begin(), ranked_profiles.end(),
        [](const auto& left, const auto& right) {
          if (left.second.score != right.second.score) {
            return left.second.score > right.second.score;
          }
          if (left.second.mentions != right.second.mentions) {
            return left.second.mentions > right.second.mentions;
          }
          if (left.second.hits != right.second.hits) {
            return left.second.hits > right.second.hits;
          }
          return left.first < right.first;
        });
    std::unordered_set<std::uint32_t> selected_offsets;
    for (const auto& profile : ranked_profiles) {
      if (!selected_offsets.insert(WindowOffset(profile.first)).second) {
        continue;
      }
      selected_profiles[stream_class].push_back(profile);
      if (selected_profiles[stream_class].size() == 7) break;
    }
    for (std::size_t rank = 0;
         rank < selected_profiles[stream_class].size(); ++rank) {
      const auto& profile = selected_profiles[stream_class][rank];
      profiles << stream_class << ',' << rank << ','
          << WindowOffset(profile.first) << ',' << WindowLength(profile.first)
          << ',' << profile.second.mentions << ',' << profile.second.score
          << ',' << profile.second.hits << '\n';
    }
  }
  for (const Pack& pack : packs) {
    if (pack.offset + pack.length > stop_offset) break;
    unsigned int rank = 0;
    for (const auto& profile : selected_profiles[pack.stream_class]) {
      const std::uint32_t donor_offset = WindowOffset(profile.first);
      const std::uint32_t donor_length = WindowLength(profile.first);
      if (static_cast<std::uint64_t>(donor_offset) + donor_length >
          pack.offset) {
        continue;
      }
      shared << pack.id << ',' << pack.offset << ',' << pack.length << ','
          << rank++ << ',' << donor_offset << ',' << donor_length << ','
          << profile.second.score << ',' << profile.second.hits << ','
          << static_cast<unsigned int>(pack.stream_class) << ','
          << static_cast<unsigned int>(pack.stream_class) << '\n';
    }
  }

  candidates.close();
  recipients.close();
  profiles.close();
  shared.close();
  munmap(const_cast<std::uint8_t*>(stream), stream_size);
  close(fd);
  std::cerr << "\nranked " << recipient_scores.size()
            << " page recipients; "
            << "candidate_csv=" << argv[3]
            << " recipient_csv=" << recipient_path
            << " profile_csv=" << profile_path
            << " shared_csv=" << shared_path << '\n';
  return candidates.good() && recipients.good() &&
      profiles.good() && shared.good() ? 0 : 1;
}

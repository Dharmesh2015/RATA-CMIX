// Build classified complete-page donor packs in the post-R1 coordinate space.
//
// The tool decodes WRT only far enough to recognize exact <page> boundaries,
// then maps their encoded post-WRT offsets through the F4RM1 segment map. It
// never writes the decoded enwik9 stream and is not part of the submission
// executable.
#include "../src/preprocess/dictionary.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr unsigned char kTextSegment = 7;
constexpr char kMapMagic[8] = {'F', '4', 'R', 'M', '1', '\r', '\n', '\0'};

struct MapSegment {
  std::uint64_t source = 0;
  std::uint64_t destination = 0;
  std::uint64_t length = 0;
};

struct Page {
  std::uint64_t decoded_start = 0;
  std::uint64_t decoded_end = 0;
  std::uint64_t pre_start = 0;
  std::uint64_t pre_end = 0;
  std::uint64_t post_start = 0;
  std::uint64_t post_end = 0;
  bool mapped = false;
  bool complete = false;
  std::uint8_t stream_class = 10;
  std::uint32_t feature_mask = 0;
  std::uint64_t letter_bytes = 0;
  std::uint64_t digit_bytes = 0;
  std::uint32_t xml_tags = 0;
  std::uint32_t template_markers = 0;
  std::uint32_t table_markers = 0;
  std::uint32_t reference_markers = 0;
  std::uint32_t url_markers = 0;
  std::uint32_t list_lines = 0;
};

enum StreamClass : std::uint8_t {
  kProse = 0,
  kXml = 1,
  kNumber = 2,
  kDate = 3,
  kTable = 4,
  kReference = 5,
  kUrl = 6,
  kIdentifier = 7,
  kTemplate = 8,
  kList = 9,
  kMixed = 10,
};

const char* StreamClassName(std::uint8_t stream_class) {
  static const char* names[] = {
      "prose", "xml", "number", "date", "table", "reference",
      "url", "identifier", "template", "list", "mixed"};
  return stream_class <= kMixed ? names[stream_class] : names[kMixed];
}

bool ReadU64(std::ifstream* input, std::uint64_t* value) {
  unsigned char bytes[8];
  input->read(reinterpret_cast<char*>(bytes), sizeof(bytes));
  if (input->gcount() != static_cast<std::streamsize>(sizeof(bytes))) {
    return false;
  }
  std::uint64_t result = 0;
  for (unsigned int i = 0; i < 8; ++i) {
    result |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
  }
  *value = result;
  return true;
}

bool ReadMap(const std::string& path, std::vector<MapSegment>* segments,
    std::uint64_t* post_wrt_size, std::uint64_t* post_r1_size) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) return false;
  char magic[8];
  input.read(magic, sizeof(magic));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(magic)) ||
      std::memcmp(magic, kMapMagic, sizeof(magic)) != 0) {
    return false;
  }
  std::uint64_t side_size = 0;
  std::uint64_t segment_count = 0;
  if (!ReadU64(&input, post_wrt_size) ||
      !ReadU64(&input, post_r1_size) ||
      !ReadU64(&input, &side_size) ||
      !ReadU64(&input, &segment_count)) {
    return false;
  }
  if (segment_count > 10000000ull) return false;
  segments->resize(static_cast<std::size_t>(segment_count));
  for (MapSegment& segment : *segments) {
    if (!ReadU64(&input, &segment.source) ||
        !ReadU64(&input, &segment.destination) ||
        !ReadU64(&input, &segment.length)) {
      return false;
    }
  }
  *post_r1_size += side_size + 16u;
  std::sort(segments->begin(), segments->end(),
      [](const MapSegment& a, const MapSegment& b) {
        return a.source < b.source;
      });
  return true;
}

const MapSegment* FindSegment(const std::vector<MapSegment>& segments,
    std::uint64_t offset, bool allow_end) {
  auto it = std::upper_bound(segments.begin(), segments.end(), offset,
      [](std::uint64_t value, const MapSegment& segment) {
        return value < segment.source;
      });
  if (it == segments.begin()) return nullptr;
  --it;
  const std::uint64_t end = it->source + it->length;
  if (offset < end || (allow_end && offset == end)) return &*it;
  return nullptr;
}

bool MapRange(const std::vector<MapSegment>& segments,
    std::uint64_t source_start, std::uint64_t source_end,
    std::uint64_t* destination_start, std::uint64_t* destination_end) {
  if (source_end < source_start) return false;
  const MapSegment* first = FindSegment(segments, source_start, false);
  const MapSegment* last = FindSegment(segments, source_end, true);
  if (!first || first != last ||
      source_end > first->source + first->length) {
    return false;
  }
  *destination_start =
      first->destination + (source_start - first->source);
  *destination_end =
      first->destination + (source_end - first->source);
  return true;
}

class BoundaryScanner {
 public:
  void Add(unsigned char byte, std::uint64_t source_before,
      std::uint64_t source_after) {
    bytes_[next_] = static_cast<char>(byte);
    before_[next_] = source_before;
    after_[next_] = source_after;
    next_ = (next_ + 1) % bytes_.size();
    if (count_ < bytes_.size()) ++count_;
    ++decoded_offset_;

    if (open_) Observe(byte);
    if (!open_ && EndsWith("<page>")) {
      open_ = true;
      current_.decoded_start = decoded_offset_ - 6;
      current_.pre_start = BeforeFromEnd(6);
      line_start_ = false;
    }
    if (open_ && EndsWith("</page>")) {
      current_.decoded_end = decoded_offset_;
      current_.pre_end = AfterFromEnd(1);
      current_.complete = true;
      FinishClass();
      pages_.push_back(current_);
      current_ = Page();
      open_ = false;
    }
  }

  void Finish(std::uint64_t source_end) {
    if (!open_) return;
    current_.decoded_end = decoded_offset_;
    current_.pre_end = source_end;
    current_.complete = false;
    FinishClass();
    pages_.push_back(current_);
    current_ = Page();
    open_ = false;
  }

  const std::vector<Page>& pages() const { return pages_; }
  std::uint64_t decoded_size() const { return decoded_offset_; }
  bool page_open() const { return open_; }

 private:
  void Observe(unsigned char byte) {
    if ((byte >= 'A' && byte <= 'Z') ||
        (byte >= 'a' && byte <= 'z')) {
      ++current_.letter_bytes;
    }
    if (byte >= '0' && byte <= '9') ++current_.digit_bytes;
    if (byte == '<') ++current_.xml_tags;
    if (line_start_ && (byte == '*' || byte == '#' || byte == ';' ||
        byte == ':')) {
      ++current_.list_lines;
    }
    line_start_ = byte == '\n' || byte == '\r';

    if (EndsWith("{{")) ++current_.template_markers;
    if (EndsWith("{|")) ++current_.table_markers;
    if (EndsWith("<ref") || EndsWith("</ref") || EndsWith("==References") ||
        EndsWith("==External links")) {
      ++current_.reference_markers;
    }
    if (EndsWith("http://") || EndsWith("https://") || EndsWith("www.")) {
      ++current_.url_markers;
    }
    if (EndsWith("<timestamp>") || EndsWith("</timestamp>")) {
      feature_mask_accumulator_ |= 1u << kDate;
    }
    if (EndsWith("<id>") || EndsWith("<sha1>") || EndsWith("ISBN")) {
      feature_mask_accumulator_ |= 1u << kIdentifier;
    }
  }

  void FinishClass() {
    if (current_.template_markers) feature_mask_accumulator_ |= 1u << kTemplate;
    if (current_.table_markers) feature_mask_accumulator_ |= 1u << kTable;
    if (current_.reference_markers) feature_mask_accumulator_ |= 1u << kReference;
    if (current_.url_markers) feature_mask_accumulator_ |= 1u << kUrl;
    if (current_.list_lines) feature_mask_accumulator_ |= 1u << kList;
    if (current_.xml_tags) feature_mask_accumulator_ |= 1u << kXml;
    if (current_.letter_bytes) feature_mask_accumulator_ |= 1u << kProse;
    if (current_.digit_bytes) feature_mask_accumulator_ |= 1u << kNumber;
    current_.feature_mask = feature_mask_accumulator_;

    const std::uint64_t decoded_length =
        current_.decoded_end - current_.decoded_start;
    std::array<std::uint64_t, 11> score{};
    score[kProse] = current_.letter_bytes / 8u;
    score[kXml] = static_cast<std::uint64_t>(current_.xml_tags) * 16u;
    score[kNumber] = current_.digit_bytes > decoded_length / 5u
        ? current_.digit_bytes / 2u : current_.digit_bytes / 16u;
    score[kDate] = (feature_mask_accumulator_ & (1u << kDate))
        ? 384u + current_.digit_bytes / 4u : 0u;
    score[kTable] = static_cast<std::uint64_t>(current_.table_markers) * 192u;
    score[kReference] =
        static_cast<std::uint64_t>(current_.reference_markers) * 160u;
    score[kUrl] = static_cast<std::uint64_t>(current_.url_markers) * 192u;
    score[kIdentifier] = (feature_mask_accumulator_ & (1u << kIdentifier))
        ? 320u + current_.digit_bytes / 8u : 0u;
    score[kTemplate] =
        static_cast<std::uint64_t>(current_.template_markers) * 128u;
    score[kList] = static_cast<std::uint64_t>(current_.list_lines) * 96u;
    score[kMixed] = 1;
    current_.stream_class = static_cast<std::uint8_t>(
        std::max_element(score.begin(), score.end()) - score.begin());
    feature_mask_accumulator_ = 0;
  }

  bool EndsWith(const char* pattern) const {
    const std::size_t length = std::strlen(pattern);
    if (length > count_) return false;
    for (std::size_t i = 0; i < length; ++i) {
      const std::size_t index =
          (next_ + bytes_.size() - length + i) % bytes_.size();
      if (bytes_[index] != pattern[i]) return false;
    }
    return true;
  }

  std::uint64_t BeforeFromEnd(std::size_t distance) const {
    const std::size_t index =
        (next_ + bytes_.size() - distance) % bytes_.size();
    return before_[index];
  }

  std::uint64_t AfterFromEnd(std::size_t distance) const {
    const std::size_t index =
        (next_ + bytes_.size() - distance) % bytes_.size();
    return after_[index];
  }

  std::array<char, 32> bytes_{};
  std::array<std::uint64_t, 32> before_{};
  std::array<std::uint64_t, 32> after_{};
  std::size_t next_ = 0;
  std::size_t count_ = 0;
  std::uint64_t decoded_offset_ = 0;
  bool open_ = false;
  bool line_start_ = true;
  std::uint32_t feature_mask_accumulator_ = 0;
  Page current_;
  std::vector<Page> pages_;
};

bool ReadLength(FILE* input, std::uint32_t* length) {
  std::uint32_t value = 0;
  for (unsigned int i = 0; i < 4; ++i) {
    const int byte = std::getc(input);
    if (byte == EOF) return false;
    value = (value << 8) | static_cast<unsigned int>(byte);
  }
  *length = value;
  return true;
}

bool ScanPages(const std::string& dictionary_path,
    const std::string& stream_path, BoundaryScanner* scanner) {
  FILE* dictionary_file = std::fopen(dictionary_path.c_str(), "rb");
  FILE* stream = std::fopen(stream_path.c_str(), "rb");
  if (!dictionary_file || !stream) {
    if (dictionary_file) std::fclose(dictionary_file);
    if (stream) std::fclose(stream);
    return false;
  }
  preprocessor::Dictionary dictionary(dictionary_file, false, true);
  std::uint64_t segment_index = 0;
  while (true) {
    const int type = std::getc(stream);
    if (type == EOF) break;
    std::uint32_t decoded_length = 0;
    if (!ReadLength(stream, &decoded_length)) {
      std::cerr << "truncated segment header at segment "
                << segment_index << '\n';
      std::fclose(stream);
      std::fclose(dictionary_file);
      return false;
    }
    bool use_dictionary = false;
    if (type == kTextSegment) {
      const int reset = std::getc(stream);
      if (reset == EOF) {
        std::cerr << "missing text reset byte at segment "
                  << segment_index << '\n';
        std::fclose(stream);
        std::fclose(dictionary_file);
        return false;
      }
      use_dictionary = reset != 0;
    }
    for (std::uint32_t i = 0; i < decoded_length; ++i) {
      const long long source_before = static_cast<long long>(ftello(stream));
      const int decoded = use_dictionary
          ? static_cast<int>(dictionary.Decode(stream)) : std::getc(stream);
      const long long source_after = static_cast<long long>(ftello(stream));
      if (decoded == EOF || source_before < 0 || source_after < 0) {
        std::cerr << "decode failure at segment " << segment_index
                  << " byte " << i << " type " << type
                  << " source " << source_before << '\n';
        std::fclose(stream);
        std::fclose(dictionary_file);
        return false;
      }
      scanner->Add(static_cast<unsigned char>(decoded),
          static_cast<std::uint64_t>(source_before),
          static_cast<std::uint64_t>(source_after));
    }
    ++segment_index;
  }
  const long long final_source = static_cast<long long>(ftello(stream));
  if (final_source >= 0) scanner->Finish(final_source);
  std::fclose(stream);
  std::fclose(dictionary_file);
  std::cerr << "scan segments=" << segment_index
            << " source_end=" << final_source
            << " pages=" << scanner->pages().size()
            << " page_open=" << (scanner->page_open() ? 1 : 0) << '\n';
  return !scanner->page_open();
}

bool WritePages(const std::string& path, const std::vector<Page>& pages) {
  std::ofstream output(path, std::ios::trunc);
  if (!output.is_open()) return false;
  output << "page_id,decoded_start,decoded_end,decoded_length,"
            "pre_r1_start,pre_r1_end,pre_r1_length,"
            "post_r1_start,post_r1_end,post_r1_length,mapped,complete,"
            "stream_class,class_name,feature_mask,letter_bytes,digit_bytes,"
            "xml_tags,template_markers,table_markers,reference_markers,"
            "url_markers,list_lines\n";
  for (std::size_t i = 0; i < pages.size(); ++i) {
    const Page& page = pages[i];
    output << i << ',' << page.decoded_start << ',' << page.decoded_end << ','
        << page.decoded_end - page.decoded_start << ','
        << page.pre_start << ',' << page.pre_end << ','
        << page.pre_end - page.pre_start << ','
        << page.post_start << ',' << page.post_end << ','
        << (page.mapped ? page.post_end - page.post_start : 0) << ','
        << (page.mapped ? 1 : 0) << ','
        << (page.complete ? 1 : 0) << ','
        << static_cast<unsigned int>(page.stream_class) << ','
        << StreamClassName(page.stream_class) << ',' << page.feature_mask << ','
        << page.letter_bytes << ',' << page.digit_bytes << ',' << page.xml_tags
        << ',' << page.template_markers << ',' << page.table_markers << ','
        << page.reference_markers << ',' << page.url_markers << ','
        << page.list_lines << '\n';
  }
  return output.good();
}

bool WritePageRecipients(
    const std::string& path, const std::vector<Page>& pages) {
  std::ofstream output(path, std::ios::trunc);
  if (!output.is_open()) return false;
  output << "pack_id,first_page,page_count,post_r1_start,post_r1_end,"
            "post_r1_length,decoded_length,stream_class,class_name,"
            "feature_mask\n";
  std::vector<std::size_t> order;
  order.reserve(pages.size());
  for (std::size_t page_id = 0; page_id < pages.size(); ++page_id) {
    const Page& page = pages[page_id];
    if (page.mapped && page.complete && page.post_end > page.post_start) {
      order.push_back(page_id);
    }
  }
  std::stable_sort(order.begin(), order.end(),
      [&](std::size_t left, std::size_t right) {
        if (pages[left].post_start != pages[right].post_start) {
          return pages[left].post_start < pages[right].post_start;
        }
        return left < right;
      });
  std::uint64_t previous_end = 0;
  for (std::size_t recipient = 0; recipient < order.size(); ++recipient) {
    const std::size_t page_id = order[recipient];
    const Page& page = pages[page_id];
    if (page.post_start < previous_end) return false;
    output << recipient << ',' << page_id << ",1," << page.post_start << ','
        << page.post_end << ',' << page.post_end - page.post_start << ','
        << page.decoded_end - page.decoded_start << ','
        << static_cast<unsigned int>(page.stream_class) << ','
        << StreamClassName(page.stream_class) << ',' << page.feature_mask
        << '\n';
    previous_end = page.post_end;
  }
  return output.good();
}

bool WritePacks(const std::string& path, const std::vector<Page>& pages,
    std::uint64_t target_size, std::size_t* pack_count) {
  std::ofstream output(path, std::ios::trunc);
  if (!output.is_open()) return false;
  output << "pack_id,first_page,page_count,post_r1_start,post_r1_end,"
            "post_r1_length,decoded_length\n";

  // R1 changes page order. Build packs in the mapped entropy-stream order,
  // while retaining the original page id for traceability.
  std::vector<std::size_t> order;
  order.reserve(pages.size());
  for (std::size_t page_id = 0; page_id < pages.size(); ++page_id) {
    const Page& page = pages[page_id];
    if (page.mapped && page.complete && page.post_end > page.post_start) {
      order.push_back(page_id);
    }
  }
  std::stable_sort(order.begin(), order.end(),
      [&](std::size_t left, std::size_t right) {
        if (pages[left].post_start != pages[right].post_start) {
          return pages[left].post_start < pages[right].post_start;
        }
        if (pages[left].post_end != pages[right].post_end) {
          return pages[left].post_end < pages[right].post_end;
        }
        return left < right;
      });

  std::size_t id = 0;
  std::size_t begin = 0;
  while (begin < order.size()) {
    const std::size_t first_page = order[begin];
    std::size_t end = begin + 1;
    std::uint64_t post_end = pages[first_page].post_end;
    std::uint64_t decoded_length =
        pages[first_page].decoded_end - pages[first_page].decoded_start;
    while (end < order.size()) {
      const Page& next = pages[order[end]];
      if (next.post_start < post_end) return false;
      const std::uint64_t proposed =
          next.post_end - pages[first_page].post_start;
      if (proposed > target_size && end > begin) break;
      post_end = next.post_end;
      decoded_length += next.decoded_end - next.decoded_start;
      ++end;
    }
    output << id++ << ',' << first_page << ',' << end - begin << ','
        << pages[first_page].post_start << ',' << post_end << ','
        << post_end - pages[first_page].post_start << ','
        << decoded_length << '\n';
    begin = end;
  }
  *pack_count = id;
  return output.good();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    std::cerr << "usage: postr1_page_index DICTIONARY PRE_R1_STREAM "
                 "R1_MAP OUT_PREFIX [TARGET_PACK_BYTES]\n";
    return 2;
  }
  std::uint64_t target_size = 262144;
  if (argc == 6) {
    char* end = nullptr;
    target_size = std::strtoull(argv[5], &end, 10);
    if (!end || *end != '\0' || target_size == 0) {
      std::cerr << "invalid target pack size\n";
      return 2;
    }
  }

  BoundaryScanner scanner;
  if (!ScanPages(argv[1], argv[2], &scanner)) {
    std::cerr << "WRT page scan failed\n";
    return 1;
  }

  std::vector<MapSegment> map;
  std::uint64_t post_wrt_size = 0;
  std::uint64_t post_r1_size = 0;
  if (!ReadMap(argv[3], &map, &post_wrt_size, &post_r1_size)) {
    std::cerr << "R1 map read failed\n";
    return 1;
  }

  std::vector<Page> pages = scanner.pages();
  std::size_t mapped_pages = 0;
  for (Page& page : pages) {
    page.mapped = MapRange(map, page.pre_start, page.pre_end,
        &page.post_start, &page.post_end);
    if (page.mapped) ++mapped_pages;
  }
  if (!WritePages(std::string(argv[4]) + ".pages.csv", pages)) {
    std::cerr << "page table write failed\n";
    return 1;
  }
  if (!WritePageRecipients(
          std::string(argv[4]) + ".page_recipients.csv", pages)) {
    std::cerr << "page recipient table write failed\n";
    return 1;
  }
  std::size_t pack_count = 0;
  if (!WritePacks(std::string(argv[4]) + ".packs.csv", pages,
      target_size, &pack_count)) {
    std::cerr << "pack table write failed\n";
    return 1;
  }

  std::cout << "decoded_bytes=" << scanner.decoded_size() << '\n'
            << "post_wrt_bytes=" << post_wrt_size << '\n'
            << "post_r1_bytes=" << post_r1_size << '\n'
            << "pages=" << pages.size() << '\n'
            << "mapped_pages=" << mapped_pages << '\n'
            << "packs=" << pack_count << '\n'
            << "target_pack_bytes=" << target_size << '\n';
  return pages.size() == 243426 && mapped_pages != 0 ? 0 : 1;
}


#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <zstd.h>

enum {
  BUFFER_SIZE = 1024 * 1024,
  TRAILER_SIZE = 32,
};

static const unsigned char trailer_magic[8] = {
    'H', 'P', 'J', 'E', 'X', '0', '0', '1',
};
static const char *partial_output;

static void remove_partial_output(void) {
  if (partial_output != NULL) unlink(partial_output);
}

static void fail(const char *message) {
  fprintf(stderr, "example-codec: %s\n", message);
  remove_partial_output();
  exit(1);
}

static void fail_errno(const char *operation) {
  fprintf(stderr, "example-codec: %s: %s\n", operation, strerror(errno));
  remove_partial_output();
  exit(1);
}

static void check_zstd(size_t result, const char *operation) {
  if (ZSTD_isError(result)) {
    fprintf(stderr, "example-codec: %s: %s\n", operation,
            ZSTD_getErrorName(result));
    remove_partial_output();
    exit(1);
  }
}

static void put_u64_le(unsigned char *destination, uint64_t value) {
  unsigned int index;
  for (index = 0; index < 8; ++index) {
    destination[index] = (unsigned char)(value & 0xffU);
    value >>= 8;
  }
}

static uint64_t get_u64_le(const unsigned char *source) {
  uint64_t value = 0;
  int index;
  for (index = 7; index >= 0; --index) {
    value = (value << 8) | source[index];
  }
  return value;
}

static void write_all(FILE *output, const void *data, size_t size,
                      const char *operation) {
  if (size != 0 && fwrite(data, 1, size, output) != size) {
    fail_errno(operation);
  }
}

static uint64_t copy_file(FILE *input, FILE *output, unsigned char *buffer) {
  uint64_t total = 0;
  for (;;) {
    size_t count = fread(buffer, 1, BUFFER_SIZE, input);
    if (count != 0) {
      write_all(output, buffer, count, "write executable prefix");
      total += count;
    }
    if (count != BUFFER_SIZE) {
      if (ferror(input)) fail_errno("read executable prefix");
      break;
    }
  }
  return total;
}

static void compress_archive(const char *self_path, const char *input_path,
                             const char *output_path) {
  FILE *self = NULL;
  FILE *input = NULL;
  FILE *output = NULL;
  ZSTD_CCtx *context = NULL;
  unsigned char *input_buffer = NULL;
  unsigned char *output_buffer = NULL;
  unsigned char trailer[TRAILER_SIZE];
  struct stat input_status;
  uint64_t prefix_size;
  uint64_t input_size = 0;
  uint64_t payload_size;
  off_t payload_end;
  int output_fd;

  self = fopen(self_path, "rb");
  if (self == NULL) fail_errno("open compressor executable");
  input = fopen(input_path, "rb");
  if (input == NULL) fail_errno("open input");
  if (fstat(fileno(input), &input_status) != 0) fail_errno("stat input");
  if (!S_ISREG(input_status.st_mode) || input_status.st_size < 0) {
    fail("input must be a regular file");
  }

  output_fd = open(output_path, O_WRONLY | O_CREAT | O_EXCL, 0755);
  if (output_fd < 0) fail_errno("create archive");
  partial_output = output_path;
  output = fdopen(output_fd, "wb");
  if (output == NULL) fail_errno("open archive stream");

  input_buffer = malloc(BUFFER_SIZE);
  output_buffer = malloc(BUFFER_SIZE);
  if (input_buffer == NULL || output_buffer == NULL) fail("out of memory");

  prefix_size = copy_file(self, output, input_buffer);
  if (prefix_size == 0) fail("compressor executable is empty");
  if (fclose(self) != 0) fail_errno("close compressor executable");
  self = NULL;

  context = ZSTD_createCCtx();
  if (context == NULL) fail("cannot create Zstandard compression context");
  check_zstd(ZSTD_CCtx_setParameter(context, ZSTD_c_compressionLevel, 1),
             "set compression level");
  check_zstd(ZSTD_CCtx_setParameter(context, ZSTD_c_checksumFlag, 1),
             "enable frame checksum");
  check_zstd(ZSTD_CCtx_setPledgedSrcSize(
                 context, (unsigned long long)input_status.st_size),
             "set input size");

  for (;;) {
    size_t count = fread(input_buffer, 1, BUFFER_SIZE, input);
    int at_end = count != BUFFER_SIZE;
    ZSTD_inBuffer zstd_input = {input_buffer, count, 0};
    size_t remaining;

    if (at_end && ferror(input)) fail_errno("read input");
    input_size += count;
    do {
      ZSTD_outBuffer zstd_output = {output_buffer, BUFFER_SIZE, 0};
      remaining = ZSTD_compressStream2(
          context, &zstd_output, &zstd_input,
          at_end ? ZSTD_e_end : ZSTD_e_continue);
      check_zstd(remaining, "compress input");
      write_all(output, output_buffer, zstd_output.pos,
                "write compressed payload");
    } while (zstd_input.pos != zstd_input.size || (at_end && remaining != 0));

    if (at_end) break;
  }

  if (input_size != (uint64_t)input_status.st_size) {
    fail("input size changed during compression");
  }
  if (fclose(input) != 0) fail_errno("close input");
  input = NULL;
  ZSTD_freeCCtx(context);
  context = NULL;

  payload_end = ftello(output);
  if (payload_end < 0 || (uint64_t)payload_end < prefix_size) {
    fail_errno("measure compressed payload");
  }
  payload_size = (uint64_t)payload_end - prefix_size;

  memcpy(trailer, trailer_magic, sizeof(trailer_magic));
  put_u64_le(trailer + 8, prefix_size);
  put_u64_le(trailer + 16, input_size);
  put_u64_le(trailer + 24, payload_size);
  write_all(output, trailer, sizeof(trailer), "write archive trailer");
  if (fflush(output) != 0) fail_errno("flush archive");
  if (fchmod(fileno(output), 0555) != 0) fail_errno("set archive mode");
  if (fclose(output) != 0) fail_errno("close archive");
  output = NULL;

  free(input_buffer);
  free(output_buffer);
  partial_output = NULL;
}

static void decompress_archive(const char *self_path) {
  const char *output_path = "data9";
  FILE *self = NULL;
  FILE *output = NULL;
  ZSTD_DCtx *context = NULL;
  unsigned char *input_buffer = NULL;
  unsigned char *output_buffer = NULL;
  unsigned char trailer[TRAILER_SIZE];
  struct stat self_status;
  uint64_t prefix_size;
  uint64_t expected_size;
  uint64_t payload_size;
  uint64_t payload_remaining;
  uint64_t output_size = 0;
  size_t frame_remaining = 1;
  int output_fd;

  self = fopen(self_path, "rb");
  if (self == NULL) fail_errno("open archive executable");
  if (fstat(fileno(self), &self_status) != 0) fail_errno("stat archive");
  if (!S_ISREG(self_status.st_mode) || self_status.st_size < TRAILER_SIZE) {
    fail("archive executable is too small");
  }
  if (fseeko(self, self_status.st_size - TRAILER_SIZE, SEEK_SET) != 0) {
    fail_errno("seek archive trailer");
  }
  if (fread(trailer, 1, sizeof(trailer), self) != sizeof(trailer)) {
    fail_errno("read archive trailer");
  }
  if (memcmp(trailer, trailer_magic, sizeof(trailer_magic)) != 0) {
    fail("archive trailer magic is absent");
  }

  prefix_size = get_u64_le(trailer + 8);
  expected_size = get_u64_le(trailer + 16);
  payload_size = get_u64_le(trailer + 24);
  if (prefix_size == 0 || payload_size == 0 ||
      prefix_size > (uint64_t)self_status.st_size ||
      payload_size > (uint64_t)self_status.st_size ||
      prefix_size + payload_size + TRAILER_SIZE !=
          (uint64_t)self_status.st_size) {
    fail("archive trailer contains invalid sizes");
  }
  if (fseeko(self, (off_t)prefix_size, SEEK_SET) != 0) {
    fail_errno("seek compressed payload");
  }

  output_fd = open(output_path, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (output_fd < 0) fail_errno("create decompressed output");
  partial_output = output_path;
  output = fdopen(output_fd, "wb");
  if (output == NULL) fail_errno("open decompressed output stream");

  input_buffer = malloc(BUFFER_SIZE);
  output_buffer = malloc(BUFFER_SIZE);
  if (input_buffer == NULL || output_buffer == NULL) fail("out of memory");
  context = ZSTD_createDCtx();
  if (context == NULL) fail("cannot create Zstandard decompression context");
  check_zstd(ZSTD_initDStream(context), "initialize decompressor");

  payload_remaining = payload_size;
  while (payload_remaining != 0) {
    size_t requested = payload_remaining < BUFFER_SIZE
                           ? (size_t)payload_remaining
                           : (size_t)BUFFER_SIZE;
    size_t count = fread(input_buffer, 1, requested, self);
    ZSTD_inBuffer zstd_input = {input_buffer, count, 0};
    if (count != requested) fail_errno("read compressed payload");
    payload_remaining -= count;

    while (zstd_input.pos != zstd_input.size) {
      ZSTD_outBuffer zstd_output = {output_buffer, BUFFER_SIZE, 0};
      frame_remaining =
          ZSTD_decompressStream(context, &zstd_output, &zstd_input);
      check_zstd(frame_remaining, "decompress payload");
      if (output_size > expected_size ||
          zstd_output.pos > expected_size - output_size) {
        fail("decompressed output exceeds declared size");
      }
      write_all(output, output_buffer, zstd_output.pos,
                "write decompressed output");
      output_size += zstd_output.pos;
      if (frame_remaining == 0 &&
          (zstd_input.pos != zstd_input.size || payload_remaining != 0)) {
        fail("compressed frame ends before the declared payload");
      }
    }
  }

  if (frame_remaining != 0) fail("compressed frame is incomplete");
  if (output_size != expected_size) fail("decompressed output has wrong size");
  if (fclose(self) != 0) fail_errno("close archive");
  self = NULL;
  ZSTD_freeDCtx(context);
  context = NULL;
  if (fflush(output) != 0) fail_errno("flush decompressed output");
  if (fclose(output) != 0) fail_errno("close decompressed output");
  output = NULL;

  free(input_buffer);
  free(output_buffer);
  partial_output = NULL;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    decompress_archive(argv[0]);
    return 0;
  }
  if (argc == 4 && strcmp(argv[1], "-e") == 0) {
    compress_archive(argv[0], argv[2], argv[3]);
    return 0;
  }
  fprintf(stderr, "usage: %s [-e INPUT ARCHIVE]\n", argv[0]);
  return 2;
}

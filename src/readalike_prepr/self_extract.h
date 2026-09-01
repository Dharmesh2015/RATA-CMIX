#ifndef SELF_EXTRACT_H 
#define SELF_EXTRACT_H 

#include <stdlib.h>
#include <malloc.h>

#include <string>

struct HeaderInfo {
  int dict_size;
  int new_article_order_size;
  int decomp_input_size;
#ifdef KH_ISBN_FOLD
  // Size (bytes) of the ISBN fold escape side-stream carried in archive9,
  // between the cmix payload and the trailing HeaderInfo. Set at -e time.
  int isbn_side_size;
#endif
#ifdef KH_BITLSTM32_ARCHIVE
  // Size (bytes) of the raw bitlstm32 head weight blob carried in archive9
  // (single-binary shape: the blob rides as an appended asset x1 instead of
  // being embedded in the binary, which would ship twice -- in the packaged
  // compressor AND in archive9's .decomp_bin). Set at -e time from the
  // KH_BITLSTM32 env blob. Extracted to .head_blob_decomp at decode time;
  // the Decoder uses that file as its default weight source when the env
  // var is unset (bare judge decode).
  int head_blob_size;
  // The compressor also needs the same head. Keep this separate from
  // head_blob_size, which describes the copy stored in archive9 for decode.
  int s1_head_blob_size;
#endif
#ifdef KH_OBIAS_ARCHIVE
  // Size (bytes) of the raw obias upstream weight blob carried in archive9
  // (same single-binary shape as head_blob_size; rides immediately after the
  // head blob, before the trailing HeaderInfo). Set at -e time from the
  // KH_OBIAS env blob. Extracted to .obias_blob_decomp at decode time; the
  // Predictor uses that file as its default weight source when the env var
  // is unset (bare judge decode). 0 = obias-less encode.
  int obias_blob_size;
#endif
#ifdef KH_RESIDUAL_LSTM96_ARCHIVE
  // Compact row-quantized byte-step LSTM-96 model. It is extracted before
  // the dictionary helper decode, because that decode uses the same terminal
  // probability path as the main payload.
  int residual_lstm96_blob_size;
#endif
#ifdef KH_TRANSFORMER6M_ARCHIVE
  // Losslessly compressed FX2TFWC2 transformer model. It is appended as-is
  // to S1 and archive9, then loaded directly by the CPU inference path.
  int transformer6m_weights_size;
#endif
};

void write(const std::string& file_name, HeaderInfo& data) {
  FILE *out = fopen(file_name.c_str() , "wb" );
  fwrite(&data , 1 , sizeof(HeaderInfo) , out );
  fclose(out);
}

void read(const std::string& file_name, HeaderInfo& data) {
  FILE *in = fopen(file_name.c_str() , "rb" );
  fread(&data , 1 , sizeof(HeaderInfo) , in );
  fclose(in);
}


// This function splits the ./cmix binary file into 3 parts:
// 1) actual compressor/decompressor binary
// 2) dictionary (get's it in compressed form and decompresses it)
// 3) new order of articles (get's it in compressed form and decompresses it) 
int selfextract_comp() {
  HeaderInfo header;

// open itslef to read auxilary data (dictionary and neworder)
  FILE *f = NULL, *fo = NULL;
  f = fopen("cmix", "rb");
  if (f == NULL) {
    perror("selfextract failed to open ./cmix");
    return 1;
  }

  // get the size of the whole binary
  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  unsigned char *p1 = (unsigned char *)malloc(fsize);
  // read the whole binary to memory
  fread(p1, fsize, 1, f);
  fclose(f);

  // read header info
  fo = fopen("test.dat", "wb");
  memcpy(&header, p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo));
  fwrite(p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo), 1, fo);
  fclose(fo);

  //Remove dictionary if present
  remove(".dict");
  
  size_t transformer6m_weights_size = 0;
#ifdef KH_TRANSFORMER6M_ARCHIVE
  transformer6m_weights_size =
      static_cast<size_t>(header.transformer6m_weights_size);
#endif
  size_t s1_head_blob_size = 0;
#ifdef KH_BITLSTM32_ARCHIVE
  s1_head_blob_size = static_cast<size_t>(header.s1_head_blob_size);
#endif
  size_t decmpressor_binary_size = fsize - header.dict_size -
      header.new_article_order_size - transformer6m_weights_size -
      s1_head_blob_size -
      sizeof(HeaderInfo);

// produce actual decompressor binary
  fo = fopen(".decomp_bin", "wb");
  fwrite(p1, decmpressor_binary_size, 1, fo);
  fclose(fo);

  // KH_DECODE_ONLY wire-up (exp_split_decoder / win4): if a purpose-built
  // decode-only binary has been staged next to "cmix", embed that instead of
  // the self-sliced full binary -- archive9 only ever runs its decode/self-
  // extract path, so it does not need the encoder-only code. This is the only
  // change that ships inside the full `cmix` (~+104 B, paid x1); the archive9
  // copy of .decomp_bin shrinks by ~10+ KB (paid x1 there too, net win).
  if (FILE* alt = fopen("decode_only_bin", "rb")) {
    fclose(alt);
    system("cp -f decode_only_bin .decomp_bin");
  }

// produce dictionary and decompress it
  fo = fopen(".dict.comp", "wb");
  fwrite(p1 + decmpressor_binary_size, header.dict_size, 1, fo);
  fclose(fo);


// produce article order and decompress it
  fo = fopen(".new_article_order.comp", "wb");
  fwrite(p1 + decmpressor_binary_size + header.dict_size, header.new_article_order_size, 1, fo);
  fclose(fo);

#ifdef KH_TRANSFORMER6M_ARCHIVE
  fo = fopen(".tfweights", "wb");
  fwrite(p1 + decmpressor_binary_size + header.dict_size +
             header.new_article_order_size,
      transformer6m_weights_size, 1, fo);
  fclose(fo);
#endif

#ifdef KH_BITLSTM32_ARCHIVE
  if (s1_head_blob_size == 0) {
    fprintf(stderr, "S1 is missing the BitLSTM32 model\n");
    return 1;
  }
  fo = fopen(".head_blob_s1", "wb");
  fwrite(p1 + decmpressor_binary_size + header.dict_size +
             header.new_article_order_size + transformer6m_weights_size,
      s1_head_blob_size, 1, fo);
  fclose(fo);
  // Helper-stream decoders and the main encoder inherit this exact model.
  setenv("KH_BITLSTM32", ".head_blob_s1", 1);
#endif
//  std::cout << "Decompressing the file with the new article order..." << std::endl;
  // The article-order/dictionary helper streams are small binary side data
  // (article order has on the order of 15 distinct byte values) and cannot
  // satisfy the transformer's 205-symbol vocabulary requirement. system()
  // forks a shell that inherits this process's environment, so a
  // FX4_ENABLE_TRANSFORMER6M=1 set for the main enwik9 payload would
  // otherwise leak into these unrelated helper decodes and hard-fail them
  // (FX4_TRANSFORMER6M_REQUIRED builds exit(2) on an incompatible stream,
  // which system() reports back here as status 512 = 2 << 8).
  unsetenv("FX4_ENABLE_TRANSFORMER6M");
  int status = system("./cmix -d .new_article_order.comp .new_article_order");
  if (status != 0) {
    fprintf(stderr, "selfextract failed: article order decode status=%d\n", status);
    free(p1);
    return 1;
  }

//  std::cout << "Decompressing dictionary..." << std::endl;
  status = system("./cmix -d .dict.comp .dict");
  if (status != 0) {
    fprintf(stderr, "selfextract failed: dictionary decode status=%d\n", status);
    free(p1);
    return 1;
  }
  free(p1);
  malloc_trim(0);
  return 0;
}

// Same as previous function, but used in decompressor
// This function splits the ./cmix binary file into 3 parts:
// 1) actual compressor/decompressor binary
// 2) dictionary (get's it in compressed form and decompresses it)
// 3) new order of articles (get's it in compressed form and decompresses it) 
int selfextract_decomp() {
  HeaderInfo header;
  FILE *f = NULL, *fo = NULL;
  f = fopen("archive9", "rb");

  fseek(f, 0, SEEK_END);
  size_t fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  unsigned char *p1 = (unsigned char *)malloc(fsize);
  fread(p1, fsize, 1, f);
  fclose(f);

  // read header info
  fo = fopen("test.dat", "wb");
  fwrite(p1 + fsize - sizeof(HeaderInfo), sizeof(HeaderInfo), 1, fo);
  fclose(fo);
  read("test.dat", header);

  //Remove dictionary if present
  remove(".dict");

  size_t isbn_side_size = 0;
#ifdef KH_ISBN_FOLD
  isbn_side_size = (size_t)header.isbn_side_size;
#endif
  size_t head_blob_size = 0;
#ifdef KH_BITLSTM32_ARCHIVE
  head_blob_size = (size_t)header.head_blob_size;
#endif
  size_t obias_blob_size = 0;
#ifdef KH_OBIAS_ARCHIVE
  obias_blob_size = (size_t)header.obias_blob_size;
#endif
  size_t residual_lstm96_blob_size = 0;
#ifdef KH_RESIDUAL_LSTM96_ARCHIVE
  residual_lstm96_blob_size = (size_t)header.residual_lstm96_blob_size;
#endif
  size_t transformer6m_weights_size = 0;
#ifdef KH_TRANSFORMER6M_ARCHIVE
  transformer6m_weights_size =
      static_cast<size_t>(header.transformer6m_weights_size);
#endif
  size_t decmpressor_binary_size = fsize - header.dict_size -
      transformer6m_weights_size - header.decomp_input_size -
      isbn_side_size - head_blob_size - obias_blob_size -
      residual_lstm96_blob_size - sizeof(HeaderInfo);
  const size_t transformer_offset =
      decmpressor_binary_size + header.dict_size;
  const size_t payload_offset =
      transformer_offset + transformer6m_weights_size;
  const size_t isbn_offset = payload_offset + header.decomp_input_size;
  const size_t head_offset = isbn_offset + isbn_side_size;
  const size_t obias_offset = head_offset + head_blob_size;
  const size_t residual_lstm96_offset = obias_offset + obias_blob_size;

#ifdef KH_TRANSFORMER6M_ARCHIVE
  // Extract before the helper dictionary decode: that subprocess uses the
  // same Predictor and must load the exact model from its first byte.
  fo = fopen(".tfweights", "wb");
  fwrite(p1 + transformer_offset, transformer6m_weights_size, 1, fo);
  fclose(fo);
#endif

#ifdef KH_BITLSTM32_ARCHIVE
  // Head weight blob rides after the isbn side-stream, before the header.
  // Extract it FIRST: the dictionary decode below runs `./archive9 -d` as a
  // subprocess whose Decoder already needs the default weight source when
  // the env var is unset (bare judge decode).
  fo = fopen(".head_blob_decomp", "wb");
  fwrite(p1 + head_offset, head_blob_size, 1, fo);
  fclose(fo);
#endif

#ifdef KH_RESIDUAL_LSTM96_ARCHIVE
  fo = fopen(".residual_lstm96_blob_decomp", "wb");
  fwrite(p1 + residual_lstm96_offset, residual_lstm96_blob_size, 1, fo);
  fclose(fo);
#endif

#ifdef KH_OBIAS_ARCHIVE
  // Obias upstream blob rides after the head blob, before the header. Same
  // early-extraction rationale: the dictionary decode below runs as a
  // subprocess whose Predictor needs the default weight source when the env
  // var is unset (bare judge decode).
  fo = fopen(".obias_blob_decomp", "wb");
  fwrite(p1 + obias_offset, obias_blob_size, 1, fo);
  fclose(fo);
#endif

  fo = fopen(".dict.comp_decomp", "wb");
  fwrite(p1 + decmpressor_binary_size, header.dict_size, 1, fo);
  fclose(fo);

  // See the matching comment in selfextract_comp(): this helper stream's
  // vocabulary cannot satisfy the transformer's 205-symbol requirement, and
  // system() would otherwise leak an inherited FX4_ENABLE_TRANSFORMER6M=1
  // into it.
  unsetenv("FX4_ENABLE_TRANSFORMER6M");
  int status = system("./archive9 -d .dict.comp_decomp .dict");//_decomp
  if (status != 0) {
    fprintf(stderr, "selfextract failed: dictionary decode status=%d\n", status);
    free(p1);
    return 1;
  }

  fo = fopen(".ready4cmix_decomp", "wb");
  fwrite(p1 + payload_offset, header.decomp_input_size, 1, fo);
  fclose(fo);

#ifdef KH_ISBN_FOLD
  // ISBN fold side-stream rides between the cmix payload and the header.
  fo = fopen(".isbn_side_decomp", "wb");
  fwrite(p1 + isbn_offset, isbn_side_size, 1, fo);
  fclose(fo);
#endif

  free(p1);
  malloc_trim(0);
  return 0;
}

#endif // PREPR_H

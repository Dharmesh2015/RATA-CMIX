
#define COMP_INTRO_END_LINE 29
#define COMP_MAIN_END_LINE  13146932
#define COMP_CODA_END_LINE  13147025

#define DECOMP_MAIN_END_LINE  13146905
#define DECOMP_INTRO_END_LINE 13146934 
#define DECOMP_CODA_END_LINE  13147027

void split4Comp(char const *enwik9_filename) {
  FILE* ifile = fopen(enwik9_filename, "rb");
  FILE* ofile1 = fopen(".intro", "wb");
  FILE* ofile2 = fopen(".main", "wb");
  FILE* ofile3 = fopen(".coda", "wb");  
  int line_count = 0;
  
  do {
    int c=getc(ifile);
    if (c==EOF) break;
    if (line_count < COMP_INTRO_END_LINE) {
      putc(c,ofile1);
    } else if (line_count < COMP_MAIN_END_LINE) {
      putc(c,ofile2);
    } else if (line_count < COMP_CODA_END_LINE) {
      putc(c,ofile3);
    } else {
      putc(c,ofile3);
    }
    if (c==10)
    line_count++;
  }
  while (!feof(ifile));
  fclose(ifile);
  fclose(ofile1);
  fclose(ofile2);
  fclose(ofile3);
}

void split4Decomp() {
  FILE* ifile = fopen(".input_decomp", "rb");
  FILE* ofile1 = fopen(".intro_decomp", "wb");
  FILE* ofile2 = fopen(".main_decomp", "wb");
  FILE* ofile3 = fopen(".coda_decomp", "wb");  
  int line_count = 0;
  // Small-input (sub-enwik9) roundtrip support, ported from the fable_wins
  // harness tree: when the decoded stream has fewer lines than the enwik9
  // main section, split by the actual line count (main = everything except
  // the trailing 29 intro lines; no coda) instead of the enwik9 constants.
  // IMPORTANT: the non-prefix (enwik9) path keeps this tree's original
  // literals 13146906/13146935/13147027 byte-for-byte (the fable_wins tree
  // uses 13146905/13146934 there -- an off-by-one vs this record-proven
  // lineage; do NOT adopt those values).
  int total_lines = 0;
  { int c; while ((c=getc(ifile))!=EOF) if(c==10) total_lines++; rewind(ifile); }
  const int use_prefix = (total_lines < 13146906);
  const int p_main_end = use_prefix ? (total_lines > COMP_INTRO_END_LINE ? total_lines - COMP_INTRO_END_LINE : total_lines) : 13146906;
  const int p_intro_end = use_prefix ? total_lines : 13146935;
  const int p_coda_end = use_prefix ? total_lines : 13147027;

  do {
      int c=getc(ifile);
      if (c==EOF) break;
    if (line_count < p_main_end) {
      putc(c,ofile2);
    } else if (line_count < p_intro_end) {
        putc(c,ofile1);
    } else if (line_count < p_coda_end) {
        putc(c,ofile3);
    } else {
        putc(c,ofile3);
    }
    if (c==10)
    line_count++;
  }
  while (!feof(ifile));
  fclose(ifile);
  fclose(ofile1);
  fclose(ofile2);
  fclose(ofile3);
  
}

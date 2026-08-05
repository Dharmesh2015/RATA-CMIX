#ifndef FX4_DONOR_WINNER_SEARCH_H
#define FX4_DONOR_WINNER_SEARCH_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

class DonorPlan;

bool RunDonorWinnerSearch(
    const std::string& input_path,
    const std::string& scratch_output_path,
    uint64_t input_bytes,
    const std::vector<bool>& vocab,
    FILE* dictionary,
    bool pretrain_dictionary,
    DonorPlan* donor_plan,
    const std::string& ledger_path,
    uint64_t* output_bytes);

#endif

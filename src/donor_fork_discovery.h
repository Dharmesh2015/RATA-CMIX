#ifndef FX4_DONOR_FORK_DISCOVERY_H
#define FX4_DONOR_FORK_DISCOVERY_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

class DonorPlan;

bool RunDonorForkDiscovery(
    const std::string& input_path,
    const std::string& scratch_output_path,
    uint64_t input_bytes,
    const std::vector<bool>& vocab,
    FILE* dictionary,
    bool pretrain_dictionary,
    bool enable_transformer6m,
    DonorPlan* donor_plan,
    uint64_t* output_bytes);

bool DonorForkDiscoveryCompleted();

#endif

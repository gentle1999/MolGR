#pragma once

#include "molgr/utils/utils.h"

#include <map>
#include <string>

namespace molgr::equivalence
{
    struct Options
    {
        bool use_chirality = false;
        bool ignore_metal_coordination_bonds = true;
        int max_resonance = 1000;
        int max_mappings = 10000;
        int max_mapping_nodes = 100000;
        int max_resonance_steps = 24;
    };

    struct Summary
    {
        std::string formula, heavy_formula;
        int atom_count = 0, hydrogen_count = 0;
        int charge = 0, radicals = 0;
        int organic_atoms = 0, organic_charge = 0, organic_radicals = 0;
        int known_fields = 0, deferred_centers = 0;
        std::vector<std::pair<int, int>> metal_states;
    };

    struct Move
    {
        std::string kind;
        int a = -1, b = -1, c = -1; // zero-based indices in the first input
        int side = 0; // 0: left -> meeting state; 1: right -> meeting state
        std::vector<int> cycle; // closed alternating trail; c is its first bond delta
    };

    struct Result
    {
        std::string decision = "inconclusive", method, reason;
        // 1 = passed, 0 = not evaluated, -1 = failed.
        std::map<std::string, int> invariants;
        Summary left, right;
        std::vector<int> mapping;
        std::vector<Move> witness;
        int mappings = 0, mapping_nodes = 0;
        int states_left = 0, states_right = 0, regions = 0, pruned_moves = 0;
        int resonance_cache_hits = 0, region_cache_hits = 0, region_constraint_rejections = 0;
        int comparable_fields = 0;
        bool mapping_limit = false, resonance_limit = false;
        bool resonance_attempted = false, validation_failed = false;
        std::string stop_reason = "not_attempted";
        double preparation_ms = 0, mapping_ms = 0, resonance_ms = 0;
    };

    // Pure native graph/electron kernel. No toolkit objects or callbacks inside BFS.
    Result Evaluate(const utils::MoleculeData &left, const utils::MoleculeData &right,
                    const Options &options = {});

    // Compare stereo under a caller-supplied zero-based bijection, without
    // resonance search or a different symmetry mapping hiding information loss.
    int CompareMappedStereo(const utils::MoleculeData &left, const utils::MoleculeData &right,
                            const std::vector<int> &mapping);
}

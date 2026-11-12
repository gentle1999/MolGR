#include "bindings.h"
#include "molgr/utils/equivalence.h"

namespace
{
    py::dict Summary(const molgr::equivalence::Summary &s)
    {
        py::dict out;
        out["formula"] = s.formula; out["heavy_formula"] = s.heavy_formula;
        out["atom_count"] = s.atom_count; out["hydrogen_count"] = s.hydrogen_count;
        out["charge"] = s.charge; out["radicals"] = s.radicals;
        out["organic_atoms"] = s.organic_atoms; out["organic_charge"] = s.organic_charge;
        out["organic_radicals"] = s.organic_radicals; out["metal_states"] = s.metal_states;
        out["known_fields"] = s.known_fields; out["deferred_centers"] = s.deferred_centers;
        return out;
    }
}

void molgr::bind::bind_equivalence(py::module_ &m)
{
    m.def("compare_mapped_stereo", &equivalence::CompareMappedStereo,
          py::arg("left"), py::arg("right"), py::arg("mapping"),
          "Compare stereo under a fixed zero-based bijection: 1 equal, 0 conflicting, -1 incomplete.");
    m.def("evaluate", [](const utils::MoleculeData &left, const utils::MoleculeData &right,
                         bool use_chirality, int max_resonance, bool ignore_coordination,
                         int max_mappings, int max_mapping_nodes, int max_steps)
    {
        equivalence::Options options;
        options.use_chirality = use_chirality;
        options.max_resonance = max_resonance;
        options.ignore_metal_coordination_bonds = ignore_coordination;
        options.max_mappings = max_mappings;
        options.max_mapping_nodes = max_mapping_nodes;
        options.max_resonance_steps = max_steps;
        equivalence::Result result;
        {
            py::gil_scoped_release release;
            result = equivalence::Evaluate(left, right, options);
        }
        py::dict out;
        out["decision"] = result.decision; out["method"] = result.method;
        out["reason"] = result.reason; out["invariants"] = result.invariants;
        out["left"] = Summary(result.left); out["right"] = Summary(result.right);
        out["mapping"] = result.mapping;
        py::list witness;
        for (const auto &move : result.witness)
        {
            py::dict item;
            item["kind"] = move.kind; item["atoms"] = py::make_tuple(move.a, move.b, move.c);
            item["side"] = move.side; witness.append(item);
            if (!move.cycle.empty()) item["cycle"] = move.cycle;
        }
        out["witness"] = witness;
        out["mappings"] = result.mappings; out["mapping_nodes"] = result.mapping_nodes;
        out["mapping_limit"] = result.mapping_limit;
        out["states_left"] = result.states_left; out["states_right"] = result.states_right;
        out["regions"] = result.regions; out["pruned_moves"] = result.pruned_moves;
        out["resonance_cache_hits"] = result.resonance_cache_hits;
        out["region_cache_hits"] = result.region_cache_hits;
        out["region_constraint_rejections"] = result.region_constraint_rejections;
        out["comparable_fields"] = result.comparable_fields;
        out["resonance_limit"] = result.resonance_limit;
        out["resonance_attempted"] = result.resonance_attempted;
        out["validation_failed"] = result.validation_failed;
        out["stop_reason"] = result.stop_reason;
        out["preparation_ms"] = result.preparation_ms;
        out["mapping_ms"] = result.mapping_ms; out["resonance_ms"] = result.resonance_ms;
        return out;
    }, py::arg("left"), py::arg("right"), py::arg("use_chirality") = false,
       py::arg("max_resonance") = 5000, py::arg("ignore_metal_coordination_bonds") = true,
       py::arg("max_mappings") = 10000, py::arg("max_mapping_nodes") = 100000,
       py::arg("max_resonance_steps") = 24,
       "Compare explicit-H MoleculeData in C++; returns a decision and a joint mapping/witness. The GIL is released during comparison.");
}

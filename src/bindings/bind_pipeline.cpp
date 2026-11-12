#include "bindings.h"

#include "molgr/pipeline/reconstruct_with_metals.h"
#include "molgr/pipeline/reconstruct_without_metals.h"
#include "molgr/pipeline/resonance.h"
#include "molgr/python_config.h"
#include "molgr/stages/break_bond.h"
#include "molgr/stages/clean.h"
#include "molgr/stages/eliminate.h"
#include "molgr/stages/fresh.h"
#include "molgr/stages/preprocess.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/force_field.h"
#include "molgr/utils/metals/scoring.h"
#include "molgr/utils/metals/search.h"
#include "molgr/utils/metals/preparation.h"
#include "molgr/utils/no_metals/neighbor_radicals.h"
#include "molgr/utils/no_metals/preparation.h"
#include "molgr/utils/utils.h"
#include "molgr/vendor/forcefielduff.h"
#include "molgr/vendor/openbabel_conversion.h"
#include "molgr/vendor/openbabel_copy.h"
#include "molgr/vendor/openbabel_threading.h"
#include "molgr/utils/logger.h"

#include <openbabel/bond.h>
#include "molgr/compat/openbabel_iter.h"
#include "molgr/diagnostics.h"
#include "molgr/pipeline/reconstruct_batch.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace molgr
{
    namespace bind
    {
        namespace
        {
            thread_local molgr::diagnostics::ReconstructionDiagnostics
                t_last_reconstruction_diagnostics;

            py::dict reconstruction_diagnostics_signature(
                const molgr::diagnostics::ReconstructionDiagnostics &diagnostics)
            {
                py::dict out;
                if (diagnostics.code.empty())
                {
                    return out;
                }
                out["code"] = diagnostics.code;
                out["stage"] = diagnostics.stage;
                out["backend"] = "cpp";
                out["message"] = diagnostics.message;
                py::dict counts;
                for (const auto &[key, value] : diagnostics.counts)
                {
                    counts[key.c_str()] = value;
                }
                out["counts"] = std::move(counts);
                py::dict details;
                for (const auto &[key, value] : diagnostics.details)
                {
                    details[key.c_str()] = value;
                }
                out["details"] = std::move(details);
                out["cause_type"] = "";
                out["cause_message"] = "";
                return out;
            }

            py::dict reconstruction_batch_result_signature(
                molgr::pipeline::reconstruct_batch::ReconstructionBatchResult result)
            {
                py::dict out;
                out["index"] = result.index;
                if (result.molecule)
                {
                    out["molecule_data"] = py::cast(std::move(result.molecule));
                }
                else
                {
                    out["molecule_data"] = py::none();
                }
                out["diagnostics"] = reconstruction_diagnostics_signature(result.diagnostics);
                return out;
            }

            std::vector<molgr::pipeline::reconstruct_batch::ReconstructionBatchRequest>
            parse_reconstruction_batch_requests(const py::iterable &requests)
            {
                std::vector<molgr::pipeline::reconstruct_batch::ReconstructionBatchRequest> parsed;
                for (const py::handle item : requests)
                {
                    if (!py::isinstance<py::sequence>(item) || py::isinstance<py::str>(item))
                    {
                        throw py::type_error(
                            "batch requests must be sequences of "
                            "(xyz_block, total_charge, total_radical_electrons)");
                    }
                    const auto sequence = py::reinterpret_borrow<py::sequence>(item);
                    if (py::len(sequence) != 3)
                    {
                        throw py::value_error(
                            "each batch request must contain exactly three values: "
                            "xyz_block, total_charge, total_radical_electrons");
                    }
                    molgr::pipeline::reconstruct_batch::ReconstructionBatchRequest request;
                    request.xyz_block = py::cast<std::string>(sequence[0]);
                    request.total_charge = py::cast<int>(sequence[1]);
                    request.total_radical_electrons = py::cast<int>(sequence[2]);
                    parsed.push_back(std::move(request));
                }
                return parsed;
            }

            OpenBabel::OBMol *require_obmol_ptr(intptr_t mol_ptr)
            {
                molgr::WarnUnsafeOpenBabelUse(
                    "molgr.dev.pipeline.*_ptr",
                    "OBMol pointers are not thread-safe; keep ownership and all operations on one thread, and free owned pointers exactly once.");
                if (mol_ptr == 0)
                {
                    throw std::runtime_error("null OBMol pointer");
                }
                return reinterpret_cast<OpenBabel::OBMol *>(mol_ptr);
            }

            py::tuple metal_state_signature(const molgr::metal::MetalAtomPosition &metal_state)
            {
                py::tuple item(8);
                item[0] = metal_state.idx;
                item[1] = metal_state.symbol;
                item[2] = metal_state.element_idx;
                item[3] = metal_state.valence;
                item[4] = metal_state.radical_num;
                item[5] = static_cast<long long>(std::llround(metal_state.position_x * 1000000.0));
                item[6] = static_cast<long long>(std::llround(metal_state.position_y * 1000000.0));
                item[7] = static_cast<long long>(std::llround(metal_state.position_z * 1000000.0));
                return item;
            }

            py::list metal_state_choice_signature(
                const std::vector<molgr::metal::MetalAtomPosition> &choice)
            {
                py::list out;
                for (const auto &metal_state : choice)
                {
                    out.append(metal_state_signature(metal_state));
                }
                return out;
            }

            py::list metal_state_search_group_signature(
                const molgr::metal::search::MetalStateChoiceGroup &group)
            {
                py::list out;
                for (const auto &choice : group)
                {
                    out.append(metal_state_choice_signature(choice));
                }
                return out;
            }

            py::list metal_state_search_layer_signature(
                const molgr::metal::search::MetalStateSearchLayer &layer)
            {
                py::list out;
                for (const auto &group : layer)
                {
                    out.append(metal_state_search_group_signature(group));
                }
                return out;
            }

            py::list phase_history_signature(const std::vector<std::string> &phase_history)
            {
                py::list out;
                for (const auto &phase : phase_history)
                {
                    out.append(phase);
                }
                return out;
            }

            std::unique_ptr<OpenBabel::OBMol> mol_from_smiles(const std::string &smiles)
            {
                auto mol = std::make_unique<OpenBabel::OBMol>();
                if (!molgr::vendor::openbabel_conversion::ReadSmiles(smiles, mol.get()))
                {
                    return nullptr;
                }
                return mol;
            }

            void bind_reconstruct_with_metals_public_ns(py::module_ &ns)
            {
                py::class_<molgr::pipeline::reconstruct_batch::ReconstructionBatchIterator>(
                    ns,
                    "ReconstructionBatchIterator")
                    .def(
                        "__iter__",
                        [](molgr::pipeline::reconstruct_batch::ReconstructionBatchIterator &iterator)
                            -> molgr::pipeline::reconstruct_batch::ReconstructionBatchIterator &
                        { return iterator; },
                        py::return_value_policy::reference_internal)
                    .def(
                        "__next__",
                        [](molgr::pipeline::reconstruct_batch::ReconstructionBatchIterator &iterator)
                        {
                            std::optional<
                                molgr::pipeline::reconstruct_batch::ReconstructionBatchResult>
                                result;
                            {
                                py::gil_scoped_release release;
                                result = iterator.Next();
                            }
                            if (!result.has_value())
                            {
                                throw py::stop_iteration();
                            }
                            return reconstruction_batch_result_signature(std::move(*result));
                        })
                    .def(
                        "close",
                        &molgr::pipeline::reconstruct_batch::ReconstructionBatchIterator::Close);

                ns.def(
                    "xyz2omol",
                    [](const std::string &xyz_block,
                       int total_charge,
                       int total_radical_electrons,
                       py::object config)
                        -> std::unique_ptr<molgr::utils::MoleculeData>
                    {
                        auto runtime_config = molgr::config::FromPython(config);
                        t_last_reconstruction_diagnostics.Reset();
                        py::gil_scoped_release release;
                        auto mol_data = molgr::pipeline::reconstruct_with_metals::Xyz2OmolMolData(
                            xyz_block,
                            total_charge,
                            total_radical_electrons,
                            runtime_config,
                            &t_last_reconstruction_diagnostics);
                        if (!mol_data)
                        {
                            return nullptr;
                        }
                        return std::move(mol_data);
                    },
                    "Reconstruct molecule data from XYZ with metal-aware pipeline.",
                    py::arg("xyz_block"),
                    py::arg("total_charge") = 0,
                    py::arg("total_radical_electrons") = 0,
                    py::kw_only(),
                    py::arg("config") = py::none());

                ns.def(
                    "batch_xyz2omol",
                    [](const py::iterable &requests,
                       py::object config,
                       std::size_t max_workers,
                       std::size_t queue_size,
                       bool ordered)
                        -> std::unique_ptr<
                            molgr::pipeline::reconstruct_batch::ReconstructionBatchIterator>
                    {
                        auto runtime_config = molgr::config::FromPython(config);
                        auto parsed_requests = parse_reconstruction_batch_requests(requests);
                        return std::make_unique<
                            molgr::pipeline::reconstruct_batch::ReconstructionBatchIterator>(
                            std::move(parsed_requests),
                            runtime_config,
                            max_workers,
                            queue_size,
                            ordered);
                    },
                    "Reconstruct a finite batch with a bounded native worker queue.",
                    py::arg("requests"),
                    py::kw_only(),
                    py::arg("config") = py::none(),
                    py::arg("max_workers") = 0,
                    py::arg("queue_size") = 16,
                    py::arg("ordered") = false);
            }

            void bind_reconstruct_with_metals_dev_ns(py::module_ &ns)
            {
                py::class_<molgr::metal::MetalAtomPosition>(ns, "MetalAtomPosition")
                    .def(py::init<>())
                    .def_readwrite("idx", &molgr::metal::MetalAtomPosition::idx)
                    .def_readwrite("symbol", &molgr::metal::MetalAtomPosition::symbol)
                    .def_readwrite("element_idx", &molgr::metal::MetalAtomPosition::element_idx)
                    .def_readwrite("valence", &molgr::metal::MetalAtomPosition::valence)
                    .def_readwrite("radical_num", &molgr::metal::MetalAtomPosition::radical_num)
                    .def_readwrite("position_x", &molgr::metal::MetalAtomPosition::position_x)
                    .def_readwrite("position_y", &molgr::metal::MetalAtomPosition::position_y)
                    .def_readwrite("position_z", &molgr::metal::MetalAtomPosition::position_z)
                    .def("__repr__", [](const molgr::metal::MetalAtomPosition &metal)
                         { return "<MetalAtomPosition " + metal.symbol +
                                   " val=" + std::to_string(metal.valence) +
                                   " rad=" + std::to_string(metal.radical_num) + ">"; },
                         "Return a concise debug representation of MetalAtomPosition.");

                ns.def(
                    "build_metal_states_ptr",
                    [](intptr_t mol_ptr, int atom_idx) -> std::vector<molgr::metal::MetalAtomPosition>
                    {
                        auto *mol = require_obmol_ptr(mol_ptr);
                        auto *atom = mol->GetAtom(atom_idx);
                        if (!atom)
                        {
                            throw std::runtime_error("invalid atom index");
                        }
                        return molgr::metal::preparation::BuildMetalStates(*atom);
                    },
                    "Build candidate metal states for a specific atom index on an OBMol pointer.",
                    py::arg("mol_ptr"),
                    py::arg("atom_idx"),
                    py::call_guard<py::gil_scoped_release>());

                ns.def(
                    "combine_metal_with_omol_ptr",
                    [](intptr_t mol_ptr, const std::vector<molgr::metal::MetalAtomPosition> &metals)
                    {
                        auto *mol = require_obmol_ptr(mol_ptr);
                        molgr::metal::preparation::CombineMetalWithOmol(*mol, metals);
                    },
                    "Combine metal states with an existing OBMol pointer in place.",
                    py::arg("mol_ptr"),
                    py::arg("metals"),
                    py::call_guard<py::gil_scoped_release>());

                ns.def(
                    "debug_metal_search_summaries",
                    [](const std::string &xyz_block,
                       int total_charge,
                       int total_radical_electrons,
                       py::object config)
                    {
                        auto runtime_config = molgr::config::FromPython(config);
                        py::dict out;
                        const auto base_state = molgr::metal::preparation::PrepareMetalState(
                            xyz_block,
                            total_charge,
                            total_radical_electrons,
                            runtime_config);

                        py::list available_states;
                        for (const auto &state_options : base_state.available_valence_radical_states)
                        {
                            py::list state_options_out;
                            for (const auto &metal_state : state_options)
                            {
                                state_options_out.append(metal_state_signature(metal_state));
                            }
                            available_states.append(std::move(state_options_out));
                        }
                        out["available_valence_radical_states"] = std::move(available_states);
                        out["base_phase_history"] = phase_history_signature(base_state.phase_history);

                        const auto state_search_groups =
                            molgr::metal::search::BuildMetalStateSearchGroups(
                                base_state.available_valence_radical_states,
                                runtime_config);
                        py::list state_search_groups_out;
                        for (const auto &group : state_search_groups)
                        {
                            state_search_groups_out.append(metal_state_search_group_signature(group));
                        }
                        out["state_search_groups"] = std::move(state_search_groups_out);

                        const auto layered_state_search_groups =
                            molgr::metal::search::BuildLayeredMetalStateSearchGroups(
                                state_search_groups,
                                total_radical_electrons,
                                runtime_config);
                        py::list layers_out;
                        py::list target_buckets_by_layer;
                        for (const auto &layer : layered_state_search_groups)
                        {
                            layers_out.append(metal_state_search_layer_signature(layer));

                            auto grouped_candidates =
                                molgr::metal::search::GroupCandidatesByTargetDp(
                                    base_state.phase_history,
                                    layer,
                                    total_charge,
                                    total_radical_electrons,
                                    runtime_config);
                            py::list layer_buckets;
                            for (const auto &target_entry : grouped_candidates)
                            {
                                py::dict bucket;
                                bucket["target"] = py::make_tuple(
                                    target_entry.first.no_metal_charge,
                                    target_entry.first.no_metal_radicals);
                                py::list candidates_out;
                                for (const auto &candidate : target_entry.second)
                                {
                                    py::dict candidate_out;
                                    candidate_out["combination_index"] =
                                        molgr::metal::scoring::CandidateCombinationIndex(candidate);
                                    candidate_out["no_metal_charge_target"] =
                                        candidate.no_metal_charge_target;
                                    candidate_out["no_metal_radical_target"] =
                                        candidate.no_metal_radical_target;
                                    candidate_out["metal_assignment_rank"] =
                                        molgr::metal::scoring::MetadataDouble(
                                            candidate,
                                            "metal_assignment_rank");
                                    candidate_out["metal_states"] =
                                        metal_state_choice_signature(candidate.metal_states);
                                    candidate_out["phase_history"] =
                                        phase_history_signature(candidate.phase_history);
                                    candidates_out.append(std::move(candidate_out));
                                }
                                bucket["candidates"] = std::move(candidates_out);
                                layer_buckets.append(std::move(bucket));
                            }
                            target_buckets_by_layer.append(std::move(layer_buckets));
                        }
                        out["layered_state_search_groups"] = std::move(layers_out);
                        out["target_buckets_by_layer"] = std::move(target_buckets_by_layer);
                        return out;
                    },
                    "Return C++ metal-search groups, layers, and target buckets for parity debugging.",
                    py::arg("xyz_block"),
                    py::arg("total_charge") = 0,
                    py::arg("total_radical_electrons") = 0,
                    py::kw_only(),
                    py::arg("config") = py::none());

                ns.def(
                    "debug_scored_candidate_summaries",
                    [](const std::string &xyz_block,
                       int total_charge,
                       int total_radical_electrons,
                       py::object config)
                    {
                        auto runtime_config = molgr::config::FromPython(config);
                        py::dict out;
                        const auto base_state = molgr::metal::preparation::PrepareMetalState(
                            xyz_block,
                            total_charge,
                            total_radical_electrons,
                            runtime_config);
                        if (base_state.phase_history.empty())
                        {
                            out["layer_index"] = py::none();
                            out["candidates"] = py::list();
                            return out;
                        }

                        const auto state_search_groups =
                            molgr::metal::search::BuildMetalStateSearchGroups(
                                base_state.available_valence_radical_states,
                                runtime_config);
                        const auto layered_state_search_groups =
                            molgr::metal::search::BuildLayeredMetalStateSearchGroups(
                                state_search_groups,
                                total_radical_electrons,
                                runtime_config);

                        std::vector<molgr::state::MetalCandidateState> selected_layer_candidates;
                        std::optional<std::size_t> selected_layer_index;
                        for (std::size_t layer_index = 0; layer_index < layered_state_search_groups.size();
                             ++layer_index)
                        {
                            auto grouped_candidates = molgr::metal::search::GroupCandidatesByTargetDp(
                                base_state.phase_history,
                                layered_state_search_groups[layer_index],
                                total_charge,
                                total_radical_electrons,
                                runtime_config);
                            if (grouped_candidates.empty())
                            {
                                continue;
                            }

                            std::vector<molgr::state::MetalCandidateState> current_layer_candidates;
                            for (auto &target_entry : grouped_candidates)
                            {
                                if (target_entry.second.empty())
                                {
                                    continue;
                                }
                                const auto &prototype = target_entry.second.front();
                                auto no_metal_state =
                                    molgr::pipeline::reconstruct_without_metals::XyzToOmolNoMetalState(
                                        base_state.no_metal_xyz_block,
                                        prototype.no_metal_charge_target,
                                        prototype.no_metal_radical_target,
                                        runtime_config);
                                if (!no_metal_state.has_value())
                                {
                                    continue;
                                }
                                auto shared_no_metal_state =
                                    std::make_shared<molgr::state::ReconstructionState>(
                                        std::move(*no_metal_state));
                                for (const auto &candidate : target_entry.second)
                                {
                                    current_layer_candidates.push_back(
                                        molgr::metal::scoring::PrepareCandidateWithNoMetalState(
                                            candidate,
                                            shared_no_metal_state,
                                            runtime_config));
                                }
                            }
                            if (current_layer_candidates.empty())
                            {
                                continue;
                            }
                            selected_layer_candidates = std::move(current_layer_candidates);
                            selected_layer_index = layer_index;
                            break;
                        }

                        out["layer_index"] = selected_layer_index.has_value()
                                                 ? py::object(py::int_(static_cast<int>(*selected_layer_index)))
                                                 : py::object(py::none());
                        py::list candidates_out;
                        auto selected_candidate =
                            molgr::metal::scoring::SelectBestCandidateInPlace(
                                &selected_layer_candidates,
                                runtime_config);
                        const int selected_combination_index =
                            selected_candidate.has_value()
                                ? molgr::metal::scoring::CandidateCombinationIndex(*selected_candidate)
                                : -1;
                        for (const auto &candidate : selected_layer_candidates)
                        {
                            py::dict item;
                            item["combination_index"] =
                                molgr::metal::scoring::CandidateCombinationIndex(candidate);
                            item["selected"] =
                                molgr::metal::scoring::CandidateCombinationIndex(candidate) ==
                                selected_combination_index;
                            item["no_metal_charge_target"] = candidate.no_metal_charge_target;
                            item["no_metal_radical_target"] = candidate.no_metal_radical_target;
                            if (candidate.no_metal_state)
                            {
                                item["no_metal_smiles"] = molgr::reconstruct::SmilesFirstToken(
                                    candidate.no_metal_state->Mol());
                                item["no_metal_score_key"] = molgr::scoring::BuildScoreKey(
                                    candidate.no_metal_state->Mol());
                                item["no_metal_total_charge"] = candidate.no_metal_state->total_charge;
                                item["no_metal_total_radical_electrons"] =
                                    candidate.no_metal_state->total_radical_electrons;
                                py::list atom_signature;
                                OpenBabel::OBMol &no_metal_mol =
                                    const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
                                FOR_ATOMS_OF_MOL(atom_iter, no_metal_mol)
                                {
                                    py::tuple atom_item(10);
                                    atom_item[0] = static_cast<int>(atom_iter->GetAtomicNum());
                                    atom_item[1] = atom_iter->GetFormalCharge();
                                    atom_item[2] = molgr::utils::GetUnpairedElectronCount(*atom_iter);
                                    atom_item[3] = molgr::utils::GetLonePairCount(*atom_iter);
                                    atom_item[4] =
                                        molgr::utils::HasUnresolvedTwoElectronCenter(*atom_iter);
                                    atom_item[5] = atom_iter->GetHyb();
                                    atom_item[6] = static_cast<long long>(
                                        std::llround(atom_iter->GetX() * 1000000.0));
                                    atom_item[7] = static_cast<long long>(
                                        std::llround(atom_iter->GetY() * 1000000.0));
                                    atom_item[8] = static_cast<long long>(
                                        std::llround(atom_iter->GetZ() * 1000000.0));
                                    atom_item[9] = atom_iter->IsAromatic();
                                    atom_signature.append(std::move(atom_item));
                                }
                                item["no_metal_atom_signature"] = std::move(atom_signature);

                                std::vector<std::tuple<int, int, int, bool>> bond_signature;
                                FOR_BONDS_OF_MOL(bond_iter, no_metal_mol)
                                {
                                    int begin_idx = bond_iter->GetBeginAtom()->GetIdx();
                                    int end_idx = bond_iter->GetEndAtom()->GetIdx();
                                    if (begin_idx > end_idx)
                                    {
                                        std::swap(begin_idx, end_idx);
                                    }
                                    bond_signature.emplace_back(
                                        begin_idx,
                                        end_idx,
                                        bond_iter->GetBondOrder(),
                                        bond_iter->IsAromatic());
                                }
                                std::sort(bond_signature.begin(), bond_signature.end());
                                py::list bond_signature_out;
                                for (const auto &bond_item_source : bond_signature)
                                {
                                    py::tuple bond_item(4);
                                    bond_item[0] = std::get<0>(bond_item_source);
                                    bond_item[1] = std::get<1>(bond_item_source);
                                    bond_item[2] = std::get<2>(bond_item_source);
                                    bond_item[3] = std::get<3>(bond_item_source);
                                    bond_signature_out.append(std::move(bond_item));
                                }
                                item["no_metal_bond_signature"] = std::move(bond_signature_out);
                            }
                            else
                            {
                                item["no_metal_smiles"] = py::none();
                                item["no_metal_score_key"] = py::none();
                                item["no_metal_total_charge"] = py::none();
                                item["no_metal_total_radical_electrons"] = py::none();
                                item["no_metal_atom_signature"] = py::none();
                                item["no_metal_bond_signature"] = py::none();
                            }
                            item["score"] = candidate.score.has_value()
                                                ? py::object(py::float_(*candidate.score))
                                                : py::object(py::none());
                            const auto selection_key_it = candidate.metadata.find("selection_key");
                            if (selection_key_it == candidate.metadata.end())
                            {
                                item["selection_key"] = py::none();
                            }
                            else if (const auto *selection_key =
                                         std::get_if<std::string>(&selection_key_it->second))
                            {
                                std::vector<std::string> selection_parts;
                                std::size_t part_start = 0;
                                while (part_start <= selection_key->size())
                                {
                                    const std::size_t comma = selection_key->find(',', part_start);
                                    if (comma == std::string::npos)
                                    {
                                        selection_parts.push_back(selection_key->substr(part_start));
                                        break;
                                    }
                                    selection_parts.push_back(
                                        selection_key->substr(part_start, comma - part_start));
                                    part_start = comma + 1;
                                }
                                if (selection_parts.size() >= 2)
                                {
                                    py::tuple parsed_selection_key(selection_parts.size());
                                    for (std::size_t part_index = 0;
                                         part_index < selection_parts.size();
                                         ++part_index)
                                    {
                                        if (part_index + 1 == selection_parts.size())
                                        {
                                            parsed_selection_key[part_index] =
                                                std::atoi(selection_parts[part_index].c_str());
                                        }
                                        else
                                        {
                                            parsed_selection_key[part_index] =
                                                std::strtod(selection_parts[part_index].c_str(), nullptr);
                                        }
                                    }
                                    item["selection_key"] = std::move(parsed_selection_key);
                                }
                                else
                                {
                                    item["selection_key"] = *selection_key;
                                }
                            }
                            else
                            {
                                item["selection_key"] = py::none();
                            }
                            item["metal_assignment_rank"] =
                                molgr::metal::scoring::MetadataDouble(candidate, "metal_assignment_rank");
                            item["organic_aromatic_atom_count"] =
                                molgr::metal::scoring::MetadataInt(candidate, "organic_aromatic_atom_count");
                            item["organic_aromatic_ring_count"] =
                                molgr::metal::scoring::MetadataInt(candidate, "organic_aromatic_ring_count");
                            item["organic_aromatic_stability_score"] =
                                molgr::metal::scoring::MetadataDouble(candidate, "organic_aromatic_stability_score");
                            item["organic_conjugated_atom_count"] =
                                molgr::metal::scoring::MetadataInt(candidate, "organic_conjugated_atom_count");
                            item["organic_conjugated_bond_count"] =
                                molgr::metal::scoring::MetadataInt(candidate, "organic_conjugated_bond_count");
                            item["organic_max_conjugated_component_size"] =
                                molgr::metal::scoring::MetadataInt(candidate, "organic_max_conjugated_component_size");
                            item["organic_hyperconjugative_donor_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "organic_hyperconjugative_donor_count");
                            item["organic_hyperconjugation_score"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "organic_hyperconjugation_score");
                            item["organic_hyperconjugation_max_score"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "organic_hyperconjugation_max_score");
                            item["organic_hyperconjugation_deficit"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "organic_hyperconjugation_deficit");
                            item["organic_charge_localization_penalty"] =
                                molgr::metal::scoring::MetadataDouble(candidate, "organic_charge_localization_penalty");
                            item["organic_charge_localization_component_cancellation"] =
                                molgr::metal::scoring::MetadataDouble(
                                    candidate,
                                    "organic_charge_localization_component_cancellation");
                            item["organic_charge_localization_polarity_inversion_penalty"] =
                                molgr::metal::scoring::MetadataDouble(
                                    candidate,
                                    "organic_charge_localization_polarity_inversion_penalty");
                            item["organic_charge_localization_reference_penalty"] =
                                molgr::metal::scoring::MetadataDouble(
                                    candidate,
                                    "organic_charge_localization_reference_penalty");
                            item["organic_charge_localization_selection_margin"] =
                                molgr::metal::scoring::MetadataDouble(
                                    candidate,
                                    "organic_charge_localization_selection_margin");
                            item["organic_charge_localization_margin_difference"] =
                                molgr::metal::scoring::MetadataDouble(
                                    candidate,
                                    "organic_charge_localization_margin_difference");
                            item["organic_charge_localization_margin_exceeded"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "organic_charge_localization_margin_exceeded") != 0;
                            item["organic_charge_localization_reference_metal_valence_max_delta"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "organic_charge_localization_reference_metal_valence_max_delta");
                            item["organic_charge_localization_metal_valence_jump_exceeded"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "organic_charge_localization_metal_valence_jump_exceeded") != 0;
                            item["organic_singlet_two_electron_center_count"] =
                                molgr::metal::scoring::MetadataInt(candidate, "organic_singlet_two_electron_center_count");
                            item["organic_visible_singlet_carbene_count"] =
                                molgr::metal::scoring::MetadataInt(candidate, "organic_visible_singlet_carbene_count");
                            item["organic_radical_localization_penalty"] =
                                molgr::metal::scoring::MetadataDouble(candidate, "organic_radical_localization_penalty");
                            item["metal_discordance_structural_count"] =
                                molgr::metal::scoring::MetadataDouble(candidate, "metal_discordance_structural_count");
                            item["metal_discordance_count"] =
                                molgr::metal::scoring::MetadataDouble(candidate, "metal_discordance_count");
                            item["metal_discordance_inner_visible_diradical_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_inner_visible_diradical_count");
                            item["metal_discordance_excess_visible_singlet_two_electron_center_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_excess_visible_singlet_two_electron_center_count");
                            item["metal_discordance_bent_cumulated_ring_allene_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_bent_cumulated_ring_allene_count");
                            item["metal_discordance_outer_or_invisible_adjacent_double_charge_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_outer_or_invisible_adjacent_double_charge_count");
                            item["metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count");
                            item["metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count");
                            item["metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count");
                            item["metal_discordance_inner_visible_adjacent_carbanion_pair_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_inner_visible_adjacent_carbanion_pair_count");
                            item["metal_discordance_inner_visible_conjugated_carbanion_pair_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_inner_visible_conjugated_carbanion_pair_count");
                            item["metal_discordance_inner_visible_same_sign_charge_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_inner_visible_same_sign_charge_count");
                            item["metal_discordance_negative_metal_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_negative_metal_count");
                            item["metal_discordance_negative_metal_penalty"] =
                                molgr::metal::scoring::MetadataDouble(
                                    candidate,
                                    "metal_discordance_negative_metal_penalty");
                            item["metal_discordance_zero_valent_metals_with_organic_cation_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_zero_valent_metals_with_organic_cation_count");
                            item["metal_discordance_unsaturated_organic_cation_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_unsaturated_organic_cation_count");
                            for (const char *exception_key : {
                                     "metal_discordance_negative_metal_outer_sphere_cation_exception",
                                     "metal_discordance_negative_metal_positive_metal_counterion_exception"})
                            {
                                const auto exception_it = candidate.metadata.find(exception_key);
                                if (exception_it == candidate.metadata.end())
                                {
                                    item[exception_key] = py::none();
                                }
                                else if (const auto *exception_value =
                                             std::get_if<bool>(&exception_it->second))
                                {
                                    item[exception_key] = *exception_value;
                                }
                                else
                                {
                                    item[exception_key] = py::none();
                                }
                            }
                            item["metal_discordance_conjugated_atom_deficit_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_conjugated_atom_deficit_count");
                            item["metal_discordance_conjugated_bond_deficit_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_conjugated_bond_deficit_count");
                            item["metal_discordance_aromatic_atom_deficit_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_aromatic_atom_deficit_count");
                            item["metal_discordance_aromatic_ring_deficit_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_aromatic_ring_deficit_count");
                            item["metal_discordance_aromatic_stability_deficit"] =
                                molgr::metal::scoring::MetadataDouble(
                                    candidate,
                                    "metal_discordance_aromatic_stability_deficit");
                            item["metal_discordance_repeated_component_charge_asymmetry_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_repeated_component_charge_asymmetry_count");
                            item["metal_discordance_haptic_arene_reduction_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_haptic_arene_reduction_count");
                            item["metal_discordance_coordination_geometry_count"] =
                                molgr::metal::scoring::MetadataInt(
                                    candidate,
                                    "metal_discordance_coordination_geometry_count");
                            const auto passes_filter_it =
                                candidate.metadata.find("passes_metal_discordance_filter");
                            if (passes_filter_it == candidate.metadata.end())
                            {
                                item["passes_metal_discordance_filter"] = py::none();
                            }
                            else if (const auto *passes_filter =
                                         std::get_if<bool>(&passes_filter_it->second))
                            {
                                item["passes_metal_discordance_filter"] = *passes_filter;
                            }
                            else
                            {
                                item["passes_metal_discordance_filter"] = py::none();
                            }
                            py::list metal_states;
                            for (const auto &metal_state : candidate.metal_states)
                            {
                                py::dict metal_item;
                                metal_item["idx"] = metal_state.idx;
                                metal_item["symbol"] = metal_state.symbol;
                                metal_item["valence"] = metal_state.valence;
                                metal_item["radical_num"] = metal_state.radical_num;
                                metal_states.append(std::move(metal_item));
                            }
                            item["metal_states"] = std::move(metal_states);
                            candidates_out.append(std::move(item));
                        }
                        out["candidates"] = std::move(candidates_out);
                        return out;
                    },
                    "Return scored metal candidates for the first successful search layer.",
                    py::arg("xyz_block"),
                    py::arg("total_charge") = 0,
                    py::arg("total_radical_electrons") = 0,
                    py::kw_only(),
                    py::arg("config") = py::none());
            }

            void bind_reconstruct_without_metals_dev_ns(py::module_ &ns)
            {
                const auto state_summary = [](molgr::state::ReconstructionState &state)
                {
                    py::dict item;
                    item["smiles"] = molgr::reconstruct::SmilesFirstToken(state.Mol());
                    item["given_charge"] = state.given_charge;
                    item["total_charge"] = state.total_charge;
                    item["total_radical_electrons"] = state.total_radical_electrons;
                    item["valid"] = molgr::reconstruct::ValidateOmol(
                        state.MutableMol(),
                        state.total_charge,
                        state.total_radical_electrons);
                    py::list phases;
                    for (const auto &phase : state.phase_history)
                    {
                        phases.append(phase);
                    }
                    item["phase_history"] = std::move(phases);
                    return item;
                };

                ns.def(
                    "debug_prepared_no_metal_seed",
                    [state_summary](const std::string &xyz_block,
                                    int total_charge,
                                    int total_radical_electrons) -> py::object
                    {
                        auto seed_state = molgr::no_metals::preparation::SeedState(
                            xyz_block,
                            total_charge,
                            total_radical_electrons);
                        if (!seed_state.omol)
                        {
                            return py::none();
                        }
                        auto prepared =
                            molgr::no_metals::preparation::PrepareNoMetalSeed(seed_state);
                        return state_summary(prepared);
                    },
                    "Return the production C++ prepared no-metal seed.",
                    py::arg("xyz_block"),
                    py::arg("total_charge") = 0,
                    py::arg("total_radical_electrons") = 0);

                ns.def(
                    "debug_neighbor_radical_seeds",
                    [state_summary](const std::string &xyz_block,
                                    int total_charge,
                                    int total_radical_electrons,
                                    std::optional<int> exact_discrepancy) -> py::object
                    {
                        auto seed_state = molgr::no_metals::preparation::SeedState(
                            xyz_block,
                            total_charge,
                            total_radical_electrons);
                        if (!seed_state.omol)
                        {
                            return py::none();
                        }
                        auto prepared =
                            molgr::no_metals::preparation::PrepareNoMetalSeed(seed_state);
                        auto states =
                            molgr::no_metals::neighbor_radicals::EnumerateNeighborRadicalSeeds(
                                prepared,
                                exact_discrepancy);
                        py::list out;
                        for (auto &state : states)
                        {
                            py::dict item = state_summary(state);
                            const auto resolution_it =
                                state.metadata.find("neighbor_radical_resolution");
                            if (resolution_it != state.metadata.end())
                            {
                                if (const auto *resolution =
                                        std::get_if<std::string>(&resolution_it->second))
                                {
                                    item["neighbor_radical_resolution"] = *resolution;
                                }
                            }
                            const auto positive_it = state.metadata.find("positive_atom_idx");
                            if (positive_it != state.metadata.end())
                            {
                                if (const auto *positive_idx =
                                        std::get_if<int>(&positive_it->second))
                                {
                                    item["positive_atom_idx"] = *positive_idx;
                                }
                            }
                            const auto actions_it =
                                state.metadata.find("neighbor_radical_actions");
                            if (actions_it != state.metadata.end())
                            {
                                if (const auto *actions =
                                        std::get_if<std::string>(&actions_it->second))
                                {
                                    item["neighbor_radical_actions"] = *actions;
                                }
                            }
                            out.append(std::move(item));
                        }
                        return out;
                    },
                    "Return production C++ neighboring-radical seeds.",
                    py::arg("xyz_block"),
                    py::arg("total_charge") = 0,
                    py::arg("total_radical_electrons") = 0,
                    py::arg("exact_discrepancy") = py::none());

                ns.def(
                    "debug_resonance_candidate_summaries",
                    [](const std::string &xyz_block,
                       int total_charge,
                       int total_radical_electrons,
                       py::object config)
                    {
                        auto runtime_config = molgr::config::FromPython(config);
                        std::vector<
                            molgr::pipeline::reconstruct_without_metals::DebugNoMetalCandidateSummary>
                            summaries;
                        {
                            py::gil_scoped_release release;
                            summaries = molgr::pipeline::reconstruct_without_metals::
                                DebugNoMetalResonanceCandidateSummaries(
                                    xyz_block,
                                    total_charge,
                                    total_radical_electrons,
                                    runtime_config);
                        }
                        py::list out;
                        for (const auto &summary : summaries)
                        {
                            py::dict item;
                            item["smiles"] = summary.smiles;
                            item["resonance_index"] = summary.resonance_index;
                            item["score"] = summary.score;
                            item["aromatic_stability_score"] =
                                summary.aromatic_stability_score;
                            item["aromatic_atom_count"] = summary.aromatic_atom_count;
                            item["aromatic_ring_count"] = summary.aromatic_ring_count;
                            item["max_conjugated_component_size"] =
                                summary.max_conjugated_component_size;
                            item["conjugated_atom_count"] = summary.conjugated_atom_count;
                            item["conjugated_bond_count"] = summary.conjugated_bond_count;
                            item["hyperconjugative_donor_count"] =
                                summary.hyperconjugative_donor_count;
                            item["hyperconjugation_score"] = summary.hyperconjugation_score;
                            item["formal_charge_absolute_sum"] =
                                summary.formal_charge_absolute_sum;
                            item["conjugation_charge_penalty"] =
                                summary.conjugation_charge_penalty;
                            item["adjusted_max_conjugated_component_size"] =
                                summary.adjusted_max_conjugated_component_size;
                            item["adjusted_conjugated_atom_count"] =
                                summary.adjusted_conjugated_atom_count;
                            item["adjusted_conjugated_bond_count"] =
                                summary.adjusted_conjugated_bond_count;
                            item["excess_radical_labels"] = summary.excess_radical_labels;
                            out.append(std::move(item));
                        }
                        return out;
                    },
                    "Return production C++ no-metal resonance candidate summaries.",
                    py::arg("xyz_block"),
                    py::arg("total_charge") = 0,
                    py::arg("total_radical_electrons") = 0,
                    py::kw_only(),
                    py::arg("config") = py::none());
            }
            void bind_resonance_dev_ns(py::module_ &ns)
            {
                ns.def(
                    "get_radical_resonances_smi",
                    [](const std::string &smiles) -> std::vector<std::string>
                    {
                        auto mol = mol_from_smiles(smiles);
                        if (!mol)
                        {
                            throw std::runtime_error("failed to parse SMILES");
                        }
                        const auto resonances = molgr::reconstruct::GetRadicalResonances(*mol);
                        std::vector<std::string> out;
                        out.reserve(resonances.size());
                        for (const auto &resonance : resonances)
                        {
                            out.push_back(molgr::reconstruct::SmilesFirstToken(resonance));
                        }
                        return out;
                    },
                    "Enumerate radical resonance structures from a SMILES string.",
                    py::arg("smiles"),
                    py::call_guard<py::gil_scoped_release>());

                ns.def(
                    "process_resonance_smi",
                    [](const std::string &smiles,
                       int charge,
                       int total_charge,
                       int total_radical_electrons) -> std::tuple<std::string, int>
                    {
                        auto mol = mol_from_smiles(smiles);
                        if (!mol)
                        {
                            throw std::runtime_error("failed to parse SMILES");
                        }
                        auto processed = molgr::reconstruct::ProcessResonance(
                            *mol,
                            charge,
                            total_charge,
                            total_radical_electrons);
                        return std::make_tuple(
                            molgr::reconstruct::SmilesFirstToken(processed.first),
                            processed.second);
                    },
                    "Process one resonance step on SMILES and return updated token plus charge.",
                    py::arg("smiles"),
                    py::arg("charge"),
                    py::arg("total_charge") = 0,
                    py::arg("total_radical_electrons") = 0,
                    py::call_guard<py::gil_scoped_release>());

                ns.def(
                    "get_radical_resonances_ptr",
                    [](intptr_t mol_ptr) -> std::vector<intptr_t>
                    {
                        auto *mol = require_obmol_ptr(mol_ptr);
                        py::gil_scoped_release release;
                        const auto resonances = molgr::reconstruct::GetRadicalResonances(*mol);
                        std::vector<intptr_t> out;
                        out.reserve(resonances.size());
                        for (const auto &resonance : resonances)
                        {
                            auto *res_ptr = new OpenBabel::OBMol(
                                molgr::vendor::openbabel_threading::CopyMol(resonance));
                            out.push_back(reinterpret_cast<intptr_t>(res_ptr));
                        }
                        return out;
                    },
                    "Get radical resonance OBMol pointers for an input OBMol pointer.",
                    py::arg("mol_ptr"));

                ns.def(
                    "process_resonance_ptr",
                    [](intptr_t mol_ptr,
                       int charge,
                       int total_charge,
                       int total_radical_electrons) -> py::tuple
                    {
                        auto *mol = require_obmol_ptr(mol_ptr);
                        const auto processed = [&]()
                        {
                            py::gil_scoped_release release;
                            return molgr::reconstruct::ProcessResonance(
                                *mol,
                                charge,
                                total_charge,
                                total_radical_electrons);
                        }();
                        auto *res_ptr = new OpenBabel::OBMol(
                            molgr::vendor::openbabel_threading::CopyMol(processed.first));
                        return py::make_tuple(reinterpret_cast<intptr_t>(res_ptr), processed.second);
                    },
                    "Process resonance on an OBMol pointer and return (new_ptr, updated_charge).",
                    py::arg("mol_ptr"),
                    py::arg("charge"),
                    py::arg("total_charge") = 0,
                    py::arg("total_radical_electrons") = 0);

                ns.def(
                    "smiles_token_ptr",
                    [](intptr_t mol_ptr) -> std::string
                    {
                        auto *mol = require_obmol_ptr(mol_ptr);
                        py::gil_scoped_release release;
                        return molgr::reconstruct::SmilesFirstToken(*mol);
                    },
                    "Return canonical first-token SMILES string for an OBMol pointer.",
                    py::arg("mol_ptr"));
            }
        }

        void bind_pipeline(py::module_ &m)
        {
            auto reconstruct_with_metals = m.def_submodule(
                "reconstruct_with_metals",
                "Fallback-aligned reconstruction helpers with metals");

            auto reconstruct_without_metals = m.def_submodule(
                "reconstruct_without_metals",
                "Fallback-aligned no-metal reconstruction helpers");

            bind_reconstruct_with_metals_public_ns(reconstruct_with_metals);

            m.def(
                "xyz2omol",
                [](const std::string &xyz_block,
                   int total_charge,
                   int total_radical_electrons,
                   py::object config)
                    -> std::unique_ptr<molgr::utils::MoleculeData>
                {
                    auto runtime_config = molgr::config::FromPython(config);
                    t_last_reconstruction_diagnostics.Reset();
                    py::gil_scoped_release release;
                    auto mol_data = molgr::pipeline::reconstruct_with_metals::Xyz2OmolMolData(
                        xyz_block,
                        total_charge,
                        total_radical_electrons,
                        runtime_config,
                        &t_last_reconstruction_diagnostics);
                    if (!mol_data)
                    {
                        return nullptr;
                    }
                    return std::move(mol_data);
                },
                "Reconstruct molecule data from XYZ with metal-aware pipeline.",
                py::arg("xyz_block"),
                py::arg("total_charge") = 0,
                py::arg("total_radical_electrons") = 0,
                py::kw_only(),
                py::arg("config") = py::none());

            m.def(
                "get_last_reconstruction_diagnostics",
                []() -> py::dict
                {
                    return reconstruction_diagnostics_signature(t_last_reconstruction_diagnostics);
                },
                "Return structured diagnostics from the most recent C++ reconstruction failure.");

            reconstruct_without_metals.def(
                "xyz_to_omol_no_metal",
                [](const std::string &xyz_block,
                   int total_charge,
                   int total_radical_electrons,
                   py::object config)
                    -> std::unique_ptr<molgr::utils::MoleculeData>
                {
                    auto runtime_config = molgr::config::FromPython(config);
                    py::gil_scoped_release release;
                    auto mol_data = molgr::pipeline::reconstruct_without_metals::XyzToMolDataNoMetal(
                        xyz_block,
                        total_charge,
                        total_radical_electrons,
                        runtime_config);
                    if (!mol_data)
                    {
                        return nullptr;
                    }
                    return std::move(mol_data);
                },
                "Reconstruct molecule data from XYZ without metal handling.",
                py::arg("xyz_block"),
                py::arg("total_charge") = 0,
                py::arg("total_radical_electrons") = 0,
                py::kw_only(),
                py::arg("config") = py::none());

            m.def(
                "get_last_run_timing_breakdown_ms",
                []() -> py::dict
                {
                    const auto timing = molgr::pipeline::perf::GetRunTimingBreakdown();
                    py::dict out;
                    out["no_metal_pipeline_ms"] = timing.no_metal_pipeline_ms;
                    out["no_metal_linear_pipeline_ms"] = timing.no_metal_linear_pipeline_ms;
                    out["no_metal_validate_ms"] = timing.no_metal_validate_ms;
                    out["resonance_handling_enumeration_ms"] = timing.resonance_handling_enumeration_ms;
                    out["resonance_walk_ms"] = timing.resonance_walk_ms;
                    out["resonance_prepare_ms"] = timing.resonance_prepare_ms;
                    out["resonance_dedup_score_ms"] = timing.resonance_dedup_score_ms;
                    out["resonance_score_ms"] = timing.resonance_score_ms;
                    out["resonance_topology_ms"] = timing.resonance_topology_ms;
                    out["resonance_raw_candidates"] = timing.resonance_raw_candidates;
                    out["resonance_pruned_expansions"] = timing.resonance_pruned_expansions;
                    out["resonance_prepared_candidates"] = timing.resonance_prepared_candidates;
                    out["resonance_valid_candidates"] = timing.resonance_valid_candidates;
                    out["resonance_dedup_candidates"] = timing.resonance_dedup_candidates;
                    out["metal_enumeration_combination_ms"] = timing.metal_enumeration_combination_ms;
                    out["force_field_total_ms"] = timing.force_field_total_ms;
                    out["force_field_cache_key_ms"] = timing.force_field_cache_key_ms;
                    out["force_field_prepare_ms"] = timing.force_field_prepare_ms;
                    out["force_field_setup_key_ms"] = timing.force_field_setup_key_ms;
                    out["force_field_setup_ms"] = timing.force_field_setup_ms;
                    out["force_field_energy_ms"] = timing.force_field_energy_ms;
                    out["force_field_calls"] = timing.force_field_calls;
                    return out;
                },
                "Return timing breakdown (milliseconds) for the most recent reconstruction run.");
        }

        void bind_dev_pipeline(py::module_ &m)
        {
            auto reconstruct_with_metals = m.def_submodule(
                "reconstruct_with_metals",
                "Development-only helpers for metal reconstruction internals");
            auto reconstruct_without_metals = m.def_submodule(
                "reconstruct_without_metals",
                "Development-only helpers for no-metal reconstruction internals");
            auto resonance = m.def_submodule(
                "resonance",
                "Development-only helpers for resonance internals");

            bind_reconstruct_with_metals_dev_ns(reconstruct_with_metals);
            bind_reconstruct_without_metals_dev_ns(reconstruct_without_metals);
            bind_resonance_dev_ns(resonance);

            m.def(
                "clear_force_field_evaluation_cache",
                []()
                {
                    molgr::scoring::ForceFieldEvaluationCacheClear();
                },
                "Clear the C++ force-field evaluation cache.");
            m.def(
                "clear_uff_atom_typing_cache",
                []()
                {
                    OpenBabel::ClearMolgrUffAtomTypeAssignmentCache();
                },
                "Clear the C++ UFF atom-typing assignment cache.");
            m.def(
                "get_uff_atom_typing_cache_info",
                []()
                {
                    const auto [hits, misses, size] =
                        OpenBabel::MolgrUffAtomTypeAssignmentCacheInfo();
                    py::dict out;
                    out["hits"] = hits;
                    out["misses"] = misses;
                    out["size"] = size;
                    return out;
                },
                "Return C++ UFF atom-typing cache hit/miss/size counters.");
        }
    }
}

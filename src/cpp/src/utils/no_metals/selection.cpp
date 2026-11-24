#include "molgr/utils/no_metals/selection.h"
#include "molgr/utils/bond_geometry.h"

#include "molgr/utils/electrons.h"
#include "molgr/utils/organic_topology.h"

#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace molgr
{
    namespace no_metals
    {
        namespace selection
        {
            namespace
            {
                /// Encode atom and bond labels for deterministic selection among otherwise tied ligand
                /// graphs.
                /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
                /// @return Deterministic atom and bond ordering key.
                NoMetalGraphTieBreakKey GraphTieBreakKey(const OpenBabel::OBMol &mol)
                {
                    NoMetalGraphTieBreakKey key;
                    key.reserve(static_cast<std::size_t>(mol.NumAtoms()) * 6 +
                                static_cast<std::size_t>(mol.NumBonds()) * 3 + 2);
                    key.push_back(mol.NumAtoms());
                    for (int atom_idx = 1; atom_idx <= mol.NumAtoms(); ++atom_idx)
                    {
                        const auto *atom = mol.GetAtom(atom_idx);
                        if (atom == nullptr)
                        {
                            continue;
                        }
                        key.push_back(atom_idx);
                        key.push_back(static_cast<int>(atom->GetAtomicNum()));
                        key.push_back(atom->GetFormalCharge());
                        key.push_back(molgr::utils::GetUnpairedElectronCount(*atom));
                        key.push_back(molgr::utils::GetLonePairCount(*atom));
                        key.push_back(
                            molgr::utils::HasUnresolvedTwoElectronCenter(*atom) ? 1 : 0);
                    }

                    std::vector<std::tuple<int, int, int>> bonds;
                    bonds.reserve(static_cast<std::size_t>(mol.NumBonds()));
                    FOR_BONDS_OF_MOL(bond_iter, const_cast<OpenBabel::OBMol &>(mol))
                    {
                        const int begin_idx = bond_iter->GetBeginAtomIdx();
                        const int end_idx = bond_iter->GetEndAtomIdx();
                        bonds.emplace_back(
                            std::min(begin_idx, end_idx),
                            std::max(begin_idx, end_idx),
                            bond_iter->GetBondOrder());
                    }
                    std::sort(bonds.begin(), bonds.end());
                    key.push_back(static_cast<int>(bonds.size()));
                    for (const auto &[begin_idx, end_idx, order] : bonds)
                    {
                        key.push_back(begin_idx);
                        key.push_back(end_idx);
                        key.push_back(order);
                    }
                    return key;
                }

                /// Measure charge separation on the ligand, counting both ends of each formal ion pair.
                /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
                /// @return Sum of the absolute values of the atom formal charges.
                int FormalChargeAbsoluteSum(const OpenBabel::OBMol &mol)
                {
                    int formal_charge_absolute_sum = 0;
                    for (int atom_idx = 1; atom_idx <= mol.NumAtoms(); ++atom_idx)
                    {
                        const OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
                        if (atom == nullptr)
                        {
                            continue;
                        }
                        formal_charge_absolute_sum += std::abs(atom->GetFormalCharge());
                    }
                    return formal_charge_absolute_sum;
                }

                /// Count explicit unpaired electrons above the requested ligand radical budget.
                /// @param candidate Candidate electronic assignment being evaluated.
                /// @return Number of unpaired electrons in excess of the target.
                int ExcessRadicalLabels(const molgr::state::ReconstructionState &candidate)
                {
                    int radical_sum = 0;
                    for (int atom_idx = 1; atom_idx <= candidate.Mol().NumAtoms(); ++atom_idx)
                    {
                        const auto *atom = candidate.Mol().GetAtom(atom_idx);
                        if (atom != nullptr)
                        {
                            radical_sum += molgr::utils::GetUnpairedElectronCount(*atom);
                        }
                    }
                    candidate.metadata["organic_radical_label_sum"] = radical_sum;
                    const int excess = std::max(
                        0,
                        radical_sum - candidate.total_radical_electrons);
                    candidate.metadata["organic_excess_radical_labels"] = excess;
                    return excess;
                }
            }

            molgr::organic_topology::OrganicTopologyMetrics AnnotateNoMetalCandidateTopology(
                molgr::state::ReconstructionState &candidate,
                const molgr::config::MolGRConfig &config)
            {
                const auto metrics = molgr::organic_topology::ComputeOrganicTopologyMetrics(
                    candidate.Mol(),
                    config.organic_topology);
                candidate.metadata["organic_aromatic_atom_count"] = metrics.aromatic_atom_count;
                candidate.metadata["organic_aromatic_ring_count"] = metrics.aromatic_ring_count;
                candidate.metadata["organic_aromatic_stability_score"] = metrics.aromatic_stability_score;
                candidate.metadata["organic_conjugated_atom_count"] = metrics.conjugated_atom_count;
                candidate.metadata["organic_conjugated_bond_count"] = metrics.conjugated_bond_count;
                candidate.metadata["organic_max_conjugated_component_size"] =
                    metrics.max_conjugated_component_size;
                candidate.metadata["organic_hyperconjugative_donor_count"] =
                    metrics.hyperconjugative_donor_count;
                candidate.metadata["organic_hyperconjugation_score"] =
                    metrics.hyperconjugation_score;
                return metrics;
            }

            double ScoreReconstructionCandidate(
                molgr::state::ReconstructionState &candidate,
                const molgr::config::MolGRConfig &config)
            {
                const double score = candidate.FullScore(config);
                candidate.metadata["score"] = score;
                return score;
            }

            std::tuple<int,int,int> NoMetalCandidateGeometryKey(
                molgr::state::ReconstructionState &candidate)
            {
                int long_carbonyl_count = 0;
                auto &mol = const_cast<OpenBabel::OBMol &>(candidate.Mol());
                FOR_BONDS_OF_MOL(bond, mol)
                {
                    if (bond->GetBondOrder() == 2 && molgr::bond_geometry::IsElongatedCarbonylPair(
                            *bond->GetBeginAtom(), *bond->GetEndAtom()))
                        ++long_carbonyl_count;
                }
                candidate.metadata["organic_long_carbonyl_multiple_bond_count"] = long_carbonyl_count;
                const int azinium_count = molgr::bond_geometry::AziniumGeometryConflictCount(mol);
                candidate.metadata["organic_azinium_geometry_conflict_count"] = azinium_count;
                const int quinodiimine_count = molgr::bond_geometry::ShortQuinodiimineGeometryConflictCount(mol);
                candidate.metadata["organic_short_quinodiimine_geometry_conflict_count"] = quinodiimine_count;
                return {azinium_count + quinodiimine_count, FormalChargeAbsoluteSum(mol), long_carbonyl_count};
            }

            NoMetalTopologySelectionKey NoMetalCandidateTopologySelectionKey(
                molgr::state::ReconstructionState &candidate,
                const molgr::config::MolGRConfig &config)
            {
                const auto metrics = AnnotateNoMetalCandidateTopology(candidate, config);
                const int formal_charge_absolute_sum = FormalChargeAbsoluteSum(candidate.Mol());
                const double conjugation_charge_penalty =
                    static_cast<double>(formal_charge_absolute_sum) / 2.0;
                const double adjusted_max_conjugated_component_size =
                    static_cast<double>(metrics.max_conjugated_component_size) -
                    conjugation_charge_penalty;
                const double adjusted_conjugated_atom_count =
                    static_cast<double>(metrics.conjugated_atom_count) - conjugation_charge_penalty;
                const double adjusted_conjugated_bond_count =
                    static_cast<double>(metrics.conjugated_bond_count) - conjugation_charge_penalty;
                candidate.metadata["organic_formal_charge_absolute_sum"] =
                    formal_charge_absolute_sum;
                candidate.metadata["organic_conjugation_charge_penalty"] =
                    conjugation_charge_penalty;
                candidate.metadata["organic_adjusted_max_conjugated_component_size"] =
                    adjusted_max_conjugated_component_size;
                candidate.metadata["organic_adjusted_conjugated_atom_count"] =
                    adjusted_conjugated_atom_count;
                candidate.metadata["organic_adjusted_conjugated_bond_count"] =
                    adjusted_conjugated_bond_count;
                const auto [geometry, absolute_charge, long_carbonyl_count] =
                    NoMetalCandidateGeometryKey(candidate);
                return {
                    geometry,
                    absolute_charge,
                    long_carbonyl_count,
                    -metrics.aromatic_atom_count,
                    -metrics.aromatic_ring_count,
                    -metrics.aromatic_stability_score,
                    -adjusted_max_conjugated_component_size,
                    -adjusted_conjugated_atom_count,
                    -adjusted_conjugated_bond_count,
                };
            }

            std::tuple<int, int, int, int, int, double, double, double, double, int, int>
            NoMetalCandidateStructuralKey(
                molgr::state::ReconstructionState &candidate,
                const molgr::config::MolGRConfig &config)
            {
                const auto topology_key = NoMetalCandidateTopologySelectionKey(candidate, config);
                const int excess_radical_labels = ExcessRadicalLabels(candidate);
                const auto hyperconjugation_it =
                    candidate.metadata.find("organic_hyperconjugation_score");
                const int hyperconjugation_score =
                    hyperconjugation_it != candidate.metadata.end() &&
                            std::get_if<int>(&hyperconjugation_it->second) != nullptr
                        ? *std::get_if<int>(&hyperconjugation_it->second)
                        : 0;
                return {
                    std::get<0>(topology_key),
                    std::get<1>(topology_key),
                    std::get<2>(topology_key),
                    std::get<3>(topology_key),
                    std::get<4>(topology_key),
                    std::get<5>(topology_key),
                    std::get<6>(topology_key),
                    std::get<7>(topology_key),
                    std::get<8>(topology_key),
                    excess_radical_labels,
                    -hyperconjugation_score,
                };
            }

            std::tuple<int, int, int, int, int, double, double, double, double, int, int, double>
            NoMetalCandidateSelectionKey(
                molgr::state::ReconstructionState &candidate,
                const molgr::config::MolGRConfig &config)
            {
                const auto prefix = NoMetalCandidateStructuralKey(candidate, config);
                return std::tuple_cat(prefix, std::make_tuple(ScoreReconstructionCandidate(candidate,config)));
            }

            NoMetalGraphTieBreakKey NoMetalCandidateGraphTieBreakKey(
                const molgr::state::ReconstructionState &candidate)
            {
                return GraphTieBreakKey(candidate.Mol());
            }
        }
    }
}

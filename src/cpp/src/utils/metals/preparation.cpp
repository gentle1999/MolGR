#include "molgr/utils/metals/preparation.h"

#include "molgr/utils/consts.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/logger.h"
#include "molgr/utils/metal_radical_inference.h"
#include "molgr/utils/xyz.h"

#include <openbabel/bond.h>
#include <openbabel/elements.h>
#include "molgr/compat/openbabel_iter.h"

#include <set>

namespace molgr
{
    namespace metal
    {
        namespace preparation
        {
            std::set<int> GetPossibleMetalRadicals(const std::string &metal_symbol, int valence)
            {
                return molgr::GetPossibleMetalRadicals(metal_symbol, valence);
            }

            std::vector<molgr::metal::MetalAtomPosition> BuildMetalStates(
                OpenBabel::OBAtom &obatom,
                const molgr::config::MolGRConfig &config)
            {
                const int atomic_num = static_cast<int>(obatom.GetAtomicNum());
                const std::string symbol = OpenBabel::OBElements::GetSymbol(atomic_num);

                /// Retain the input metal position and use neutral, zero-unpaired labels when no state is
                /// available.
                const auto default_state = [&]()
                {
                    return molgr::metal::MetalAtomPosition{
                        static_cast<int>(obatom.GetIdx()),
                        symbol,
                        atomic_num,
                        0,
                        0,
                        obatom.GetX(),
                        obatom.GetY(),
                        obatom.GetZ()};
                };

                std::vector<int> valences;
                std::set<int> seen_valences;
                /// Append each new metal valence once, preserving the preference order of the source list.
                const auto add_valences = [&](const std::vector<int> &source)
                {
                    for (const int valence : source)
                    {
                        if (seen_valences.insert(valence).second)
                        {
                            valences.push_back(valence);
                        }
                    }
                };

                if (kMetalValencePrior.count(symbol))
                {
                    add_valences(kMetalValencePrior.at(symbol));
                }
                if (kMetalValenceMinor.count(symbol))
                {
                    add_valences(kMetalValenceMinor.at(symbol));
                }
                if (valences.empty())
                {
                    valences.push_back(0);
                }

                if (!kMetalFDSP.count(symbol))
                {
                    return {default_state()};
                }

                std::vector<molgr::metal::MetalAtomPosition> states;
                for (const int valence : valences)
                {
                    const auto radicals =
                        molgr::metal_radical_inference::InferMetalRadicalCounts(
                            obatom,
                            valence,
                            config);
                    for (const int radical_num : radicals)
                    {
                        states.push_back(
                            molgr::metal::MetalAtomPosition{
                                static_cast<int>(obatom.GetIdx()),
                                symbol,
                                atomic_num,
                                valence,
                                radical_num,
                                obatom.GetX(),
                                obatom.GetY(),
                                obatom.GetZ()});
                    }
                }

                if (states.empty())
                {
                    return {default_state()};
                }
                return states;
            }

            // Electron bookkeeping: copy the chosen metal radical_num only into
            // the real unpaired-electron field. Metals receive no active lone-pair
            // or unresolved-center classification; organic fields survive cloning.
            void ReinsertMetalStates(
                OpenBabel::OBMol &mol,
                const std::vector<molgr::metal::MetalAtomPosition> &metals)
            {
                mol.BeginModify();

                int num_organic = mol.NumAtoms();
                int num_metals = static_cast<int>(metals.size());
                int total_atoms = num_organic + num_metals;

                for (const auto &m : metals)
                {
                    OpenBabel::OBAtom *atom = mol.NewAtom();
                    atom->SetAtomicNum(m.element_idx);
                    atom->SetFormalCharge(m.valence);
                    molgr::utils::SetUnpairedElectronCount(*atom, m.radical_num);
                    atom->SetVector(m.position_x, m.position_y, m.position_z);
                }

                std::vector<int> new_order(total_atoms, 0);
                bool error_flag = false;

                for (int i = 0; i < num_metals; ++i)
                {
                    int current_idx = num_organic + 1 + i;
                    int target_slot = metals[i].idx - 1;

                    if (target_slot >= 0 && target_slot < total_atoms)
                    {
                        if (new_order[target_slot] != 0)
                        {
                            LOG_ERROR("[ReinsertMetalStates] Index collision at slot " << target_slot);
                            error_flag = true;
                        }
                        new_order[target_slot] = current_idx;
                    }
                    else
                    {
                        LOG_ERROR("[ReinsertMetalStates] Original index out of bounds: " << metals[i].idx);
                        error_flag = true;
                    }
                }

                int current_organic_idx = 1;
                for (int i = 0; i < total_atoms; ++i)
                {
                    if (new_order[i] == 0)
                    {
                        if (current_organic_idx <= num_organic)
                        {
                            new_order[i] = current_organic_idx;
                            current_organic_idx++;
                        }
                        else
                        {
                            LOG_ERROR("[ReinsertMetalStates] Not enough organic atoms to fill slots.");
                            error_flag = true;
                        }
                    }
                }

                for (int idx : new_order)
                {
                    if (idx == 0)
                    {
                        LOG_ERROR("[ReinsertMetalStates] Invalid 0 index in renumber map. Aborting renumber.");
                        error_flag = true;
                        break;
                    }
                }

                if (!error_flag)
                {
                    mol.RenumberAtoms(new_order);
                }

                mol.EndModify();
            }

            void CombineMetalWithOmol(
                OpenBabel::OBMol &mol,
                const std::vector<molgr::metal::MetalAtomPosition> &metals)
            {
                ReinsertMetalStates(mol, metals);
            }

            molgr::state::MetalPreparationState PrepareMetalState(
                const std::string &xyz_block,
                int total_charge,
                int total_radical_electrons,
                const molgr::config::MolGRConfig &config)
            {
                OpenBabel::OBMol mol;
                if (!molgr::utils::ReadXyzBlockToMol(xyz_block, &mol))
                {
                    return {};
                }

                molgr::state::MetalPreparationState state;
                std::vector<OpenBabel::OBAtom *> removable_metal_atoms;
                std::vector<std::vector<molgr::metal::MetalAtomPosition>> available_states;
                FOR_ATOMS_OF_MOL(atom_iter, mol)
                {
                    OpenBabel::OBAtom *atom = &(*atom_iter);
                    if (!atom->IsMetal())
                    {
                        continue;
                    }
                    state.metal_sites.push_back({static_cast<int>(atom->GetAtomicNum()),
                        atom->GetX(), atom->GetY(), atom->GetZ()});
                    removable_metal_atoms.push_back(atom);
                    available_states.push_back(BuildMetalStates(*atom, config));
                }

                for (OpenBabel::OBAtom *atom : removable_metal_atoms)
                {
                    mol.DeleteAtom(atom);
                }

                state.no_metal_xyz_block = molgr::utils::WriteXyzBlock(mol);
                state.available_valence_radical_states = std::move(available_states);
                state.total_charge = total_charge;
                state.total_radical_electrons = total_radical_electrons;
                state.phase_history = {
                    "read_xyz",
                    "build_metal_state_options",
                    "remove_metal_atoms",
                    "serialize_no_metal_xyz",
                };
                state.metadata["metal_atom_count"] = static_cast<int>(removable_metal_atoms.size());
                return state;
            }
        }
    }
}

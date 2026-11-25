#include "molgr/stages/fresh.h"

#include "molgr/stages/internal_helpers.h"

#include "molgr/utils/consts.h"
#include "molgr/utils/bond_geometry.h"
#include "molgr/utils/electrons.h"

#include <openbabel/atom.h>
#include <openbabel/obfunctions.h>
#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <utility>

namespace molgr
{
    namespace reconstruct
    {
        using namespace OpenBabel;

        const ElementInfo *GetElementInfo(int atomic_num)
        {
            auto it = kNonMetalDict.find(atomic_num);
            if (it != kNonMetalDict.end())
                return &(it->second);
            return nullptr;
        }

        // Electron bookkeeping: return a bond-valence deficit, not an already
        // classified unpaired-electron count. Boron uses its nonbonding electron
        // budget minus inactive anionic lone pairs, retaining an odd electron
        // on a three-bond B- radical just as in the Python fallback.
        int AssignRadicalDots(OBAtom &atom)
        {
            const int total_valence = static_cast<int>(atom.GetTotalValence());
            if (atom.GetAtomicNum() == 5)
            {
                const int charge = atom.GetFormalCharge();
                const int nonbonding_electrons = std::max(0, 3 - charge - total_valence);
                const int inactive_lone_pairs = std::min(
                    std::max(0, -charge), nonbonding_electrons / 2);
                return nonbonding_electrons - 2 * inactive_lone_pairs;
            }
            const int typical_valence = GetTypicalValence(
                atom.GetAtomicNum(),
                total_valence,
                atom.GetFormalCharge());
            return std::max(0, typical_valence - total_valence);
        }

        // Electron bookkeeping: distribute active electrons over valence-orbital
        // slots derived from sigma degree and pi bond increments. The pair is
        // {real unpaired electrons, reconstruction-active lone pairs}; it is not
        // inferred from atom-wide parity and does not use cached hybridization.
        /// Classify active electrons into unpaired electrons and active lone pairs.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param electron_count Number of locally available active electrons to classify.
        /// @return Pair of unpaired-electron and active lone-pair counts, respectively.
        std::pair<int, int> InferActiveElectronOccupancy(
            OBAtom &atom,
            int electron_count)
        {
            if (electron_count <= 0)
            {
                return {0, 0};
            }

            const int total_valence = static_cast<int>(atom.GetTotalValence());
            const int sigma_bond_count = static_cast<int>(atom.GetTotalDegree());
            const int pi_bond_count = std::max(0, total_valence - sigma_bond_count);
            const int available_valence_orbitals =
                std::max(0, 4 - sigma_bond_count - pi_bond_count);

            int lone_pair_count = 0;
            int unpaired_electron_count = 0;

            const int valence_electrons =
                std::min(electron_count, 2 * available_valence_orbitals);
            if (valence_electrons <= available_valence_orbitals)
            {
                unpaired_electron_count += valence_electrons;
            }
            else
            {
                lone_pair_count += valence_electrons - available_valence_orbitals;
                unpaired_electron_count +=
                    2 * available_valence_orbitals - valence_electrons;
            }
            electron_count -= valence_electrons;

            // Do not use Open Babel hybridization to interpret excess active
            // electrons; it can be stale after reconstruction changes bonds.
            lone_pair_count += electron_count / 2;
            unpaired_electron_count += electron_count % 2;
            return {unpaired_electron_count, lone_pair_count};
        }

        // Electron bookkeeping: completely overwrite unpaired and active-lone-pair
        // counts from one local deficit. The unresolved marker is managed by the
        // higher-level atom assignment function.
        /// Replace both explicit occupancy fields from one local electron deficit.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param electron_count Number of locally available active electrons to classify.
        void AssignActiveElectronOccupancy(OBAtom &atom, int electron_count)
        {
            const auto occupancy = InferActiveElectronOccupancy(atom, electron_count);
            molgr::utils::SetUnpairedElectronCount(atom, occupancy.first);
            molgr::utils::SetLonePairCount(atom, occupancy.second);
        }

        // Electron bookkeeping: rebuild one atom's charge and explicit electron
        // classification. Neutral C/N/Si/P two-electron deficits become unresolved;
        // already resolved (unpaired=0, lone_pair=1) and (2,0) states survive while
        // the same topology remains. Other states overwrite both occupancy fields
        // and may also normalize formal charge.
        bool AssignChargeRadicalForAtom(OBAtom &atom)
        {
            const int before_charge = atom.GetFormalCharge();
            const int before_unpaired = molgr::utils::GetUnpairedElectronCount(atom);
            const int before_lone_pairs = molgr::utils::GetLonePairCount(atom);
            const bool before_unresolved_center =
                molgr::utils::HasUnresolvedTwoElectronCenter(atom);
            /// Compare charge and active electron labels with their values before local valence rebuilding.
            auto changed = [&]() -> bool
            {
                return atom.GetFormalCharge() != before_charge ||
                       molgr::utils::GetUnpairedElectronCount(atom) != before_unpaired ||
                       molgr::utils::GetLonePairCount(atom) != before_lone_pairs ||
                       molgr::utils::HasUnresolvedTwoElectronCenter(atom) !=
                           before_unresolved_center;
            };

            const int atomic_num = atom.GetAtomicNum();
            if (before_charge == -1 && molgr::bond_geometry::AllowsSeesawPhosphoranide(atom))
            {
                // Preserve an explored P- hypothesis and its inactive closed-shell
                // pair. The ordinary seed remains P+ until a branch is requested.
                molgr::utils::SetUnpairedElectronCount(atom, 0);
                molgr::utils::SetLonePairCount(atom, 0);
                molgr::utils::SetUnresolvedTwoElectronCenter(atom, false);
                return changed();
            }
            const int current_val = atom.GetTotalValence();
            const ElementInfo *info = GetElementInfo(atomic_num);
            if (!info)
            {
                return changed();
            }

            if (info->num_outer_electrons < current_val)
            {
                atom.SetFormalCharge(info->num_outer_electrons - current_val);
            }
            else
            {
                const int total_elec =
                    info->num_outer_electrons + current_val;
                if (total_elec > 8 && total_elec % 8 <=
                    info->num_outer_electrons - current_val)
                {
                    atom.SetFormalCharge(total_elec % 8);
                }
            }

            const int rad = AssignRadicalDots(atom);
            const bool is_two_electron_center =
                (atomic_num == 6 || atomic_num == 7 || atomic_num == 14 || atomic_num == 15) &&
                atom.GetFormalCharge() == 0 && rad == 2;
            if (is_two_electron_center)
            {
                const std::pair<int, int> occupancy{
                    molgr::utils::GetUnpairedElectronCount(atom),
                    molgr::utils::GetLonePairCount(atom),
                };
                const bool explicit_occupancy =
                    occupancy == std::pair<int, int>{0, 1} ||
                    occupancy == std::pair<int, int>{2, 0};
                if (before_unresolved_center || !explicit_occupancy)
                {
                    molgr::utils::SetUnpairedElectronCount(atom, 0);
                    molgr::utils::SetLonePairCount(atom, 0);
                    molgr::utils::SetUnresolvedTwoElectronCenter(atom, true);
                }
                return molgr::utils::GetUnpairedElectronCount(atom) != before_unpaired ||
                       molgr::utils::GetLonePairCount(atom) != before_lone_pairs ||
                       molgr::utils::HasUnresolvedTwoElectronCenter(atom) !=
                           before_unresolved_center;
            }
            molgr::utils::SetUnresolvedTwoElectronCenter(atom, false);
            AssignActiveElectronOccupancy(atom, rad);
            return changed();
        }

        // Electron bookkeeping: apply AssignChargeRadicalForAtom to every atom.
        // This may replace all three electron fields and formal charges, so local
        // resonance transforms should prefer endpoint-only refreshes.
        bool FreshOmolChargeRadical(OBMol &mol)
        {
            bool hit = false;
            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                hit = AssignChargeRadicalForAtom(*atom_iter) || hit;
            }
            return hit;
        }
    }
}

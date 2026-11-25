#include "molgr/stages/break_bond.h"

#include "molgr/utils/electrons.h"
#include "molgr/utils/logger.h"
#include "molgr/utils/smarts.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/elements.h>
#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace molgr
{
    namespace reconstruct
    {
        using namespace OpenBabel;

        namespace
        {
            /// Count the electronic defects available to a bond-breaking step.
            /// @param mol Molecular graph on which the operation acts.
            /// @return Total bond-breaking electron budget for the graph.
            int CurrentBondBreakingElectronBudget(OBMol &mol)
            {
                int sum = 0;
                FOR_ATOMS_OF_MOL(atom, mol)
                {
                    sum += molgr::utils::GetUnpairedElectronCount(*atom);
                    if (molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                    {
                        sum += 2;
                    }
                }
                return sum;
            }
        }

        // Electron bookkeeping: reduce one deformed multiple bond order and add
        // one real unpaired electron to each endpoint (homolytic pi-bond cleavage).
        // Deferred two-electron centers count toward the stopping budget so their
        // occupied electrons are not created again by another bond cleavage.
        bool BreakDeformedEne(OBMol &mol, int allowed_charge, int allowed_radical, double tolerance)
        {
            bool hit = false;

            /// Candidate strained alkene with the indices and geometry used to rank a bond-order reduction.
            struct EneCandidate
            {
                int idx1;
                int idx2;
                double deviation;
            };
            std::vector<EneCandidate> candidates;

            {
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::BREAK_DEFORMED_ENE_A);
                for (const auto &idxs : matches)
                {
                    OBBond *bond = mol.GetBond(idxs[1], idxs[2]);
                    if (!bond || bond->IsRotor() || bond->GetBondOrder() == 1)
                        continue;

                    double torsion = std::abs(mol.GetTorsion(idxs[0], idxs[1], idxs[2], idxs[3]));
                    double deviation = std::min(torsion, 180.0 - torsion);
                    if (deviation > tolerance)
                    {
                        candidates.push_back(EneCandidate{idxs[1], idxs[2], deviation});
                    }
                }
            }

            {
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::BREAK_DEFORMED_ENE_B);
                for (const auto &idxs : matches)
                {
                    OBBond *bond = mol.GetBond(idxs[1], idxs[2]);
                    if (!bond || bond->IsRotor() || bond->GetBondOrder() == 1)
                        continue;

                    double torsion = std::abs(mol.GetTorsion(idxs[0], idxs[1], idxs[2], idxs[3]));
                    double deviation = std::min(torsion, 180.0 - torsion);
                    if (deviation > tolerance)
                    {
                        candidates.push_back(EneCandidate{idxs[1], idxs[2], deviation});
                    }
                }
            }

            std::sort(candidates.begin(), candidates.end(), [](const EneCandidate &a, const EneCandidate &b)
                      { return a.deviation > b.deviation; });

            for (const auto &candidate : candidates)
            {
                if (CurrentBondBreakingElectronBudget(mol) >=
                    std::abs(allowed_charge) + allowed_radical)
                    return hit;
                OBBond *bond = mol.GetBond(candidate.idx1, candidate.idx2);
                if (!bond || bond->IsRotor() || bond->GetBondOrder() == 1)
                    continue;
                bond->SetBondOrder(bond->GetBondOrder() - 1);
                OBAtom *begin_atom = bond->GetBeginAtom();
                OBAtom *end_atom = bond->GetEndAtom();
                molgr::utils::SetUnpairedElectronCount(*begin_atom, molgr::utils::GetUnpairedElectronCount(*begin_atom) + 1);
                molgr::utils::SetUnpairedElectronCount(*end_atom, molgr::utils::GetUnpairedElectronCount(*end_atom) + 1);
                hit = true;
                LOG_DEBUG("[BreakDeformedEne] Broken sorted candidate");
            }
            return hit;
        }

        // Electron bookkeeping: ordinary reductions/deletions create one unpaired
        // electron per endpoint. The cation template creates one radical only on
        // the neutral endpoint and lowers the positive endpoint charge, an explicit
        // charge-transfer heuristic. Deferred two-electron centers count toward
        // the stopping budget but remain unresolved and are not consumed here.
        // Recover overbonded N/P cations before cleaving neutral multiple bonds;
        // otherwise an intact carbonyl can absorb an unrelated electron deficit.
        bool BreakOneBond(OBMol &mol, int &charge, int allowed_radical)
        {
            bool hit = false;
            /// Stop breaking bonds when the graph already supplies the radicals and charge reserve
            /// required.
            auto check_cond = [&]()
            {
                return CurrentBondBreakingElectronBudget(mol) >=
                       std::abs(charge) + allowed_radical;
            };

            while (true)
            {
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::BREAK_ONE_BOND_CATION);
                if (matches.empty())
                    break;
                if (check_cond())
                    return hit;
                const auto &idxs = matches.front();
                OBBond *bond = mol.GetBond(idxs[0], idxs[1]);
                if (!bond)
                    break;
                bond->SetBondOrder(bond->GetBondOrder() - 1);
                OBAtom *idx0_atom = mol.GetAtom(idxs[0]);
                OBAtom *idx1_atom = mol.GetAtom(idxs[1]);
                if (!idx0_atom || !idx1_atom)
                    break;
                molgr::utils::SetUnpairedElectronCount(*idx1_atom, molgr::utils::GetUnpairedElectronCount(*idx1_atom) + 1);
                idx0_atom->SetFormalCharge(idx0_atom->GetFormalCharge() - 1);
                charge += 1;
                hit = true;
                LOG_DEBUG("[BreakOneBond] Broken charge-transfer candidate");
            }

            while (true)
            {
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::BREAK_ONE_BOND_MULTIPLE);
                if (matches.empty())
                    break;
                if (check_cond())
                    return hit;
                const auto &idxs = matches.front();
                OBBond *bond = mol.GetBond(idxs[0], idxs[1]);
                if (!bond)
                    break;
                bond->SetBondOrder(bond->GetBondOrder() - 1);
                OBAtom *begin_atom = bond->GetBeginAtom();
                OBAtom *end_atom = bond->GetEndAtom();
                molgr::utils::SetUnpairedElectronCount(*begin_atom, molgr::utils::GetUnpairedElectronCount(*begin_atom) + 1);
                molgr::utils::SetUnpairedElectronCount(*end_atom, molgr::utils::GetUnpairedElectronCount(*end_atom) + 1);
                hit = true;
                LOG_DEBUG("[BreakOneBond] Broken triple/double candidate");
            }

            {
                auto matches = molgr::smarts::FindAll(mol, molgr::smarts::PatternId::BREAK_ONE_BOND_AROMATIC);
                if (!matches.empty())
                {
                    if (check_cond())
                        return hit;
                    for (const auto &idxs : matches)
                    {
                        OBBond *bond = mol.GetBond(idxs[0], idxs[1]);
                        if (!bond)
                            continue;
                        if (bond->GetBondOrder() == 1)
                            continue;
                        bond->SetBondOrder(bond->GetBondOrder() - 1);
                        OBAtom *begin_atom = bond->GetBeginAtom();
                        OBAtom *end_atom = bond->GetEndAtom();
                        molgr::utils::SetUnpairedElectronCount(*begin_atom, molgr::utils::GetUnpairedElectronCount(*begin_atom) + 1);
                        molgr::utils::SetUnpairedElectronCount(*end_atom, molgr::utils::GetUnpairedElectronCount(*end_atom) + 1);
                        hit = true;
                        LOG_DEBUG("[BreakOneBond] Broken aromatic candidate");
                        break;
                    }
                }
            }

            if (check_cond())
                return hit;

            bool all_single = true;
            FOR_BONDS_OF_MOL(b, mol)
            if (b->GetBondOrder() != 1) all_single = false;

            if (all_single)
            {
                std::vector<OBBond *> single_bonds;
                /// Rank a stretched bond by length beyond the sum of its endpoint covalent radii, in
                /// angstroms.
                const auto excess_length = [](OBBond *bond)
                {
                    return bond->GetLength() -
                           OBElements::GetCovalentRad(bond->GetBeginAtom()->GetAtomicNum()) -
                           OBElements::GetCovalentRad(bond->GetEndAtom()->GetAtomicNum());
                };
                FOR_BONDS_OF_MOL(b, mol)
                {
                    // An electronic target does not justify cleaving compact
                    // XYZ sigma bonds (e.g. producing hydride from a methyl).
                    if (mol.GetDimension() != 3 || excess_length(&*b) > 0.20)
                        single_bonds.push_back(&*b);
                }
                if (mol.GetDimension() == 3)
                {
                    std::stable_sort(single_bonds.begin(), single_bonds.end(),
                                     [&](OBBond *a, OBBond *b)
                                     { return excess_length(a) > excess_length(b); });
                }
                if (!single_bonds.empty())
                {
                    if (check_cond())
                        return hit;
                    OBBond *single_bond = single_bonds.front();
                    OBAtom *b_at = single_bond->GetBeginAtom();
                    OBAtom *e_at = single_bond->GetEndAtom();
                    molgr::utils::SetUnpairedElectronCount(*b_at, molgr::utils::GetUnpairedElectronCount(*b_at) + 1);
                    molgr::utils::SetUnpairedElectronCount(*e_at, molgr::utils::GetUnpairedElectronCount(*e_at) + 1);
                    mol.DeleteBond(single_bond);
                    hit = true;
                    LOG_DEBUG("[BreakOneBond] Deleted single bond");
                }
            }
            return hit;
        }
    }
}

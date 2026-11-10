#include "molgr/vendor/openbabel_threading.h"
#include "molgr/vendor/openbabel_copy.h"

// Narrow thread-safe vendor implementations of the OpenBabel XYZ seed
// perception routines used by MolGR. Local perception remains concurrent;
// only native lazy perception calls that can write the process-global logger
// share the narrow lock used by native molecule copies.

#include <openbabel/elements.h>
#include <openbabel/generic.h>
#include <openbabel/kekulize.h>
#include <openbabel/parsmart.h>

#include "molgr/compat/openbabel_iter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    int NativeTotalCharge(OpenBabel::OBMol &mol)
    {
        std::lock_guard<std::recursive_mutex> lock(
            molgr::vendor::openbabel_threading::NativeOpenBabelMutex());
        return mol.GetTotalCharge();
    }

    struct AtomDistanceEntry
    {
        OpenBabel::OBAtom *atom = nullptr;
        double radius = 0.0;
    };

    bool SortAtomByZ(const AtomDistanceEntry &lhs, const AtomDistanceEntry &rhs)
    {
        return lhs.atom->GetZ() < rhs.atom->GetZ();
    }

    double Squared(double value)
    {
        return value * value;
    }

    bool IsApprox(double lhs, double rhs, double tolerance)
    {
        return std::fabs(lhs - rhs) < tolerance;
    }

    double CorrectedBondRadius(unsigned int elem, unsigned int hyb)
    {
        const double radius = OpenBabel::OBElements::GetCovalentRad(elem);
        if (hyb == 2)
        {
            return radius * 0.95;
        }
        if (hyb == 1)
        {
            return radius * 0.90;
        }
        return radius;
    }

    bool CanAddBondToNeighbor(OpenBabel::OBAtom &atom, OpenBabel::OBAtom &neighbor)
    {
        if (atom.GetExplicitValence() == 5 &&
            atom.GetAtomicNum() == OpenBabel::OBElements::Phosphorus)
        {
            return neighbor.GetAtomicNum() == OpenBabel::OBElements::Fluorine ||
                   neighbor.GetAtomicNum() == OpenBabel::OBElements::Chlorine;
        }
        return true;
    }

    std::vector<OpenBabel::OBBond *> AtomBonds(OpenBabel::OBAtom &atom)
    {
        std::vector<OpenBabel::OBBond *> bonds;
        OpenBabel::OBBondIterator bond_iter;
        for (OpenBabel::OBBond *bond = atom.BeginBond(bond_iter);
             bond != nullptr;
             bond = atom.NextBond(bond_iter))
        {
            bonds.push_back(bond);
        }
        return bonds;
    }

    int FindRingsLocal(
        OpenBabel::OBAtom *root,
        std::vector<int> &atom_visit_depth,
        std::vector<unsigned char> &bond_visited,
        unsigned int &closure_count,
        int root_depth)
    {
        struct Frame
        {
            OpenBabel::OBAtom *atom = nullptr;
            int depth = 0;
            int result = -1;
            OpenBabel::OBBondIterator iterator;
            OpenBabel::OBBond *bond = nullptr;
        };

        std::vector<Frame> stack;
        stack.reserve(64);
        Frame root_frame;
        root_frame.atom = root;
        root_frame.depth = root_depth;
        root_frame.bond = root->BeginBond(root_frame.iterator);
        stack.push_back(root_frame);

        int child_result = 0;
        bool returning = false;
        while (!stack.empty())
        {
            Frame &current = stack.back();

            if (returning)
            {
                returning = false;
                const int neighbor_visit = child_result;
                if (neighbor_visit > 0 && neighbor_visit <= current.depth)
                {
                    current.bond->SetInRing();
                    if (current.result < 0 || neighbor_visit < current.result)
                    {
                        current.result = neighbor_visit;
                    }
                }
                current.bond = current.atom->NextBond(current.iterator);
            }

            bool recursed = false;
            while (current.bond != nullptr)
            {
                OpenBabel::OBBond *bond = current.bond;
                const unsigned int bond_idx = bond->GetIdx();
                if (bond_idx >= bond_visited.size())
                {
                    current.bond = current.atom->NextBond(current.iterator);
                    continue;
                }
                if (bond_visited[bond_idx] == 0)
                {
                    bond_visited[bond_idx] = 1;
                    OpenBabel::OBAtom *neighbor = bond->GetNbrAtom(current.atom);
                    const unsigned int neighbor_idx = neighbor->GetIdx();
                    const int neighbor_visit =
                        neighbor_idx < atom_visit_depth.size()
                            ? atom_visit_depth[neighbor_idx]
                            : 0;
                    if (neighbor_visit == 0)
                    {
                        if (neighbor_idx < atom_visit_depth.size())
                        {
                            atom_visit_depth[neighbor_idx] = current.depth + 1;
                        }
                        Frame next_frame;
                        next_frame.atom = neighbor;
                        next_frame.depth = current.depth + 1;
                        next_frame.bond = neighbor->BeginBond(next_frame.iterator);
                        stack.push_back(next_frame);
                        recursed = true;
                        break;
                    }

                    if (current.result < 0 || neighbor_visit < current.result)
                    {
                        current.result = neighbor_visit;
                    }
                    bond->SetClosure();
                    bond->SetInRing();
                    ++closure_count;
                }
                current.bond = current.atom->NextBond(current.iterator);
            }

            if (recursed)
            {
                continue;
            }

            if (current.result > 0 && current.result <= current.depth)
            {
                current.atom->SetInRing();
            }
            child_result = current.result;
            returning = true;
            stack.pop_back();
        }

        return child_result;
    }

    unsigned int FindRingAtomsAndBondsLocal(OpenBabel::OBMol &mol)
    {
        mol.SetRingAtomsAndBondsPerceived();
        mol.SetClosureBondsPerceived();

        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            atom_iter->SetInRing(false);
        }
        FOR_BONDS_OF_MOL(bond_iter, mol)
        {
            bond_iter->SetInRing(false);
            bond_iter->SetClosure(false);
        }

        std::vector<unsigned char> bond_visited(
            static_cast<std::size_t>(mol.NumBonds()) + 1,
            0);
        std::vector<int> atom_visit_depth(
            static_cast<std::size_t>(mol.NumAtoms()) + 1,
            0);

        unsigned int closure_count = 0;
        for (unsigned int atom_idx = 1; atom_idx <= mol.NumAtoms(); ++atom_idx)
        {
            if (atom_visit_depth[atom_idx] != 0)
            {
                continue;
            }
            OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
            if (atom == nullptr)
            {
                continue;
            }
            atom_visit_depth[atom_idx] = 1;
            FindRingsLocal(atom, atom_visit_depth, bond_visited, closure_count, 1);
        }
        return closure_count;
    }

    unsigned int DetermineFrerejacqueLocal(OpenBabel::OBMol &mol)
    {
        if (!mol.HasClosureBondsPerceived())
        {
            return FindRingAtomsAndBondsLocal(mol);
        }

        unsigned int closure_count = 0;
        FOR_BONDS_OF_MOL(bond_iter, mol)
        {
            if (bond_iter->IsClosure())
            {
                ++closure_count;
            }
        }
        return closure_count;
    }

    OpenBabel::OBRingData *GetSssrRingData(OpenBabel::OBMol &mol)
    {
        OpenBabel::OBGenericData *existing = mol.GetData("SSSR");
        return dynamic_cast<OpenBabel::OBRingData *>(existing);
    }

    void EnsureSSSRLocal(OpenBabel::OBMol &mol)
    {
        if (mol.HasSSSRPerceived() && GetSssrRingData(mol) != nullptr)
        {
            return;
        }

        mol.SetSSSRPerceived();
        if (mol.HasData("SSSR"))
        {
            mol.DeleteData("SSSR");
        }

        std::vector<OpenBabel::OBRing *> rings;
        const unsigned int frerejacque = DetermineFrerejacqueLocal(mol);
        if (frerejacque > 0)
        {
            std::vector<OpenBabel::OBBond *> closure_bonds;
            FOR_BONDS_OF_MOL(bond_iter, mol)
            {
                if (bond_iter->IsClosure())
                {
                    closure_bonds.push_back(&*bond_iter);
                }
            }

            if (!closure_bonds.empty())
            {
                OpenBabel::OBRingSearch ring_search;
                for (OpenBabel::OBBond *bond : closure_bonds)
                {
                    ring_search.AddRingFromClosure(mol, bond);
                }
                ring_search.SortRings();
                ring_search.RemoveRedundant(static_cast<int>(frerejacque));

                for (auto ring_iter = ring_search.BeginRings();
                     ring_iter != ring_search.EndRings();
                     ++ring_iter)
                {
                    auto *ring = new OpenBabel::OBRing(
                        (*ring_iter)->_path,
                        static_cast<int>(mol.NumAtoms()) + 1);
                    ring->SetParent(&mol);
                    rings.push_back(ring);
                }
            }
        }

        auto *ring_data = new OpenBabel::OBRingData();
        ring_data->SetOrigin(OpenBabel::perceived);
        ring_data->SetAttribute("SSSR");
        ring_data->SetData(rings);
        mol.SetData(ring_data);
    }

    void RemoveExcessLocalBonds(OpenBabel::OBMol &mol, const std::vector<int> &initial_bond_counts)
    {
        bool removed = true;
        while (removed)
        {
            removed = false;
            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                OpenBabel::OBAtom &atom = *atom_iter;
                const bool over_valence =
                    atom.GetExplicitValence() >
                    OpenBabel::OBElements::GetMaxBonds(atom.GetAtomicNum());
                const bool too_small_angle =
                    atom.GetExplicitDegree() >= 2 && atom.SmallestBondAngle() < 45.0;
                if (!over_valence && !too_small_angle)
                {
                    continue;
                }

                std::vector<OpenBabel::OBBond *> bonds = AtomBonds(atom);
                const int first_new_bond_index =
                    std::min<int>(
                        initial_bond_counts.at(static_cast<std::size_t>(atom.GetIdx() - 1)),
                        static_cast<int>(bonds.size()));
                if (first_new_bond_index >= static_cast<int>(bonds.size()))
                {
                    continue;
                }

                OpenBabel::OBBond *bond_to_delete = nullptr;
                if (atom.GetAtomicNum() == OpenBabel::OBElements::Hydrogen)
                {
                    for (std::size_t idx = static_cast<std::size_t>(first_new_bond_index);
                         idx < bonds.size();
                         ++idx)
                    {
                        OpenBabel::OBAtom *neighbor = bonds[idx]->GetNbrAtom(&atom);
                        if (neighbor != nullptr &&
                            neighbor->GetAtomicNum() == OpenBabel::OBElements::Hydrogen)
                        {
                            bond_to_delete = bonds[idx];
                            break;
                        }
                    }
                }
                if (bond_to_delete == nullptr)
                {
                    bond_to_delete = bonds[static_cast<std::size_t>(first_new_bond_index)];
                    double max_length = bond_to_delete->GetLength();
                    for (std::size_t idx = static_cast<std::size_t>(first_new_bond_index) + 1;
                         idx < bonds.size();
                         ++idx)
                    {
                        const double length = bonds[idx]->GetLength();
                        if (length > max_length)
                        {
                            bond_to_delete = bonds[idx];
                            max_length = length;
                        }
                    }
                }

                if (bond_to_delete != nullptr)
                {
                    mol.DeleteBond(bond_to_delete);
                    removed = true;
                    break;
                }
            }
        }
    }

    void EstimateHybridizationsFromGeometry(OpenBabel::OBMol &mol)
    {
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            OpenBabel::OBAtom &atom = *atom_iter;
            const double angle = atom.AverageBondAngle();
            if (angle > 155.0)
            {
                atom.SetHyb(1);
            }
            else if (angle > 115.0)
            {
                atom.SetHyb(2);
            }

            if (atom.GetAtomicNum() == OpenBabel::OBElements::Nitrogen &&
                atom.ExplicitHydrogenCount() == 1 &&
                atom.GetExplicitDegree() == 2 &&
                angle > 109.5)
            {
                atom.SetHyb(2);
            }
            else if (atom.GetAtomicNum() == OpenBabel::OBElements::Nitrogen &&
                     atom.GetExplicitDegree() == 2 &&
                     atom.IsInRing())
            {
                atom.SetHyb(2);
            }
        }
        mol.SetHybridizationPerceived();
    }

    void MarkPlanarSmallRingsSp2(OpenBabel::OBMol &mol)
    {
        EnsureSSSRLocal(mol);
        for (OpenBabel::OBRing *ring : mol.GetSSSR())
        {
            if (ring == nullptr)
            {
                continue;
            }
            const std::vector<int> &path = ring->_path;
            if (path.size() == 5)
            {
                const double torsions =
                    (std::fabs(mol.GetTorsion(path[0], path[1], path[2], path[3])) +
                     std::fabs(mol.GetTorsion(path[1], path[2], path[3], path[4])) +
                     std::fabs(mol.GetTorsion(path[2], path[3], path[4], path[0])) +
                     std::fabs(mol.GetTorsion(path[3], path[4], path[0], path[1])) +
                     std::fabs(mol.GetTorsion(path[4], path[0], path[1], path[2]))) /
                    5.0;
                if (torsions <= 7.5)
                {
                    for (const int atom_idx : path)
                    {
                        OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
                        if (atom != nullptr && atom->GetExplicitDegree() == 2)
                        {
                            atom->SetHyb(2);
                        }
                    }
                }
            }
            else if (path.size() == 6)
            {
                const double torsions =
                    (std::fabs(mol.GetTorsion(path[0], path[1], path[2], path[3])) +
                     std::fabs(mol.GetTorsion(path[1], path[2], path[3], path[4])) +
                     std::fabs(mol.GetTorsion(path[2], path[3], path[4], path[5])) +
                     std::fabs(mol.GetTorsion(path[3], path[4], path[5], path[0])) +
                     std::fabs(mol.GetTorsion(path[4], path[5], path[0], path[1])) +
                     std::fabs(mol.GetTorsion(path[5], path[0], path[1], path[2]))) /
                    6.0;
                if (torsions <= 12.0)
                {
                    for (const int atom_idx : path)
                    {
                        OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
                        if (atom != nullptr &&
                            (atom->GetExplicitDegree() == 2 || atom->GetExplicitDegree() == 3))
                        {
                            atom->SetHyb(2);
                        }
                    }
                }
            }
        }
    }

    void RelaxIsolatedUnsaturatedHybridizations(OpenBabel::OBMol &mol)
    {
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            OpenBabel::OBAtom &atom = *atom_iter;
            if (atom.GetHyb() != 2 && atom.GetHyb() != 1)
            {
                continue;
            }

            bool open_neighbor = false;
            OpenBabel::OBBondIterator bond_iter;
            for (OpenBabel::OBAtom *neighbor = atom.BeginNbrAtom(bond_iter);
                 neighbor != nullptr;
                 neighbor = atom.NextNbrAtom(bond_iter))
            {
                if (neighbor->GetHyb() < 3 || neighbor->GetExplicitDegree() == 1)
                {
                    open_neighbor = true;
                    break;
                }
            }
            if (!open_neighbor && atom.GetHyb() == 2)
            {
                atom.SetHyb(3);
            }
            else if (!open_neighbor && atom.GetHyb() == 1)
            {
                atom.SetHyb(2);
            }
        }
    }

    struct FunctionalGroupBondRule
    {
        const char *smarts;
        std::vector<int> assignments;
    };

    const std::vector<FunctionalGroupBondRule> &FunctionalGroupBondRules()
    {
        static const std::vector<FunctionalGroupBondRule> rules = {
            FunctionalGroupBondRule{R"SMARTS([X2,X3]1[#6]([#7D3]2)[#6][#6][#6]2[X2,X3][#6]([#7D3]3)[#6][#6][#6]3[X2,X3][#6]([#7D3]4)[#6][#6][#6]4[X2,X3][#6]([#7D3]5)[#6][#6][#6]51)SMARTS", {0, 1, 2, 1, 2, 1, 1, 3, 1, 3, 4, 2, 4, 5, 1, 5, 2, 1, 5, 6, 2, 6, 7, 1, 7, 8, 2, 7, 9, 1, 9, 10, 2, 10, 11, 1, 11, 8, 1, 11, 12, 2, 12, 13, 1, 13, 14, 1, 13, 15, 2, 15, 16, 1, 16, 17, 2, 17, 14, 1, 17, 18, 1, 18, 19, 2, 19, 20, 1, 19, 21, 1, 21, 22, 2, 22, 23, 1, 23, 20, 2}},
            FunctionalGroupBondRule{R"SMARTS([X2,X3]1[#6]([#7D3]2)[#6][#6][#6]2[X2,X3][#6]([#7]3)[#6][#6][#6]3[X2,X3][#6]([#7D3]4)[#6][#6][#6]4[X2,X3][#6]([#7]5)[#6][#6][#6]51)SMARTS", {0, 1, 2, 1, 2, 1, 1, 3, 1, 3, 4, 2, 4, 5, 1, 5, 2, 1, 5, 6, 2, 6, 7, 1, 7, 8, 2, 7, 9, 1, 9, 10, 2, 10, 11, 1, 11, 8, 1, 11, 12, 2, 12, 13, 1, 13, 14, 1, 13, 15, 2, 15, 16, 1, 16, 17, 2, 17, 14, 1, 17, 18, 1, 18, 19, 2, 19, 20, 1, 19, 21, 1, 21, 22, 2, 22, 23, 1, 23, 20, 2}},
            FunctionalGroupBondRule{R"SMARTS([X2,X3]1[#6]([#7]2)[#6][#6][#6]2[X2,X3][#6]([#7]3)[#6][#6][#6]3[X2,X3][#6]([#7]4)[#6][#6][#6]4[X2,X3][#6]([#7]5)[#6][#6][#6]51)SMARTS", {0, 1, 2, 1, 2, 1, 1, 3, 1, 3, 4, 2, 4, 5, 1, 5, 2, 1, 5, 6, 2, 6, 7, 1, 7, 8, 2, 7, 9, 1, 9, 10, 2, 10, 11, 1, 11, 8, 1, 11, 12, 2, 12, 13, 1, 13, 14, 1, 13, 15, 2, 15, 16, 1, 16, 17, 2, 17, 14, 1, 17, 18, 1, 18, 19, 2, 19, 20, 1, 19, 21, 1, 21, 22, 2, 22, 23, 1, 23, 20, 2}},
            FunctionalGroupBondRule{R"SMARTS([#7D2][#7D2^1][#7D1])SMARTS", {0, 1, 2, 1, 2, 2}},
            FunctionalGroupBondRule{R"SMARTS([#8D1][#7D3^2]([#8D1])*)SMARTS", {0, 1, 2, 1, 2, 2, 1, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#16D4]([#8D1])([#8D1])([*!#8])([*!#8]))SMARTS", {0, 1, 2, 0, 2, 2, 0, 3, 1, 0, 4, 1}},
            FunctionalGroupBondRule{R"SMARTS([#16D4]([#8D1])([#8D1])([#8-,#8D1])([#8-,#8D1]))SMARTS", {0, 1, 2, 0, 2, 2, 0, 3, 1, 0, 4, 1}},
            FunctionalGroupBondRule{R"SMARTS([#16D4]([#16D1])([#8D1])([#8-,#8])([#8-,#8]))SMARTS", {0, 1, 2, 0, 2, 2, 0, 3, 1, 0, 4, 1}},
            FunctionalGroupBondRule{R"SMARTS([#16D3]([#8D1])([*!#8])([*!#8]))SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#16D3]([#8D1])([#8D1-])([#8D1-]))SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#16D3]([#8D1])([#8D1])([#8D1]))SMARTS", {0, 1, 2, 0, 2, 2, 0, 3, 2}},
            FunctionalGroupBondRule{R"SMARTS([#16D3]([#8D1])([#8])([#8]))SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#16D2]([#8D1])([#16D1]))SMARTS", {0, 1, 2, 0, 2, 2}},
            FunctionalGroupBondRule{R"SMARTS([#16D2]([#8D1])([*!#8]))SMARTS", {0, 1, 2, 0, 2, 1}},
            FunctionalGroupBondRule{R"SMARTS([#16D2]([#8D1])([#8D1]))SMARTS", {0, 1, 2, 0, 2, 2}},
            FunctionalGroupBondRule{R"SMARTS([#15D3]([#8D1])([#8D1])([#8D2]))SMARTS", {0, 1, 2, 0, 2, 2, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#7D2]([#8D1])([#1]))SMARTS", {0, 1, 2, 0, 2, 1}},
            FunctionalGroupBondRule{R"SMARTS([#15D4]([#8D1])(*)(*)(*))SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1, 0, 4, 1}},
            FunctionalGroupBondRule{R"SMARTS([#6D3^2]([#8D1])([#8])*)SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#8D1][#6D2^1][#8D1])SMARTS", {0, 1, 2, 1, 2, 2}},
            FunctionalGroupBondRule{R"SMARTS([#6D3^2]([#8D1;!-])([#7])*)SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#34D3^2]([#8D1])([#8])*)SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#6D3^2]([#8D1])([#16])*)SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#6D3^2]([#16D1])([#16])*)SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([CD3^2]([#16D1])([N])*)SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#6^2][#6D2^1][#6^2])SMARTS", {0, 1, 2, 1, 2, 2}},
            FunctionalGroupBondRule{R"SMARTS([#6^2][#6D2^1][#8D1])SMARTS", {0, 1, 2, 1, 2, 2}},
            FunctionalGroupBondRule{R"SMARTS([#6D1][#7D2^1]*)SMARTS", {0, 1, 3, 1, 2, 1}},
            FunctionalGroupBondRule{R"SMARTS([Nv2R][#6v3^2][#8v2])SMARTS", {0, 1, 2, 1, 2, 1}},
            FunctionalGroupBondRule{R"SMARTS([Nv2R][#6v3^2][Nv2])SMARTS", {0, 1, 2, 1, 2, 1}},
            FunctionalGroupBondRule{R"SMARTS([#6D3^2;!R]([#7D2;!R])([#7D1;!R])~[#7D1;!R])SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#6D3^2;!R]([#7D1H0;!R])([#7;!R])*)SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#6D3^2;!R]([#7D2H1;!R])([#7;!R])*)SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#6D3^2;!R]([#7D3H2;!R])([#7;!R])*)SMARTS", {0, 1, 2, 0, 2, 1, 0, 3, 1}},
            FunctionalGroupBondRule{R"SMARTS([#6D3^2;!R]([#1,#6])([#1,#6])[#7D3^2;!R]([#1])[#6])SMARTS", {0, 1, 1, 0, 2, 1, 0, 3, 2, 3, 4, 1, 3, 5, 1}},
        };
        return rules;
    }

    std::vector<std::unique_ptr<OpenBabel::OBSmartsPattern>> &ThreadLocalFunctionalGroupPatterns()
    {
        // Do not invoke Open Babel SMARTS destructors from TLS teardown.
        // This cache is bounded per worker and intentionally process-lived.
        thread_local std::vector<std::unique_ptr<OpenBabel::OBSmartsPattern>> *patterns = []()
        {
            auto *compiled_patterns =
                new std::vector<std::unique_ptr<OpenBabel::OBSmartsPattern>>();
            const auto &rules = FunctionalGroupBondRules();
            compiled_patterns->reserve(rules.size());
            for (const FunctionalGroupBondRule &rule : rules)
            {
                auto pattern = std::make_unique<OpenBabel::OBSmartsPattern>();
                if (!pattern->Init(rule.smarts))
                {
                    throw std::runtime_error(
                        std::string("Invalid MolGR vendor functional-group SMARTS: ") +
                        rule.smarts);
                }
                compiled_patterns->push_back(std::move(pattern));
            }
            return compiled_patterns;
        }();
        return *patterns;
    }

    std::vector<std::vector<int>> MatchUMapList(
        OpenBabel::OBSmartsPattern &pattern,
        OpenBabel::OBMol &mol)
    {
        if (!pattern.Match(mol))
        {
            return {};
        }
        return pattern.GetUMapList();
    }

    void AssignFunctionalGroupBondsLocal(OpenBabel::OBMol &mol)
    {
        const auto &rules = FunctionalGroupBondRules();
        auto &patterns = ThreadLocalFunctionalGroupPatterns();
        for (std::size_t rule_idx = 0; rule_idx < rules.size(); ++rule_idx)
        {
            std::vector<std::vector<int>> match_list =
                MatchUMapList(*patterns[rule_idx], mol);
            if (match_list.empty())
            {
                continue;
            }

            const FunctionalGroupBondRule &rule = rules[rule_idx];
            for (const std::vector<int> &match : match_list)
            {
                for (std::size_t assignment_idx = 0;
                     assignment_idx + 2 < rule.assignments.size();
                     assignment_idx += 3)
                {
                    const int match_atom_a = rule.assignments[assignment_idx];
                    const int match_atom_b = rule.assignments[assignment_idx + 1];
                    const int bond_order = rule.assignments[assignment_idx + 2];
                    if (match_atom_a < 0 ||
                        match_atom_b < 0 ||
                        static_cast<std::size_t>(match_atom_a) >= match.size() ||
                        static_cast<std::size_t>(match_atom_b) >= match.size())
                    {
                        continue;
                    }

                    OpenBabel::OBAtom *atom_a = mol.GetAtom(match[static_cast<std::size_t>(match_atom_a)]);
                    OpenBabel::OBAtom *atom_b = mol.GetAtom(match[static_cast<std::size_t>(match_atom_b)]);
                    if (atom_a == nullptr || atom_b == nullptr)
                    {
                        continue;
                    }
                    OpenBabel::OBBond *bond = atom_a->GetBond(atom_b);
                    if (bond != nullptr)
                    {
                        bond->SetBondOrder(bond_order);
                    }
                }
            }
        }

        OpenBabel::OBSmartsPattern carbo;
        carbo.Init("[#8D1;!-][#6](*)(*)");
        std::vector<std::vector<int>> match_list = MatchUMapList(carbo, mol);
        if (!match_list.empty())
        {
            for (const std::vector<int> &match : match_list)
            {
                OpenBabel::OBAtom *oxygen = mol.GetAtom(match[0]);
                OpenBabel::OBAtom *carbon = mol.GetAtom(match[1]);
                if (oxygen == nullptr || carbon == nullptr)
                {
                    continue;
                }
                const double angle = carbon->AverageBondAngle();
                const double distance = oxygen->GetDistance(carbon);
                if (angle > 115.0 && angle < 150.0 && distance < 1.28 && !oxygen->HasDoubleBond())
                {
                    OpenBabel::OBBond *bond = oxygen->GetBond(carbon);
                    if (bond != nullptr)
                    {
                        bond->SetBondOrder(2);
                    }
                }
            }
        }

        OpenBabel::OBSmartsPattern thione;
        thione.Init("[#16D1][#6](*)(*)");
        match_list = MatchUMapList(thione, mol);
        if (!match_list.empty())
        {
            for (const std::vector<int> &match : match_list)
            {
                OpenBabel::OBAtom *sulfur = mol.GetAtom(match[0]);
                OpenBabel::OBAtom *carbon = mol.GetAtom(match[1]);
                if (sulfur == nullptr || carbon == nullptr)
                {
                    continue;
                }
                const double angle = carbon->AverageBondAngle();
                const double distance = sulfur->GetDistance(carbon);
                if (angle > 115.0 && angle < 150.0 && distance < 1.72 && !sulfur->HasDoubleBond())
                {
                    OpenBabel::OBBond *bond = sulfur->GetBond(carbon);
                    if (bond != nullptr)
                    {
                        bond->SetBondOrder(2);
                    }
                }
            }
        }

        OpenBabel::OBSmartsPattern isocyanate;
        isocyanate.Init("[#8,#16;D1][#6D2][#7D2]");
        match_list = MatchUMapList(isocyanate, mol);
        if (!match_list.empty())
        {
            for (const std::vector<int> &match : match_list)
            {
                OpenBabel::OBAtom *terminal = mol.GetAtom(match[0]);
                OpenBabel::OBAtom *carbon = mol.GetAtom(match[1]);
                OpenBabel::OBAtom *nitrogen = mol.GetAtom(match[2]);
                if (terminal == nullptr || carbon == nullptr || nitrogen == nullptr)
                {
                    continue;
                }
                const double angle = carbon->AverageBondAngle();
                const double terminal_distance = terminal->GetDistance(carbon);
                const double nitrogen_distance = carbon->GetDistance(nitrogen);
                const bool terminal_distance_ok =
                    terminal->GetAtomicNum() == OpenBabel::OBElements::Oxygen
                        ? terminal_distance < 1.28
                        : terminal_distance < 1.72;
                if (angle > 150.0 && terminal_distance_ok && nitrogen_distance < 1.34)
                {
                    OpenBabel::OBBond *terminal_bond = terminal->GetBond(carbon);
                    OpenBabel::OBBond *nitrogen_bond = carbon->GetBond(nitrogen);
                    if (terminal_bond != nullptr && nitrogen_bond != nullptr)
                    {
                        terminal_bond->SetBondOrder(2);
                        nitrogen_bond->SetBondOrder(2);
                    }
                }
            }
        }

        OpenBabel::OBSmartsPattern oxime;
        oxime.Init("[#6D3][#7D2][#8D2]");
        match_list = MatchUMapList(oxime, mol);
        if (!match_list.empty())
        {
            for (const std::vector<int> &match : match_list)
            {
                OpenBabel::OBAtom *carbon = mol.GetAtom(match[0]);
                OpenBabel::OBAtom *nitrogen = mol.GetAtom(match[1]);
                if (carbon == nullptr || nitrogen == nullptr)
                {
                    continue;
                }
                const double angle = nitrogen->AverageBondAngle();
                const double distance = carbon->GetDistance(nitrogen);
                if (angle > 110.0 && angle < 150.0 && distance < 1.4 && !carbon->HasDoubleBond())
                {
                    OpenBabel::OBBond *bond = carbon->GetBond(nitrogen);
                    if (bond != nullptr)
                    {
                        bond->SetBondOrder(2);
                    }
                }
            }
        }

        OpenBabel::OBSmartsPattern oxidopyr;
        oxidopyr.Init("[#8D1][#7D3r6]");
        match_list = MatchUMapList(oxidopyr, mol);
        if (!match_list.empty())
        {
            for (const std::vector<int> &match : match_list)
            {
                OpenBabel::OBAtom *oxygen = mol.GetAtom(match[0]);
                OpenBabel::OBAtom *nitrogen = mol.GetAtom(match[1]);
                if (oxygen == nullptr || nitrogen == nullptr)
                {
                    continue;
                }
                const double angle = nitrogen->AverageBondAngle();
                const double distance = oxygen->GetDistance(nitrogen);
                if (angle > 110.0 && angle < 150.0 && distance < 1.35)
                {
                    oxygen->SetFormalCharge(-1);
                    nitrogen->SetFormalCharge(+1);
                }
            }
        }
    }

    void KekulizePlanarAromaticRings(OpenBabel::OBMol &mol)
    {
        EnsureSSSRLocal(mol);
        bool needs_kekulization = false;
        for (OpenBabel::OBRing *ring : mol.GetSSSR())
        {
            if (ring == nullptr)
            {
                continue;
            }
            const std::vector<int> &path = ring->_path;
            if (path.size() != 5 && path.size() != 6 && path.size() != 7)
            {
                continue;
            }

            bool typed = false;
            for (const int atom_idx : path)
            {
                OpenBabel::OBAtom *atom = mol.GetAtom(atom_idx);
                if (atom == nullptr ||
                    atom->HasBondOfOrder(2) ||
                    atom->HasBondOfOrder(3) ||
                    atom->GetHyb() != 2)
                {
                    typed = true;
                    break;
                }
            }

            if (typed)
            {
                continue;
            }
            for (std::size_t idx = 0; idx < path.size(); ++idx)
            {
                OpenBabel::OBBond *bond = mol.GetBond(path[idx], path[(idx + 1) % path.size()]);
                if (bond != nullptr)
                {
                    bond->SetAromatic();
                    needs_kekulization = true;
                }
            }
        }

        if (!needs_kekulization)
        {
            return;
        }

        mol.SetAromaticPerceived();
        FOR_BONDS_OF_MOL(bond_iter, mol)
        {
            if (bond_iter->IsAromatic())
            {
                bond_iter->GetBeginAtom()->SetAromatic();
                bond_iter->GetEndAtom()->SetAromatic();
            }
        }
        OpenBabel::OBKekulize(&mol);
        mol.SetAromaticPerceived(false);
    }

    enum class ExocyclicAtom
    {
        NONE,
        OXYGEN,
        NONOXYGEN,
    };

    ExocyclicAtom FindExocyclicAtom(OpenBabel::OBAtom *atom)
    {
        OpenBabel::OBBondIterator bond_iter;
        for (OpenBabel::OBBond *bond = atom->BeginBond(bond_iter);
             bond != nullptr;
             bond = atom->NextBond(bond_iter))
        {
            if (bond->GetBondOrder() != 2 || bond->IsInRing())
            {
                continue;
            }
            const unsigned int atomic_num = bond->GetNbrAtom(atom)->GetAtomicNum();
            if (atomic_num == OpenBabel::OBElements::Oxygen)
            {
                return ExocyclicAtom::OXYGEN;
            }
            return ExocyclicAtom::NONOXYGEN;
        }
        return ExocyclicAtom::NONE;
    }

    bool HasExocyclicBondToOxygenMinus(OpenBabel::OBAtom *atom)
    {
        OpenBabel::OBBondIterator bond_iter;
        for (OpenBabel::OBBond *bond = atom->BeginBond(bond_iter);
             bond != nullptr;
             bond = atom->NextBond(bond_iter))
        {
            if (bond->GetBondOrder() != 1 || bond->IsInRing())
            {
                continue;
            }
            OpenBabel::OBAtom *neighbor = bond->GetNbrAtom(atom);
            if (neighbor->GetAtomicNum() == OpenBabel::OBElements::Oxygen &&
                neighbor->GetFormalCharge() == -1)
            {
                return true;
            }
        }
        return false;
    }

    bool HasExocyclicDoubleBondToOxygen(OpenBabel::OBAtom *atom)
    {
        OpenBabel::OBBondIterator bond_iter;
        for (OpenBabel::OBBond *bond = atom->BeginBond(bond_iter);
             bond != nullptr;
             bond = atom->NextBond(bond_iter))
        {
            if (bond->GetBondOrder() == 2 &&
                !bond->IsInRing() &&
                bond->GetNbrAtom(atom)->GetAtomicNum() == OpenBabel::OBElements::Oxygen)
            {
                return true;
            }
        }
        return false;
    }

    bool HasExocyclicDoubleBondToHeteroatom(OpenBabel::OBAtom *atom)
    {
        OpenBabel::OBBondIterator bond_iter;
        for (OpenBabel::OBBond *bond = atom->BeginBond(bond_iter);
             bond != nullptr;
             bond = atom->NextBond(bond_iter))
        {
            if (bond->GetBondOrder() != 2 || bond->IsInRing())
            {
                continue;
            }
            const unsigned int atomic_num = bond->GetNbrAtom(atom)->GetAtomicNum();
            if (atomic_num != OpenBabel::OBElements::Carbon &&
                atomic_num != OpenBabel::OBElements::Hydrogen)
            {
                return true;
            }
        }
        return false;
    }

    bool AssignOpenBabelAromaticityContribution(
        OpenBabel::OBAtom *atom,
        int &min_electrons,
        int &max_electrons)
    {
        if (!atom->IsInRing())
        {
            min_electrons = 0;
            max_electrons = 0;
            return false;
        }

        const unsigned int elem = atom->GetAtomicNum();
        const int charge = atom->GetFormalCharge();
        const unsigned int degree = atom->GetExplicitDegree() + atom->GetImplicitHCount();
        const unsigned int valence = atom->GetExplicitValence() + atom->GetImplicitHCount();

        switch (elem)
        {
        case OpenBabel::OBElements::Carbon:
            switch (charge)
            {
            case 0:
                if (valence == 4 && degree == 3)
                {
                    min_electrons = HasExocyclicDoubleBondToHeteroatom(atom) ? 0 : 1;
                    max_electrons = min_electrons;
                    return true;
                }
                break;
            case 1:
                if (valence == 3)
                {
                    if (degree == 3)
                    {
                        min_electrons = 0;
                        max_electrons = 0;
                        return true;
                    }
                    if (degree == 2)
                    {
                        min_electrons = 1;
                        max_electrons = 1;
                        return true;
                    }
                }
                break;
            case -1:
                if (valence == 3)
                {
                    if (degree == 3)
                    {
                        min_electrons = 2;
                        max_electrons = 2;
                        return true;
                    }
                    if (degree == 2)
                    {
                        min_electrons = 1;
                        max_electrons = 1;
                        return true;
                    }
                }
                break;
            default:
                break;
            }
            break;

        case OpenBabel::OBElements::Nitrogen:
        case OpenBabel::OBElements::Phosphorus:
            switch (charge)
            {
            case 0:
                if (valence == 3)
                {
                    if (degree == 3)
                    {
                        min_electrons = 2;
                        max_electrons = 2;
                        return true;
                    }
                    if (degree == 2)
                    {
                        min_electrons = 1;
                        max_electrons = 1;
                        return true;
                    }
                }
                else if (valence == 5 && degree == 3)
                {
                    const ExocyclicAtom exocyclic_atom = FindExocyclicAtom(atom);
                    if (exocyclic_atom == ExocyclicAtom::OXYGEN)
                    {
                        min_electrons = 1;
                        max_electrons = 1;
                        return true;
                    }
                    if (exocyclic_atom == ExocyclicAtom::NONOXYGEN)
                    {
                        min_electrons = 2;
                        max_electrons = 2;
                        return true;
                    }
                }
                break;
            case 1:
                if (valence == 4 && degree == 3)
                {
                    min_electrons = 1;
                    max_electrons = 1;
                    return true;
                }
                break;
            case -1:
                if (valence == 2 && degree == 2)
                {
                    min_electrons = 2;
                    max_electrons = 2;
                    return true;
                }
                break;
            default:
                break;
            }
            break;

        case OpenBabel::OBElements::Oxygen:
        case OpenBabel::OBElements::Selenium:
            if (charge == 0 && valence == 2 && degree == 2)
            {
                min_electrons = 2;
                max_electrons = 2;
                return true;
            }
            if (charge == 1 && valence == 3 && degree == 2)
            {
                min_electrons = 1;
                max_electrons = 1;
                return true;
            }
            break;

        case OpenBabel::OBElements::Sulfur:
            if (charge == 0)
            {
                if (valence == 2 && degree == 2)
                {
                    min_electrons = 2;
                    max_electrons = 2;
                    return true;
                }
                if (valence == 4 && degree == 3 && HasExocyclicDoubleBondToOxygen(atom))
                {
                    min_electrons = 2;
                    max_electrons = 2;
                    return true;
                }
            }
            else if (charge == 1 && valence == 3)
            {
                if (degree == 2)
                {
                    min_electrons = 1;
                    max_electrons = 1;
                    return true;
                }
                if (degree == 3 && HasExocyclicBondToOxygenMinus(atom))
                {
                    min_electrons = 2;
                    max_electrons = 2;
                    return true;
                }
            }
            break;

        case OpenBabel::OBElements::Boron:
            if (charge == 0 && valence == 3)
            {
                if (degree == 2)
                {
                    min_electrons = 1;
                    max_electrons = 1;
                    return true;
                }
                if (degree == 3)
                {
                    min_electrons = 0;
                    max_electrons = 0;
                    return true;
                }
            }
            break;

        case OpenBabel::OBElements::Arsenic:
            if (charge == 0 && valence == 3)
            {
                if (degree == 2)
                {
                    min_electrons = 1;
                    max_electrons = 1;
                    return true;
                }
                if (degree == 3)
                {
                    min_electrons = 2;
                    max_electrons = 2;
                    return true;
                }
            }
            else if (charge == 1 && valence == 4 && degree == 3)
            {
                min_electrons = 1;
                max_electrons = 1;
                return true;
            }
            break;

        case 0:
            if (charge == 0)
            {
                if (valence == 2 && (degree == 2 || degree == 3))
                {
                    min_electrons = 0;
                    max_electrons = 2;
                    return true;
                }
                if (valence == 3 && (degree == 2 || degree == 3))
                {
                    min_electrons = 0;
                    max_electrons = 1;
                    return true;
                }
            }
            break;

        default:
            break;
        }

        min_electrons = 0;
        max_electrons = 0;
        return false;
    }

    class LocalAromaticityState
    {
    public:
        explicit LocalAromaticityState(OpenBabel::OBMol &mol_in)
            : mol(mol_in),
              potential_aromatic(mol.NumAtoms() + 1),
              visited(mol.NumAtoms() + 1),
              root(mol.NumAtoms() + 1),
              electron_range(mol.NumAtoms() + 1)
        {
        }

        void Assign()
        {
            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                atom_iter->SetAromatic(false);
            }
            FOR_BONDS_OF_MOL(bond_iter, mol)
            {
                bond_iter->SetAromatic(false);
            }

            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                const unsigned int idx = atom_iter->GetIdx();
                potential_aromatic[idx] = AssignOpenBabelAromaticityContribution(
                    &(*atom_iter),
                    electron_range[idx].first,
                    electron_range[idx].second);
            }

            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                if (potential_aromatic[atom_iter->GetIdx()])
                {
                    PropagatePotentialAromatic(&(*atom_iter));
                }
            }

            SelectRootAtoms(true);
            ExcludeSmallRing();

            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                if (root[atom_iter->GetIdx()])
                {
                    CheckAromaticity(&(*atom_iter), 14);
                }
            }
        }

    private:
        OpenBabel::OBMol &mol;
        std::vector<bool> potential_aromatic;
        std::vector<bool> visited;
        std::vector<bool> root;
        std::vector<std::pair<int, int>> electron_range;

        bool TraverseCycle(
            OpenBabel::OBAtom *root_atom,
            OpenBabel::OBAtom *atom,
            OpenBabel::OBBond *previous_bond,
            std::pair<int, int> &range,
            int depth)
        {
            if (atom == root_atom)
            {
                for (int electrons = range.first; electrons <= range.second; ++electrons)
                {
                    if (electrons % 4 == 2 && electrons > 2)
                    {
                        return true;
                    }
                }
                return false;
            }

            if (depth == 0 || !potential_aromatic[atom->GetIdx()] || visited[atom->GetIdx()])
            {
                return false;
            }

            bool result = false;
            --depth;
            range.first += electron_range[atom->GetIdx()].first;
            range.second += electron_range[atom->GetIdx()].second;
            visited[atom->GetIdx()] = true;

            OpenBabel::OBBondIterator bond_iter;
            for (OpenBabel::OBAtom *neighbor = atom->BeginNbrAtom(bond_iter);
                 neighbor != nullptr;
                 neighbor = atom->NextNbrAtom(bond_iter))
            {
                OpenBabel::OBBond *bond = *bond_iter;
                if (bond == previous_bond ||
                    !bond->IsInRing() ||
                    !potential_aromatic[neighbor->GetIdx()])
                {
                    continue;
                }
                if (TraverseCycle(root_atom, neighbor, bond, range, depth))
                {
                    result = true;
                    bond->SetAromatic();
                }
            }

            visited[atom->GetIdx()] = false;
            if (result)
            {
                atom->SetAromatic();
            }
            range.first -= electron_range[atom->GetIdx()].first;
            range.second -= electron_range[atom->GetIdx()].second;
            return result;
        }

        void CheckAromaticity(OpenBabel::OBAtom *atom, int depth)
        {
            OpenBabel::OBBondIterator bond_iter;
            for (OpenBabel::OBAtom *neighbor = atom->BeginNbrAtom(bond_iter);
                 neighbor != nullptr;
                 neighbor = atom->NextNbrAtom(bond_iter))
            {
                OpenBabel::OBBond *bond = *bond_iter;
                if (!bond->IsInRing())
                {
                    continue;
                }
                std::pair<int, int> range = electron_range[atom->GetIdx()];
                if (TraverseCycle(atom, neighbor, bond, range, depth - 1))
                {
                    atom->SetAromatic();
                    bond->SetAromatic();
                }
            }
        }

        void PropagatePotentialAromatic(OpenBabel::OBAtom *atom)
        {
            int count = 0;
            OpenBabel::OBBondIterator bond_iter;
            for (OpenBabel::OBAtom *neighbor = atom->BeginNbrAtom(bond_iter);
                 neighbor != nullptr;
                 neighbor = atom->NextNbrAtom(bond_iter))
            {
                OpenBabel::OBBond *bond = *bond_iter;
                if (bond->IsInRing() && potential_aromatic[neighbor->GetIdx()])
                {
                    ++count;
                }
            }

            if (count >= 2)
            {
                return;
            }
            potential_aromatic[atom->GetIdx()] = false;
            if (count == 1)
            {
                OpenBabel::OBBondIterator recurse_iter;
                for (OpenBabel::OBAtom *neighbor = atom->BeginNbrAtom(recurse_iter);
                     neighbor != nullptr;
                     neighbor = atom->NextNbrAtom(recurse_iter))
                {
                    OpenBabel::OBBond *bond = *recurse_iter;
                    if (bond->IsInRing() && potential_aromatic[neighbor->GetIdx()])
                    {
                        PropagatePotentialAromatic(neighbor);
                    }
                }
            }
        }

        void SelectRootAtoms(bool avoid_inner_ring_atoms)
        {
            std::vector<OpenBabel::OBBond *> closure_bonds;
            std::vector<int> root_atom_candidates;
            std::vector<std::vector<OpenBabel::OBRing *>> rings_by_atom;

            OpenBabel::OBBondIterator bond_iter;
            for (OpenBabel::OBBond *bond = mol.BeginBond(bond_iter);
                 bond != nullptr;
                 bond = mol.NextBond(bond_iter))
            {
                if (!bond->IsClosure())
                {
                    continue;
                }
                closure_bonds.push_back(bond);
                if (avoid_inner_ring_atoms)
                {
                    root_atom_candidates.push_back(static_cast<int>(bond->GetBeginAtomIdx()));
                }
            }

            if (avoid_inner_ring_atoms)
            {
                rings_by_atom.resize(static_cast<std::size_t>(mol.NumAtoms()) + 1);
                for (OpenBabel::OBRing *ring : mol.GetSSSR())
                {
                    if (ring == nullptr)
                    {
                        continue;
                    }
                    for (const int atom_idx : ring->_path)
                    {
                        rings_by_atom[static_cast<std::size_t>(atom_idx)].push_back(ring);
                    }
                }
            }

            for (OpenBabel::OBBond *bond : closure_bonds)
            {
                const int root_atom_idx = static_cast<int>(bond->GetBeginAtomIdx());
                root[static_cast<std::size_t>(root_atom_idx)] = true;
                if (!avoid_inner_ring_atoms)
                {
                    continue;
                }

                OpenBabel::OBAtom *atom = mol.GetAtom(root_atom_idx);
                if (atom == nullptr)
                {
                    continue;
                }

                int ring_neighbor_count = 0;
                OpenBabel::OBBondIterator neighbor_iter;
                for (OpenBabel::OBAtom *neighbor = atom->BeginNbrAtom(neighbor_iter);
                     neighbor != nullptr;
                     neighbor = atom->NextNbrAtom(neighbor_iter))
                {
                    if (neighbor->GetAtomicNum() == OpenBabel::OBElements::Hydrogen)
                    {
                        continue;
                    }
                    if (neighbor->IsInRing())
                    {
                        ++ring_neighbor_count;
                    }

                    if (ring_neighbor_count <= 2)
                    {
                        continue;
                    }

                    int new_root = -1;
                    for (OpenBabel::OBRing *ring :
                         rings_by_atom[static_cast<std::size_t>(root_atom_idx)])
                    {
                        if (ring == nullptr)
                        {
                            continue;
                        }

                        int ring_root_count = 0;
                        for (const int candidate_idx : root_atom_candidates)
                        {
                            if (ring->IsInRing(candidate_idx))
                            {
                                ++ring_root_count;
                                if (ring_root_count >= 2)
                                {
                                    break;
                                }
                            }
                        }
                        if (ring_root_count >= 2)
                        {
                            continue;
                        }

                        bool check_this_ring = false;
                        for (const int ring_atom_idx : ring->_path)
                        {
                            if (ring_atom_idx == root_atom_idx)
                            {
                                check_this_ring = true;
                            }
                            else if (root[static_cast<std::size_t>(ring_atom_idx)])
                            {
                                check_this_ring = false;
                                break;
                            }
                        }
                        if (!check_this_ring)
                        {
                            continue;
                        }

                        for (const int ring_atom_idx : ring->_path)
                        {
                            OpenBabel::OBAtom *ring_atom = mol.GetAtom(ring_atom_idx);
                            if (ring_atom == nullptr)
                            {
                                continue;
                            }
                            int candidate_ring_neighbors = 0;
                            OpenBabel::OBBondIterator ring_neighbor_iter;
                            for (OpenBabel::OBAtom *ring_neighbor =
                                     ring_atom->BeginNbrAtom(ring_neighbor_iter);
                                 ring_neighbor != nullptr;
                                 ring_neighbor = ring_atom->NextNbrAtom(ring_neighbor_iter))
                            {
                                if (ring_neighbor->GetAtomicNum() != OpenBabel::OBElements::Hydrogen &&
                                    ring_neighbor->IsInRing())
                                {
                                    ++candidate_ring_neighbors;
                                }
                            }
                            if (candidate_ring_neighbors <= 2 && ring->IsInRing(ring_atom->GetIdx()))
                            {
                                new_root = ring_atom_idx;
                            }
                        }
                    }

                    if (new_root != -1 && root_atom_idx != new_root)
                    {
                        root[static_cast<std::size_t>(root_atom_idx)] = false;
                        root[static_cast<std::size_t>(new_root)] = true;
                    }
                }
            }
        }

        void ExcludeSmallRing()
        {
            FOR_ATOMS_OF_MOL(atom_iter, mol)
            {
                OpenBabel::OBAtom *atom = &(*atom_iter);
                if (!root[atom->GetIdx()])
                {
                    continue;
                }
                OpenBabel::OBBondIterator nbr1_iter;
                for (OpenBabel::OBAtom *nbr1 = atom->BeginNbrAtom(nbr1_iter);
                     nbr1 != nullptr;
                     nbr1 = atom->NextNbrAtom(nbr1_iter))
                {
                    OpenBabel::OBBond *bond1 = *nbr1_iter;
                    if (!bond1->IsInRing() || !potential_aromatic[nbr1->GetIdx()])
                    {
                        continue;
                    }
                    OpenBabel::OBBondIterator nbr2_iter;
                    for (OpenBabel::OBAtom *nbr2 = nbr1->BeginNbrAtom(nbr2_iter);
                         nbr2 != nullptr;
                         nbr2 = nbr1->NextNbrAtom(nbr2_iter))
                    {
                        OpenBabel::OBBond *bond2 = *nbr2_iter;
                        if (nbr2 != atom &&
                            bond2->IsInRing() &&
                            potential_aromatic[nbr2->GetIdx()] &&
                            atom->IsConnected(nbr2))
                        {
                            root[atom->GetIdx()] = false;
                        }
                    }
                }
            }
        }
    };

    void AssignAromaticFlagsLocal(OpenBabel::OBMol &mol)
    {
        if (mol.HasAromaticPerceived())
        {
            return;
        }
        EnsureSSSRLocal(mol);
        mol.SetAromaticPerceived();
        LocalAromaticityState state(mol);
        state.Assign();
    }

    std::vector<OpenBabel::OBAtom *> ElectronegativitySortedAtoms(OpenBabel::OBMol &mol)
    {
        std::vector<std::pair<OpenBabel::OBAtom *, double>> sortable_atoms;
        sortable_atoms.reserve(static_cast<std::size_t>(mol.NumAtoms()));
        FOR_ATOMS_OF_MOL(atom_iter, mol)
        {
            OpenBabel::OBAtom *atom = &(*atom_iter);
            double shortest_bond = 1.0e5;
            OpenBabel::OBBondIterator bond_iter;
            for (OpenBabel::OBAtom *neighbor = atom->BeginNbrAtom(bond_iter);
                 neighbor != nullptr;
                 neighbor = atom->NextNbrAtom(bond_iter))
            {
                if (neighbor->GetAtomicNum() != OpenBabel::OBElements::Hydrogen)
                {
                    OpenBabel::OBBond *bond = atom->GetBond(neighbor);
                    if (bond != nullptr)
                    {
                        shortest_bond = std::min(shortest_bond, bond->GetLength());
                    }
                }
            }
            sortable_atoms.emplace_back(
                atom,
                OpenBabel::OBElements::GetElectroNeg(atom->GetAtomicNum()) * 1.0e6 +
                    shortest_bond);
        }

        std::sort(
            sortable_atoms.begin(),
            sortable_atoms.end(),
            [](const auto &lhs, const auto &rhs)
            {
                return lhs.second < rhs.second;
            });

        std::vector<OpenBabel::OBAtom *> atoms;
        atoms.reserve(sortable_atoms.size());
        for (const auto &item : sortable_atoms)
        {
            atoms.push_back(item.first);
        }
        return atoms;
    }

    void AssignRemainingMultipleBonds(OpenBabel::OBMol &mol)
    {
        const std::vector<OpenBabel::OBAtom *> sorted_atoms = ElectronegativitySortedAtoms(mol);
        for (OpenBabel::OBAtom *atom : sorted_atoms)
        {
            if (atom == nullptr)
            {
                continue;
            }

            if ((atom->GetHyb() == 1 || atom->GetExplicitDegree() == 1) &&
                atom->GetExplicitValence() + 2 <=
                    OpenBabel::OBElements::GetMaxBonds(atom->GetAtomicNum()))
            {
                if (atom->HasNonSingleBond() ||
                    (atom->GetAtomicNum() == OpenBabel::OBElements::Nitrogen &&
                     atom->GetExplicitValence() + 2 > 3))
                {
                    continue;
                }

                OpenBabel::OBAtom *selected = nullptr;
                double max_electronegativity = 0.0;
                double shortest_bond = 5000.0;
                OpenBabel::OBBondIterator bond_iter;
                for (OpenBabel::OBAtom *neighbor = atom->BeginNbrAtom(bond_iter);
                     neighbor != nullptr;
                     neighbor = atom->NextNbrAtom(bond_iter))
                {
                    const double current_electronegativity =
                        OpenBabel::OBElements::GetElectroNeg(neighbor->GetAtomicNum());
                    OpenBabel::OBBond *bond = atom->GetBond(neighbor);
                    if (bond == nullptr)
                    {
                        continue;
                    }
                    if ((neighbor->GetHyb() == 1 || neighbor->GetExplicitDegree() == 1) &&
                        neighbor->GetExplicitValence() + 2 <=
                            OpenBabel::OBElements::GetMaxBonds(neighbor->GetAtomicNum()) &&
                        (current_electronegativity > max_electronegativity ||
                         (IsApprox(current_electronegativity, max_electronegativity, 1.0e-6) &&
                          bond->GetLength() < shortest_bond)))
                    {
                        if (neighbor->HasNonSingleBond() ||
                            (neighbor->GetAtomicNum() == OpenBabel::OBElements::Nitrogen &&
                             neighbor->GetExplicitValence() + 2 > 3))
                        {
                            continue;
                        }

                        const double bond_length = bond->GetLength();
                        if (atom->GetExplicitDegree() == 1 || neighbor->GetExplicitDegree() == 1)
                        {
                            const double test_length =
                                CorrectedBondRadius(atom->GetAtomicNum(), atom->GetHyb()) +
                                CorrectedBondRadius(neighbor->GetAtomicNum(), neighbor->GetHyb());
                            if (bond_length > 0.9 * test_length)
                            {
                                continue;
                            }
                        }

                        shortest_bond = bond_length;
                        max_electronegativity = current_electronegativity;
                        selected = neighbor;
                    }
                }
                if (selected != nullptr)
                {
                    atom->GetBond(selected)->SetBondOrder(3);
                }
            }
            else if ((atom->GetHyb() == 2 || atom->GetExplicitDegree() == 1) &&
                     atom->GetExplicitValence() + 1 <=
                         OpenBabel::OBElements::GetMaxBonds(atom->GetAtomicNum()))
            {
                if (atom->HasNonSingleBond() ||
                    (atom->GetAtomicNum() == OpenBabel::OBElements::Nitrogen &&
                     atom->GetExplicitValence() + 1 > 3))
                {
                    continue;
                }

                if (atom->IsInRing() && atom->GetAtomicNum() == OpenBabel::OBElements::Sulfur)
                {
                    if (NativeTotalCharge(mol) > 1 && atom->GetFormalCharge() == 0)
                    {
                        atom->SetFormalCharge(+1);
                    }
                    else
                    {
                        continue;
                    }
                }

                OpenBabel::OBAtom *selected = nullptr;
                double max_electronegativity = 0.0;
                double shortest_bond = 5000.0;
                OpenBabel::OBBondIterator bond_iter;
                for (OpenBabel::OBAtom *neighbor = atom->BeginNbrAtom(bond_iter);
                     neighbor != nullptr;
                     neighbor = atom->NextNbrAtom(bond_iter))
                {
                    const double current_electronegativity =
                        OpenBabel::OBElements::GetElectroNeg(neighbor->GetAtomicNum());
                    OpenBabel::OBBond *bond = atom->GetBond(neighbor);
                    if (bond == nullptr)
                    {
                        continue;
                    }
                    if ((neighbor->GetHyb() == 2 || neighbor->GetExplicitDegree() == 1) &&
                        neighbor->GetExplicitValence() + 1 <=
                            OpenBabel::OBElements::GetMaxBonds(neighbor->GetAtomicNum()) &&
                        bond->IsDoubleBondGeometry() &&
                        (current_electronegativity > max_electronegativity ||
                         IsApprox(current_electronegativity, max_electronegativity, 1.0e-6)))
                    {
                        if (neighbor->HasNonSingleBond() ||
                            (neighbor->GetAtomicNum() == OpenBabel::OBElements::Nitrogen &&
                             neighbor->GetExplicitValence() + 1 > 3))
                        {
                            continue;
                        }

                        if (neighbor->IsInRing() &&
                            neighbor->GetAtomicNum() == OpenBabel::OBElements::Sulfur)
                        {
                            if (NativeTotalCharge(mol) > 1 && neighbor->GetFormalCharge() == 0)
                            {
                                neighbor->SetFormalCharge(+1);
                            }
                            else
                            {
                                continue;
                            }
                        }

                        const double bond_length = bond->GetLength();
                        if (atom->GetExplicitDegree() == 1 || neighbor->GetExplicitDegree() == 1)
                        {
                            const double test_length =
                                CorrectedBondRadius(atom->GetAtomicNum(), atom->GetHyb()) +
                                CorrectedBondRadius(neighbor->GetAtomicNum(), neighbor->GetHyb());
                            if (bond_length > 0.93 * test_length)
                            {
                                continue;
                            }
                        }

                        const double difference = shortest_bond - bond_length;
                        if (difference > 0.1 ||
                            (difference > -0.01 &&
                             ((!atom->IsInRing() ||
                               selected == nullptr ||
                               !selected->IsInRing() ||
                               neighbor->IsInRing()) ||
                              (atom->IsInRing() &&
                               selected != nullptr &&
                               !selected->IsInRing() &&
                               neighbor->IsInRing()))))
                        {
                            shortest_bond = bond_length;
                            max_electronegativity = current_electronegativity;
                            selected = neighbor;
                        }
                    }
                }
                if (selected != nullptr)
                {
                    atom->GetBond(selected)->SetBondOrder(2);
                }
            }
        }
    }
}

namespace molgr
{
    namespace vendor
    {
        namespace openbabel_threading
        {
            void ConnectTheDotsLocal(OpenBabel::OBMol &mol)
            {
                if (mol.Empty() || mol.GetDimension() != 3)
                {
                    return;
                }

                std::vector<int> initial_bond_counts;
                initial_bond_counts.reserve(static_cast<std::size_t>(mol.NumAtoms()));

                std::vector<AtomDistanceEntry> sorted_atoms;
                sorted_atoms.reserve(static_cast<std::size_t>(mol.NumAtoms()));
                double max_radius = 0.0;

                FOR_ATOMS_OF_MOL(atom_iter, mol)
                {
                    OpenBabel::OBAtom *atom = &(*atom_iter);
                    initial_bond_counts.push_back(static_cast<int>(atom->GetExplicitDegree()));

                    if (atom->GetExplicitValence() >=
                        OpenBabel::OBElements::GetMaxBonds(atom->GetAtomicNum()))
                    {
                        continue;
                    }
                    if (atom->GetAtomicNum() == OpenBabel::OBElements::Nitrogen &&
                        atom->GetFormalCharge() == 0 &&
                        atom->GetExplicitValence() >= 3)
                    {
                        continue;
                    }

                    const double radius = OpenBabel::OBElements::GetCovalentRad(atom->GetAtomicNum());
                    sorted_atoms.push_back(AtomDistanceEntry{atom, radius});
                    max_radius = std::max(max_radius, radius);
                }

                std::sort(sorted_atoms.begin(), sorted_atoms.end(), SortAtomByZ);

                for (std::size_t left_idx = 0; left_idx < sorted_atoms.size(); ++left_idx)
                {
                    OpenBabel::OBAtom *left_atom = sorted_atoms[left_idx].atom;
                    const double left_radius = sorted_atoms[left_idx].radius;
                    const double max_cutoff = Squared(left_radius + max_radius + 0.45);

                    for (std::size_t right_idx = left_idx + 1; right_idx < sorted_atoms.size(); ++right_idx)
                    {
                        OpenBabel::OBAtom *right_atom = sorted_atoms[right_idx].atom;
                        const double right_radius = sorted_atoms[right_idx].radius;
                        const double z_distance_squared =
                            Squared(left_atom->GetZ() - right_atom->GetZ());
                        if (z_distance_squared > max_cutoff)
                        {
                            break;
                        }

                        const double cutoff = Squared(left_radius + right_radius + 0.45);
                        double distance_squared = Squared(left_atom->GetX() - right_atom->GetX());
                        if (distance_squared > cutoff)
                        {
                            continue;
                        }
                        distance_squared += Squared(left_atom->GetY() - right_atom->GetY());
                        if (distance_squared > cutoff)
                        {
                            continue;
                        }
                        distance_squared += z_distance_squared;
                        if (distance_squared > cutoff || distance_squared < 0.16)
                        {
                            continue;
                        }
                        if (left_atom->IsConnected(right_atom))
                        {
                            continue;
                        }
                        if (!CanAddBondToNeighbor(*left_atom, *right_atom) ||
                            !CanAddBondToNeighbor(*right_atom, *left_atom))
                        {
                            continue;
                        }

                        mol.AddBond(
                            static_cast<int>(left_atom->GetIdx()),
                            static_cast<int>(right_atom->GetIdx()),
                            1);
                    }
                }

                RemoveExcessLocalBonds(mol, initial_bond_counts);
            }

            void ConnectTheDotsAndPerceiveBondOrders(OpenBabel::OBMol &mol)
            {
                ConnectTheDotsLocal(mol);
                PerceiveBondOrdersLocal(mol);
            }

            void PerceiveBondOrdersLocal(OpenBabel::OBMol &mol)
            {
                if (mol.Empty() || mol.GetDimension() != 3)
                {
                    return;
                }

                EstimateHybridizationsFromGeometry(mol);
                MarkPlanarSmallRingsSp2(mol);
                RelaxIsolatedUnsaturatedHybridizations(mol);
                AssignFunctionalGroupBondsLocal(mol);
                KekulizePlanarAromaticRings(mol);
                AssignRemainingMultipleBonds(mol);
                mol.SetHybridizationPerceived(false);
                mol.SetAromaticPerceived(false);
                mol.SetAtomTypesPerceived(false);
            }

            void FindRingAtomsAndBonds(OpenBabel::OBMol &mol)
            {
                if (!mol.HasRingAtomsAndBondsPerceived())
                {
                    FindRingAtomsAndBondsLocal(mol);
                }
            }

            void PrepareForSmartsMatching(OpenBabel::OBMol &mol)
            {
                FindRingAtomsAndBonds(mol);
                if (!mol.HasAromaticPerceived())
                {
                    ResetAndAssignAromaticFlags(mol);
                }
            }

            void EnsureHybridizationPerceived(OpenBabel::OBMol &mol)
            {
                if (mol.HasHybridizationPerceived())
                {
                    return;
                }

                // GetHyb may invoke OpenBabel's native typer and global logger.
                std::lock_guard<std::recursive_mutex> lock(NativeOpenBabelMutex());
                FOR_ATOMS_OF_MOL(atom_iter, mol)
                {
                    static_cast<void>(atom_iter->GetHyb());
                }
            }

            void SetAromaticPerceived(OpenBabel::OBMol &mol, bool perceived)
            {
                mol.SetAromaticPerceived(perceived);
            }

            void ResetAndAssignAromaticFlags(OpenBabel::OBMol &mol)
            {
                FindRingAtomsAndBonds(mol);
                mol.SetAromaticPerceived(false);
                AssignAromaticFlagsLocal(mol);
            }

            bool AtomIsAromatic(OpenBabel::OBAtom &atom)
            {
                OpenBabel::OBMol *mol = atom.GetParent();
                if (mol == nullptr)
                {
                    return false;
                }
                if (mol != nullptr && !mol->HasAromaticPerceived())
                {
                    ResetAndAssignAromaticFlags(*mol);
                }
                return atom.IsAromatic();
            }

            bool BondIsAromatic(OpenBabel::OBBond &bond)
            {
                OpenBabel::OBMol *mol = bond.GetParent();
                if (mol == nullptr)
                {
                    return false;
                }
                if (mol != nullptr && !mol->HasAromaticPerceived())
                {
                    ResetAndAssignAromaticFlags(*mol);
                }
                return bond.IsAromatic();
            }

            bool RingIsAromatic(OpenBabel::OBRing &ring)
            {
                OpenBabel::OBMol *mol = ring.GetParent();
                if (mol != nullptr && !mol->HasAromaticPerceived())
                {
                    ResetAndAssignAromaticFlags(*mol);
                }
                for (int atom_idx : ring._path)
                {
                    if (mol == nullptr)
                    {
                        return false;
                    }
                    OpenBabel::OBAtom *atom = mol->GetAtom(atom_idx);
                    if (atom == nullptr || !atom->IsAromatic())
                    {
                        return false;
                    }
                }
                return true;
            }
        }
    }
}

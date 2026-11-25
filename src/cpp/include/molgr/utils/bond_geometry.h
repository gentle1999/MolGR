#pragma once

#include "molgr/compat/openbabel_iter.h"
#include "molgr/utils/electrons.h"
#include <openbabel/atom.h>
#include <openbabel/elements.h>
#include <openbabel/bond.h>
#include <openbabel/mol.h>
#include <openbabel/ring.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

namespace molgr::bond_geometry
{
    /// Recognize a short C–N single bond to a neutral, closed-shell trivalent N.
    /// @param carbon Carbon atom at the candidate bonding site.
    /// @return True for a qualifying short C-N bond to a neutral trivalent donor.
    inline bool HasConjugatedThioamideDonor(OpenBabel::OBAtom *carbon)
    {
        FOR_NBORS_OF_ATOM(atom, carbon)
        {
            const double distance = carbon->GetDistance(&*atom);
            if (atom->GetAtomicNum() == 7 && atom->GetFormalCharge() == 0 &&
                atom->GetExplicitDegree() == 3 && atom->GetTotalValence() == 3 &&
                atom->GetImplicitHCount() == 0 && !molgr::utils::GetUnpairedElectronCount(*atom) &&
                !molgr::utils::HasUnresolvedTwoElectronCenter(*atom) &&
                carbon->GetBond(&*atom)->GetBondOrder() == 1 &&
                distance > 1.30 && distance < 1.42) return true;
        }
        return false;
    }

    /// Six ring carbons followed by the two exocyclic imine nitrogens.
    /// Recognition uses only the nonmetal geometry, never a metal charge target.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Ordered ring carbon indices followed by the two exocyclic nitrogen indices for each site.
    inline std::vector<std::array<int, 8>> ShortOrthoQuinodiimineSites(OpenBabel::OBMol &mol)
    {
        std::vector<std::array<int, 8>> sites;
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &path = ring->_path;
            if (path.size() != 6) continue;
            for (int start = 0; start < 6; ++start)
                for (const int direction : {-1, 1})
                {
                    std::array<int, 8> ids{};
                    bool valid = true;
                    for (int j = 0; j < 6; ++j) ids[j] = path[(start + direction * j + 6) % 6];
                    if (ids[0] > ids[1]) continue;
                    std::array<double, 6> lengths{};
                    for (int j = 0; j < 6; ++j)
                    {
                        auto *c = mol.GetAtom(ids[j]);
                        lengths[j] = c->GetDistance(mol.GetAtom(ids[(j + 1) % 6]));
                        const double torsion = std::abs(mol.GetTorsion(
                            ids[j], ids[(j + 1) % 6], ids[(j + 2) % 6], ids[(j + 3) % 6]));
                        if (c->GetAtomicNum() != 6 || c->GetExplicitDegree() != 3 ||
                            c->GetImplicitHCount() || !(1.34 < lengths[j] && lengths[j] < 1.50) ||
                            !std::isfinite(torsion) || std::min(torsion, std::abs(180.0 - torsion)) >= 15)
                            valid = false;
                    }
                    // The N-bearing C-C edge is long, and the two opposite
                    // double-bond sites are shorter than their flanking edges.
                    if (!valid || !(lengths[0] > 1.43 && lengths[2] < 1.41 && lengths[4] < 1.41 &&
                        lengths[1] - lengths[2] > 0.015 && lengths[3] - lengths[2] > 0.015 &&
                        lengths[3] - lengths[4] > 0.015 && lengths[5] - lengths[4] > 0.015)) continue;
                    for (int j = 0; j < 2; ++j)
                    {
                        auto *c = mol.GetAtom(ids[j]);
                        OpenBabel::OBAtom *n = nullptr;
                        FOR_NBORS_OF_ATOM(a, c)
                            if (std::find(path.begin(), path.end(), a->GetIdx()) == path.end()) n = &*a;
                        if (!n || n->GetAtomicNum() != 7 || n->GetExplicitDegree() != 2 ||
                            n->GetImplicitHCount() || !(1.27 < n->GetDistance(c) && n->GetDistance(c) < 1.35))
                        { valid = false; break; }
                        ids[6 + j] = n->GetIdx();
                    }
                    if (valid) sites.push_back(ids);
                }
        }
        return sites;
    }

    /// Count reduced or outward-facing imine assignments that contradict both short inner C–N bonds.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Number of recognized rings with an electronically incompatible assignment.
    inline int ShortQuinodiimineGeometryConflictCount(OpenBabel::OBMol &mol)
    {
        int count = 0;
        for (const auto &ids : ShortOrthoQuinodiimineSites(mol))
        {
            bool reduced = true;
            bool outward_imines = true;
            for (int j = 0; j < 2; ++j)
            {
                auto *n = mol.GetAtom(ids[6 + j]);
                auto *inner = mol.GetAtom(ids[j]);
                reduced &= n->GetFormalCharge() == -1 && n->GetTotalValence() == 2 &&
                    n->GetBond(inner)->GetBondOrder() == 1 &&
                    !molgr::utils::GetUnpairedElectronCount(*n) &&
                    !molgr::utils::HasUnresolvedTwoElectronCenter(*n);
                // A neutral N can also leave the short inner edge single by
                // putting its double bond on the other substituent. Penalize
                // this direction only when both longer outer C-N edges oppose
                // the independently recognized central quinoid geometry.
                bool outward = false;
                if (n->GetFormalCharge() == 0 && n->GetTotalValence() == 3 &&
                    n->GetBond(inner)->GetBondOrder() == 1 &&
                    !molgr::utils::GetUnpairedElectronCount(*n) &&
                    !molgr::utils::HasUnresolvedTwoElectronCenter(*n))
                {
                    FOR_NBORS_OF_ATOM(outer, n)
                        if (&*outer != inner && outer->GetAtomicNum() == 6 &&
                            n->GetBond(&*outer)->GetBondOrder() == 2 &&
                            n->GetDistance(&*outer) - n->GetDistance(inner) > 0.02)
                            outward = true;
                }
                outward_imines &= outward;
            }
            count += reduced || outward_imines;
        }
        return count;
    }

    /// Atom indices for one N–N–aryl arm and its optional phenoxy oxygen.
    struct FormazanArm
    {
        // Inner N, outer N, N-bound aryl C, O-bound aryl C, terminal O.
        // The last two indices are zero for an unsubstituted phenyl arm.
        std::array<int, 5> atoms;
        std::vector<int> ring;
    };

    /// The central carbon and two aryl arms of a planar formazan ligand.
    struct FormazanSite
    {
        int center;
        std::array<FormazanArm, 2> arms;
        std::vector<int> atom_indices;
    };

    /// Accept syn- and anti-coplanar segments of a formazan conjugation path.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @param a First one-based atom index in the interaction.
    /// @param b Second one-based atom index in the interaction.
    /// @param c Third one-based atom index in the interaction.
    /// @param d Fourth one-based atom index in the interaction.
    /// @return True for a finite torsion within the planar angular window.
    inline bool FormazanPlanarTorsion(OpenBabel::OBMol &mol, int a, int b, int c, int d)
    {
        const double t = std::abs(mol.GetTorsion(a, b, c, d));
        return std::min(t, std::abs(180.0 - t)) < 15.0;
    }

    /// Recognize a complete mono- or bis(ortho-phenoxy) formazan without using
    /// bond order to infer hybridization. A mono site requires a plain phenyl
    /// second arm; an invalid phenoxy arm cannot fall back to a plain arm.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @param required_phenoxy_arms Number of ortho-phenoxy arms required on the formazan ligand.
    /// @return Matched central carbon and ordered ligand arms for each supported planar site.
    inline std::vector<FormazanSite> PlanarPhenoxyFormazanSites(
        OpenBabel::OBMol &mol, int required_phenoxy_arms)
    {
        std::vector<std::vector<int>> rings;
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &ids = ring->_path;
            if (ids.size() != 6) continue;
            bool valid = true;
            double shortest = 100.0, longest = 0.0;
            for (int i = 0; i < 6; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const double length = atom->GetDistance(mol.GetAtom(ids[(i + 1) % 6]));
                const double angle = mol.GetAngle(mol.GetAtom(ids[(i + 5) % 6]), atom,
                    mol.GetAtom(ids[(i + 1) % 6]));
                if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 3 ||
                    atom->GetImplicitHCount() || !(length > 1.35 && length < 1.46) ||
                    !(angle > 90.0 && angle < 150.0) ||
                    !FormazanPlanarTorsion(mol, ids[i], ids[(i + 1) % 6],
                        ids[(i + 2) % 6], ids[(i + 3) % 6])) { valid = false; break; }
                shortest = std::min(shortest, length);
                longest = std::max(longest, length);
            }
            if (valid && longest - shortest <= 0.06) rings.push_back(ids);
        }
        std::vector<FormazanSite> sites;
        FOR_ATOMS_OF_MOL(center, mol)
        {
            if (center->GetAtomicNum() != 6 || center->GetExplicitDegree() != 3 ||
                center->GetImplicitHCount()) continue;
            std::vector<OpenBabel::OBAtom *> inners;
            FOR_NBORS_OF_ATOM(atom, &*center)
                if (atom->GetAtomicNum() == 7) inners.push_back(&*atom);
            if (inners.size() != 2) continue;
            std::sort(inners.begin(), inners.end(), [](auto *a, auto *b) { return a->GetIdx() < b->GetIdx(); });
            std::vector<FormazanArm> arms;
            for (auto *inner : inners)
            {
                if (inner->GetExplicitDegree() != 2 || inner->GetImplicitHCount()) break;
                OpenBabel::OBAtom *outer = nullptr, *carbon = nullptr;
                FOR_NBORS_OF_ATOM(atom, inner)
                    if (atom->GetIdx() != center->GetIdx()) outer = &*atom;
                if (!outer || outer->GetAtomicNum() != 7 || outer->GetExplicitDegree() != 2 ||
                    outer->GetImplicitHCount()) break;
                FOR_NBORS_OF_ATOM(atom, outer)
                    if (atom->GetIdx() != inner->GetIdx()) carbon = &*atom;
                if (!carbon) break;
                const double nn = inner->GetDistance(outer), cn = center->GetDistance(inner),
                    aryl_cn = outer->GetDistance(carbon);
                const double inner_angle = mol.GetAngle(&*center, inner, outer),
                    outer_angle = mol.GetAngle(inner, outer, carbon);
                if (!(nn > 1.24 && nn < 1.33 && cn > 1.32 && cn < 1.43 &&
                      aryl_cn > 1.36 && aryl_cn < 1.46 && inner_angle > 100.0 &&
                      inner_angle < 140.0 && outer_angle > 100.0 && outer_angle < 140.0)) break;
                std::vector<FormazanArm> matches;
                for (const auto &ids : rings)
                {
                    if (std::find(ids.begin(), ids.end(), carbon->GetIdx()) == ids.end()) continue;
                    bool plain_phenyl = required_phenoxy_arms == 1;
                    for (int i : ids)
                        FOR_NBORS_OF_ATOM(substituent, mol.GetAtom(i))
                            if (std::find(ids.begin(), ids.end(), substituent->GetIdx()) == ids.end() &&
                                !(i == carbon->GetIdx() && &*substituent == outer) &&
                                substituent->GetAtomicNum() != 1) plain_phenyl = false;
                    FOR_NBORS_OF_ATOM(adjacent, carbon)
                    {
                        if (std::find(ids.begin(), ids.end(), adjacent->GetIdx()) == ids.end()) continue;
                        FOR_NBORS_OF_ATOM(oxygen, &*adjacent)
                        {
                            const double co = adjacent->GetDistance(&*oxygen);
                            if (oxygen->GetAtomicNum() == 8 && oxygen->GetExplicitDegree() == 1 &&
                                !oxygen->GetImplicitHCount() && co > 1.30 && co < 1.40)
                                matches.push_back({{static_cast<int>(inner->GetIdx()), static_cast<int>(outer->GetIdx()),
                                    static_cast<int>(carbon->GetIdx()), static_cast<int>(adjacent->GetIdx()),
                                    static_cast<int>(oxygen->GetIdx())}, ids});
                        }
                    }
                    if (plain_phenyl)
                        matches.push_back({{static_cast<int>(inner->GetIdx()), static_cast<int>(outer->GetIdx()),
                            static_cast<int>(carbon->GetIdx()), 0, 0}, ids});
                }
                if (matches.size() != 1) break;
                arms.push_back(matches[0]);
            }
            if (arms.size() != 2) continue;
            if (static_cast<int>(arms[0].atoms[4] != 0) +
                static_cast<int>(arms[1].atoms[4] != 0) != required_phenoxy_arms) continue;
            if (std::any_of(arms[0].ring.begin(), arms[0].ring.end(), [&](int i) {
                return std::find(arms[1].ring.begin(), arms[1].ring.end(), i) != arms[1].ring.end();
            })) continue;
            const auto &left = arms[0].atoms, &right = arms[1].atoms;
            const double angle = mol.GetAngle(inners[0], &*center, inners[1]);
            if (!(angle > 100.0 && angle < 150.0) ||
                !FormazanPlanarTorsion(mol, left[1], left[0], center->GetIdx(), right[0]) ||
                !FormazanPlanarTorsion(mol, left[0], center->GetIdx(), right[0], right[1])) continue;
            const double nn_left = mol.GetAtom(left[0])->GetDistance(mol.GetAtom(left[1])),
                nn_right = mol.GetAtom(right[0])->GetDistance(mol.GetAtom(right[1]));
            if (std::abs(nn_left - nn_right) > 0.04 ||
                std::abs(center->GetDistance(inners[0]) - center->GetDistance(inners[1])) > 0.04) continue;
            std::set<int> indices{static_cast<int>(center->GetIdx())};
            for (const auto &arm : arms)
            {
                for (int i : arm.atoms) if (i) indices.insert(i);
                indices.insert(arm.ring.begin(), arm.ring.end());
            }
            if (std::any_of(indices.begin(), indices.end(), [&](int i) {
                auto *atom = mol.GetAtom(i);
                return atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*atom);
            })) continue;
            sites.push_back({static_cast<int>(center->GetIdx()), {arms[0], arms[1]},
                std::vector<int>(indices.begin(), indices.end())});
        }
        return sites;
    }

    /// Require a terminal phenoxy oxygen on each arm of the planar formazan.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Matched central carbon and ordered ligand arms for each supported planar site.
    inline std::vector<FormazanSite> PlanarBisPhenoxyFormazanSites(OpenBabel::OBMol &mol)
    {
        return PlanarPhenoxyFormazanSites(mol, 2);
    }

    /// Endpoint motif for a folded C8 ring: a locally planar, equalized C4
    /// segment opposite a bond-alternating segment. This does not require the
    /// whole ring to be planar or infer aromaticity from its electron count.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Eight ring indices in order, starting with the locally equalized four-carbon segment.
    inline std::vector<std::array<int, 8>> FoldedEightCarbonSegments(OpenBabel::OBMol &mol)
    {
        std::vector<std::array<int, 8>> result;
        std::set<std::array<int, 4>> seen;
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &path = ring->_path;
            if (path.size() != 8) continue;
            bool valid = true;
            for (int index : path)
            {
                auto *atom = mol.GetAtom(index);
                if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 3 ||
                    atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                    { valid = false; break; }
            }
            if (!valid) continue;
            for (int shift = 0; shift < 8; ++shift)
            {
                std::array<int, 8> ids;
                std::array<double, 8> lengths;
                for (int i = 0; i < 8; ++i) ids[i] = path[(shift + i) % 8];
                for (int i = 0; i < 8; ++i)
                    lengths[i] = mol.GetAtom(ids[i])->GetDistance(mol.GetAtom(ids[(i + 1) % 8]));
                if (!std::all_of(lengths.begin(), lengths.begin() + 3,
                        [](double x) { return x > 1.40 && x < 1.46; }) ||
                    *std::max_element(lengths.begin(), lengths.begin() + 3) -
                    *std::min_element(lengths.begin(), lengths.begin() + 3) > 0.03 ||
                    !(lengths[3] > 1.43 && lengths[3] < 1.55 &&
                      lengths[7] > 1.43 && lengths[7] < 1.55 &&
                      lengths[4] > 1.30 && lengths[4] < 1.40 &&
                      lengths[6] > 1.30 && lengths[6] < 1.40 &&
                      lengths[5] > 1.40 && lengths[5] < 1.50)) continue;
                /// Measure the torsion beginning at ring offset i as degrees from the nearer planar
                /// orientation.
                const auto deviation = [&](int i) {
                    const double t = std::abs(mol.GetTorsion(ids[i], ids[(i + 1) % 8],
                        ids[(i + 2) % 8], ids[(i + 3) % 8]));
                    return std::min(t, std::abs(180.0 - t));
                };
                if (!(deviation(0) < 15.0 && deviation(1) > 25.0 && deviation(1) < 75.0 &&
                      deviation(7) > 25.0 && deviation(7) < 75.0)) continue;
                valid = true;
                for (int i = 0; i < 8; ++i)
                {
                    const double angle = mol.GetAngle(mol.GetAtom(ids[(i + 7) % 8]),
                        mol.GetAtom(ids[i]), mol.GetAtom(ids[(i + 1) % 8]));
                    if (!(angle > 90.0 && angle < 150.0)) { valid = false; break; }
                }
                std::array<int, 4> site{ids[0], ids[1], ids[2], ids[3]};
                std::sort(site.begin(), site.end());
                if (valid && seen.insert(site).second) result.push_back(ids);
            }
        }
        return result;
    }

    /// Screen a cyclic sequence for short perimeter bonds, nondegenerate angles and small torsions.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @param ids Ordered atom indices defining the candidate motif.
    /// @return True when all perimeter distances, angles and torsions pass the geometry screen.
    inline bool CompactPlanarCircuit(OpenBabel::OBMol &mol, const std::vector<int> &ids)
    {
        const auto size = ids.size();
        for (std::size_t i = 0; i < size; ++i)
        {
            auto *atom = mol.GetAtom(ids[i]);
            const double length = atom->GetDistance(mol.GetAtom(ids[(i + 1) % size]));
            const double angle = mol.GetAngle(mol.GetAtom(ids[(i + size - 1) % size]),
                atom, mol.GetAtom(ids[(i + 1) % size]));
            const double torsion = std::abs(mol.GetTorsion(ids[i], ids[(i + 1) % size],
                ids[(i + 2) % size], ids[(i + 3) % size]));
            if (!(length > 1.25 && length < 1.50 && angle > 60.0 && angle < 170.0 &&
                  std::min(torsion, std::abs(180.0 - torsion)) < 15.0)) return false;
        }
        return true;
    }

    /// Bounded recognition of a tetrapyrrole inner circuit, not an oxidation
    /// state assignment. Keep geometry/degree windows identical to Python.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Ordered atom indices of recognized tetrapyrrole circuits.
    inline std::vector<std::vector<int>> PlanarTetrapyrroleCircuits(OpenBabel::OBMol &mol)
    {
        std::vector<std::vector<int>> sites;
        const auto &rings = mol.GetSSSR();
        for (const auto *ring : rings)
        {
            const auto &ids = ring->_path;
            const auto size = ids.size();
            if (size != 15 && size != 16) continue;
            std::vector<OpenBabel::OBAtom *> nitrogens;
            bool valid = true;
            for (std::size_t i = 0; i < size; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const int z = atom->GetAtomicNum();
                if ((z != 6 && z != 7) || atom->GetImplicitHCount() != 0 ||
                    atom->GetExplicitDegree() != (z == 7 ? 2 : 3)) { valid = false; break; }
                if (z == 7) nitrogens.push_back(atom);
            }
            if (!valid || nitrogens.size() != 4 || !CompactPlanarCircuit(mol, ids)) continue;
            for (auto *nitrogen : nitrogens)
            {
                bool pyrrole = false;
                for (const auto *small : rings)
                {
                    const auto &path = small->_path;
                    if (path.size() != 5 ||
                        std::find(path.begin(), path.end(), nitrogen->GetIdx()) == path.end()) continue;
                    int shared = 0;
                    std::vector<int> elements;
                    for (int index : path)
                    {
                        if (std::find(ids.begin(), ids.end(), index) != ids.end()) ++shared;
                        elements.push_back(mol.GetAtom(index)->GetAtomicNum());
                    }
                    std::sort(elements.begin(), elements.end());
                    if (shared == 3 && elements == std::vector<int>({6, 6, 6, 6, 7})) pyrrole = true;
                }
                if (!pyrrole) { valid = false; break; }
            }
            if (valid) sites.push_back(ids);
        }
        return sites;
    }

    /// Ring atom indices and candidate nitrogen donors of a conjugated macrocycle.
    struct AzaMacrocycleCircuit
    {
        std::vector<int> atom_indices;
        std::set<int> donor_indices;
    };

    /// Three pyrrole rings and one C4N2 six-ring, sharing three atoms each
    /// with a compact planar 15-member circuit. Bridging N is not a donor by
    /// geometry alone; only the four degree-two N in these fused rings qualify.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Recognized macrocycle circuits and their donor indices.
    inline std::vector<AzaMacrocycleCircuit> PlanarAzaMacrocycleCircuits(OpenBabel::OBMol &mol)
    {
        std::vector<AzaMacrocycleCircuit> sites;
        const auto &rings = mol.GetSSSR();
        for (const auto *ring : rings)
        {
            const auto &ids = ring->_path;
            if (ids.size() != 15) continue;
            bool valid = true;
            for (int index : ids)
            {
                auto *atom = mol.GetAtom(index);
                const int z = atom->GetAtomicNum(), degree = atom->GetExplicitDegree();
                if ((z != 6 && z != 7) || atom->GetImplicitHCount() != 0 ||
                    (z == 6 ? degree != 3 : (degree != 2 && degree != 3)))
                    { valid = false; break; }
            }
            if (!valid || !CompactPlanarCircuit(mol, ids)) continue;
            int pyrroles = 0, diazines = 0;
            std::set<int> donors;
            for (const auto *small : rings)
            {
                const auto &path = small->_path;
                if (path.size() != 5 && path.size() != 6) continue;
                int shared = 0;
                std::vector<int> elements;
                for (int index : path)
                {
                    if (std::find(ids.begin(), ids.end(), index) != ids.end()) ++shared;
                    elements.push_back(mol.GetAtom(index)->GetAtomicNum());
                }
                if (shared != 3) continue;
                std::sort(elements.begin(), elements.end());
                if (elements == std::vector<int>({6, 6, 6, 6, 7})) ++pyrroles;
                else if (elements == std::vector<int>({6, 6, 6, 6, 7, 7})) ++diazines;
                else continue;
                for (int index : path)
                {
                    auto *atom = mol.GetAtom(index);
                    if (std::find(ids.begin(), ids.end(), index) != ids.end() &&
                        atom->GetAtomicNum() == 7 && atom->GetExplicitDegree() == 2)
                        donors.insert(index);
                }
            }
            if (pyrroles == 3 && diazines == 1 && donors.size() == 4)
                sites.push_back({ids, std::move(donors)});
        }
        return sites;
    }

    /// C8N8 inner circuit, including phthalocyanines. Only the four N atoms
    /// shared with C4N five-rings are donors; bridging nitrogens remain separate.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Recognized macrocycle circuits and their donor indices.
    inline std::vector<AzaMacrocycleCircuit> PlanarTetraazaporphyrinCircuits(OpenBabel::OBMol &mol)
    {
        std::vector<AzaMacrocycleCircuit> sites;
        const auto &rings = mol.GetSSSR();
        for (const auto *ring : rings)
        {
            const auto &ids = ring->_path;
            if (ids.size() != 16) continue;
            bool valid = true;
            int nitrogens = 0;
            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const int z = atom->GetAtomicNum();
                nitrogens += z == 7;
                if ((z != 6 && z != 7) || atom->GetImplicitHCount() ||
                    atom->GetExplicitDegree() != (z == 6 ? 3u : 2u) ||
                    z == mol.GetAtom(ids[(i + 1) % 16])->GetAtomicNum()) valid = false;
            }
            if (!valid || nitrogens != 8 || !CompactPlanarCircuit(mol, ids)) continue;
            int pyrroles = 0;
            std::set<int> donors;
            for (const auto *small : rings)
            {
                if (small->_path.size() != 5) continue;
                int shared = 0;
                std::vector<int> elements;
                for (int index : small->_path)
                {
                    shared += std::find(ids.begin(), ids.end(), index) != ids.end();
                    elements.push_back(mol.GetAtom(index)->GetAtomicNum());
                }
                std::sort(elements.begin(), elements.end());
                if (shared != 3 || elements != std::vector<int>({6, 6, 6, 6, 7})) continue;
                ++pyrroles;
                for (int index : small->_path)
                    if (mol.GetAtom(index)->GetAtomicNum() == 7 &&
                        std::find(ids.begin(), ids.end(), index) != ids.end()) donors.insert(index);
            }
            if (pyrroles == 4 && donors.size() == 4) sites.push_back({ids, std::move(donors)});
        }
        return sites;
    }

    /// Positive evidence for a delocalized, noncyclic three-carbon hypothesis.
    /// This does not reject ordinary stretched or metal-bound alkynes.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Triples of borrowed atom pointers in endpoint-center-endpoint order.
    inline std::vector<std::array<OpenBabel::OBAtom *, 3>> EqualizedThreeCarbonChains(
        OpenBabel::OBMol &mol)
    {
        std::vector<std::array<OpenBabel::OBAtom *, 3>> sites;
        FOR_ATOMS_OF_MOL(middle, mol)
        {
            if (middle->GetAtomicNum() != 6 || middle->GetExplicitDegree() != 2) continue;
            std::vector<OpenBabel::OBAtom *> ends;
            FOR_NBORS_OF_ATOM(n, &*middle) ends.push_back(&*n);
            std::sort(ends.begin(), ends.end(), [](auto *a, auto *b) { return a->GetIdx() < b->GetIdx(); });
            std::array<OpenBabel::OBAtom *, 3> atoms{ends[0], &*middle, ends[1]};
            if (std::any_of(atoms.begin(), atoms.end(), [](auto *a) {
                return a->GetAtomicNum() != 6 || a->GetExplicitDegree() != 2 ||
                    a->GetImplicitHCount() != 0 || a->IsInRing();
            })) continue;
            const double first = middle->GetDistance(ends[0]);
            const double second = middle->GetDistance(ends[1]);
            if (!(first > 1.35 && first < 1.50 && second > 1.35 && second < 1.50) ||
                std::abs(first - second) > 0.05) continue;
            const double angle = mol.GetAngle(ends[0], &*middle, ends[1]);
            if (!(angle > 110.0 && angle < 150.0)) continue;
            bool allowed = true;
            for (auto *end : ends)
            {
                bool bent = false;
                FOR_NBORS_OF_ATOM(n, end)
                {
                    if (n->GetIdx() == middle->GetIdx()) continue;
                    const double terminal = mol.GetAngle(&*n, end, &*middle);
                    if (terminal > 110.0 && terminal < 150.0) bent = true;
                }
                if (!bent) allowed = false;
            }
            if (allowed) sites.push_back(atoms);
        }
        return sites;
    }

    /// Positive evidence for an additional P(-1) hypothesis, not a charge
    /// assignment. Require the same C/F or O/N seesaw windows as Python.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True when the local four-coordinate phosphorus geometry supports the anion hypothesis.
    inline bool AllowsSeesawPhosphoranide(OpenBabel::OBAtom &atom)
    {
        if (atom.GetAtomicNum() != 15 || atom.GetExplicitDegree() != 4 ||
            atom.GetImplicitHCount() != 0) return false;
        std::vector<OpenBabel::OBAtom *> neighbors;
        bool carbon_fluorine = true, oxygen_nitrogen = true;
        FOR_NBORS_OF_ATOM(n, &atom)
        {
            const int z = n->GetAtomicNum();
            if (atom.GetBond(&*n)->GetBondOrder() != 1) return false;
            carbon_fluorine = carbon_fluorine && (z == 6 || z == 9);
            oxygen_nitrogen = oxygen_nitrogen && (z == 7 || z == 8);
            neighbors.push_back(&*n);
        }
        if (!carbon_fluorine && !oxygen_nitrogen) return false;
        const bool has_oxygen = std::any_of(neighbors.begin(), neighbors.end(),
            [](auto *n) { return n->GetAtomicNum() == 8; });
        const int axial_element = carbon_fluorine ? 9 : (has_oxygen ? 8 : 7);
        for (auto *n : neighbors)
        {
            const int z = n->GetAtomicNum();
            const double distance = atom.GetDistance(n);
            // The P-O window is expressed relative to the neighbor radius for P-N.
            const double offset = z == 7 ? OpenBabel::OBElements::GetCovalentRad(7) -
                                           OpenBabel::OBElements::GetCovalentRad(8) : 0.0;
            const double lower = oxygen_nitrogen ? 1.55 + offset : (z == 9 ? 1.45 : 1.60);
            const double upper = oxygen_nitrogen ? 1.90 + offset : (z == 9 ? 1.85 : 2.05);
            if (!(distance > lower && distance < upper)) return false;
        }
        auto *mol = atom.GetParent();
        for (std::size_t i = 0; i < neighbors.size(); ++i)
            for (std::size_t j = i + 1; j < neighbors.size(); ++j)
            {
                // Fluorophosphoranides can have C trans to F. Test the
                // axial pair symmetrically so atom order cannot select the state.
                const bool first_axial = neighbors[i]->GetAtomicNum() == axial_element;
                const bool second_axial = neighbors[j]->GetAtomicNum() == axial_element;
                if (carbon_fluorine ? !(first_axial || second_axial) : !(first_axial && second_axial))
                    continue;
                const double axial_angle = mol->GetAngle(neighbors[i], &atom, neighbors[j]);
                if (!(axial_angle > 165.0 && axial_angle <= 180.0)) continue;
                std::vector<OpenBabel::OBAtom *> equatorial;
                for (std::size_t k = 0; k < neighbors.size(); ++k)
                    if (k != i && k != j) equatorial.push_back(neighbors[k]);
                const double angle = mol->GetAngle(equatorial[0], &atom, equatorial[1]);
                if (!(angle > 80.0 && angle < 140.0)) continue;
                bool allowed = true;
                for (auto *axial : {neighbors[i], neighbors[j]})
                    for (auto *lateral : equatorial)
                    {
                        const double cross_angle = mol->GetAngle(axial, &atom, lateral);
                        if (!(cross_angle > 75.0 && cross_angle < 105.0)) allowed = false;
                    }
                if (allowed) return true;
            }
        return false;
    }

    /// Reject C--N=N--C only when all three lengths support C=N--N=C.
    /// Ligand-local screen for resolved, long and bent C--N(+)#N. This does
    /// not prescribe the charge of the retained alternative or use metal positions.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool BentDiazoniumGeometryAllowed(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        if (order != 3 || first.GetAtomicNum() != 7 || second.GetAtomicNum() != 7) return true;
        const double length = first.GetDistance(&second);
        if (!(length > 1.24 && length < 1.50)) return true;
        for (auto *atom : {&first, &second})
            if (atom->IsInRing() || atom->GetImplicitHCount() ||
                molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return true;
        for (auto *center : {&first, &second})
        {
            auto *terminal = center == &first ? &second : &first;
            if (center->GetFormalCharge() != 1 || center->GetExplicitDegree() != 2 ||
                terminal->GetFormalCharge() != 0 || terminal->GetExplicitDegree() != 1) continue;
            FOR_NBORS_OF_ATOM(carbon, center)
            {
                if (carbon->GetIdx() == terminal->GetIdx() || carbon->GetAtomicNum() != 6 ||
                    center->GetBond(&*carbon)->GetBondOrder() != 1) continue;
                const double cn_length = center->GetDistance(&*carbon);
                if (!(cn_length > 1.15 && cn_length < 1.60)) continue;
                const double angle = center->GetParent()->GetAngle(&*carbon, center, terminal);
                if (angle > 0.0 && angle < 150.0) return false;
            }
        }
        return true;
    }

    /// Resolved acyclic C=N(+)=N requires a nearly linear central N. Terminal
    /// diazo N(-), radicals, deferred labels and incomplete geometry abstain.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool BentCNNCumuleneGeometryAllowed(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        if (order != 2 || first.GetAtomicNum() != 7 || second.GetAtomicNum() != 7) return true;
        const double length = first.GetDistance(&second);
        if (!(length > 1.20 && length < 1.50)) return true;
        for (auto *atom : {&first, &second})
            if (atom->IsInRing() || atom->GetImplicitHCount() ||
                molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return true;
        for (auto *center : {&first, &second})
        {
            auto *outer = center == &first ? &second : &first;
            if (center->GetFormalCharge() != 1 || center->GetExplicitDegree() != 2 ||
                !((outer->GetFormalCharge() == 0 && outer->GetExplicitDegree() == 2) ||
                  (outer->GetFormalCharge() == -1 && outer->GetExplicitDegree() == 1))) continue;
            FOR_NBORS_OF_ATOM(c, center)
            {
                auto *carbon = &*c;
                if (carbon->GetIdx() == outer->GetIdx() || carbon->GetAtomicNum() != 6 ||
                    carbon->GetFormalCharge() != 0 || carbon->GetExplicitDegree() != 3 ||
                    carbon->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*carbon) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*carbon) ||
                    center->GetBond(carbon)->GetBondOrder() != 2) continue;
                const double cn_length = center->GetDistance(carbon);
                if (!(cn_length > 1.15 && cn_length < 1.50)) continue;
                const double angle = center->GetParent()->GetAngle(carbon, center, outer);
                if (angle > 1e-6 && angle < 150.0) return false;
            }
        }
        return true;
    }

    /// Inspect either C=N of resolved C=N--N=C; a short N--N and two long
    /// C--N bonds jointly support an azo endpoint instead.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool ShortAzoHydrazoneGeometryAllowed(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        if (order != 2 || !((first.GetAtomicNum() == 6 && second.GetAtomicNum() == 7) ||
                           (first.GetAtomicNum() == 7 && second.GetAtomicNum() == 6))) return true;
        auto *nitrogen = first.GetAtomicNum() == 7 ? &first : &second;
        FOR_NBORS_OF_ATOM(o, nitrogen)
        {
            auto *other = &*o;
            const double length = nitrogen->GetDistance(other);
            if (other->GetAtomicNum() != 7 || nitrogen->GetBond(other)->GetBondOrder() != 1 ||
                !(length > 1.20 && length < 1.31)) continue;
            bool matches = true;
            for (auto *n : {nitrogen, other})
            {
                if (n->GetFormalCharge() != 0 || n->GetExplicitDegree() != 2 || n->IsInRing() ||
                    n->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*n) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*n))
                {
                    matches = false;
                    break;
                }
                auto *partner = n == nitrogen ? other : nitrogen;
                FOR_NBORS_OF_ATOM(c, n)
                {
                    if (c->GetIdx() == partner->GetIdx()) continue;
                    const double cn = n->GetDistance(&*c);
                    if (c->GetAtomicNum() != 6 || c->GetFormalCharge() != 0 ||
                        c->GetExplicitDegree() != 3 || c->GetImplicitHCount() ||
                        molgr::utils::GetUnpairedElectronCount(*c) ||
                        molgr::utils::HasUnresolvedTwoElectronCenter(*c) ||
                        n->GetBond(&*c)->GetBondOrder() != 2 || !(cn > 1.35 && cn < 1.50)) matches = false;
                    // Formazans and other heteroatom branches extend the
                    // conjugated unit beyond the isolated C=N--N=C motif.
                    FOR_NBORS_OF_ATOM(substituent, &*c)
                        if (substituent->GetIdx() != n->GetIdx() &&
                            substituent->GetAtomicNum() != 1 && substituent->GetAtomicNum() != 6)
                            matches = false;
                }
            }
            if (matches) return false;
        }
        return true;
    }

    /// Reject a long N=N+ at a strongly pyramidal dialkylamino nitrogen.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool PyramidalHydrazoniumGeometryAllowed(
        OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        if (order != 2 || first.GetAtomicNum() != 7 || second.GetAtomicNum() != 7) return true;
        for (auto *donor : {&first, &second})
        {
            auto *other = donor == &first ? &second : &first;
            if (donor->GetFormalCharge() != 1 || other->GetFormalCharge() != 0 ||
                donor->GetExplicitDegree() != 3 || other->GetExplicitDegree() != 2 ||
                donor->GetImplicitHCount() || other->GetImplicitHCount() ||
                molgr::utils::GetUnpairedElectronCount(*donor) ||
                molgr::utils::GetUnpairedElectronCount(*other) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*donor) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*other) ||
                !(donor->GetDistance(other) > 1.38 && donor->GetDistance(other) < 1.60) ||
                donor->GetBond(other)->IsInRing()) continue;
            std::vector<OpenBabel::OBAtom *> neighbors;
            bool valid = true;
            FOR_NBORS_OF_ATOM(n, donor)
            {
                neighbors.push_back(&*n);
                if (n->GetIdx() == other->GetIdx()) continue;
                if (n->GetAtomicNum() != 6 || donor->GetBond(&*n)->GetBondOrder() != 1 ||
                    !(donor->GetDistance(&*n) > 1.20 && donor->GetDistance(&*n) < 1.70)) valid = false;
            }
            double sum = 0;
            for (std::size_t i = 0; i < neighbors.size(); ++i)
                for (std::size_t j = i + 1; j < neighbors.size(); ++j)
                {
                    const double angle = donor->GetParent()->GetAngle(neighbors[i], donor, neighbors[j]);
                    if (!(angle > 90 && angle < 125)) valid = false;
                    sum += angle;
                }
            if (valid && sum > 300 && sum < 340) return false;
        }
        return true;
    }

    /// A long N--N distance alone does not exclude an azo assignment. Ring bonds
    /// abstain because this chain-local screen does not model cyclic delocalization.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool NitrogenNitrogenOrderAllowed(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        if (!BentDiazoniumGeometryAllowed(first, second, order)) return false;
        if (!BentCNNCumuleneGeometryAllowed(first, second, order)) return false;
        if (!ShortAzoHydrazoneGeometryAllowed(first, second, order)) return false;
        if (!PyramidalHydrazoniumGeometryAllowed(first, second, order)) return false;
        if (order != 2) return true;
        for (auto *atom : {&first, &second})
            if (atom->GetAtomicNum() != 7 || atom->GetExplicitDegree() != 2 ||
                atom->GetImplicitHCount() != 0) return true;
        const double length = first.GetDistance(&second);
        if (!(length > 1.30 && length < 1.50)) return true;
        auto *bond = first.GetBond(&second);
        if (!bond || bond->IsInRing()) return true;
        bool both_short = true;
        for (auto *atom : {&first, &second})
        {
            bool short_cn = false;
            FOR_NBORS_OF_ATOM(neighbor, atom)
            {
                const double cn_length = atom->GetDistance(&*neighbor);
                if (neighbor->GetAtomicNum() == 6 && atom->GetBond(&*neighbor)->GetBondOrder() == 1 &&
                    cn_length > 1.20 && cn_length < 1.35) short_cn = true;
            }
            if (!short_cn) both_short = false;
        }
        if (both_short) return false;
        // The asymmetric thiohydrazone branch has one short C--N and a long
        // terminal C=S. Require resolved, neutral C--N=N--C(=S) labels.
        for (auto *atom : {&first, &second})
        {
            if (atom->GetFormalCharge() != 0 || molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return true;
            bool carbon_neighbor = false;
            FOR_NBORS_OF_ATOM(n, atom) if (n->GetAtomicNum() == 6) carbon_neighbor = true;
            if (!carbon_neighbor) return true;
        }
        for (auto *nitrogen : {&first, &second})
            FOR_NBORS_OF_ATOM(c, nitrogen)
            {
                auto *carbon = &*c;
                const double cn = nitrogen->GetDistance(carbon);
                if (carbon->GetAtomicNum() != 6 || carbon->GetFormalCharge() != 0 ||
                    carbon->GetExplicitDegree() != 3 || carbon->GetImplicitHCount() != 0 ||
                    molgr::utils::GetUnpairedElectronCount(*carbon) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*carbon) ||
                    nitrogen->GetBond(carbon)->GetBondOrder() != 1 || !(cn > 1.20 && cn < 1.35)) continue;
                FOR_NBORS_OF_ATOM(s, carbon)
                {
                    const double cs = carbon->GetDistance(&*s);
                    if (s->GetAtomicNum() == 16 && s->GetFormalCharge() == 0 &&
                        s->GetExplicitDegree() == 1 && s->GetImplicitHCount() == 0 &&
                        !molgr::utils::GetUnpairedElectronCount(*s) &&
                        !molgr::utils::HasUnresolvedTwoElectronCenter(*s) &&
                        carbon->GetBond(&*s)->GetBondOrder() == 2 && cs > 1.72 && cs < 1.90)
                        return false;
                }
            }
        return true;
    }

    /// Exclude bent, resolved closed-shell O=N(+)=O at normal N--O lengths.
    /// @param nitrogen Nitrogen center whose valence and geometry are tested.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool NitroniumGeometryAllowed(OpenBabel::OBAtom &nitrogen)
    {
        if (nitrogen.GetAtomicNum() != 7 || nitrogen.GetFormalCharge() != 1 ||
            nitrogen.GetExplicitDegree() != 2 || nitrogen.GetImplicitHCount() != 0) return true;
        std::vector<OpenBabel::OBAtom *> oxygens;
        FOR_NBORS_OF_ATOM(o, &nitrogen)
        {
            const double length = nitrogen.GetDistance(&*o);
            if (o->GetAtomicNum() != 8 || o->GetFormalCharge() != 0 ||
                o->GetExplicitDegree() != 1 || o->GetImplicitHCount() != 0 ||
                nitrogen.GetBond(&*o)->GetBondOrder() != 2 || !(length > 1.05 && length < 1.40))
                return true;
            oxygens.push_back(&*o);
        }
        for (auto *atom : {&nitrogen, oxygens[0], oxygens[1]})
            if (molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return true;
        // Open Babel reports a small nonzero angle for coincident directions.
        if (oxygens[0]->GetDistance(oxygens[1]) < 1e-6) return true;
        const double angle = nitrogen.GetParent()->GetAngle(oxygens[0], &nitrogen, oxygens[1]);
        return !(angle > 0.0 && angle < 150.0);
    }

    /// Screen bent nitronium and nitroso C--N=O with conflicting geometry.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool NitrogenOxygenOrderAllowed(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        if (order != 2) return true;
        auto *nitrogen = first.GetAtomicNum() == 7 ? &first : &second;
        auto *oxygen = first.GetAtomicNum() == 7 ? &second : &first;
        if (!NitroniumGeometryAllowed(*nitrogen)) return false;
        if (nitrogen->GetAtomicNum() != 7 || oxygen->GetAtomicNum() != 8 ||
            nitrogen->GetExplicitDegree() != 2 || nitrogen->GetImplicitHCount() != 0 ||
            oxygen->GetImplicitHCount() != 0)
            return true;
        const double no_length = nitrogen->GetDistance(oxygen);
        if (!(no_length > 1.30 && no_length < 1.50)) return true;
        bool borate_capped = false;
        if (oxygen->GetExplicitDegree() == 2)
            FOR_NBORS_OF_ATOM(neighbor, oxygen)
            {
                if (neighbor->GetAtomicNum() != 5 || neighbor->GetExplicitDegree() != 4 ||
                    oxygen->GetBond(&*neighbor)->GetBondOrder() != 1) continue;
                bool singles = true;
                FOR_BONDS_OF_ATOM(bond, &*neighbor)
                    if (bond->GetBondOrder() != 1) singles = false;
                if (singles) borate_capped = true;
            }
        if (oxygen->GetExplicitDegree() != 1 && !borate_capped) return true;
        FOR_NBORS_OF_ATOM(neighbor, nitrogen)
        {
            const double cn_length = nitrogen->GetDistance(&*neighbor);
            // A long oxime N-O and short C-N cannot support a bent C=N+=O
            // cumulene. Keep linear nitrosoiminium and shorter N=O alternatives.
            if (neighbor->GetAtomicNum() == 6 && neighbor->GetFormalCharge() == 0 &&
                oxygen->GetFormalCharge() == 0 && oxygen->GetExplicitDegree() == 1 &&
                nitrogen->GetFormalCharge() == 1 && nitrogen->GetTotalValence() == 4 &&
                nitrogen->GetBond(&*neighbor)->GetBondOrder() == 2 &&
                cn_length > 1.20 && cn_length < 1.34)
            {
                const double angle = nitrogen->GetParent()->GetAngle(&*neighbor, nitrogen, oxygen);
                if (angle > 0.0 && angle < 150.0) return false;
            }
            if (neighbor->GetAtomicNum() == 6 && nitrogen->GetBond(&*neighbor)->GetBondOrder() == 1 &&
                cn_length > 1.20 && cn_length < 1.34) return false;
        }
        return true;
    }

    /// Match the narrow local C--O candidate screens in the Python backend.
    /// No charge/oxidation state is inferred here; isolated CO is excluded.
    /// A bent ketene alone is insufficient: require the neighboring ring-bound
    /// long carbonyl and short C--C, exactly as in the Python geometry screen.
    /// @param carbon Carbon atom at the candidate bonding site.
    /// @param oxygen Oxygen partner in the proposed C-O assignment.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool PairedCarbonylKeteneGeometryAllowed(OpenBabel::OBAtom &carbon, OpenBabel::OBAtom &oxygen)
    {
        /// Accept neutral atoms with explicit hydrogens and fully assigned, closed-shell electron labels.
        const auto resolved = [](OpenBabel::OBAtom &atom) {
            return atom.GetFormalCharge() == 0 && atom.GetImplicitHCount() == 0 &&
                !molgr::utils::GetUnpairedElectronCount(atom) &&
                !molgr::utils::HasUnresolvedTwoElectronCenter(atom);
        };
        const double co = carbon.GetDistance(&oxygen);
        if (carbon.GetAtomicNum() != 6 || oxygen.GetAtomicNum() != 8 ||
            carbon.GetExplicitDegree() != 2 || oxygen.GetExplicitDegree() != 1 ||
            !resolved(carbon) || !resolved(oxygen) || !(co > 1.15 && co < 1.35)) return true;
        FOR_NBORS_OF_ATOM(mid, &carbon)
        {
            auto *middle = &*mid;
            const double cc = carbon.GetDistance(middle);
            const double angle = carbon.GetParent()->GetAngle(middle, &carbon, &oxygen);
            if (middle->GetAtomicNum() != 6 || middle->GetExplicitDegree() != 3 ||
                !resolved(*middle) || !middle->IsInRing() ||
                carbon.GetBond(middle)->GetBondOrder() != 2 ||
                !(cc > 1.44 && cc < 1.65 && angle > 90 && angle < 150)) continue;
            FOR_NBORS_OF_ATOM(next, middle)
            {
                auto *other = &*next;
                const double ring_cc = middle->GetDistance(other);
                if (other->GetIdx() == carbon.GetIdx() || other->GetAtomicNum() != 6 ||
                    other->GetExplicitDegree() != 3 || !resolved(*other) || !other->IsInRing() ||
                    !middle->GetBond(other)->IsInRing() || middle->GetBond(other)->GetBondOrder() != 1 ||
                    !(ring_cc > 1.38 && ring_cc < 1.45)) continue;
                FOR_NBORS_OF_ATOM(terminal, other)
                {
                    const double carbonyl = other->GetDistance(&*terminal);
                    if (terminal->GetAtomicNum() == 8 && terminal->GetExplicitDegree() == 1 &&
                        resolved(*terminal) && other->GetBond(&*terminal)->GetBondOrder() == 2 &&
                        carbonyl > 1.30 && carbonyl < 1.42) return false;
                }
            }
        }
        return true;
    }

    /// Match the metal-free Python screen: only a long, pyramidal, neutral
    /// carbon/hydrogen-substituted Si=C is excluded. Short pyramidal silenes,
    /// rings, hypercoordination and other electronic states remain available.
    /// This does not impose a silyl charge or a metal oxidation state.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool SiliconCarbonOrderAllowed(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        if (order != 2 || !((first.GetAtomicNum() == 14 && second.GetAtomicNum() == 6) ||
                           (first.GetAtomicNum() == 6 && second.GetAtomicNum() == 14))) return true;
        auto &si = first.GetAtomicNum() == 14 ? first : second;
        auto &carbon = first.GetAtomicNum() == 6 ? first : second;
        for (auto *atom : {&si, &carbon})
            if (atom->GetFormalCharge() != 0 || atom->GetExplicitDegree() != 3 ||
                atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return true;
        const double length = si.GetDistance(&carbon);
        if (!(length > 1.88 && length < 2.10)) return true;
        auto *bond = si.GetBond(&carbon);
        if (!bond || bond->IsInRing()) return true;
        std::vector<OpenBabel::OBAtom *> neighbors;
        FOR_NBORS_OF_ATOM(atom, si)
        {
            if (atom->GetAtomicNum() != 1 && atom->GetAtomicNum() != 6) return true;
            neighbors.push_back(&*atom);
        }
        FOR_NBORS_OF_ATOM(atom, carbon)
            if (atom->GetIdx() != si.GetIdx() && atom->GetAtomicNum() != 1 &&
                atom->GetAtomicNum() != 6) return true;
        double sum = 0;
        for (size_t i = 0; i < neighbors.size(); ++i)
            for (size_t j = i + 1; j < neighbors.size(); ++j)
            {
                const double angle = si.GetParent()->GetAngle(neighbors[i], &si, neighbors[j]);
                if (!(angle > 80 && angle < 140)) return true;
                sum += angle;
            }
        return !(sum < 335);
    }

    /// Match the Python bounded biaryl-like geometry screen. No metal states or
    /// charges are inferred: only an incompatible final C=C candidate is rejected.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool TwistedRingLinkOrderAllowed(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        if (order != 2 || first.GetAtomicNum() != 6 || second.GetAtomicNum() != 6) return true;
        for (auto *atom : {&first, &second})
            if (atom->GetExplicitDegree() != 3 || !atom->IsInRing() || atom->GetImplicitHCount() ||
                atom->GetFormalCharge() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return true;
        auto *bond = first.GetBond(&second);
        const double length = first.GetDistance(&second);
        if (!bond || bond->IsInRing() || !(length > 1.47 && length < 1.65)) return true;
        auto &mol = *first.GetParent();
        std::array<std::vector<OpenBabel::OBAtom *>, 2> groups;
        std::array<OpenBabel::OBAtom *, 2> centers{&first, &second};
        for (size_t i = 0; i < centers.size(); ++i)
        {
            FOR_NBORS_OF_ATOM(n, centers[i])
                if (n->GetIdx() != centers[1 - i]->GetIdx()) groups[i].push_back(&*n);
            if (groups[i].size() != 2) return true;
            for (auto *n : groups[i]) if (n->GetAtomicNum() != 6) return true;
            const double angle = mol.GetAngle(groups[i][0], centers[i], groups[i][1]);
            if (!(angle > 90 && angle < 150)) return true;
        }
        for (auto *a : groups[0])
            for (auto *b : groups[1])
            {
                const double t = std::abs(mol.GetTorsion(a->GetIdx(), first.GetIdx(), second.GetIdx(), b->GetIdx()));
                const double deviation = std::min(t, std::abs(180.0 - t));
                if (!std::isfinite(t) || !(deviation > 45 && deviation <= 90)) return true;
            }
        for (auto *ring : mol.GetSSSR())
        {
            const auto &ids = ring->_path;
            if (ids.size() != 6) continue;
            const bool has_first = std::find(ids.begin(), ids.end(), first.GetIdx()) != ids.end();
            const bool has_second = std::find(ids.begin(), ids.end(), second.GetIdx()) != ids.end();
            if (has_first == has_second) continue;
            bool valid = true;
            double minimum = 10.0, maximum = 0.0;
            for (size_t i = 0; i < ids.size(); ++i)
            {
                auto *a = mol.GetAtom(ids[i]);
                if (a->GetAtomicNum() != 6 || a->GetExplicitDegree() != 3 || a->GetImplicitHCount())
                { valid = false; break; }
                const double d = a->GetDistance(mol.GetAtom(ids[(i + 1) % 6]));
                const double t = std::abs(mol.GetTorsion(ids[i], ids[(i + 1) % 6], ids[(i + 2) % 6], ids[(i + 3) % 6]));
                if (!(d > 1.35 && d < 1.45) || !std::isfinite(t) || !(std::min(t, std::abs(180.0 - t)) < 15))
                { valid = false; break; }
                minimum = std::min(minimum, d);
                maximum = std::max(maximum, d);
            }
            if (valid && maximum - minimum < 0.06) return false;
        }
        return true;
    }

    /// A long C=O+ at a tetrahedral carbon cannot describe an oxocarbenium.
    /// Require both length and local pyramidalization; short or planar C=O+
    /// and charge-separated/open-shell alternatives remain available.
    /// @param carbon Carbon atom at the candidate bonding site.
    /// @param oxygen Oxygen partner in the proposed C-O assignment.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool PyramidalOxoniumGeometryAllowed(OpenBabel::OBAtom &carbon, OpenBabel::OBAtom &oxygen)
    {
        if (carbon.GetFormalCharge() != 0 || oxygen.GetFormalCharge() != 1 ||
            carbon.GetExplicitDegree() != 3 || oxygen.GetExplicitDegree() != 2 ||
            carbon.GetTotalValence() != 4 || oxygen.GetTotalValence() != 3 ||
            carbon.IsInRing() || oxygen.IsInRing()) return true;
        for (auto *atom : {&carbon, &oxygen})
            if (atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return true;
        const double length = carbon.GetDistance(&oxygen);
        if (!(length > 1.35 && length < 1.65)) return true;
        std::vector<OpenBabel::OBAtom *> neighbors;
        FOR_NBORS_OF_ATOM(neighbor, &carbon)
        {
            if (neighbor->GetIdx() != oxygen.GetIdx() && neighbor->GetAtomicNum() != 1 &&
                neighbor->GetAtomicNum() != 6) return true;
            neighbors.push_back(&*neighbor);
        }
        double sum = 0.0;
        for (size_t i = 0; i < neighbors.size(); ++i)
            for (size_t j = i + 1; j < neighbors.size(); ++j)
            {
                const double angle = carbon.GetParent()->GetAngle(neighbors[i], &carbon, neighbors[j]);
                if (!(angle > 90.0 && angle < 130.0)) return true;
                sum += angle;
            }
        return !(sum > 300.0 && sum < 345.0);
    }

    /// Screen scoped C--O multiple-bond patterns using explicit local geometry.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool CarbonOxygenOrderAllowed(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        auto *carbon = first.GetAtomicNum() == 6 ? &first : &second;
        auto *oxygen = first.GetAtomicNum() == 6 ? &second : &first;
        if (carbon->GetAtomicNum() != 6 || oxygen->GetAtomicNum() != 8) return true;
        if (order == 2 && !PairedCarbonylKeteneGeometryAllowed(*carbon, *oxygen)) return false;
        if (order == 2 && !PyramidalOxoniumGeometryAllowed(*carbon, *oxygen)) return false;
        const double length = carbon->GetDistance(oxygen);
        if (order == 2 && oxygen->GetExplicitDegree() == 2 && length > 1.42 && length < 1.65)
            return false;
        // Short C--O distances do not justify a bent substituted acylium.
        // Isolated CO has no nonmetal substituent and remains unaffected.
        if (order == 3 && carbon->GetExplicitDegree() > 1 && oxygen->GetExplicitDegree() == 1 &&
            length > 1.05 && length < 1.60)
        {
            FOR_NBORS_OF_ATOM(neighbor, carbon)
            {
                if (neighbor->GetIdx() == oxygen->GetIdx() || carbon->GetDistance(&*neighbor) <= 0.0)
                    continue;
                const double angle = carbon->GetParent()->GetAngle(&*neighbor, carbon, oxygen);
                if (angle > 0.0 && angle < 150.0) return false;
            }
        }
        return true;
    }

    /// Strong C=O geometry evidence; exclude isolated CO and substituted O.
    /// @param carbon Carbon atom at the candidate bonding site.
    /// @param oxygen Oxygen partner in the proposed C-O assignment.
    /// @return True for a short terminal C=O matching the scoped carbonyl environment.
    inline bool IsShortTerminalCarbonyl(OpenBabel::OBAtom &carbon, OpenBabel::OBAtom &oxygen)
    {
        const double distance = carbon.GetDistance(&oxygen);
        return carbon.GetAtomicNum() == 6 && carbon.GetExplicitDegree() >= 2 &&
               oxygen.GetAtomicNum() == 8 && oxygen.GetExplicitDegree() == 1 &&
               distance > 1.10 && distance < 1.30;
    }

    /// Preserve a short carbonyl beside a terminal S or diazo carbon center.
    /// @param carbon Carbon atom at the candidate bonding site.
    /// @param oxygen Oxygen partner in the proposed C-O assignment.
    /// @param alpha Carbonyl-adjacent atom inspected for a competing terminal bond.
    /// @return True when the existing carbonyl must be preserved in this local motif.
    inline bool ProtectShortCarbonyl(OpenBabel::OBAtom &carbon, OpenBabel::OBAtom &oxygen, OpenBabel::OBAtom &alpha)
    {
        if (!IsShortTerminalCarbonyl(carbon, oxygen) || alpha.GetAtomicNum() != 6) return false;
        // Joint short C=O / long C-C evidence preserves a localized carbonyl
        // during electron cleanup, independent of the alpha substituent.
        // Cyclic conjugation needs a whole-ring electronic assignment; this
        // local guard must not constrain its alternative electron counts.
        // Intermediate enolates and long, activated carbonyls remain eligible.
        const double co = carbon.GetDistance(&oxygen), cc = carbon.GetDistance(&alpha);
        const double angle = carbon.GetParent()->GetAngle(&oxygen, &carbon, &alpha);
        if (!carbon.IsInRing() && co < 1.26 && cc > 1.47 && cc < 1.65 && angle > 100.0 && angle < 150.0 &&
            carbon.GetFormalCharge() == 0 && oxygen.GetFormalCharge() == 0 &&
            (alpha.GetFormalCharge() == 0 || alpha.GetFormalCharge() == -1) &&
            !carbon.GetImplicitHCount() && !oxygen.GetImplicitHCount() && !alpha.GetImplicitHCount())
            return true;
        FOR_NBORS_OF_ATOM(n, &alpha)
        {
            if (n->GetAtomicNum() == 16 && n->GetExplicitDegree() == 1) return true;
            if (n->GetAtomicNum() == 7 && n->GetExplicitDegree() == 2)
                FOR_NBORS_OF_ATOM(a, &*n)
                    if (a->GetAtomicNum() == 7 && a->GetExplicitDegree() == 1) return true;
        }
        return false;
    }

    /// Reject a new ring pi bond that would put a bent C=C=C center in a ring.
    /// @param center Atom or site occupying this position in the local chemical motif.
    /// @param partner Atom or site occupying this position in the local chemical motif.
    /// @return True when the new multiple bond would create a geometrically bent ring allene.
    inline bool WouldFormBentRingAllene(OpenBabel::OBAtom &center, OpenBabel::OBAtom &partner)
    {
        if (center.GetAtomicNum() != 6 || !center.IsInRing() || !partner.IsInRing())
            return false;
        FOR_BONDS_OF_ATOM(bond, &center)
        {
            auto *neighbor = bond->GetNbrAtom(&center);
            if (neighbor->GetIdx() == partner.GetIdx() || !neighbor->IsInRing() || bond->GetBondOrder() != 2)
                continue;
            const double angle = center.GetParent()->GetAngle(neighbor, &center, &partner);
            if (angle > 0.0 && angle < 150.0)
                return true;
        }
        return false;
    }

    /// Geometry screen for azinium preference; five-membered NHCs are excluded.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Six-membered C/N rings inside the compact planar geometry window.
    inline std::vector<std::vector<int>> CompactPlanarCNSixRings(OpenBabel::OBMol &mol)
    {
        std::vector<std::vector<int>> result;
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &ids = ring->_path;
            if (ids.size() != 6)
                continue;
            bool valid = true;
            for (int i = 0; i < 6; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const double length = atom->GetDistance(mol.GetAtom(ids[(i + 1) % 6]));
                const double torsion = std::abs(mol.GetTorsion(ids[i], ids[(i + 1) % 6], ids[(i + 2) % 6], ids[(i + 3) % 6]));
                if ((atom->GetAtomicNum() != 6 && atom->GetAtomicNum() != 7) ||
                    !(length >= 1.25 && length <= 1.44) ||
                    !std::isfinite(torsion) || std::min(torsion, 180.0 - torsion) > 15.0)
                    valid = false;
            }
            if (valid)
                result.push_back(ids);
        }
        return result;
    }

    /// Whether donating the N lone pair completes an alternating six-membered ring.
    /// @param nitrogen Nitrogen center whose valence and geometry are tested.
    /// @param radical Neighbor bearing the unpaired electron available for pi-bond formation.
    /// @return True when the proposed donation completes the aromatic ring bond pattern.
    inline bool RestoresAziniumRing(OpenBabel::OBAtom &nitrogen, OpenBabel::OBAtom &radical)
    {
        if (nitrogen.GetAtomicNum() != 7 || nitrogen.GetTotalValence() != 3)
            return false;
        auto &mol = *nitrogen.GetParent();
        for (const auto &ids : CompactPlanarCNSixRings(mol))
        {
            if (std::find(ids.begin(), ids.end(), nitrogen.GetIdx()) == ids.end() ||
                std::find(ids.begin(), ids.end(), radical.GetIdx()) == ids.end())
                continue;
            std::array<int, 6> orders;
            for (int i = 0; i < 6; ++i)
            {
                const int a = ids[i], b = ids[(i + 1) % 6];
                const bool changed = (a == static_cast<int>(nitrogen.GetIdx()) && b == static_cast<int>(radical.GetIdx())) ||
                                     (b == static_cast<int>(nitrogen.GetIdx()) && a == static_cast<int>(radical.GetIdx()));
                orders[i] = mol.GetBond(a, b)->GetBondOrder() + changed;
            }
            if (orders == std::array<int, 6>{1, 2, 1, 2, 1, 2} || orders == std::array<int, 6>{2, 1, 2, 1, 2, 1})
                return true;
        }
        return false;
    }

    /// Count compact planar reduced azinium rings competing with aromatic candidates.
    /// @param mol Ligand graph with explicit atom coordinates in angstroms.
    /// @return Number of recognized rings with an electronically incompatible assignment.
    inline int AziniumGeometryConflictCount(OpenBabel::OBMol &mol)
    {
        int count = 0;
        for (const auto &ids : CompactPlanarCNSixRings(mol))
        {
            std::array<int, 6> charges, orders;
            bool carbanion = false;
            for (int i = 0; i < 6; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                charges[i] = atom->GetFormalCharge();
                orders[i] = mol.GetBond(ids[i], ids[(i + 1) % 6])->GetBondOrder();
                carbanion = carbanion || (atom->GetAtomicNum() == 6 && charges[i] == -1);
            }
            auto sorted_orders = orders;
            std::sort(charges.begin(), charges.end());
            std::sort(sorted_orders.begin(), sorted_orders.end());
            if (!carbanion || charges != std::array<int, 6>{-1, 0, 0, 0, 0, 0} ||
                sorted_orders != std::array<int, 6>{1, 1, 1, 1, 2, 2})
                continue;
            for (int i = 0; i < 6; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                if (atom->GetAtomicNum() == 7 && atom->GetFormalCharge() == 0 &&
                    atom->GetTotalValence() == 3 && atom->GetExplicitDegree() + atom->GetImplicitHCount() == 3 &&
                    orders[i] == 1 && orders[(i + 5) % 6] == 1)
                {
                    ++count;
                    break;
                }
            }
        }
        return count;
    }

    /// Shared generation/scoring window for stretched aldehyde/ketone C--O.
    /// Exclude CO/CO2, ester/amide and phenoxide/enolate substitution patterns.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @return True for an eligible carbonyl pair in the elongated distance window.
    inline bool IsElongatedCarbonylPair(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second)
    {
        auto *carbon = first.GetAtomicNum() == 6 ? &first : &second;
        auto *oxygen = first.GetAtomicNum() == 6 ? &second : &first;
        if (carbon->GetAtomicNum() != 6 || oxygen->GetAtomicNum() != 8 ||
            carbon->GetExplicitDegree() + carbon->GetImplicitHCount() != 3 ||
            oxygen->GetExplicitDegree() != 1 || oxygen->GetImplicitHCount() != 0)
            return false;
        const double length = carbon->GetDistance(oxygen);
        if (!(length >= 1.35 && length <= 1.60))
            return false;
        FOR_NBORS_OF_ATOM(neighbor, carbon)
        {
            if (neighbor->GetIdx() == oxygen->GetIdx())
                continue;
            if ((neighbor->GetAtomicNum() != 1 && neighbor->GetAtomicNum() != 6) ||
                carbon->GetBond(&*neighbor)->GetBondOrder() != 1)
                return false;
        }
        return true;
    }
}

namespace molgr::bond_geometry {
    /// Joint evidence for a long P+=C at an acyclic, pyramidal three-carbon P.
    /// Preserve rings, heteroatom substitution, open shells and C-=P+ states.
    /// This neither assigns metal charge nor changes cationic donor penalties.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool PhosphorusCarbonOrderAllowed(OpenBabel::OBAtom &first,
                                             OpenBabel::OBAtom &second, int order)
    {
        auto *phosphorus = first.GetAtomicNum() == 15 ? &first : &second;
        auto *carbon = first.GetAtomicNum() == 15 ? &second : &first;
        if (order != 2 || phosphorus->GetAtomicNum() != 15 || carbon->GetAtomicNum() != 6)
            return true;
        if (phosphorus->GetFormalCharge() != 1 || phosphorus->GetExplicitDegree() != 3 ||
            phosphorus->GetTotalValence() != 4 || phosphorus->IsInRing() ||
            carbon->GetFormalCharge() != 0)
            return true;
        for (auto *atom : {phosphorus, carbon})
            if (atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom))
                return true;
        std::vector<OpenBabel::OBAtom *> neighbors;
        FOR_NBORS_OF_ATOM(neighbor, phosphorus)
        {
            if (neighbor->GetAtomicNum() != 6)
                return true;
            neighbors.push_back(&*neighbor);
        }
        const double distance = phosphorus->GetDistance(carbon);
        if (!(1.78 < distance && distance < 2.10))
            return true;
        double sum = 0;
        for (std::size_t i = 0; i < neighbors.size(); ++i)
            for (std::size_t j = i + 1; j < neighbors.size(); ++j)
            {
                const double angle = phosphorus->GetParent()->GetAngle(neighbors[i], phosphorus, neighbors[j]);
                if (!(80 < angle && angle < 130))
                    return true;
                sum += angle;
            }
        return !(300 < sum && sum < 330);
    }

    /// Find a planar iminium contributor supported by the surrounding conjugated system.
    /// @param carbon Carbon atom at the candidate bonding site.
    /// @param nitrogen Nitrogen center whose valence and geometry are tested.
    /// @return True if a conjugated alternative supplies a planar iminium witness.
    inline bool HasPlanarIminiumResonanceWitness(
        OpenBabel::OBAtom &carbon, OpenBabel::OBAtom &nitrogen)
    {
        // An azomethine ylide can move this pi bond onto a short, planar
        // adjacent carbanion: C=N+--[C-] <-> [C-]--N+=C. This transfers one
        // existing negative charge without changing the total electron budget.
        // A degree-two NHC center is not such a witness: it would instead
        // create a new cumulene. Check only resolved, three-coordinate C-.
        FOR_NBORS_OF_ATOM(alternate, &nitrogen)
        {
            auto *other = &*alternate;
            if (other->GetIdx() == carbon.GetIdx() || other->GetAtomicNum() != 6 ||
                other->GetFormalCharge() != -1 || other->GetExplicitDegree() != 3 ||
                other->GetTotalValence() != 3 || other->GetImplicitHCount() ||
                molgr::utils::GetUnpairedElectronCount(*other) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*other) ||
                nitrogen.GetBond(other)->GetBondOrder() != 1) continue;
            const double alternate_length = nitrogen.GetDistance(other);
            if (!(alternate_length > 1.15 && alternate_length < 1.40)) continue;
            std::vector<OpenBabel::OBAtom *> alternate_neighbors;
            bool valid = true;
            FOR_NBORS_OF_ATOM(neighbor, other)
            {
                if (neighbor->IsMetal()) valid = false;
                alternate_neighbors.push_back(&*neighbor);
            }
            double alternate_sum = 0.0;
            for (std::size_t i = 0; i < alternate_neighbors.size(); ++i)
                for (std::size_t j = i + 1; j < alternate_neighbors.size(); ++j)
                {
                    const double angle = other->GetParent()->GetAngle(
                        alternate_neighbors[i], other, alternate_neighbors[j]);
                    if (!(angle > 90.0 && angle < 150.0)) valid = false;
                    alternate_sum += angle;
                }
            if (valid && alternate_sum >= 345.0 && alternate_sum <= 360.0 + 1e-6) return true;
        }
        return false;
    }

    /// Joint evidence for a long C=N+ at a tetrahedral carbon. Inspect only
    /// the metal-free graph, including explicit H vertices. A long bond alone
    /// does not reject a planar iminium or a short, pyramidal pi ligand.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool PyramidalIminiumGeometryAllowed(
        OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        auto *carbon = first.GetAtomicNum() == 6 ? &first : &second;
        auto *nitrogen = first.GetAtomicNum() == 6 ? &second : &first;
        if (order != 2 || carbon->GetAtomicNum() != 6 || nitrogen->GetAtomicNum() != 7 ||
            carbon->GetFormalCharge() != 0 || nitrogen->GetFormalCharge() != 1 ||
            carbon->GetExplicitDegree() != 3 || nitrogen->GetExplicitDegree() != 3 ||
            carbon->GetTotalValence() != 4 || nitrogen->GetTotalValence() != 4 ||
            carbon->IsInRing()) return true;
        for (auto *atom : {carbon, nitrogen})
            if (atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return true;
        const double length = carbon->GetDistance(nitrogen);
        if (!(length > 1.40 && length < 1.65)) return true;
        std::vector<OpenBabel::OBAtom *> neighbors;
        FOR_NBORS_OF_ATOM(neighbor, carbon)
        {
            if (neighbor->GetIdx() != nitrogen->GetIdx() && neighbor->GetAtomicNum() != 1 &&
                neighbor->GetAtomicNum() != 6) return true;
            neighbors.push_back(&*neighbor);
        }
        double sum = 0.0;
        for (std::size_t i = 0; i < neighbors.size(); ++i)
            for (std::size_t j = i + 1; j < neighbors.size(); ++j)
            {
                const double angle = carbon->GetParent()->GetAngle(neighbors[i], carbon, neighbors[j]);
                if (!(angle > 90.0 && angle < 130.0)) return true;
                sum += angle;
            }
        if (!(sum > 300.0 && sum < 345.0)) return true;
        return HasPlanarIminiumResonanceWitness(*carbon, *nitrogen);
    }

    /// A long N+=C bond and a strongly bent central carbon jointly oppose
    /// a keteniminium assignment. Equivalent local ylide contributors stay
    /// eligible; ring cumulenes and short N=C bonds use their separate rules.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool BentKeteniminiumGeometryAllowed(
        OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order)
    {
        auto *carbon = first.GetAtomicNum() == 6 ? &first : &second;
        auto *nitrogen = first.GetAtomicNum() == 6 ? &second : &first;
        if (order != 2 || carbon->GetAtomicNum() != 6 || nitrogen->GetAtomicNum() != 7 ||
            carbon->GetFormalCharge() != 0 || nitrogen->GetFormalCharge() != 1 ||
            carbon->GetExplicitDegree() != 2 || nitrogen->GetExplicitDegree() != 3 ||
            carbon->GetTotalValence() != 4 || nitrogen->GetTotalValence() != 4 ||
            carbon->IsInRing()) return true;
        for (auto *atom : {carbon, nitrogen})
            if (atom->GetImplicitHCount() || molgr::utils::GetUnpairedElectronCount(*atom) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*atom)) return true;
        const double length = carbon->GetDistance(nitrogen);
        if (!(length > 1.40 && length < 1.65)) return true;
        FOR_NBORS_OF_ATOM(partner, carbon)
        {
            if (partner->GetIdx() == nitrogen->GetIdx()) continue;
            if (partner->GetAtomicNum() != 6 || partner->GetFormalCharge() != 0 ||
                partner->GetTotalValence() != 4 || partner->GetImplicitHCount() ||
                molgr::utils::GetUnpairedElectronCount(*partner) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*partner) ||
                carbon->GetBond(&*partner)->GetBondOrder() != 2) return true;
            const double cc_length = carbon->GetDistance(&*partner);
            if (!(cc_length > 1.15 && cc_length < 1.60)) return true;
            const double angle = carbon->GetParent()->GetAngle(nitrogen, carbon, &*partner);
            if (angle > 1e-6 && angle < 150.0)
                return HasPlanarIminiumResonanceWitness(*carbon, *nitrogen);
        }
        return true;
    }

    /// Abstain except for long, acyclic nitrilium triples bent at both ends.
    /// @param first First endpoint of the bond being tested.
    /// @param second Other endpoint of the same bond.
    /// @param order Proposed integer bond order.
    /// @return True when the proposed assignment passes the local geometry screen, including cases outside
    ///     its scope.
    inline bool NitriliumGeometryAllowed(OpenBabel::OBAtom &first, OpenBabel::OBAtom &second, int order) {
        auto *carbon=first.GetAtomicNum()==6 ? &first : &second;
        auto *nitrogen=first.GetAtomicNum()==6 ? &second : &first;
        if(order!=3 || carbon->GetAtomicNum()!=6 || nitrogen->GetAtomicNum()!=7) return true;
        if(carbon->GetFormalCharge()!=0 || nitrogen->GetFormalCharge()!=1) return true;
        for(auto *a:{carbon,nitrogen}) {
            if(a->GetExplicitDegree()!=2 || a->IsInRing() || a->GetImplicitHCount() ||
               molgr::utils::GetUnpairedElectronCount(*a) || molgr::utils::HasUnresolvedTwoElectronCenter(*a)) return true;
        }
        const double d=carbon->GetDistance(nitrogen);
        if(!(1.26<d && d<1.50)) return true;
        for(auto *center:{carbon,nitrogen}) {
            auto *other=center==carbon ? nitrogen : carbon;
            FOR_NBORS_OF_ATOM(n,center) {
                if(n->GetIdx()==other->GetIdx()) continue;
                if(n->GetAtomicNum()!=6 || center->GetDistance(&*n)<=0) return true;
                const double angle=center->GetParent()->GetAngle(&*n,center,other);
                if(!(0<angle && angle<150)) return true;
            }
        }
        return false;
    }
}

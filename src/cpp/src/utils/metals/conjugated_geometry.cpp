#include "molgr/utils/metals/conjugated_geometry.h"
#include "molgr/utils/metals/scoring.h"
#include "molgr/utils/electrons.h"
#include "molgr/vendor/openbabel_copy.h"
#include "molgr/compat/openbabel_iter.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/ring.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
    using Candidate = molgr::state::MetalCandidateState;
    using Site = std::vector<int>;
    using Sites = std::set<Site>;
    using Signature = std::pair<std::vector<std::array<int, 6>>, std::vector<std::array<int, 3>>>;
    using molgr::metal::scoring::MetadataDouble;
    constexpr const char *QuinodiimineKey = "metal_quinodiimine_geometry_conflict_count";
    constexpr const char *PolyenamineKey = "metal_polyenamine_geometry_conflict_count";
    constexpr std::size_t MaxVariants = 32, MaxPathAtoms = 32, MaxPathStates = 4096;
    /// An alternating-path search exceeded its budget before enumerating all resonance witnesses.
    struct NormalizationBoundExceeded {};

    /// Require explicit hydrogens and exclude radicals and unresolved two-electron centers.
    /// @param atom Atom whose local chemical environment is examined.
    /// @return True for a site without implicit hydrogens, unpaired electrons or a deferred center.
    bool Resolved(OpenBabel::OBAtom *atom)
    {
        return atom->GetImplicitHCount() == 0 && molgr::utils::GetUnpairedElectronCount(*atom) == 0 &&
            !molgr::utils::HasUnresolvedTwoElectronCenter(*atom);
    }

    /// @param mol Molecular graph on which the operation acts.
    /// @param a First one-based atom index in the interaction.
    /// @param b Second one-based atom index in the interaction.
    /// @param c Third one-based atom index in the interaction.
    /// @param d Fourth one-based atom index in the interaction.
    /// @return True when the four atom positions define a torsion within 15 degrees of coplanarity.
    bool PlanarTorsion(OpenBabel::OBMol &mol, int a, int b, int c, int d)
    {
        const double angle = std::abs(mol.GetTorsion(a, b, c, d));
        return std::isfinite(angle) && std::min(angle, std::abs(180.0 - angle)) < 15.0;
    }

    /// Decode the groups of atom indices visible to each metal from the candidate metadata.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Donor-index sets in stored metal order.
    std::vector<std::set<int>> VisibleGroups(const Candidate &candidate)
    {
        std::vector<std::set<int>> groups;
        const auto found = candidate.metadata.find("metal_visible_atom_groups");
        if (found == candidate.metadata.end()) return groups;
        const auto *text = std::get_if<std::string>(&found->second);
        if (!text) return groups;
        std::istringstream stream(*text);
        std::string group;
        while (std::getline(stream, group, ';'))
        {
            std::set<int> atoms;
            std::istringstream parts(group);
            std::string index;
            while (std::getline(parts, index, ','))
                if (!index.empty()) atoms.insert(std::stoi(index));
            groups.push_back(std::move(atoms));
        }
        return groups;
    }

    /// Include bonds crossing the motif boundary; unrelated changes abstain.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param inside Motif atom indices whose local electronic labels are excluded from the spectator
    ///     comparison.
    /// @return Spectator atom labels and sorted bond labels, including bonds crossing the motif boundary.
    Signature OutsideSignature(OpenBabel::OBMol &mol, const std::set<int> &inside)
    {
        Signature signature;
        FOR_ATOMS_OF_MOL(atom, mol)
        {
            if (inside.count(atom->GetIdx())) continue;
            signature.first.push_back({static_cast<int>(atom->GetIdx()), static_cast<int>(atom->GetAtomicNum()),
                atom->GetFormalCharge(), molgr::utils::GetUnpairedElectronCount(*atom),
                static_cast<int>(molgr::utils::HasUnresolvedTwoElectronCenter(*atom)),
                static_cast<int>(atom->GetImplicitHCount())});
        }
        FOR_BONDS_OF_MOL(bond, mol)
        {
            const int a = bond->GetBeginAtomIdx(), b = bond->GetEndAtomIdx();
            if (!inside.count(a) || !inside.count(b))
                signature.second.push_back({std::min(a, b), std::max(a, b), static_cast<int>(bond->GetBondOrder())});
        }
        std::sort(signature.second.begin(), signature.second.end());
        return signature;
    }

    /// Include bonds crossing the motif boundary; unrelated changes abstain.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param inside Motif atom indices whose local electronic labels are excluded from the spectator
    ///     comparison.
    /// @return Spectator atom labels and sorted bond labels, including bonds crossing the motif boundary.
    Signature OutsideSignature(const Candidate &candidate, const std::set<int> &inside)
    {
        return OutsideSignature(const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol()), inside);
    }

    /// Subtract the previously recorded motif contribution from both candidate conflict scores.
    /// @param candidates Metal-ligand candidates whose motif conflicts are updated in place.
    /// @param key Lookup key for the recorded state or property.
    void Clear(std::vector<Candidate> &candidates, const char *key)
    {
        for (auto &candidate : candidates)
        {
            const double previous = MetadataDouble(candidate, key);
            for (const auto *field : {"metal_discordance_structural_count", "metal_discordance_count"})
                candidate.metadata[field] = MetadataDouble(candidate, field) - previous;
            candidate.metadata[key] = 0;
        }
    }

    /// Accumulate geometry conflicts under the specified candidate metadata key.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param key Lookup key for the recorded state or property.
    /// @param count Number of work items or occurrences.
    void AddConflicts(Candidate &candidate, const char *key, std::size_t count)
    {
        candidate.metadata[key] = static_cast<int>(count);
        for (const auto *field : {"metal_discordance_structural_count", "metal_discordance_count"})
            candidate.metadata[field] = MetadataDouble(candidate, field) + count;
    }

    /// Check whether the two single-metal candidates can be compared as local redox alternatives.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param partner Second metal-ligand candidate, carrying the alternative motif assignment.
    /// @param count Current candidate conflict score compared with the proposed partner.
    /// @param otherCount Conflict score of the proposed partner.
    /// @return True if the metal identity, spin budget and comparison penalties permit pairing.
    bool Compatible(const Candidate &candidate, const Candidate &partner, double count, double otherCount)
    {
        return candidate.metal_states.size() == 1 && partner.metal_states.size() == 1 &&
            candidate.no_metal_radical_target == partner.no_metal_radical_target && otherCount <= count &&
            MetadataDouble(candidate, "nitrosyl_geometry_penalty") == MetadataDouble(partner, "nitrosyl_geometry_penalty") &&
            candidate.metal_states[0].idx == partner.metal_states[0].idx &&
            candidate.metal_states[0].radical_num == partner.metal_states[0].radical_num;
    }

    constexpr const char *PyramidalPSKey = "metal_pyramidal_ps_geometry_conflict_count";
    constexpr const char *PyramidalPNKey = "metal_pyramidal_pn_geometry_conflict_count";

    /// Identify long P+=N / P–N(-) partners at pyramidal cyclic RPN2 sites.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return P/N site assignments. True marks P-N(-), false marks P(+)=N.
    std::map<Site, bool> PyramidalPNSites(const Candidate &candidate)
    {
        // Long P+=N at a pyramidal cyclic RPN2 center requires a complete
        // P(0)-N(-) redox partner; never forbid general phosphazene bonding.
        std::map<Site, bool> sites;
        if (!candidate.no_metal_state) return sites;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        std::set<int> visible;
        // Keep the same visibility field as the P/S screen and Python backend.
        const auto found = candidate.metadata.find("metal_visible_atom_indices");
        if (found != candidate.metadata.end())
            if (const auto *value = std::get_if<std::string>(&found->second))
            {
                std::istringstream stream(*value);
                std::string index;
                while (std::getline(stream, index, ','))
                    if (!index.empty()) visible.insert(std::stoi(index));
            }
        FOR_ATOMS_OF_MOL(atom, mol)
        {
            auto *p = &*atom;
            if (p->GetAtomicNum() != 15 || p->GetExplicitDegree() != 3 ||
                !p->IsInRing() || !visible.count(p->GetIdx()) || !Resolved(p)) continue;
            std::vector<OpenBabel::OBAtom *> neighbors;
            std::vector<int> elements;
            bool valid = true;
            FOR_NBORS_OF_ATOM(n, p)
            {
                neighbors.push_back(&*n);
                elements.push_back(n->GetAtomicNum());
                if (!Resolved(&*n) || (n->GetAtomicNum() == 7 && !n->IsInRing())) valid = false;
            }
            std::sort(elements.begin(), elements.end());
            if (!valid || elements != std::vector<int>{6, 7, 7}) continue;
            // Abstain for a pi-active carbon substituent: its coupled redox
            // alternatives are outside this P/N-local model (COLPOI control).
            for (auto *carbon : neighbors)
                if (carbon->GetAtomicNum() == 6)
                {
                    if (carbon->GetFormalCharge() != 0 || carbon->GetExplicitDegree() != 4 ||
                        carbon->GetTotalValence() != 4) valid = false;
                    FOR_BONDS_OF_ATOM(bond, carbon)
                        if (bond->GetBondOrder() != 1) valid = false;
                }
            if (!valid) continue;
            double angleSum = 0;
            for (std::size_t i = 0; i < neighbors.size(); ++i)
                for (std::size_t j = i + 1; j < neighbors.size(); ++j)
                {
                    const double angle = mol.GetAngle(neighbors[i], p, neighbors[j]);
                    if (!(80 < angle && angle < 130)) valid = false;
                    angleSum += angle;
                }
            if (!valid || !(300 < angleSum && angleSum < 330)) continue;
            for (auto *n : neighbors)
            {
                if (n->GetAtomicNum() != 7 || n->GetExplicitDegree() != 2 ||
                    !(1.64 < p->GetDistance(n) && p->GetDistance(n) < 1.85)) continue;
                valid = true;
                FOR_NBORS_OF_ATOM(a, n)
                    if (a->GetAtomicNum() != 15 ||
                        (a->GetIdx() != p->GetIdx() && n->GetBond(&*a)->GetBondOrder() != 1)) valid = false;
                for (auto *a : neighbors)
                    if (a != n && p->GetBond(a)->GetBondOrder() != 1) valid = false;
                if (!valid) continue;
                const std::array<int, 5> labels{p->GetFormalCharge(), static_cast<int>(p->GetTotalValence()),
                    n->GetFormalCharge(), static_cast<int>(n->GetTotalValence()),
                    static_cast<int>(p->GetBond(n)->GetBondOrder())};
                const Site site{static_cast<int>(p->GetIdx()), static_cast<int>(n->GetIdx())};
                if (labels == std::array<int, 5>{1, 4, 0, 3, 2}) sites[site] = false;
                else if (labels == std::array<int, 5>{0, 3, -1, 2, 1}) sites[site] = true;
            }
        }
        return sites;
    }

    /// Recognize resolved, metal-visible R2P=S / R2P–S(-) alternatives.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return P/S site assignments. True marks P-S(-), false marks P(+)=S.
    std::map<Site, bool> PyramidalPSSites(const Candidate &candidate)
    {
        std::map<Site, bool> sites; // true: lone-pair-bearing P(0)-S(-)
        if (!candidate.no_metal_state) return sites;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        std::set<int> visible;
        const auto found = candidate.metadata.find("metal_visible_atom_indices");
        if (found != candidate.metadata.end())
            if (const auto *value = std::get_if<std::string>(&found->second))
            {
                std::istringstream stream(*value);
                std::string index;
                while (std::getline(stream, index, ','))
                    if (!index.empty()) visible.insert(std::stoi(index));
            }
        FOR_ATOMS_OF_MOL(atom, mol)
        {
            auto *p = &*atom;
            if (p->GetAtomicNum() != 15 || p->GetExplicitDegree() != 3 ||
                !visible.count(p->GetIdx()) || !Resolved(p)) continue;
            std::vector<OpenBabel::OBAtom *> neighbors;
            bool valid = true;
            FOR_NBORS_OF_ATOM(n, p)
            {
                neighbors.push_back(&*n);
                if (!Resolved(&*n)) valid = false;
            }
            double angleSum = 0.0;
            for (std::size_t i = 0; i < neighbors.size(); ++i)
                for (std::size_t j = i + 1; j < neighbors.size(); ++j)
                {
                    const double angle = mol.GetAngle(neighbors[i], p, neighbors[j]);
                    if (!(80.0 < angle && angle < 130.0)) valid = false;
                    angleSum += angle;
                }
            if (!valid || !(300.0 < angleSum && angleSum < 330.0)) continue;
            for (auto *s : neighbors)
            {
                if (s->GetAtomicNum() != 16 || s->GetExplicitDegree() != 1) continue;
                valid = true;
                for (auto *n : neighbors)
                    if (n != s && (n->GetAtomicNum() != 6 || p->GetBond(n)->GetBondOrder() != 1))
                        valid = false;
                if (!valid) continue;
                const std::array<int, 5> labels{p->GetFormalCharge(), static_cast<int>(p->GetTotalValence()),
                    s->GetFormalCharge(), static_cast<int>(s->GetTotalValence()),
                    static_cast<int>(p->GetBond(s)->GetBondOrder())};
                const Site site{static_cast<int>(p->GetIdx()), static_cast<int>(s->GetIdx())};
                if (labels == std::array<int, 5>{1, 4, 0, 2, 2}) sites[site] = false;
                else if (labels == std::array<int, 5>{0, 3, -1, 1, 1}) sites[site] = true;
            }
        }
        return sites;
    }

    /// Separate reduced C–N and neutral C=N labels on adjacent donors of an equalized six-carbon ring.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Two sets of ortho sites, first those with reduced C-N labels and then those with neutral
    ///     C=N labels.
    std::pair<Sites, Sites> QuinodiimineSites(const Candidate &candidate)
    {
        std::pair<Sites, Sites> result; // reduced / quinodiimine
        if (!candidate.no_metal_state) return result;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const auto groups = VisibleGroups(candidate);
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &ids = ring->_path;
            if (ids.size() != 6) continue;
            bool valid = true;
            std::vector<double> lengths;
            for (int i = 0; i < 6; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const double distance = atom->GetDistance(mol.GetAtom(ids[(i + 1) % 6]));
                lengths.push_back(distance);
                if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 3 || !Resolved(atom) ||
                    atom->GetFormalCharge() != 0 || atom->GetTotalValence() != 4 ||
                    !(1.35 < distance && distance < 1.45) ||
                    !PlanarTorsion(mol, ids[i], ids[(i + 1) % 6], ids[(i + 2) % 6], ids[(i + 3) % 6])) valid = false;
            }
            if (!valid || *std::max_element(lengths.begin(), lengths.end()) -
                *std::min_element(lengths.begin(), lengths.end()) > 0.06) continue;
            for (int i = 0; i < 6; ++i)
            {
                std::vector<OpenBabel::OBAtom *> nitrogens;
                std::vector<OpenBabel::OBAtom *> carbons{mol.GetAtom(ids[i]), mol.GetAtom(ids[(i + 1) % 6])};
                for (auto *carbon : carbons)
                {
                    std::vector<OpenBabel::OBAtom *> outside;
                    FOR_NBORS_OF_ATOM(neighbor, carbon)
                        if (std::find(ids.begin(), ids.end(), neighbor->GetIdx()) == ids.end()) outside.push_back(&*neighbor);
                    if (outside.size() != 1) break;
                    auto *nitrogen = outside[0];
                    const double distance = nitrogen->GetDistance(carbon);
                    if (nitrogen->GetAtomicNum() != 7 || nitrogen->GetExplicitDegree() != 2 || !Resolved(nitrogen) ||
                        !(1.36 < distance && distance < 1.46)) break;
                    nitrogens.push_back(nitrogen);
                }
                if (nitrogens.size() != 2 || !std::any_of(groups.begin(), groups.end(), [&](const auto &group) {
                        return group.count(nitrogens[0]->GetIdx()) && group.count(nitrogens[1]->GetIdx());
                    })) continue;
                Site site(ids.begin(), ids.end());
                std::sort(site.begin(), site.end());
                site.push_back(std::min(nitrogens[0]->GetIdx(), nitrogens[1]->GetIdx()));
                site.push_back(std::max(nitrogens[0]->GetIdx(), nitrogens[1]->GetIdx()));
                bool reduced = true, oxidized = true;
                for (int j = 0; j < 2; ++j)
                {
                    auto *n = nitrogens[j];
                    const int q = n->GetFormalCharge(), v = n->GetTotalValence();
                    const int order = n->GetBond(carbons[j])->GetBondOrder();
                    reduced &= ((q == 0 && v == 3) || (q == -1 && v == 2)) && order == 1;
                    oxidized &= q == 0 && v == 3 && order == 2;
                }
                if (reduced) result.first.insert(site);
                if (oxidized) result.second.insert(site);
            }
        }
        return result;
    }

    /// N=C-...-X(-) <-> N(-)-C=...=X using a closed alternating path.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param start One-based atom index at the start of the alternating electron-transfer path.
    /// @param forbidden Atoms excluded from the alternating-path search.
    /// @return Alternating charge-transfer paths beginning at the specified atom.
    /// @exception NormalizationBoundExceeded The alternating-path search exceeded its configured work
    ///     bound.
    std::vector<Site> ChargeTransferPaths(OpenBabel::OBMol &mol, int start, const std::set<int> &forbidden)
    {
        std::deque<Site> queue{{start}};
        std::vector<Site> paths;
        std::size_t visited = 0;
        while (!queue.empty())
        {
            if (++visited > MaxPathStates) throw NormalizationBoundExceeded{};
            auto path = std::move(queue.front());
            queue.pop_front();
            auto *atom = mol.GetAtom(path.back());
            const int order = path.size() % 2 ? 2 : 1;
            FOR_BONDS_OF_ATOM(bond, atom)
            {
                auto *other = bond->GetNbrAtom(atom);
                const int idx = other->GetIdx();
                if (forbidden.count(idx) || std::find(path.begin(), path.end(), idx) != path.end() ||
                    bond->GetBondOrder() != order || !Resolved(other)) continue;
                const int z = other->GetAtomicNum(), q = other->GetFormalCharge(), v = other->GetTotalValence();
                const int neutralValence = z == 6 ? 4 : z == 7 ? 3 : z == 8 ? 2 : -1;
                const bool endpoint = order == 1 && q == -1 && neutralValence > 0 && v == neutralValence - 1;
                const bool internal = q == 0 && v == neutralValence;
                if (!endpoint && !internal) continue;
                if (path.size() >= MaxPathAtoms) throw NormalizationBoundExceeded{};
                auto extended = path;
                extended.push_back(idx);
                if (endpoint)
                {
                    paths.push_back(std::move(extended));
                    if (paths.size() > MaxVariants) throw NormalizationBoundExceeded{};
                }
                else queue.push_back(std::move(extended));
            }
        }
        return paths;
    }

    /// Compare the rest of the ligand after each allowed charge shift onto the diimine site.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param changed Motif atom indices whose local electronic labels are excluded from the spectator
    ///     comparison.
    /// @return Admissible spectator signatures, empty if the bounded normalization cannot be completed.
    std::set<Signature> NormalizedOutsideSignatures(const Candidate &candidate, const Sites &changed)
    {
        // Copy-only resonance normalization; a truncated traversal must abstain.
        try
        {
            std::set<int> core, nitrogens;
            for (const auto &site : changed)
            {
                core.insert(site.begin(), site.end());
                nitrogens.insert(site.end() - 2, site.end());
            }
            using MolPtr = std::shared_ptr<OpenBabel::OBMol>;
            std::vector<MolPtr> variants{molgr::vendor::openbabel_threading::SharedMolCopy(candidate.no_metal_state->Mol())};
            for (int index : nitrogens)
            {
                std::vector<MolPtr> extended;
                for (const auto &mol : variants)
                {
                    auto *n = mol->GetAtom(index);
                    if (n->GetFormalCharge() == -1 && n->GetTotalValence() == 2) extended.push_back(mol);
                    else if (n->GetFormalCharge() == 0 && n->GetTotalValence() == 3)
                    {
                        for (const auto &path : ChargeTransferPaths(*mol, index, core))
                        {
                            auto copy = molgr::vendor::openbabel_threading::SharedMolCopy(*mol);
                            copy->GetAtom(index)->SetFormalCharge(-1);
                            copy->GetAtom(path.back())->SetFormalCharge(0);
                            for (std::size_t i = 1; i < path.size(); ++i)
                            {
                                auto *bond = copy->GetBond(path[i - 1], path[i]);
                                bond->SetBondOrder(3 - bond->GetBondOrder());
                            }
                            extended.push_back(std::move(copy));
                        }
                    }
                    if (extended.size() > MaxVariants) throw NormalizationBoundExceeded{};
                }
                variants = std::move(extended);
                if (variants.empty()) return {};
            }
            std::set<Signature> result;
            for (const auto &mol : variants) result.insert(OutsideSignature(*mol, core));
            return result;
        }
        catch (const NormalizationBoundExceeded &) { return {}; }
    }

    /// Classify planar chains with a metal-facing CH₂ end as reduced polyenamine or iminium.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Matched chains with true for the reduced terminal assignment and false for its inverted
    ///     alternative.
    std::map<Site, bool> PolyenamineSites(const Candidate &candidate)
    {
        std::map<Site, bool> result; // true: reduced; false: inverted iminium
        if (!candidate.no_metal_state) return result;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const auto groups = VisibleGroups(candidate);
        /// Recognize resolved, acyclic, three-coordinate carbon in the planar polyenamine path.
        const auto carbon = [](OpenBabel::OBAtom *atom) {
            return atom->GetAtomicNum() == 6 && atom->GetExplicitDegree() == 3 && !atom->IsInRing() && Resolved(atom);
        };
        FOR_ATOMS_OF_MOL(nitrogen, mol)
        {
            auto *n = &*nitrogen;
            if (n->GetAtomicNum() != 7 || n->GetExplicitDegree() != 3 || n->IsInRing() || !Resolved(n)) continue;
            std::vector<OpenBabel::OBAtom *> neighbors;
            bool valid = true;
            FOR_NBORS_OF_ATOM(neighbor, n)
            {
                neighbors.push_back(&*neighbor);
                const double distance = n->GetDistance(&*neighbor);
                if ((neighbor->GetAtomicNum() != 1 && neighbor->GetAtomicNum() != 6) ||
                    !(0.8 < distance && distance < 1.6)) valid = false;
            }
            double angleSum = 0.0;
            for (std::size_t i = 0; i < neighbors.size(); ++i)
                for (std::size_t j = i + 1; j < neighbors.size(); ++j)
                {
                    const double angle = mol.GetAngle(neighbors[i], n, neighbors[j]);
                    if (!(95.0 < angle && angle < 145.0)) valid = false;
                    angleSum += angle;
                }
            if (!valid || !(angleSum > 355.0)) continue;
            for (auto *first : neighbors)
            {
                const double distance = n->GetDistance(first);
                if (!carbon(first) || !(1.30 < distance && distance < 1.43)) continue;
                Site path{static_cast<int>(n->GetIdx()), static_cast<int>(first->GetIdx())};
                while (path.size() < 10)
                {
                    std::vector<int> options;
                    FOR_NBORS_OF_ATOM(neighbor, mol.GetAtom(path.back()))
                        if (neighbor->GetIdx() != path[path.size() - 2] && carbon(&*neighbor)) options.push_back(neighbor->GetIdx());
                    if (options.size() != 1 || std::find(path.begin(), path.end(), options[0]) != path.end()) break;
                    path.push_back(options[0]);
                }
                if (path.size() != 4 && path.size() != 6 && path.size() != 8 && path.size() != 10) continue;
                int hydrogens = 0;
                FOR_NBORS_OF_ATOM(neighbor, mol.GetAtom(path.back()))
                    if (neighbor->GetAtomicNum() == 1) ++hydrogens;
                if (hydrogens != 2 || !std::any_of(groups.begin(), groups.end(), [&](const auto &group) {
                        return group.count(path.back()) && group.count(path[path.size() - 2]) && !group.count(path[0]);
                    })) continue;
                valid = true;
                std::vector<double> lengths;
                for (std::size_t i = 1; i + 1 < path.size(); ++i)
                {
                    const double length = mol.GetAtom(path[i])->GetDistance(mol.GetAtom(path[i + 1]));
                    lengths.push_back(length);
                    if (!(1.30 < length && length < 1.50)) valid = false;
                }
                for (std::size_t i = 0; i < lengths.size(); i += 2)
                    if (!(lengths[i + 1] - lengths[i] > 0.02)) valid = false;
                for (std::size_t i = 0; i + 2 < path.size(); ++i)
                {
                    const double angle = mol.GetAngle(mol.GetAtom(path[i]), mol.GetAtom(path[i + 1]), mol.GetAtom(path[i + 2]));
                    if (!(105.0 < angle && angle < 145.0)) valid = false;
                }
                for (std::size_t i = 0; i + 3 < path.size(); ++i)
                    if (!PlanarTorsion(mol, path[i], path[i + 1], path[i + 2], path[i + 3])) valid = false;
                if (!valid) continue;
                bool reduced = n->GetFormalCharge() == 0 && n->GetTotalValence() == 3 &&
                    mol.GetBond(path[0], path[1])->GetBondOrder() == 1;
                bool inverted = n->GetFormalCharge() == 1 && n->GetTotalValence() == 4;
                int anions = 0;
                for (std::size_t i = 1; i < path.size(); ++i)
                {
                    auto *atom = mol.GetAtom(path[i]);
                    const int q = atom->GetFormalCharge(), v = atom->GetTotalValence();
                    const int order = mol.GetBond(path[i - 1], path[i])->GetBondOrder();
                    if (q == -1 && v == 3) ++anions;
                    reduced &= ((q == -1 && v == 3) || (q == 0 && v == 4)) && (order == 1 || order == 2);
                    inverted &= q == 0 && v == 4 && order == (i % 2 ? 2 : 1);
                }
                if (reduced && anions == 1) result[path] = true;
                if (inverted) result[path] = false;
            }
        }
        return result;
    }
}

namespace molgr::metal::geometry
{
    void ClearConjugatedGeometryConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, PolyenamineKey);
        Clear(candidates, QuinodiimineKey);
    }

    void AnnotateQuinodiimineAlternatives(std::vector<Candidate> &candidates)
    {
        Clear(candidates, QuinodiimineKey);
        std::vector<std::pair<Sites, Sites>> features;
        std::vector<double> counts;
        for (const auto &candidate : candidates)
        {
            features.push_back(QuinodiimineSites(candidate));
            counts.push_back(MetadataDouble(candidate, "metal_discordance_structural_count"));
        }
        std::map<std::pair<std::size_t, Sites>, std::set<Signature>> cache;
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            const auto &[red, neutral] = features[i];
            Sites conflicts;
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (!Compatible(candidate, partner, counts[i], counts[j])) continue;
                const auto &[otherRed, otherNeutral] = features[j];
                Sites changed, all = red, otherAll = otherRed;
                std::set_intersection(neutral.begin(), neutral.end(), otherRed.begin(), otherRed.end(), std::inserter(changed, changed.end()));
                all.insert(neutral.begin(), neutral.end());
                otherAll.insert(otherNeutral.begin(), otherNeutral.end());
                if (changed.empty() || all != otherAll || !std::includes(otherRed.begin(), otherRed.end(), red.begin(), red.end()) ||
                    partner.metal_states[0].valence - candidate.metal_states[0].valence != 2 * static_cast<int>(changed.size())) continue;
                std::set<int> core;
                for (const auto &site : changed) core.insert(site.begin(), site.end());
                const auto key = std::make_pair(j, changed);
                auto found = cache.find(key);
                if (found == cache.end()) found = cache.emplace(key, NormalizedOutsideSignatures(partner, changed)).first;
                if (found->second.count(OutsideSignature(candidate, core))) conflicts.insert(changed.begin(), changed.end());
            }
            AddConflicts(candidate, QuinodiimineKey, conflicts.size());
        }
    }

    void AnnotatePolyenamineAlternatives(std::vector<Candidate> &candidates)
    {
        Clear(candidates, PolyenamineKey);
        std::vector<std::map<Site, bool>> features;
        std::vector<double> counts;
        for (const auto &candidate : candidates)
        {
            features.push_back(PolyenamineSites(candidate));
            counts.push_back(MetadataDouble(candidate, "metal_discordance_structural_count"));
        }
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            Sites conflicts;
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (!Compatible(candidate, partner, counts[i], counts[j]) ||
                    partner.metal_states[0].valence != candidate.metal_states[0].valence + 2) continue;
                for (const auto &[site, reduced] : features[i])
                {
                    const auto alternative = features[j].find(site);
                    const std::set<int> core(site.begin(), site.end());
                    if (!reduced && alternative != features[j].end() && alternative->second &&
                        OutsideSignature(candidate, core) == OutsideSignature(partner, core)) conflicts.insert(site);
                }
            }
            AddConflicts(candidate, PolyenamineKey, conflicts.size());
        }
    }
}


namespace
{
    constexpr const char *PairedDiimineKey = "metal_paired_diimine_geometry_conflict_count";

    /// Recognize equalized cis-diimine pairs in a saturated-bridge C10N4 ring.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Map from the sorted diimine core atom indices to its formal charge.
    std::map<Site, int> PairedDiimineSites(const Candidate &candidate)
    {
        std::map<Site, int> result;
        if (!candidate.no_metal_state) return result;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const auto groups = VisibleGroups(candidate);
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &ids = ring->_path;
            if (ids.size() != 14) continue;
            std::set<int> nitrogens;
            bool valid = true;
            for (int idx : ids)
            {
                auto *atom = mol.GetAtom(idx);
                if (atom->GetAtomicNum() == 7) nitrogens.insert(idx);
                if ((atom->GetAtomicNum() != 6 && atom->GetAtomicNum() != 7) || !Resolved(atom)) valid = false;
            }
            if (!valid || nitrogens.size() != 4 || !std::any_of(groups.begin(), groups.end(), [&](const auto &group) {
                    return std::includes(group.begin(), group.end(), nitrogens.begin(), nitrogens.end());
                })) continue;
            std::vector<std::array<int, 4>> units;
            std::set<int> core;
            for (int i = 0; i < 14; ++i)
            {
                std::array<int, 4> path{ids[i], ids[(i + 1) % 14], ids[(i + 2) % 14], ids[(i + 3) % 14]};
                if (mol.GetAtom(path[0])->GetAtomicNum() == 7 && mol.GetAtom(path[1])->GetAtomicNum() == 6 &&
                    mol.GetAtom(path[2])->GetAtomicNum() == 6 && mol.GetAtom(path[3])->GetAtomicNum() == 7)
                {
                    units.push_back(path);
                    core.insert(path.begin(), path.end());
                }
            }
            if (units.size() != 2 || core.size() != 8) continue;
            // Saturated three-carbon bridges separate the two cis diimine units.
            for (int idx : ids)
            {
                if (core.count(idx)) continue;
                auto *a = mol.GetAtom(idx);
                if (a->GetAtomicNum() != 6 || a->GetExplicitDegree() != 4 ||
                    a->GetFormalCharge() != 0 || a->GetTotalValence() != 4) valid = false;
            }
            std::vector<std::array<double, 3>> lengths;
            int charge = 0;
            for (const auto &path : units)
            {
                auto *n = mol.GetAtom(path[0]), *c = mol.GetAtom(path[1]);
                auto *d = mol.GetAtom(path[2]), *k = mol.GetAtom(path[3]);
                if (n->GetExplicitDegree() != 2 || c->GetExplicitDegree() != 3 ||
                    d->GetExplicitDegree() != 3 || k->GetExplicitDegree() != 2 ||
                    c->GetFormalCharge() != 0 || d->GetFormalCharge() != 0 ||
                    c->GetTotalValence() != 4 || d->GetTotalValence() != 4) valid = false;
                const std::array<int, 3> orders{static_cast<int>(n->GetBond(c)->GetBondOrder()),
                    static_cast<int>(c->GetBond(d)->GetBondOrder()), static_cast<int>(d->GetBond(k)->GetBondOrder())};
                const bool neutral = orders == std::array<int, 3>{2, 1, 2} &&
                    n->GetFormalCharge() == 0 && k->GetFormalCharge() == 0 &&
                    n->GetTotalValence() == 3 && k->GetTotalValence() == 3;
                const bool reduced = orders == std::array<int, 3>{1, 2, 1} &&
                    n->GetFormalCharge() == -1 && k->GetFormalCharge() == -1 &&
                    n->GetTotalValence() == 2 && k->GetTotalValence() == 2;
                if (!neutral && !reduced) valid = false;
                const std::array<double, 3> distances{n->GetDistance(c), c->GetDistance(d), d->GetDistance(k)};
                if (!(1.33 < distances[0] && distances[0] < 1.38 &&
                      1.33 < distances[2] && distances[2] < 1.38 &&
                      1.38 < distances[1] && distances[1] < 1.44 &&
                      std::abs(distances[0] - distances[2]) < 0.025)) valid = false;
                // Planar trans chains do not identify this chelating motif.
                if (!(std::abs(mol.GetTorsion(path[0], path[1], path[2], path[3])) < 15.0)) valid = false;
                for (double angle : {mol.GetAngle(n, c, d), mol.GetAngle(c, d, k)})
                    if (!(100.0 < angle && angle < 130.0)) valid = false;
                lengths.push_back(distances);
                charge += n->GetFormalCharge() + k->GetFormalCharge();
            }
            for (int i = 0; i < 3; ++i)
                if (std::abs(lengths[0][i] - lengths[1][i]) >= 0.025) valid = false;
            if (valid) result[Site(core.begin(), core.end())] = charge;
        }
        return result;
    }

    using DiimineGroup = std::tuple<Site, std::vector<std::array<double, 3>>, int, int, int, double>;
    /// Group TIM assignments at fixed donor geometry, metal spin and ligand radical target.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param core Ordered diimine core atom indices defining the comparison group.
    /// @return Comparison key for candidates sharing the diimine core and metal/spin constraints.
    DiimineGroup PairedDiimineGroup(const Candidate &candidate, const Site &core)
    {
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        std::vector<std::array<double, 3>> geometry;
        for (int idx : core)
        {
            auto *a = mol.GetAtom(idx);
            geometry.push_back({a->GetX(), a->GetY(), a->GetZ()});
        }
        const auto &metal = candidate.metal_states[0];
        return {core, geometry, metal.idx, metal.radical_num, candidate.no_metal_radical_target,
            MetadataDouble(candidate, "nitrosyl_geometry_penalty")};
    }
}

namespace molgr::metal::geometry
{
    void ClearPairedDiimineConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, PairedDiimineKey);
    }

    void AnnotatePairedDiimineAlternatives(std::vector<Candidate> &candidates)
    {
        ClearPairedDiimineConflicts(candidates);
        std::vector<std::map<Site, int>> features;
        std::vector<double> counts;
        for (const auto &candidate : candidates)
        {
            features.push_back(PairedDiimineSites(candidate));
            counts.push_back(MetadataDouble(candidate, "metal_discordance_structural_count"));
        }
        std::set<DiimineGroup> supported;
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            const auto &candidate = candidates[i];
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (!Compatible(candidate, partner, counts[i], counts[j])) continue;
                for (const auto &[site, charge] : features[i])
                {
                    const auto found = features[j].find(site);
                    if (charge == -2 || found == features[j].end() || found->second != -2 ||
                        candidate.metal_states[0].valence - partner.metal_states[0].valence != -2 - charge) continue;
                    const std::set<int> core(site.begin(), site.end());
                    const auto group = PairedDiimineGroup(candidate, site);
                    if (group == PairedDiimineGroup(partner, site) &&
                        OutsideSignature(candidate, core) == OutsideSignature(partner, core)) supported.insert(group);
                }
            }
        }
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            std::size_t conflicts = 0;
            if (candidate.metal_states.size() == 1)
                for (const auto &[site, charge] : features[i])
                    if (charge != -2 && supported.count(PairedDiimineGroup(candidate, site))) ++conflicts;
            AddConflicts(candidate, PairedDiimineKey, conflicts);
        }
    }
}

namespace
{
    constexpr const char *PyridylKey = "metal_pyramidal_pyridyl_geometry_conflict_count";
    using PyridylSite = std::tuple<Site, int, int>; // full ring, N, para carbon

    /// Find pyramidal para-CH centers between two elongated bonds in a six-membered pyridyl ring.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @return Matching ring centers and their adjacent ring atoms.
    std::vector<std::array<int, 3>> PyramidalPyridylCenters(OpenBabel::OBMol &mol)
    {
        std::vector<std::array<int, 3>> result;
        std::set<int> seen;
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &ids = ring->_path;
            if (ids.size() != 6) continue;
            int nitrogen = -1, nCount = 0;
            bool valid = true;
            for (int i = 0; i < 6; ++i)
            {
                auto *a = mol.GetAtom(ids[i]);
                if (a->GetAtomicNum() == 7) { nitrogen = i; ++nCount; }
                else if (a->GetAtomicNum() != 6) valid = false;
                if (a->GetExplicitDegree() != 3 || !Resolved(a)) valid = false;
            }
            if (!valid || nCount != 1) continue;
            const int ci = (nitrogen + 3) % 6;
            auto *center = mol.GetAtom(ids[ci]);
            auto *left = mol.GetAtom(ids[(ci + 5) % 6]), *right = mol.GetAtom(ids[(ci + 1) % 6]);
            std::vector<OpenBabel::OBAtom *> neighbors, external;
            FOR_NBORS_OF_ATOM(neighbor, center)
            {
                neighbors.push_back(&*neighbor);
                if (std::find(ids.begin(), ids.end(), neighbor->GetIdx()) == ids.end()) external.push_back(&*neighbor);
            }
            if (external.size() != 1 || external[0]->GetAtomicNum() != 1 ||
                center->GetBond(external[0])->GetBondOrder() != 1) continue;
            for (auto *neighbor : {left, right})
            {
                const double distance = neighbor->GetDistance(center);
                if (!(1.44 < distance && distance < 1.55)) valid = false;
            }
            for (int sign : {-1, 1})
            {
                const double distance = mol.GetAtom(ids[(ci + sign + 6) % 6])->GetDistance(
                    mol.GetAtom(ids[(ci + 2 * sign + 6) % 6]));
                if (!(1.30 < distance && distance < 1.43)) valid = false;
            }
            double angleSum = 0;
            for (std::size_t i = 0; i < neighbors.size(); ++i)
                for (std::size_t j = i + 1; j < neighbors.size(); ++j)
                {
                    const double angle = mol.GetAngle(neighbors[i], center, neighbors[j]);
                    if (!(80 < angle && angle < 140)) valid = false;
                    angleSum += angle;
                }
            if (!valid || !(340 < angleSum && angleSum < 352) || !seen.insert(center->GetIdx()).second) continue;
            result.push_back({static_cast<int>(center->GetIdx()), static_cast<int>(left->GetIdx()), static_cast<int>(right->GetIdx())});
        }
        return result;
    }

    /// A sigma donor approaches the vacant tetrahedral direction. An eta2
    /// approach lies toward one existing C-C bond and must not trigger this rule.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param center Candidate pyramidal atom with three existing neighbors.
    /// @return True when the metal lies in the open direction opposite the three existing neighbors.
    bool VacantPyramidalDirection(const Candidate &candidate, OpenBabel::OBAtom *center)
    {
        if (candidate.metal_states.size() != 1) return false;
        const auto &metal = candidate.metal_states[0];
        const double mx = metal.position_x - center->GetX();
        const double my = metal.position_y - center->GetY();
        const double mz = metal.position_z - center->GetZ();
        const double norm = mx*mx + my*my + mz*mz;
        if (!(norm > 0) || !std::isfinite(norm)) return false;
        int count = 0;
        FOR_NBORS_OF_ATOM(neighbor, center)
        {
            const double dot = mx*(neighbor->GetX()-center->GetX()) +
                my*(neighbor->GetY()-center->GetY()) + mz*(neighbor->GetZ()-center->GetZ());
            if (!std::isfinite(dot) || !(dot < 0)) return false;
            ++count;
        }
        return count == 3;
    }

    /// Identify visible pyramidal pyridyl centers and classify their local electronic state.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Matched sites with true for the carbanion and false for a neutral-carbon alternative.
    std::map<PyridylSite, bool> PyramidalPyridylSites(const Candidate &candidate)
    {
        std::map<PyridylSite, bool> result; // true: reduced, false: oxidized
        if (!candidate.no_metal_state) return result;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const auto groups = VisibleGroups(candidate);
        for (const auto &center : PyramidalPyridylCenters(mol))
        {
            if (!std::any_of(groups.begin(), groups.end(), [&](const auto &g) { return g.count(center[0]); })) continue;
            if (!VacantPyramidalDirection(candidate, mol.GetAtom(center[0]))) continue;
            for (const auto *ring : mol.GetSSSR())
            {
                Site ids(ring->_path.begin(), ring->_path.end());
                std::sort(ids.begin(), ids.end());
                if (ids.size() != 6 || !std::all_of(center.begin(), center.end(), [&](int i) {
                        return std::binary_search(ids.begin(), ids.end(), i); })) continue;
                bool valid = true, neutralCarbons = true;
                int nitrogen = 0, nitrogenCount = 0, carbonCount = 0, charge = 0;
                for (int i : ids)
                {
                    auto *a = mol.GetAtom(i);
                    const int z = a->GetAtomicNum(), q = a->GetFormalCharge(), v = a->GetTotalValence();
                    if (!Resolved(a)) valid = false;
                    charge += q;
                    if (z == 7)
                    {
                        nitrogen = i; ++nitrogenCount;
                        if (!((q == 0 && v == 3) || (q == 1 && v == 4))) valid = false;
                    }
                    else if (z == 6)
                    {
                        ++carbonCount;
                        if (!((q == 0 && v == 4) || (q == -1 && v == 3))) valid = false;
                        neutralCarbons &= q == 0 && v == 4;
                    }
                    else valid = false;
                }
                int bonds = 0;
                FOR_BONDS_OF_MOL(b, mol)
                    if (std::binary_search(ids.begin(), ids.end(), b->GetBeginAtomIdx()) &&
                        std::binary_search(ids.begin(), ids.end(), b->GetEndAtomIdx()))
                    {
                        ++bonds;
                        if (b->GetBondOrder() != 1 && b->GetBondOrder() != 2) valid = false;
                    }
                if (!valid || nitrogenCount != 1 || carbonCount != 5 || bonds != 6) continue;
                const PyridylSite site{ids, nitrogen, center[0]};
                if (charge == -1) result[site] = true;
                else if (neutralCarbons && (charge == 0 || charge == 1)) result[site] = false;
            }
        }
        return result;
    }

    /// Collect outside labels after moving an attached cation onto the pyridyl ring nitrogen.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param site Local ligand motif under comparison.
    /// @param maxPathStates Maximum number of alternating-path states to explore.
    /// @param maxVariants Maximum number of distinct normalizations to retain.
    /// @return Admissible spectator signatures, empty if the bounded normalization cannot be completed.
    std::set<Signature> PyridylOutsideNormalizations(const Candidate &candidate, const PyridylSite &site,
                                                   std::size_t maxPathStates, std::size_t maxVariants)
    {
        const auto &[ring, nitrogen, center] = site;
        auto &source = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const std::set<int> core(ring.begin(), ring.end());
        auto *n = source.GetAtom(nitrogen);
        if (n->GetFormalCharge() == 1 && n->GetTotalValence() == 4) return {OutsideSignature(source, core)};
        if (n->GetFormalCharge() != 0 || n->GetTotalValence() != 3) return {};
        std::deque<Site> queue{{nitrogen}};
        std::vector<Site> paths;
        std::size_t visited = 0;
        while (!queue.empty())
        {
            if (++visited > maxPathStates) return {};
            auto path = std::move(queue.front()); queue.pop_front();
            auto *atom = source.GetAtom(path.back());
            const int order = path.size() % 2 ? 1 : 2;
            FOR_BONDS_OF_ATOM(bond, atom)
            {
                auto *other = bond->GetNbrAtom(atom);
                const int idx = other->GetIdx(), z = other->GetAtomicNum(), q = other->GetFormalCharge(), v = other->GetTotalValence();
                if (std::find(path.begin(), path.end(), idx) != path.end() || bond->GetBondOrder() != order || !Resolved(other)) continue;
                auto extended = path; extended.push_back(idx);
                if (!core.count(idx))
                {
                    if (order == 2 && q == 1 && ((z == 7 && v == 4) || (z == 8 && v == 3)))
                    {
                        paths.push_back(std::move(extended));
                        if (paths.size() > maxVariants) return {};
                    }
                }
                else if (z == 6 && q == 0 && v == 4) queue.push_back(std::move(extended));
            }
        }
        std::set<Signature> result;
        for (const auto &path : paths)
        {
            auto copy = molgr::vendor::openbabel_threading::SharedMolCopy(source);
            copy->GetAtom(nitrogen)->SetFormalCharge(1);
            copy->GetAtom(path.back())->SetFormalCharge(0);
            for (std::size_t i = 1; i < path.size(); ++i)
            {
                auto *bond = copy->GetBond(path[i - 1], path[i]);
                bond->SetBondOrder(3 - bond->GetBondOrder());
            }
            result.insert(OutsideSignature(*copy, core));
        }
        return result;
    }
}

namespace molgr::metal::geometry
{
    void ClearPyramidalPyridylConflicts(std::vector<Candidate> &candidates) { Clear(candidates, PyridylKey); }

    void AnnotatePyramidalPyridylAlternatives(std::vector<Candidate> &candidates,
                                           std::size_t maxPathStates, std::size_t maxVariants)
    {
        ClearPyramidalPyridylConflicts(candidates);
        std::vector<std::map<PyridylSite, bool>> features;
        std::vector<double> counts;
        for (const auto &candidate : candidates)
        {
            features.push_back(PyramidalPyridylSites(candidate));
            counts.push_back(MetadataDouble(candidate, "metal_discordance_structural_count"));
        }
        std::map<std::pair<std::size_t, PyridylSite>, std::set<Signature>> normalized;
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            std::set<PyridylSite> conflicts;
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (!Compatible(candidate, partner, counts[i], counts[j]) ||
                    partner.metal_states[0].valence != candidate.metal_states[0].valence + 2) continue;
                for (const auto &[site, reduced] : features[i])
                {
                    const auto alternative = features[j].find(site);
                    if (reduced || alternative == features[j].end() || !alternative->second) continue;
                    const auto key = std::make_pair(i, site);
                    auto found = normalized.find(key);
                    if (found == normalized.end()) found = normalized.emplace(key,
                        PyridylOutsideNormalizations(candidate, site, maxPathStates, maxVariants)).first;
                    const auto &ring = std::get<0>(site);
                    if (found->second.count(OutsideSignature(partner, std::set<int>(ring.begin(), ring.end())))) conflicts.insert(site);
                }
            }
            AddConflicts(candidate, PyridylKey, conflicts.size());
        }
    }
}

namespace {
    constexpr const char *LocalCCKey = "metal_local_cc_redox_geometry_conflict_count";
    using CCSite = std::pair<int, std::vector<int>>;
    using CCFeatures = std::map<CCSite, bool>; // true: supported reduced form

    /// Recognize the reduced aryl and the quinoid form of a visible N,C chelate.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Electronic alternatives grouped by their local ligand motif.
    CCFeatures OrthoArylAmidoCarbanionSites(const Candidate &candidate)
    {
        CCFeatures out;
        if (!candidate.no_metal_state) return out;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const auto groups = VisibleGroups(candidate);
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &ids = ring->_path;
            if (ids.size() != 6) continue;
            std::vector<double> lengths;
            std::vector<int> orders;
            bool valid = true;
            for (int i = 0; i < 6; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const double length = atom->GetDistance(mol.GetAtom(ids[(i + 1) % 6]));
                const int order = mol.GetBond(ids[i], ids[(i + 1) % 6])->GetBondOrder();
                if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 3 ||
                    atom->GetFormalCharge() != 0 || atom->GetTotalValence() != 4 || !Resolved(atom) ||
                    !(1.35 < length && length < 1.45) || (order != 1 && order != 2) ||
                    !PlanarTorsion(mol, ids[i], ids[(i + 1) % 6], ids[(i + 2) % 6], ids[(i + 3) % 6]))
                {valid = false; break;}
                lengths.push_back(length);
                orders.push_back(order);
            }
            if (!valid || *std::max_element(lengths.begin(), lengths.end()) -
                *std::min_element(lengths.begin(), lengths.end()) > 0.06) continue;
            bool alternating = true;
            int orderSum = 0;
            for (int i = 0; i < 6; ++i)
            {
                alternating = alternating && orders[i] + orders[(i + 1) % 6] == 3;
                orderSum += orders[i];
            }
            for (int i = 0; i < 6; ++i)
            {
                for (int direction : {-1, 1})
                {
                    auto *left = mol.GetAtom(ids[i]);
                    auto *right = mol.GetAtom(ids[(i + direction + 6) % 6]);
                    OpenBabel::OBAtom *nitrogen = nullptr, *carbon = nullptr;
                    FOR_NBORS_OF_ATOM(n, left)
                        if (std::find(ids.begin(), ids.end(), n->GetIdx()) == ids.end()) nitrogen = &*n;
                    FOR_NBORS_OF_ATOM(n, right)
                        if (std::find(ids.begin(), ids.end(), n->GetIdx()) == ids.end()) carbon = &*n;
                    if (!nitrogen || !carbon || nitrogen->GetAtomicNum() != 7 || carbon->GetAtomicNum() != 6 ||
                        nitrogen->GetExplicitDegree() != 2 || carbon->GetExplicitDegree() != 3 ||
                        !Resolved(nitrogen) || !Resolved(carbon)) continue;
                    const double cn = left->GetDistance(nitrogen), cc = right->GetDistance(carbon);
                    if (!(1.38 < cn && cn < 1.48) || !(1.47 < cc && cc < 1.60)) continue;
                    bool visible = false;
                    for (const auto &group : groups)
                        if (group.count(nitrogen->GetIdx()) && group.count(carbon->GetIdx())) visible = true;
                    if (!visible) continue;
                    bool singles = true;
                    for (auto *atom : {nitrogen, carbon})
                        FOR_NBORS_OF_ATOM(n, atom)
                            if (n->GetIdx() != left->GetIdx() && n->GetIdx() != right->GetIdx() &&
                                (n->GetAtomicNum() != 6 || atom->GetBond(&*n)->GetBondOrder() != 1)) singles = false;
                    if (!singles) continue;
                    auto site = ids;
                    std::sort(site.begin(), site.end());
                    site.push_back(nitrogen->GetIdx());
                    site.push_back(carbon->GetIdx());
                    const CCSite key{2, site};
                    const int nc = nitrogen->GetBond(left)->GetBondOrder(), ccc = carbon->GetBond(right)->GetBondOrder();
                    if (nc == 2 && ccc == 2 && nitrogen->GetFormalCharge() == 0 && nitrogen->GetTotalValence() == 3 &&
                        carbon->GetFormalCharge() == 0 && carbon->GetTotalValence() == 4 && orderSum == 8)
                        out[key] = false;
                    if (nc == 1 && ccc == 1 && nitrogen->GetFormalCharge() == -1 && nitrogen->GetTotalValence() == 2 &&
                        carbon->GetFormalCharge() == -1 && carbon->GetTotalValence() == 3 && alternating)
                        out[key] = true;
                }
            }
        }
        return out;
    }

    /// Test the five ring edges and torsions against the compact planar pyrrole geometry window.
    /// @param mol Ligand graph supplying the atom labels, coordinates and bond orders.
    /// @param ids Ordered atom indices defining the candidate motif.
    /// @return True for a five-membered ring within the bond-length and planarity windows.
    bool CompactPyrroleRing(OpenBabel::OBMol &mol, const std::vector<int> &ids) {
        for (int i=0; i<5; ++i) {
            const double d=mol.GetAtom(ids[i])->GetDistance(mol.GetAtom(ids[(i+1)%5]));
            if (!(1.30<d && d<1.46) || !PlanarTorsion(mol,ids[i],ids[(i+1)%5],ids[(i+2)%5],ids[(i+3)%5])) return false;
        }
        return true;
    }
    /// Compare a long inter-ring C=C with two linked pyrrolide anions carrying a single C–C bond.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Electronic alternatives grouped by their local ligand motif.
    CCFeatures PyrroleLinkSites(const Candidate &candidate) {
        CCFeatures out;
        if (!candidate.no_metal_state) return out;
        auto &mol=const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        std::vector<std::pair<std::set<int>,OpenBabel::OBAtom *>> rings;
        for (const auto *ring:mol.GetSSSR()) {
            const auto &ids=ring->_path;
            if (ids.size()!=5 || !CompactPyrroleRing(mol,ids)) continue;
            bool valid=true; int nitrogens=0; OpenBabel::OBAtom *n=nullptr;
            for (int idx:ids) {
                auto *a=mol.GetAtom(idx); const int z=a->GetAtomicNum();
                if ((z!=6 && z!=7) || !Resolved(a) || a->GetExplicitDegree()!=(z==7?2:3)) {valid=false;break;}
                if (z==6 && (a->GetFormalCharge()!=0 || a->GetTotalValence()!=4)) {valid=false;break;}
                if (z==7) {
                    ++nitrogens; n=a;
                    if (!((a->GetFormalCharge()==0 && a->GetTotalValence()==3) || (a->GetFormalCharge()==-1 && a->GetTotalValence()==2))) valid=false;
                }
            }
            if (valid && nitrogens==1) rings.push_back({std::set<int>(ids.begin(),ids.end()),n});
        }
        FOR_BONDS_OF_MOL(bond,mol) {
            auto *a=bond->GetBeginAtom(), *b=bond->GetEndAtom();
            const double d=a->GetDistance(b);
            if (a->GetAtomicNum()!=6 || b->GetAtomicNum()!=6 || !(1.47<d && d<1.58)) continue;
            for (const auto &[left,n]:rings) {
                if (!left.count(a->GetIdx()) || left.count(b->GetIdx()) || !a->GetBond(n)) continue;
                for (const auto &[right,o]:rings) {
                    if (!right.count(b->GetIdx()) || !b->GetBond(o)) continue;
                    bool overlap=false;for (int idx:left) if(right.count(idx)) overlap=true;
                    if (overlap || !PlanarTorsion(mol,n->GetIdx(),a->GetIdx(),b->GetIdx(),o->GetIdx())) continue;
                    std::set<int> core=left;core.insert(right.begin(),right.end());
                    CCSite key{0,std::vector<int>(core.begin(),core.end())};
                    if (bond->GetBondOrder()==2 && n->GetFormalCharge()==0 && o->GetFormalCharge()==0) out[key]=false;
                    if (bond->GetBondOrder()==1 && n->GetFormalCharge()==-1 && o->GetFormalCharge()==-1) out[key]=true;
                }
            }
        }
        return out;
    }
    /// Classify allylic assignments from bond-length asymmetry and endpoint charge and valence.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Electronic alternatives grouped by their local ligand motif.
    CCFeatures AllylCarbeneSites(const Candidate &candidate) {
        CCFeatures out;
        if (!candidate.no_metal_state) return out;
        auto &mol=const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        FOR_ATOMS_OF_MOL(middle,mol) {
            if (middle->GetAtomicNum()!=6 || middle->GetExplicitDegree()!=3 || middle->IsInRing() || !Resolved(&*middle) || middle->GetFormalCharge()!=0 || middle->GetTotalValence()!=4) continue;
            FOR_NBORS_OF_ATOM(first,&*middle) {
                const double length=first->GetDistance(&*middle);
                if (first->GetAtomicNum()!=6 || first->GetExplicitDegree()!=3 || first->IsInRing() || !Resolved(&*first) || !(1.46<length && length<1.60)) continue;
                FOR_NBORS_OF_ATOM(last,&*middle) {
                    const double shortLength=last->GetDistance(&*middle);
                    if (last->GetIdx()==first->GetIdx() || last->GetAtomicNum()!=6 || last->GetExplicitDegree()!=2 || last->IsInRing() || !Resolved(&*last) || !(1.28<shortLength && shortLength<1.40)) continue;
                    const double angle=mol.GetAngle(&*first,&*middle,&*last);
                    if (!(length-shortLength>0.08) || !(95<angle && angle<145)) continue;
                    bool valid=true;
                    for (auto *a:{&*first,&*middle,&*last}) {
                        FOR_NBORS_OF_ATOM(n,a) if(n->GetAtomicNum()!=1 && n->GetAtomicNum()!=6) valid=false;
                    }
                    if (!valid) continue;
                    const int x=first->GetBond(&*middle)->GetBondOrder(),y=middle->GetBond(&*last)->GetBondOrder();
                    CCSite key{1,{static_cast<int>(first->GetIdx()),static_cast<int>(middle->GetIdx()),static_cast<int>(last->GetIdx())}};
                    if (x==2 && y==1 && first->GetFormalCharge()==0 && last->GetFormalCharge()==0 && last->GetTotalValence()==2 && molgr::utils::GetLonePairCount(*last)==1) out[key]=false;
                    if (x==1 && y==2 && last->GetFormalCharge()==-1 && last->GetTotalValence()==3 &&
                        ((first->GetFormalCharge()==-1 && first->GetTotalValence()==3) ||
                         (first->GetFormalCharge()==0 && first->GetTotalValence()==4))) out[key]=true;
                }
            }
        }
        return out;
    }
    /// Encode the ligand outside the selected core after normalizing eligible cyclic bond orders.
    /// @param source Molecule supplying the initial topology and electronic labels.
    /// @param core Motif atom indices whose local electronic labels are excluded from the spectator
    ///     comparison.
    /// @return Spectator atom labels and sorted bond labels, including bonds crossing the motif boundary.
    Signature CycleNormalizedOutside(OpenBabel::OBMol &source,const std::set<int> &core) {
        auto copy=molgr::vendor::openbabel_threading::SharedMolCopy(source);
        auto &mol=*copy;
        std::map<int,int> membership;
        for(const auto *ring:mol.GetSSSR()) for(int idx:ring->_path) ++membership[idx];
        for (const auto *ring:mol.GetSSSR()) {
            auto ids=ring->_path;
            bool isolated=true;
            for(int idx:ids) if(core.count(idx) || membership[idx]!=1) isolated=false;
            if(!isolated) continue;
            if(ids.size()==6 || ids.size()==8) {
                const int size=ids.size();bool valid=true;
                for(int i=0;i<size;++i) {
                    auto *a=mol.GetAtom(ids[i]);
                    if(a->GetAtomicNum()!=6 || a->GetExplicitDegree()!=3 || !Resolved(a) || a->GetFormalCharge()!=0 || a->GetTotalValence()!=4) {valid=false;break;}
                    const int x=mol.GetBond(ids[i],ids[(i+1)%size])->GetBondOrder(),y=mol.GetBond(ids[i],ids[(i+size-1)%size])->GetBondOrder();
                    if((x!=1 && x!=2) || x+y!=3) {valid=false;break;}
                    FOR_BONDS_OF_ATOM(b,a) if(std::find(ids.begin(),ids.end(),b->GetNbrAtom(a)->GetIdx())==ids.end() && b->GetBondOrder()!=1) valid=false;
                }
                if(!valid) continue;
                std::rotate(ids.begin(),std::min_element(ids.begin(),ids.end()),ids.end());
                if(ids[1]>ids.back()) std::reverse(ids.begin()+1,ids.end());
                for(int i=0;i<size;++i) mol.GetBond(ids[i],ids[(i+1)%size])->SetBondOrder(i%2==0?2:1);
                continue;
            }
            if (ids.size()!=5) continue;
            bool valid=true;int charge=0;
            for(int i=0;i<5;++i) {
                const int idx=ids[i];auto *a=mol.GetAtom(idx);
                if(core.count(idx) || a->GetAtomicNum()!=6 || a->GetExplicitDegree()!=3 || !Resolved(a) || !((a->GetFormalCharge()==0 && a->GetTotalValence()==4) || (a->GetFormalCharge()==-1 && a->GetTotalValence()==3))) {valid=false;break;}
                charge+=a->GetFormalCharge();
                const int x=mol.GetBond(idx,ids[(i+1)%5])->GetBondOrder(),y=mol.GetBond(idx,ids[(i+4)%5])->GetBondOrder();
                if((x!=1 && x!=2) || (y!=1 && y!=2) || x+y!=(a->GetFormalCharge()==-1?2:3)) {valid=false;break;}
                FOR_BONDS_OF_ATOM(b,a) {
                    const int other=b->GetNbrAtom(a)->GetIdx();
                    if(std::find(ids.begin(),ids.end(),other)==ids.end() && b->GetBondOrder()!=1) valid=false;
                }
            }
            if(!valid || charge!=-1) continue;
            std::rotate(ids.begin(),std::min_element(ids.begin(),ids.end()),ids.end());
            if(ids[1]>ids[4]) std::reverse(ids.begin()+1,ids.end());
            for(int i=0;i<5;++i) {
                mol.GetAtom(ids[i])->SetFormalCharge(i==0?-1:0);
                mol.GetBond(ids[i],ids[(i+1)%5])->SetBondOrder(i==1 || i==3?2:1);
            }
        }
        return OutsideSignature(mol,core);
    }
    /// Encode the ligand outside the selected core after normalizing eligible cyclic bond orders.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param core Motif atom indices whose local electronic labels are excluded from the spectator
    ///     comparison.
    /// @return Spectator atom labels and sorted bond labels, including bonds crossing the motif boundary.
    Signature CycleNormalizedOutside(const Candidate &candidate,const std::set<int> &core) {
        return CycleNormalizedOutside(const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol()),core);
    }
    /// Compare full outside labels after bounded, copy-only resonance moves.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param site Local ligand motif under comparison.
    /// @param reduced Whether to compare the reduced electronic form of the site.
    /// @return Admissible spectator signatures, empty if the bounded normalization cannot be completed.
    std::set<Signature> LocalCCOutsideSignatures(const Candidate &candidate,const CCSite &site,bool reduced) {
        const std::set<int> core(site.second.begin(),site.second.end());
        if(site.first!=1) return {OutsideSignature(candidate,core)};
        auto &source=const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const int first=site.second.front();
        if(!reduced || source.GetAtom(first)->GetFormalCharge()==-1)
            return {CycleNormalizedOutside(candidate,core)};
        // A delocalized anion supplies a preference only through a complete
        // closed-shell resonance witness with identical outside labels.
        try {
            const auto paths=ChargeTransferPaths(source,first,core);
            std::set<Signature> signatures;
            for(const auto &path:paths) {
                auto mol=molgr::vendor::openbabel_threading::SharedMolCopy(source);
                mol->GetAtom(first)->SetFormalCharge(-1);
                mol->GetAtom(path.back())->SetFormalCharge(0);
                for(std::size_t i=1;i<path.size();++i) {
                    auto *bond=mol->GetBond(path[i-1],path[i]);
                    bond->SetBondOrder(3-bond->GetBondOrder());
                }
                signatures.insert(CycleNormalizedOutside(*mol,core));
            }
            return signatures;
        } catch(const NormalizationBoundExceeded &) {return {};}
    }
}
namespace molgr::metal::geometry {
    void ClearLocalCCRedoxConflicts(std::vector<Candidate> &candidates) {Clear(candidates,LocalCCKey);}
    void AnnotateLocalCCRedoxAlternatives(std::vector<Candidate> &candidates) {
        std::vector<CCFeatures> features;std::vector<double> counts;
        for(const auto &c:candidates) {
            auto f=PyrroleLinkSites(c),g=AllylCarbeneSites(c),h=OrthoArylAmidoCarbanionSites(c);
            f.insert(g.begin(),g.end());f.insert(h.begin(),h.end());features.push_back(std::move(f));
            counts.push_back(MetadataDouble(c,"metal_discordance_structural_count"));
        }
        std::map<std::pair<std::size_t,CCSite>,std::set<Signature>> cache;
        for(std::size_t i=0;i<candidates.size();++i) {
            auto &c=candidates[i];std::set<CCSite> conflicts;
            for(std::size_t j=0;j<candidates.size();++j) {
                const auto &p=candidates[j];
                if(!Compatible(c,p,counts[i],counts[j]) || p.metal_states[0].valence!=c.metal_states[0].valence+2) continue;
                for(const auto &[site,reduced]:features[i]) {
                    const auto other=features[j].find(site);
                    if(reduced || other==features[j].end() || !other->second) continue;
                    for(auto index:{i,j}) {
                        const auto key=std::make_pair(index,site);
                        if(!cache.count(key)) cache.emplace(key,LocalCCOutsideSignatures(candidates[index],site,features[index].at(site)));
                    }
                    for(const auto &signature:cache.at({i,site}))
                        if(cache.at({j,site}).count(signature)) {conflicts.insert(site);break;}
                }
            }
            AddConflicts(c,LocalCCKey,conflicts.size());
        }
    }
}

namespace
{
    constexpr const char *ArylCarbenoidKey = "metal_aryl_carbenoid_geometry_conflict_count";
    using ArylFeatures = std::map<std::vector<int>, bool>;

    /// Identify the two C–S bond-order assignments at a metal-visible aryl carbenoid.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Electronic alternatives grouped by their local ligand motif.
    ArylFeatures ArylCarbenoidSites(const Candidate &candidate)
    {
        ArylFeatures out;
        if (!candidate.no_metal_state) return out;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        std::set<int> visible;
        for (const auto &group : VisibleGroups(candidate)) visible.insert(group.begin(), group.end());
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &ids = ring->_path;
            if (ids.size() != 6) continue;
            bool valid = true;
            std::vector<double> lengths;
            std::vector<int> orders;
            int charge = 0, order_sum = 0;
            for (int i = 0; i < 6; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const int q = atom->GetFormalCharge(), v = atom->GetTotalValence();
                if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 3 || !Resolved(atom) ||
                    !((q == 0 && v == 4) || (q == -1 && v == 3))) {valid = false; break;}
                const double distance = atom->GetDistance(mol.GetAtom(ids[(i + 1) % 6]));
                if (!(1.35 < distance && distance < 1.45) ||
                    !PlanarTorsion(mol, ids[i], ids[(i + 1) % 6], ids[(i + 2) % 6], ids[(i + 3) % 6]))
                {valid = false; break;}
                lengths.push_back(distance);
                const int order = mol.GetBond(ids[i], ids[(i + 1) % 6])->GetBondOrder();
                if (order != 1 && order != 2) {valid = false; break;}
                orders.push_back(order);
                charge += q;
                order_sum += order;
            }
            if (!valid || *std::max_element(lengths.begin(), lengths.end()) -
                *std::min_element(lengths.begin(), lengths.end()) > 0.06) continue;
            bool alternating = true;
            for (int i = 0; i < 6; ++i) if (orders[i] + orders[(i + 1) % 6] != 3) alternating = false;
            for (int index : ids)
            {
                auto *aryl = mol.GetAtom(index);
                FOR_NBORS_OF_ATOM(carbon_iter, aryl)
                {
                    auto *carbon = &*carbon_iter;
                    const double cc = aryl->GetDistance(carbon);
                    if (std::find(ids.begin(), ids.end(), carbon->GetIdx()) != ids.end() ||
                        !visible.count(carbon->GetIdx()) || carbon->GetAtomicNum() != 6 ||
                        carbon->GetExplicitDegree() != 2 || carbon->GetFormalCharge() != -1 ||
                        carbon->GetTotalValence() != 3 || !Resolved(carbon) || !(1.47 < cc && cc < 1.60)) continue;
                    OpenBabel::OBAtom *sulfur = nullptr;
                    FOR_NBORS_OF_ATOM(n, carbon) if (n->GetIdx() != aryl->GetIdx()) sulfur = &*n;
                    if (!sulfur || sulfur->GetAtomicNum() != 16 || sulfur->GetExplicitDegree() != 2 ||
                        !Resolved(sulfur)) continue;
                    const double cs = carbon->GetDistance(sulfur);
                    if (!(1.60 < cs && cs < 1.85)) continue;
                    OpenBabel::OBAtom *rest = nullptr;
                    FOR_NBORS_OF_ATOM(n, sulfur) if (n->GetIdx() != carbon->GetIdx()) rest = &*n;
                    if (!rest || rest->GetAtomicNum() != 6 || sulfur->GetBond(rest)->GetBondOrder() != 1) continue;
                    const int x = aryl->GetBond(carbon)->GetBondOrder(), y = carbon->GetBond(sulfur)->GetBondOrder();
                    auto key = ids;
                    std::sort(key.begin(), key.end());
                    key.push_back(carbon->GetIdx());
                    key.push_back(sulfur->GetIdx());
                    if (x == 2 && y == 1 && sulfur->GetFormalCharge() == 0 && sulfur->GetTotalValence() == 2 &&
                        charge == -1 && order_sum == 8) out[key] = false;
                    if (x == 1 && y == 2 && sulfur->GetFormalCharge() == 1 && sulfur->GetTotalValence() == 3 &&
                        charge == 0 && order_sum == 9 && alternating) out[key] = true;
                }
            }
        }
        return out;
    }
}

namespace molgr::metal::geometry
{
    void ClearArylCarbenoidConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, ArylCarbenoidKey);
    }

    void AnnotateArylCarbenoidAlternatives(std::vector<Candidate> &candidates)
    {
        std::vector<ArylFeatures> features;
        std::vector<double> independent;
        for (const auto &candidate : candidates)
        {
            features.push_back(ArylCarbenoidSites(candidate));
            independent.push_back(MetadataDouble(candidate, "metal_discordance_structural_count") -
                MetadataDouble(candidate, "metal_discordance_inner_visible_same_sign_charge_penalty"));
        }
        std::map<std::pair<std::size_t, std::vector<int>>, Signature> outside;
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            std::set<std::vector<int>> conflicts;
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (!Compatible(candidate, partner, independent[i], independent[j]) ||
                    partner.metal_states[0].valence <= 0 ||
                    candidate.metal_states[0].valence - 2 != partner.metal_states[0].valence) continue;
                for (const auto &[site, supported] : features[i])
                {
                    const auto found = features[j].find(site);
                    if (supported || found == features[j].end() || !found->second) continue;
                    const std::set<int> core(site.begin(), site.end());
                    for (const auto index : {i, j})
                    {
                        const auto key = std::make_pair(index, site);
                        if (!outside.count(key)) outside.emplace(key, CycleNormalizedOutside(candidates[index], core));
                    }
                    if (outside.at({i, site}) == outside.at({j, site})) conflicts.insert(site);
                }
            }
            // Retain the cation donor's intermediate penalty. Penalize the long
            // exocyclic C=C only when a complete compatible alternative exists.
            AddConflicts(candidate, ArylCarbenoidKey, conflicts.size());
        }
    }
}

namespace molgr::metal::geometry
{
    void ClearPyramidalPSConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, PyramidalPSKey);
    }

    /// Compare candidates with matching local sites and record redox alternatives that conflict with
    /// geometry.
    /// @param candidates Metal-ligand candidates whose motif conflicts are updated in place.
    /// @param key Lookup key for the recorded state or property.
    /// @param getSites Extractor for the redox sites to compare across candidates.
    static void AnnotateLocalRedoxAlternatives(std::vector<Candidate> &candidates,
        const char *key, std::map<Site, bool> (*getSites)(const Candidate &))
    {
        // Keep the normal cation-donor penalties. Geometry adds a preference
        // only between complete local redox partners with identical outside graphs.
        Clear(candidates, key);
        std::vector<std::map<Site, bool>> features;
        std::vector<double> counts;
        for (const auto &candidate : candidates)
        {
            features.push_back(getSites(candidate));
            counts.push_back(MetadataDouble(candidate, "metal_discordance_structural_count"));
        }
        for (std::size_t i = 0; i < candidates.size(); ++i)
        {
            auto &candidate = candidates[i];
            Sites conflicts;
            for (std::size_t j = 0; j < candidates.size(); ++j)
            {
                const auto &partner = candidates[j];
                if (!Compatible(candidate, partner, counts[i], counts[j]) ||
                    candidate.metal_states[0].valence + 2 != partner.metal_states[0].valence) continue;
                for (const auto &[site, reduced] : features[i])
                {
                    const auto found = features[j].find(site);
                    if (reduced || found == features[j].end() || !found->second) continue;
                    const std::set<int> inside(site.begin(), site.end());
                    if (OutsideSignature(candidate, inside) == OutsideSignature(partner, inside))
                        conflicts.insert(site);
                }
            }
            AddConflicts(candidate, key, conflicts.size());
        }
    }

    void AnnotatePyramidalPSAlternatives(std::vector<Candidate> &candidates)
    {
        AnnotateLocalRedoxAlternatives(candidates, PyramidalPSKey, PyramidalPSSites);
    }

    void ClearPyramidalPNConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, PyramidalPNKey);
    }

    void AnnotatePyramidalPNAlternatives(std::vector<Candidate> &candidates)
    {
        AnnotateLocalRedoxAlternatives(candidates, PyramidalPNKey, PyramidalPNSites);
    }
}

namespace
{
    constexpr const char *OrthoDioxygenKey = "metal_ortho_dioxygen_geometry_conflict_count";

    /// Identify aromatic O/O partners supported by a planar, equalized ring.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Ortho dioxygen sites with true for the reduced aromatic form and false for the oxidized
    ///     form.
    std::map<Site, bool> OrthoDioxygenSites(const Candidate &candidate)
    {
        std::map<Site, bool> out;
        if (!candidate.no_metal_state) return out;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const auto groups = VisibleGroups(candidate);
        for (const auto *ring : mol.GetSSSR())
        {
            const auto &ids = ring->_path;
            if (ids.size() != 6) continue;
            std::vector<double> lengths;
            std::vector<int> orders;
            bool valid = true;
            for (int i = 0; i < 6; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const double length = atom->GetDistance(mol.GetAtom(ids[(i + 1) % 6]));
                const int order = mol.GetBond(ids[i], ids[(i + 1) % 6])->GetBondOrder();
                if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 3 ||
                    atom->GetFormalCharge() != 0 || atom->GetTotalValence() != 4 || !Resolved(atom) ||
                    !(1.35 < length && length < 1.45) || (order != 1 && order != 2) ||
                    !PlanarTorsion(mol, ids[i], ids[(i + 1) % 6], ids[(i + 2) % 6], ids[(i + 3) % 6]))
                {valid = false; break;}
                lengths.push_back(length);
                orders.push_back(order);
            }
            if (!valid || *std::max_element(lengths.begin(), lengths.end()) -
                *std::min_element(lengths.begin(), lengths.end()) > 0.06) continue;
            bool alternating = true;
            int order_sum = 0;
            for (int i = 0; i < 6; ++i)
            {
                if (orders[i] + orders[(i + 1) % 6] != 3) alternating = false;
                order_sum += orders[i];
            }
            for (int i = 0; i < 6; ++i)
            {
                std::vector<OpenBabel::OBAtom *> oxygens;
                std::set<int> terminals;
                bool reduced = true, oxidized = true;
                for (int k = 0; k < 2; ++k)
                {
                    auto *carbon = mol.GetAtom(ids[(i + k) % 6]);
                    std::vector<OpenBabel::OBAtom *> outside;
                    FOR_NBORS_OF_ATOM(n, carbon)
                        if (std::find(ids.begin(), ids.end(), n->GetIdx()) == ids.end()) outside.push_back(&*n);
                    if (outside.size() != 1) break;
                    auto *oxygen = outside[0];
                    const int degree = oxygen->GetExplicitDegree();
                    const double co = carbon->GetDistance(oxygen);
                    if (oxygen->GetAtomicNum() != 8 || (degree != 1 && degree != 2) ||
                        !Resolved(oxygen) || !(1.32 < co && co < 1.45)) break;
                    bool singles = true;
                    FOR_NBORS_OF_ATOM(n, oxygen)
                        if (n->GetIdx() != carbon->GetIdx() && oxygen->GetBond(&*n)->GetBondOrder() != 1) singles = false;
                    if (!singles) break;
                    oxygens.push_back(oxygen);
                    if (degree == 1) terminals.insert(oxygen->GetIdx());
                    const int q = oxygen->GetFormalCharge(), val = oxygen->GetTotalValence();
                    const int order = oxygen->GetBond(carbon)->GetBondOrder();
                    reduced = reduced && q == degree - 2 && val == degree && order == 1;
                    oxidized = oxidized && q == degree - 1 && val == degree + 1 && order == 2;
                }
                if (oxygens.size() != 2 || terminals.empty()) continue;
                bool visible = false;
                for (const auto &group : groups)
                    if (std::includes(group.begin(), group.end(), terminals.begin(), terminals.end())) visible = true;
                if (!visible) continue;
                auto site = ids;
                std::sort(site.begin(), site.end());
                site.push_back(std::min(oxygens[0]->GetIdx(), oxygens[1]->GetIdx()));
                site.push_back(std::max(oxygens[0]->GetIdx(), oxygens[1]->GetIdx()));
                if (reduced && alternating) out[site] = true;
                if (oxidized && order_sum == 8) out[site] = false;
            }
        }
        return out;
    }
}

namespace molgr::metal::geometry
{
    void ClearOrthoDioxygenConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, OrthoDioxygenKey);
    }

    void AnnotateOrthoDioxygenAlternatives(std::vector<Candidate> &candidates)
    {
        AnnotateLocalRedoxAlternatives(candidates, OrthoDioxygenKey, OrthoDioxygenSites);
    }
}

namespace
{
    constexpr const char *ArylAmidoNHCKey = "metal_aryl_amido_nhc_geometry_conflict_count";
    constexpr const char *ArylPhenoxideNHCKey = "metal_aryl_phenoxide_nhc_geometry_conflict_count";
    constexpr const char *ArylThiolateNHCKey = "metal_aryl_thiolate_nhc_geometry_conflict_count";
    constexpr const char *ArylBenzylNHCKey = "metal_aryl_benzyl_nhc_geometry_conflict_count";

    /// Compare ortho amido/phenoxide NHC arms with quinonoid partners.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @param donor_z Atomic number of the indicated bond partner or site.
    /// @return Matched chelates with true for the anionic donor/aromatic arm and false for the
    ///     quinonoid alternative.
    std::map<Site, bool> ArylDonorNHCSites(const Candidate &candidate, int donor_z)
    {
        std::map<Site, bool> out;
        if (!candidate.no_metal_state) return out;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const auto groups = VisibleGroups(candidate);
        const auto rings = mol.GetSSSR();
        for (const auto *ring : rings)
        {
            const auto &ids = ring->_path;
            if (ids.size() != 6) continue;
            bool valid = true, alternating = true;
            std::vector<double> lengths;
            std::vector<int> orders;
            int order_sum = 0;
            for (int i = 0; i < 6; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const double length = atom->GetDistance(mol.GetAtom(ids[(i + 1) % 6]));
                const int order = mol.GetBond(ids[i], ids[(i + 1) % 6])->GetBondOrder();
                if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 3 ||
                    atom->GetFormalCharge() != 0 || atom->GetTotalValence() != 4 || !Resolved(atom) ||
                    !(1.35 < length && length < 1.45) || (order != 1 && order != 2) ||
                    !PlanarTorsion(mol, ids[i], ids[(i + 1) % 6], ids[(i + 2) % 6], ids[(i + 3) % 6]))
                {valid = false; break;}
                lengths.push_back(length);
                orders.push_back(order);
                order_sum += order;
            }
            if (!valid || *std::max_element(lengths.begin(), lengths.end()) -
                *std::min_element(lengths.begin(), lengths.end()) > 0.06) continue;
            for (int i = 0; i < 6; ++i)
                if (orders[i] + orders[(i + 1) % 6] != 3) alternating = false;
            for (int i = 0; i < 6; ++i)
            {
                std::vector<OpenBabel::OBAtom *> ns, carbons;
                for (int k = 0; k < 2; ++k)
                {
                    auto *carbon = mol.GetAtom(ids[(i + k) % 6]);
                    std::vector<OpenBabel::OBAtom *> outside;
                    FOR_NBORS_OF_ATOM(n, carbon)
                        if (std::find(ids.begin(), ids.end(), n->GetIdx()) == ids.end()) outside.push_back(&*n);
                    if (outside.size() != 1) break;
                    auto *n = outside[0];
                    const double cn = n->GetDistance(carbon);
                    const bool length_ok = n->GetAtomicNum() == 7 ? (1.36 < cn && cn < 1.46) :
                        (donor_z == 6 ? (1.43 < cn && cn < 1.60) :
                         donor_z == 16 ? (1.72 < cn && cn < 1.85) : (1.30 < cn && cn < 1.45));
                    if ((n->GetAtomicNum() != 7 && n->GetAtomicNum() != donor_z) ||
                        !Resolved(n) || !length_ok) break;
                    ns.push_back(n);
                    carbons.push_back(carbon);
                }
                const int donor_degree = donor_z == 6 ? 3 : (donor_z == 7 ? 2 : 1);
                if (ns.size() != 2 || !((ns[0]->GetExplicitDegree() == donor_degree && ns[1]->GetExplicitDegree() == 3) ||
                    (ns[0]->GetExplicitDegree() == 3 && ns[1]->GetExplicitDegree() == donor_degree))) continue;
                const int a = donor_z == 6 ? (ns[0]->GetAtomicNum() == 7 ? 0 : 1) :
                    (ns[0]->GetExplicitDegree() == 3 ? 0 : 1);
                auto *arm = ns[a], *amido = ns[1 - a], *ac = carbons[a], *nc = carbons[1 - a];
                if (arm->GetAtomicNum() != 7 || amido->GetAtomicNum() != donor_z) continue;
                bool singles = true;
                FOR_NBORS_OF_ATOM(n, amido)
                    if (n->GetIdx() != nc->GetIdx() && amido->GetBond(&*n)->GetBondOrder() != 1) singles = false;
                if (!singles) continue;
                if (donor_z == 6)
                {
                    // A sigma-bound benzyl CH2 center can have a relatively
                    // short aryl-C bond. Require the explicit-H pyramid and
                    // metal approach, not bond length alone, to distinguish it
                    // from a planar exocyclic methylene/quinonoid contributor.
                    std::vector<OpenBabel::OBAtom *> neighbors;
                    int hydrogens = 0;
                    FOR_NBORS_OF_ATOM(n, amido)
                    {
                        neighbors.push_back(&*n);
                        if (n->GetAtomicNum() == 1) ++hydrogens;
                    }
                    if (amido->IsInRing() || hydrogens != 2 || neighbors.size() != 3 ||
                        !VacantPyramidalDirection(candidate, amido)) continue;
                    double angle_sum = 0;
                    bool pyramid = true;
                    for (std::size_t j = 0; j < neighbors.size(); ++j)
                        for (std::size_t k = j + 1; k < neighbors.size(); ++k)
                        {
                            const double angle = mol.GetAngle(neighbors[j], amido, neighbors[k]);
                            if (!(90 < angle && angle < 130)) pyramid = false;
                            angle_sum += angle;
                        }
                    if (!pyramid || !(300 < angle_sum && angle_sum < 350)) continue;
                }
                for (const auto *azole_ring : rings)
                {
                    const auto &azole = azole_ring->_path;
                    if (azole.size() != 5 || std::find(azole.begin(), azole.end(), arm->GetIdx()) == azole.end()) continue;
                    bool disjoint = true;
                    for (int idx : azole) if (std::find(ids.begin(), ids.end(), idx) != ids.end()) disjoint = false;
                    if (!disjoint) continue;
                    std::vector<int> elements;
                    std::set<int> nitrogen_ids;
                    std::vector<OpenBabel::OBAtom *> az;
                    bool resolved = true;
                    int az_charge = 0;
                    for (int idx : azole)
                    {
                        auto *atom = mol.GetAtom(idx);
                        az.push_back(atom);
                        elements.push_back(atom->GetAtomicNum());
                        az_charge += atom->GetFormalCharge();
                        if (!Resolved(atom)) resolved = false;
                        if (atom->GetAtomicNum() == 7) nitrogen_ids.insert(idx);
                    }
                    std::sort(elements.begin(), elements.end());
                    if (!resolved || elements != std::vector<int>({6, 6, 6, 7, 7})) continue;
                    bool valid_nitrogens = true;
                    for (int idx : nitrogen_ids)
                    {
                        auto *n = mol.GetAtom(idx);
                        const int q = n->GetFormalCharge(), v = n->GetTotalValence();
                        if (n->GetExplicitDegree() != 3 || !((q == 0 && v == 3) || (q == 1 && v == 4)))
                            valid_nitrogens = false;
                    }
                    if (!valid_nitrogens) continue;
                    std::vector<OpenBabel::OBAtom *> carbenes;
                    for (auto *atom : az)
                    {
                        if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 2 ||
                            atom->GetFormalCharge() != -1 || atom->GetTotalValence() != 3) continue;
                        std::set<int> neighbors;
                        FOR_NBORS_OF_ATOM(n, atom) neighbors.insert(n->GetIdx());
                        if (neighbors == nitrogen_ids) carbenes.push_back(atom);
                    }
                    if (carbenes.size() != 1) continue;
                    auto *carbene = carbenes[0];
                    std::vector<OpenBabel::OBAtom *> backbone;
                    for (auto *atom : az)
                        if (atom->GetAtomicNum() == 6 && atom != carbene) backbone.push_back(atom);
                    bool valid_backbone = true;
                    for (auto *atom : backbone)
                    {
                        if (atom->GetFormalCharge() != 0 || atom->GetTotalValence() != 4)
                            valid_backbone = false;
                        FOR_NBORS_OF_ATOM(n, atom)
                            if (n->GetAtomicNum() == 7 && atom->GetBond(&*n)->GetBondOrder() != 1) valid_backbone = false;
                    }
                    auto *bb = mol.GetBond(backbone[0], backbone[1]);
                    if (!valid_backbone || !bb) continue;
                    bool planar = true;
                    if (bb->GetBondOrder() == 2 && backbone[0]->GetExplicitDegree() == 3 && backbone[1]->GetExplicitDegree() == 3)
                    {
                        for (int j = 0; j < 5; ++j)
                        {
                            const double length = az[j]->GetDistance(az[(j + 1) % 5]);
                            if (!(1.30 < length && length < 1.46) ||
                                !PlanarTorsion(mol, azole[j], azole[(j + 1) % 5], azole[(j + 2) % 5], azole[(j + 3) % 5]))
                                planar = false;
                        }
                    }
                    else if ((donor_z == 8 || donor_z == 16) && bb->GetBondOrder() == 1 &&
                             backbone[0]->GetExplicitDegree() == 4 && backbone[1]->GetExplicitDegree() == 4)
                    {
                        // Saturated NHCs may pucker at the sigma-bonded backbone.
                        // The two N donors must still have planar local geometry.
                        const double cc = backbone[0]->GetDistance(backbone[1]);
                        if (!(1.45 < cc && cc < 1.65)) planar = false;
                        for (auto *atom : backbone)
                        {
                            FOR_BONDS_OF_ATOM(bond, atom)
                                if (bond->GetBondOrder() != 1) planar = false;
                        }
                        for (int idx : nitrogen_ids)
                        {
                            auto *nitrogen = mol.GetAtom(idx);
                            const double cn = nitrogen->GetDistance(carbene);
                            if (!(1.30 < cn && cn < 1.43)) planar = false;
                            std::vector<OpenBabel::OBAtom *> neighbors;
                            FOR_NBORS_OF_ATOM(n, nitrogen)
                            {
                                neighbors.push_back(&*n);
                                if (n->GetIdx() == backbone[0]->GetIdx() || n->GetIdx() == backbone[1]->GetIdx())
                                {
                                    const double length = nitrogen->GetDistance(&*n);
                                    if (!(1.42 < length && length < 1.60)) planar = false;
                                }
                            }
                            double sum = 0;
                            for (std::size_t j = 0; j < neighbors.size(); ++j)
                                for (std::size_t k = j + 1; k < neighbors.size(); ++k)
                                {
                                    const double angle = mol.GetAngle(neighbors[j], nitrogen, neighbors[k]);
                                    if (!(85 < angle && angle < 140)) planar = false;
                                    sum += angle;
                                }
                            if (!(350 < sum && sum <= 360.000001)) planar = false;
                        }
                    }
                    else continue;
                    bool visible = false;
                    for (const auto &group : groups)
                        if (group.count(amido->GetIdx()) && group.count(carbene->GetIdx())) visible = true;
                    if (!planar || !visible) continue;
                    // Include the whole NHC ring: both charge-separated C-=N+
                    // contributors are allowed without altering either graph.
                    auto site = ids, azole_ids = azole;
                    std::sort(site.begin(), site.end());
                    std::sort(azole_ids.begin(), azole_ids.end());
                    site.insert(site.end(), azole_ids.begin(), azole_ids.end());
                    site.push_back(amido->GetIdx());
                    const int q = amido->GetFormalCharge(), v = amido->GetTotalValence();
                    const int bo = amido->GetBond(nc)->GetBondOrder(), arm_bo = arm->GetBond(ac)->GetBondOrder();
                    if (q == -1 && v == donor_degree && bo == 1 && arm_bo == 1 && az_charge == 0 && alternating) out[site] = true;
                    if (q == 0 && v == donor_degree + 1 && bo == 2 && arm_bo == 2 && az_charge == 1 && order_sum == 8) out[site] = false;
                }
            }
        }
        return out;
    }

    /// Match an ortho amido nitrogen and an N-heterocyclic carbene facing the same metal.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Matched chelates with true for the anionic donor/aromatic arm and false for the
    ///     quinonoid alternative.
    std::map<Site, bool> ArylAmidoNHCSites(const Candidate &candidate)
    {
        return ArylDonorNHCSites(candidate, 7);
    }

    /// Match an ortho phenoxide oxygen and an N-heterocyclic carbene facing the same metal.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Matched chelates with true for the anionic donor/aromatic arm and false for the
    ///     quinonoid alternative.
    std::map<Site, bool> ArylPhenoxideNHCSites(const Candidate &candidate)
    {
        return ArylDonorNHCSites(candidate, 8);
    }

    /// Identify the aryl donor and N-heterocyclic carbene pair and classify its electronic assignment.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Matched chelates with true for the anionic donor/aromatic arm and false for the
    ///     quinonoid alternative.
    std::map<Site, bool> ArylThiolateNHCSites(const Candidate &candidate)
    {
        // A long C-S bond alone does not distinguish thiolate from thione.
        // Require the complete equalized aryl/NHC chelate, local N geometry,
        // and a validated charge-balanced partner with identical surroundings.
        return ArylDonorNHCSites(candidate, 16);
    }

    /// Identify the aryl donor and N-heterocyclic carbene pair and classify its electronic assignment.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Matched chelates with true for the anionic donor/aromatic arm and false for the
    ///     quinonoid alternative.
    std::map<Site, bool> ArylBenzylNHCSites(const Candidate &candidate)
    {
        return ArylDonorNHCSites(candidate, 6);
    }
}

namespace molgr::metal::geometry
{
    void ClearArylAmidoNHCConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, ArylAmidoNHCKey);
    }

    void AnnotateArylAmidoNHCAlternatives(std::vector<Candidate> &candidates)
    {
        AnnotateLocalRedoxAlternatives(candidates, ArylAmidoNHCKey, ArylAmidoNHCSites);
    }

    void ClearArylPhenoxideNHCConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, ArylPhenoxideNHCKey);
    }

    void AnnotateArylPhenoxideNHCAlternatives(std::vector<Candidate> &candidates)
    {
        AnnotateLocalRedoxAlternatives(candidates, ArylPhenoxideNHCKey, ArylPhenoxideNHCSites);
    }

    void ClearArylThiolateNHCConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, ArylThiolateNHCKey);
    }

    void AnnotateArylThiolateNHCAlternatives(std::vector<Candidate> &candidates)
    {
        AnnotateLocalRedoxAlternatives(candidates, ArylThiolateNHCKey, ArylThiolateNHCSites);
    }

    void ClearArylBenzylNHCConflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, ArylBenzylNHCKey);
    }

    void AnnotateArylBenzylNHCAlternatives(std::vector<Candidate> &candidates)
    {
        AnnotateLocalRedoxAlternatives(candidates, ArylBenzylNHCKey, ArylBenzylNHCSites);
    }
}

namespace
{
    constexpr const char *AminoC5Key = "metal_amino_c5_geometry_conflict_count";

    /// Compare amino-substituted Cp/indenyl and exocyclic iminium partners.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Matched amino-C5 sites with true for a ring anion and false for the exocyclic iminium.
    std::map<Site, bool> AminoC5Sites(const Candidate &candidate)
    {
        std::map<Site, bool> out;
        if (!candidate.no_metal_state) return out;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        const auto rings = mol.GetSSSR();
        const auto groups = VisibleGroups(candidate);
        for (const auto *ring : rings)
        {
            const auto &ids = ring->_path;
            if (ids.size() != 5) continue;
            std::set<int> core(ids.begin(), ids.end());
            bool visible = false;
            for (const auto &group : groups)
                if (std::includes(group.begin(), group.end(), core.begin(), core.end())) visible = true;
            if (!visible) continue;
            bool valid = true;
            std::vector<double> lengths;
            int ring_charge = 0;
            for (int i = 0; i < 5; ++i)
            {
                auto *atom = mol.GetAtom(ids[i]);
                const int q = atom->GetFormalCharge(), v = atom->GetTotalValence();
                const double length = atom->GetDistance(mol.GetAtom(ids[(i + 1) % 5]));
                if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 3 || !Resolved(atom) ||
                    !((q == 0 && v == 4) || (q == -1 && v == 3)) || !(1.38 < length && length < 1.48) ||
                    !PlanarTorsion(mol, ids[i], ids[(i + 1) % 5], ids[(i + 2) % 5], ids[(i + 3) % 5]))
                {valid = false; break;}
                lengths.push_back(length);
                ring_charge += q;
            }
            if (!valid || *std::max_element(lengths.begin(), lengths.end()) -
                *std::min_element(lengths.begin(), lengths.end()) > 0.06) continue;
            std::vector<std::vector<int>> fused;
            for (const auto *other : rings)
            {
                const std::set<int> other_ids(other->_path.begin(), other->_path.end());
                if (other_ids == core) continue;
                bool overlaps = false;
                for (int idx : other_ids) if (core.count(idx)) overlaps = true;
                if (overlaps) fused.push_back(other->_path);
            }
            if (fused.size() > 1) continue;
            if (!fused.empty())
            {
                const auto &fused_ids = fused[0];
                std::vector<int> shared;
                for (int idx : fused_ids) if (core.count(idx)) shared.push_back(idx);
                if (fused_ids.size() != 6 || shared.size() != 2 || !mol.GetBond(shared[0], shared[1])) continue;
                std::vector<double> fused_lengths;
                for (int i = 0; i < 6; ++i)
                {
                    auto *atom = mol.GetAtom(fused_ids[i]);
                    const double length = atom->GetDistance(mol.GetAtom(fused_ids[(i + 1) % 6]));
                    if (atom->GetAtomicNum() != 6 || atom->GetExplicitDegree() != 3 || atom->GetFormalCharge() != 0 ||
                        atom->GetTotalValence() != 4 || !Resolved(atom) || !(1.35 < length && length < 1.48) ||
                        !PlanarTorsion(mol, fused_ids[i], fused_ids[(i + 1) % 6], fused_ids[(i + 2) % 6], fused_ids[(i + 3) % 6]))
                    {valid = false; break;}
                    fused_lengths.push_back(length);
                }
                if (!valid || *std::max_element(fused_lengths.begin(), fused_lengths.end()) -
                    *std::min_element(fused_lengths.begin(), fused_lengths.end()) > 0.10) continue;
                core.insert(fused_ids.begin(), fused_ids.end());
            }
            FOR_BONDS_OF_MOL(bond, mol)
                if (core.count(bond->GetBeginAtomIdx()) && core.count(bond->GetEndAtomIdx()) &&
                    bond->GetBondOrder() != 1 && bond->GetBondOrder() != 2) valid = false;
            if (!valid) continue;
            for (int idx : ids)
            {
                auto *carbon = mol.GetAtom(idx);
                FOR_NBORS_OF_ATOM(nitrogen, carbon)
                {
                    auto *n = &*nitrogen;
                    const double cn = n->GetDistance(carbon);
                    if (core.count(n->GetIdx()) || n->GetAtomicNum() != 7 || n->GetExplicitDegree() != 3 ||
                        !Resolved(n) || !(1.36 < cn && cn < 1.46)) continue;
                    std::vector<OpenBabel::OBAtom *> neighbors;
                    bool saturated = true;
                    FOR_NBORS_OF_ATOM(neighbor, n)
                    {
                        auto *a = &*neighbor;
                        neighbors.push_back(a);
                        if (a->GetIdx() == carbon->GetIdx()) continue;
                        if (a->GetAtomicNum() != 6 || core.count(a->GetIdx()) || a->GetExplicitDegree() != 4 ||
                            a->GetFormalCharge() != 0 || a->GetTotalValence() != 4 || !Resolved(a) ||
                            n->GetBond(a)->GetBondOrder() != 1) saturated = false;
                    }
                    if (!saturated) continue;
                    double sum = 0;
                    bool planar = true;
                    for (int i = 0; i < 3; ++i)
                        for (int j = i + 1; j < 3; ++j)
                        {
                            const double angle = mol.GetAngle(neighbors[i], n, neighbors[j]);
                            if (!(100 < angle && angle < 140)) planar = false;
                            sum += angle;
                        }
                    if (!planar || sum < 355) continue;
                    // The fused six-ring is local: its Kekule bonds change with
                    // the indenyl redox assignment, but boundary bonds may not.
                    Site site(core.begin(), core.end());
                    site.push_back(n->GetIdx());
                    const int q = n->GetFormalCharge(), v = n->GetTotalValence();
                    const int order = n->GetBond(carbon)->GetBondOrder();
                    if (ring_charge == -1 && q == 0 && v == 3 && order == 1) out[site] = true;
                    if (ring_charge == 0 && q == 1 && v == 4 && order == 2) out[site] = false;
                }
            }
        }
        return out;
    }
}

namespace molgr::metal::geometry
{
    void ClearAminoC5Conflicts(std::vector<Candidate> &candidates)
    {
        Clear(candidates, AminoC5Key);
    }

    void AnnotateAminoC5Alternatives(std::vector<Candidate> &candidates)
    {
        AnnotateLocalRedoxAlternatives(candidates, AminoC5Key, AminoC5Sites);
    }
}

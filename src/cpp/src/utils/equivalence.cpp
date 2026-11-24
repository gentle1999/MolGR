#include "molgr/utils/equivalence.h"
#include "molgr/utils/equivalence_stereo.h"
#include "molgr/utils/consts.h"

#include <openbabel/elements.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <deque>
#include <functional>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace molgr::equivalence
{
    using namespace utils;
    namespace
    {
        constexpr int MaxResonanceSteps = 10000;
        using Clock = std::chrono::steady_clock;
        /// @param start Steady-clock timestamp at the beginning of the interval.
        /// @return Elapsed steady-clock time in milliseconds.
        double Milliseconds(Clock::time_point start)
        {
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
        /// @param z Atomic number to look up in the nonmetal table.
        /// @return True for an atomic number outside the reconstruction nonmetal table.
        bool Metal(int z) { return kNonMetalDict.count(z) == 0; }
        /// @param z Atomic number to look up in the nonmetal table.
        /// @return The tabulated outer electron count, or -1 for an element outside the nonmetal table.
        int Outer(int z)
        {
            auto p = kNonMetalDict.find(z);
            return p == kNonMetalDict.end() ? -1 : p->second.num_outer_electrons;
        }
        /// Indexed graph edge carrying bond order, aromaticity and coordination-bond direction.
        struct Edge
        {
            int u, v, order, kind; // kind: 0=covalent, 1=metal incident, 2=directed dative, 3=undirected
            bool aromatic;
        };
        /// Comparison graph with atom labels, edge indices and adjacency for a single reconstruction.
        struct Graph
        {
            const MoleculeData &source;
            int n;
            std::vector<int> charge, radical, valence, matrix, returned_pair_electrons;
            std::vector<Edge> edges;
            std::vector<std::vector<std::pair<int, int>>> adj;
            std::vector<bool> metal, stereo_fixed, unsupported;
            bool rearrangeable = false, paired_carbene_mode = false;
            /// @param u Zero-based graph vertex.
            /// @param v Zero-based graph vertex.
            /// @return Zero-based edge index, or -1 if the two vertices are not adjacent.
            int EdgeAt(int u, int v) const { return matrix[u * n + v]; }
            /// Initialize the graph representation from transferable molecule records.
            /// @param d Input atom, bond and stereo records.
            explicit Graph(const MoleculeData &d) : source(d), n(d.atoms.size()),
                charge(n), radical(n), valence(n, 0), matrix(n * n, -1), returned_pair_electrons(n, 0), adj(n),
                metal(n), stereo_fixed(n, false), unsupported(n, false) {}
        };

        /// @param data Transferable molecular graph records.
        /// @param heavy Whether to omit hydrogen from the composition signature.
        /// @param organic Whether to restrict the composition signature to nonmetal atoms.
        /// @return A deterministic element-count signature with the requested atom filters.
        std::string Formula(const MoleculeData &data, bool heavy, bool organic)
        {
            std::map<std::string, int> counts;
            for (const auto &a : data.atoms)
                if ((!heavy || a.atomic_num != 1) && (!organic || !Metal(a.atomic_num)))
                    ++counts[OpenBabel::OBElements::GetSymbol(a.atomic_num)];
            std::string result;
            /// Append one element count to the formula and consume it from the remaining composition.
            auto add = [&](const std::string &symbol)
            {
                auto p = counts.find(symbol);
                if (p == counts.end()) return;
                result += symbol;
                if (p->second != 1) result += std::to_string(p->second);
                counts.erase(p);
            };
            add("C"); add("H");
            while (!counts.empty()) add(counts.begin()->first);
            return result;
        }

        /// Validate molecular records before using their atom indices for equivalence matching.
        /// @param d Input atom, bond and stereo records.
        /// @exception std::invalid_argument Malformed atom, bond, stereo or electronic records, or an
        ///     input exceeding the native size limit.
        void Validate(const MoleculeData &d)
        {
            const int n = d.atoms.size();
            if (n > 4096) throw std::invalid_argument("Molecule exceeds the native graph size limit");
            int charge = 0, radicals = 0;
            for (const auto &a : d.atoms)
            {
                if (a.atomic_num <= 0 || a.atomic_num > 118 || a.isotope < 0 ||
                    a.formal_charge < -128 || a.formal_charge > 128 || a.radical_num < 0 || a.radical_num > 32 ||
                    a.lone_pair_count < 0 || a.lone_pair_count > 32 || a.electron_property_presence > 7)
                    throw std::invalid_argument("Invalid atom/electron fields");
                charge += a.formal_charge;
                radicals += a.radical_num;
            }
            if (charge != d.total_charge || radicals != d.total_radical_num)
                throw std::invalid_argument("Declared totals contradict atom fields");
            std::set<std::pair<int, int>> bonds;
            for (const auto &b : d.bonds)
            {
                if (b.kind != BondKind::Covalent && b.kind != BondKind::Dative && b.kind != BondKind::CoordinationUnspecified)
                    throw std::invalid_argument("Unsupported bond kind");
                if (b.begin_atom_idx < 1 || b.end_atom_idx < 1 || b.begin_atom_idx > n ||
                    b.end_atom_idx > n || b.begin_atom_idx == b.end_atom_idx)
                    throw std::invalid_argument("Invalid bond endpoints");
                if (!bonds.insert(std::minmax(b.begin_atom_idx, b.end_atom_idx)).second)
                    throw std::invalid_argument("Duplicate bond");
                if (b.kind == BondKind::Covalent && (b.order < 1 || b.order > 4))
                    throw std::invalid_argument("Native input requires explicit integer Kekule bond orders");
            }
            std::set<std::pair<int, int>> stereo_centers;
            for (const auto &s : d.stereo)
            {
                if (static_cast<int>(s.kind) > static_cast<int>(StereoKind::Octahedral))
                    throw std::invalid_argument("Unsupported stereo kind");
                int size = s.kind == StereoKind::TrigonalBipyramidal ? 5 : s.kind == StereoKind::Octahedral ? 6 : 4;
                int maximum = s.kind == StereoKind::SquarePlanar ? 3 : s.kind == StereoKind::TrigonalBipyramidal ? 20 : s.kind == StereoKind::Octahedral ? 30 : 2;
                if (s.center < 1 || s.center > n || s.end < 0 || s.end > n ||
                    s.refs.size() != static_cast<std::size_t>(size) || s.configuration < 0 || s.configuration > maximum)
                    throw std::invalid_argument("Invalid stereo record");
                if ((s.kind == StereoKind::DoubleBond) != (s.end != 0))
                    throw std::invalid_argument("Invalid stereo center/end");
                if (s.end && (s.center == s.end || !bonds.count(std::minmax(s.center, s.end))))
                    throw std::invalid_argument("Double-bond stereo center/end are not bonded");
                auto key = s.end ? std::minmax(s.center, s.end) : std::pair<const int &, const int &>(s.center, s.end);
                if (!stereo_centers.emplace(key.first, key.second).second)
                    throw std::invalid_argument("Duplicate stereo record");
                std::set<int> seen;
                for (int i = 0; i < size; ++i)
                {
                    // Two ends of an alkene can share a substituent (e.g. a
                    // coordinated metal). Uniqueness is per end, not per frame.
                    if (s.end && i == 2) seen.clear();
                    int ref = s.refs[i], owner = s.end && i >= 2 ? s.end : s.center;
                    if (ref < 0 || ref > n || (ref && (!seen.insert(ref).second ||
                        (s.end && (ref == s.center || ref == s.end)) ||
                        !bonds.count(std::minmax(owner, ref)))))
                        throw std::invalid_argument("Stereo references do not match the topology");
                }
            }
            std::set<int> grouped_atoms;
            for (const auto &g : d.stereo_groups)
            {
                if (static_cast<int>(g.kind) > static_cast<int>(StereoGroupKind::Or))
                    throw std::invalid_argument("Unsupported stereo group kind");
                for (int i : g.atoms)
                    if (i < 1 || i > n || !grouped_atoms.insert(i).second)
                        throw std::invalid_argument("Invalid or multiply grouped stereo atom");
            }
        }

        /// Mark potential metal-bound carbene sites in molecular atom order.
        /// @param d Input atom, bond and stereo records.
        /// @return Mask in input atom order marking potential metal-carbene sites.
        std::vector<bool> MetalCarbeneSites(const MoleculeData &d)
        {
            std::vector<bool> sites(d.atoms.size());
            std::vector<int> metal_bonds(d.atoms.size(), 0), valence(d.atoms.size(), 0);
            for (int i = 0; i < static_cast<int>(d.atoms.size()); ++i)
            {
                const auto &atom = d.atoms[i];
                sites[i] = atom.atomic_num == 6 && !atom.formal_charge && !atom.radical_num &&
                    !((atom.electron_property_presence & 4) && atom.unresolved_two_electron_center);
            }
            for (const auto &bond : d.bonds)
            {
                int u = bond.begin_atom_idx - 1, v = bond.end_atom_idx - 1;
                for (auto [carbon, other] : {std::pair<int, int>{u, v}, {v, u}})
                {
                    if (!sites[carbon]) continue;
                    if (bond.kind != BondKind::Covalent) { sites[carbon] = false; continue; }
                    if (Metal(d.atoms[other].atomic_num))
                    {
                        if (bond.order != 2) sites[carbon] = false;
                        ++metal_bonds[carbon];
                    }
                    else valence[carbon] += bond.order;
                }
            }
            for (int i = 0; i < static_cast<int>(sites.size()); ++i)
                sites[i] = sites[i] && metal_bonds[i] == 1 && valence[i] == 2;
            return sites;
        }

        /// @param d Input atom, bond and stereo records.
        /// @param sites Mask of potential carbene sites in molecular atom order.
        /// @return True when a marked carbene site carries the electron annotations needed for
        ///     normalization.
        bool HasKnownPairedMetalCarbene(const MoleculeData &d, const std::vector<bool> &sites)
        {
            for (int i = 0; i < static_cast<int>(d.atoms.size()); ++i)
                if ((d.atoms[i].electron_property_presence & 5) == 5 && sites[i]) return true;
            return false;
        }

        /// Construct the comparison graph and collect composition and electronic-state diagnostics.
        /// @param d Input atom, bond and stereo records.
        /// @param options Graph projection settings and bounds for equivalence search.
        /// @param summary Destination for the graph invariants used in equivalence comparison.
        /// @param paired_carbene_mode Whether to compare the explicit paired-electron metal-carbene
        ///     representation.
        /// @param carbene_sites Mask of atom positions eligible for metal-carbene normalization.
        /// @return Indexed graph prepared for the requested comparison.
        Graph Prepare(const MoleculeData &d, const Options &options, Summary &summary,
                      bool paired_carbene_mode, const std::vector<bool> &carbene_sites)
        {
            Graph g(d);
            g.paired_carbene_mode = paired_carbene_mode;
            summary.atom_count = g.n;
            summary.formula = Formula(d, false, false);
            summary.heavy_formula = Formula(d, true, true);
            for (int i = 0; i < g.n; ++i)
            {
                const auto &a = d.atoms[i];
                g.metal[i] = Metal(a.atomic_num);
                g.charge[i] = a.formal_charge;
                g.radical[i] = g.metal[i] ? 0 : a.radical_num;
                summary.charge += a.formal_charge;
                summary.radicals += g.radical[i];
                summary.hydrogen_count += a.atomic_num == 1;
                summary.organic_atoms += !g.metal[i];
                summary.known_fields += (g.metal[i] && (a.electron_property_presence & 1) ? 1 : 0) +
                    ((a.electron_property_presence & 2) ? 1 : 0) + ((a.electron_property_presence & 4) ? 1 : 0);
                summary.deferred_centers += (a.electron_property_presence & 4) && a.unresolved_two_electron_center;
            }
            for (const auto &b : d.bonds)
            {
                int u = b.begin_atom_idx - 1, v = b.end_atom_idx - 1;
                bool metal_edge = g.metal[u] || g.metal[v];
                if (g.metal[u] != g.metal[v] && b.kind == BondKind::Covalent)
                {
                    int m = g.metal[u] ? u : v, ligand = g.metal[u] ? v : u;
                    if (b.order == 2)
                    {
                        // A resolved, explicitly closed-shell metal carbene
                        // returns a pair to carbon when the metal bond is
                        // removed. Do not turn that pair into real radicals.
                        // Unknown legacy inputs retain the prior convention.
                        if (!paired_carbene_mode || !carbene_sites[ligand]) g.radical[ligand] += 2;
                        else g.returned_pair_electrons[ligand] = 2;
                    }
                    else
                    {
                        g.charge[m] += b.order;
                        g.charge[ligand] -= b.order;
                    }
                }
                if (metal_edge && options.ignore_metal_coordination_bonds) continue;
                int kind = metal_edge ? 1 : b.kind == BondKind::Covalent ? 0 : b.kind == BondKind::Dative ? 2 : 3;
                int index = g.edges.size();
                g.edges.push_back({u, v, kind ? 0 : b.order, kind, b.aromatic});
                g.matrix[u * g.n + v] = g.matrix[v * g.n + u] = index;
                g.adj[u].emplace_back(v, index);
                g.adj[v].emplace_back(u, index);
                if (!kind)
                {
                    g.valence[u] += b.order;
                    g.valence[v] += b.order;
                }
                else if (!metal_edge)
                    g.unsupported[u] = g.unsupported[v] = true;
            }
            // RDKit's unannotated neutral valence-two C uses two bookkeeping
            // radicals for an unspecified carbene. In a paired comparison it
            // may represent a pair, but an explicitly known unpaired count
            // or unresolved center must never be overwritten.
            if (paired_carbene_mode)
                for (int i = 0; i < g.n; ++i)
                {
                    const auto &atom = d.atoms[i];
                    if (atom.atomic_num == 6 && !g.charge[i] && g.valence[i] == 2 && g.radical[i] == 2 &&
                        !(atom.electron_property_presence & 1) &&
                        !((atom.electron_property_presence & 2) && !atom.lone_pair_count) &&
                        !((atom.electron_property_presence & 4) && atom.unresolved_two_electron_center)) g.radical[i] = 0;
                }
            // Preserve the established deterministic hypervalent normalization.
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int i = 0; i < g.n; ++i)
                {
                    int z = d.atoms[i].atomic_num;
                    if (z != 7 && z != 8 && z != 9 && z != 15 && z != 16 && z != 17 && z != 33 && z != 34 && z != 35 && z != 53) continue;
                    int limit = z == 9 || z == 17 || z == 35 || z == 53 ? 1 : 4;
                    if (g.valence[i] <= limit) continue;
                    int selected = -1, other = -1;
                    for (auto [j, e] : g.adj[i])
                    {
                        const auto &b = g.edges[e];
                        if (b.kind || b.aromatic || b.order <= 1) continue;
                        if (selected < 0 || b.order > g.edges[selected].order ||
                            (b.order == g.edges[selected].order && j < other))
                            selected = e, other = j;
                    }
                    if (selected < 0) continue;
                    --g.edges[selected].order;
                    --g.valence[i]; --g.valence[other];
                    ++g.charge[i]; --g.charge[other];
                    changed = true;
                    break;
                }
            }
            if (options.use_chirality)
                for (const auto &s : d.stereo)
                    if (s.kind == StereoKind::Tetrahedral && s.configuration &&
                        ActiveStereo(d, s, options.ignore_metal_coordination_bonds))
                        g.stereo_fixed[s.center - 1] = true;
            for (int i = 0; i < g.n; ++i)
            {
                if (g.metal[i]) summary.metal_states.emplace_back(d.atoms[i].atomic_num, g.charge[i]);
                else
                {
                    summary.organic_charge += g.charge[i];
                    summary.organic_radicals += g.radical[i];
                    if (!g.adj[i].empty() && (g.charge[i] || g.radical[i])) g.rearrangeable = true;
                }
            }
            for (auto b : g.edges) if (!b.kind && (b.order > 1 || b.aromatic)) g.rearrangeable = true;
            std::sort(summary.metal_states.begin(), summary.metal_states.end());
            return g;
        }

        /// @param g Indexed graph used for equivalence matching.
        /// @param edge Index of the graph edge to label.
        /// @param from Zero-based vertex from which the edge direction is viewed.
        /// @param exact Whether to distinguish exact covalent orders and aromaticity.
        /// @return An oriented edge label, optionally distinguishing exact covalent order and aromatic
        ///     bonds.
        int EdgeLabel(const Graph &g, int edge, int from, bool exact)
        {
            if (edge < 0) return 0;
            const auto &b = g.edges[edge];
            if (b.kind == 1) return 20;
            if (b.kind == 2) return b.u == from ? 21 : 22;
            if (b.kind == 3) return 23;
            return exact ? (b.aromatic ? 10 : b.order) : 1;
        }

        /// Compare the multiplicities of the two sets of atom refinement colors.
        /// @param left First vector of refined atom colors.
        /// @param right Second vector of refined atom colors.
        /// @return True if all color classes have matching populations.
        bool SameColors(const std::vector<int> &left, const std::vector<int> &right)
        {
            int count = 0;
            for (int c : left) count = std::max(count, c + 1);
            for (int c : right) count = std::max(count, c + 1);
            std::vector<int> balance(count, 0);
            for (int c : left) ++balance[c];
            for (int c : right) --balance[c];
            return std::all_of(balance.begin(), balance.end(), [](int n) { return n == 0; });
        }

        // One shared partition of the sigma skeleton. Electronic labels and
        // covalent orders must not split candidates that resonance can connect.
        /// Refine atom colors jointly so the resulting labels can be compared across both graphs.
        /// @param a First graph in the fixed comparison order.
        /// @param b Second graph in the fixed comparison order.
        /// @return Atom colors for the left and right graph, respectively.
        std::pair<std::vector<int>, std::vector<int>> Colors(const Graph &a, const Graph &b)
        {
            std::vector<int> left(a.n), right(b.n), next_a(a.n), next_b(b.n);
            std::map<std::vector<int>, int> table;
            /// Intern a refinement signature in the table shared by both graphs so color numbers are
            /// comparable.
            auto color = [&](const std::vector<int> &key) { auto [it, added] = table.emplace(key, table.size()); return it->second; };
            for (int side = 0; side < 2; ++side)
            {
                const auto &g = side ? b : a;
                auto &out = side ? right : left;
                for (int i = 0; i < g.n; ++i)
                    out[i] = color({g.source.atoms[i].atomic_num, g.source.atoms[i].isotope,
                        static_cast<int>(g.adj[i].size())});
            }
            std::vector<std::pair<int, int>> neighbors;
            std::vector<int> key;
            for (int round = 0; round < std::min(a.n, 12) && SameColors(left, right); ++round)
            {
                table.clear();
                for (int side = 0; side < 2; ++side)
                {
                    const auto &g = side ? b : a;
                    const auto &old = side ? right : left;
                    auto &out = side ? next_b : next_a;
                    for (int i = 0; i < g.n; ++i)
                    {
                        neighbors.clear();
                        for (auto [j, e] : g.adj[i]) neighbors.emplace_back(EdgeLabel(g, e, i, false), old[j]);
                        std::sort(neighbors.begin(), neighbors.end());
                        key.clear(); key.push_back(old[i]);
                        for (auto [label, c] : neighbors) { key.push_back(label); key.push_back(c); }
                        out[i] = color(key);
                    }
                }
                if (next_a == left && next_b == right) break;
                left.swap(next_a); right.swap(next_b);
            }
            return {left, right};
        }

        /// Test whether this atom pair and its electronic labels admit the metal-carbene normalization.
        /// @param g Indexed graph used for equivalence matching.
        /// @param u Zero-based graph vertex.
        /// @param v Zero-based graph vertex.
        /// @param q Formal charges indexed by vertex.
        /// @param r Unpaired-electron counts indexed by vertex.
        /// @param valence Current valence of each graph vertex.
        /// @return True when this metal-carbon assignment meets the normalization conditions.
        bool CarbeneSite(const Graph &g, int u, int v, const std::vector<int> &q,
                         const std::vector<int> &r, const std::vector<int> &valence)
        {
            auto element = kNonMetalDict.find(g.source.atoms[v].atomic_num);
            return g.source.atoms[u].atomic_num == 6 && element != kNonMetalDict.end() &&
                q[u] == -1 && q[v] == 1 && valence[u] == 3 &&
                valence[v] == element->second.default_valence + 1 && !r[u] && !r[v];
        }

        /// @param g Indexed graph used for equivalence matching.
        /// @return True when at least one edge permits metal-carbene normalization.
        bool CanNormalizeCarbene(const Graph &g)
        {
            for (const auto &e : g.edges)
                if (!e.kind && e.order == 2 &&
                    (CarbeneSite(g, e.u, e.v, g.charge, g.radical, g.valence) ||
                     CarbeneSite(g, e.v, e.u, g.charge, g.radical, g.valence))) return true;
            return false;
        }

        /// Connected components and color signatures used to restrict atom mapping.
        struct Components
        {
            std::vector<int> of, charge, radicals;
            std::vector<std::vector<int>> signatures;
            /// Partition the comparison graph into connected components using the supplied color labels.
            /// @param g Indexed graph used for equivalence matching.
            /// @param colors Jointly refined atom labels used to restrict possible mappings.
            Components(const Graph &g, const std::vector<int> &colors) : of(g.n, -1)
            {
                for (int seed = 0; seed < g.n; ++seed)
                {
                    if (of[seed] >= 0) continue;
                    int id = signatures.size();
                    std::vector<int> queue{seed}, signature;
                    of[seed] = id;
                    int q = 0, r = 0;
                    for (std::size_t p = 0; p < queue.size(); ++p)
                    {
                        int i = queue[p]; q += g.charge[i]; r += g.radical[i]; signature.push_back(colors[i]);
                        for (auto [j, e] : g.adj[i])
                            if (!g.edges[e].kind && of[j] < 0) { of[j] = id; queue.push_back(j); }
                    }
                    std::sort(signature.begin(), signature.end());
                    signatures.push_back(std::move(signature)); charge.push_back(q); radicals.push_back(r);
                }
            }
        };

        /// Compare available electron annotations and track whether the two sides supply comparable data.
        /// @param a First graph in the fixed comparison order.
        /// @param b Second graph in the fixed comparison order.
        /// @param i Source vertex index, zero-based.
        /// @param j Target vertex index, zero-based.
        /// @param comparable Output flag tracking whether both sides supply comparable electron
        ///     annotations.
        /// @return True if the comparable electron annotations agree.
        bool AtomAnnotations(const Graph &a, const Graph &b, int i, int j, int &comparable)
        {
            const auto &x = a.source.atoms[i], &y = b.source.atoms[j];
            int known = x.electron_property_presence & y.electron_property_presence;
            bool equal = true;
            if (a.metal[i] && (known & 1))
            {
                ++comparable;
                equal &= x.radical_num == y.radical_num;
            }
            bool deferred = ((x.electron_property_presence & 4) && x.unresolved_two_electron_center) ||
                            ((y.electron_property_presence & 4) && y.unresolved_two_electron_center);
            if ((known & 2) && !deferred)
            {
                ++comparable;
                if (x.lone_pair_count != y.lone_pair_count)
                    equal &= 2 * x.lone_pair_count + a.charge[i] + a.valence[i] + a.radical[i] + a.returned_pair_electrons[i] ==
                             2 * y.lone_pair_count + b.charge[j] + b.valence[j] + b.radical[j] + b.returned_pair_electrons[j];
            }
            return equal;
        }

        /// Compare available electron annotations and track whether the two sides supply comparable data.
        /// @param a First graph in the fixed comparison order.
        /// @param b Second graph in the fixed comparison order.
        /// @param mapping Zero-based image of each source atom in the target graph.
        /// @param comparable Output flag tracking whether both sides supply comparable electron
        ///     annotations.
        /// @return True if the comparable electron annotations agree.
        bool Annotations(const Graph &a, const Graph &b, const std::vector<int> &mapping, int &comparable)
        {
            comparable = 0;
            bool equal = true;
            for (int i = 0; i < a.n; ++i) equal &= AtomAnnotations(a, b, i, mapping[i], comparable);
            return equal;
        }

        /// @param a First graph in the fixed comparison order.
        /// @param b Second graph in the fixed comparison order.
        /// @param mapping Zero-based image of each source atom in the target graph.
        /// @return True when the mapping preserves charges, radicals and exact oriented bond labels.
        bool ExactMapping(const Graph &a, const Graph &b, const std::vector<int> &mapping)
        {
            for (int i = 0; i < a.n; ++i)
                if (a.charge[i] != b.charge[mapping[i]] || a.radical[i] != b.radical[mapping[i]]) return false;
            for (int e = 0; e < static_cast<int>(a.edges.size()); ++e)
            {
                const auto &edge = a.edges[e];
                if (EdgeLabel(a, e, edge.u, true) !=
                    EdgeLabel(b, b.EdgeAt(mapping[edge.u], mapping[edge.v]), mapping[edge.u], true)) return false;
            }
            return true;
        }

        /// Keep one proven identity mapping for each independent family of repeated components.
        /// Other charge/topology families cannot use these atoms. Fixing the entire family
        /// preserves a full bijection even when some electronic annotations are unknown.
        struct SpectatorMappings
        {
            std::vector<std::vector<int>> families;
            std::vector<int> fixed;
            /// Relative stereo can couple configurations across otherwise disconnected fragments.
            enum class StereoRole { None, Local, Relative };
            std::vector<StereoRole> left_stereo, right_stereo;

            /// @param g Prepared graph whose projected stereo records will be compared.
            /// @param options Stereo and metal-coordination projection settings.
            /// @return Centers with local stereo or enhanced stereo that may couple disconnected fragments.
            static std::vector<StereoRole> StereoRoles(const Graph &g, const Options &options)
            {
                std::vector<StereoRole> roles(g.n, StereoRole::None);
                if (!options.use_chirality) return roles;
                for (const auto &s : g.source.stereo)
                    if (ActiveStereo(g.source, s, options.ignore_metal_coordination_bonds))
                        roles[s.center - 1] = StereoRole::Local;
                for (const auto &group : g.source.stereo_groups)
                    if (group.kind != StereoGroupKind::Absolute)
                        for (int atom : group.atoms)
                            if (roles[atom - 1] != StereoRole::None) roles[atom - 1] = StereoRole::Relative;
                return roles;
            }

            /// @param a Source graph after applying the comparison's coordination policy.
            /// @param b Target graph after the same projection.
            /// @param components Covalent-component membership in the source graph.
            /// @param signatures Color multiset and conserved charge used to restrict component pairing.
            /// @param options Stereo and metal-coordination projection settings.
            SpectatorMappings(const Graph &a, const Graph &b, const Components &components,
                              const std::vector<std::vector<int>> &signatures, const Options &options)
                : fixed(a.n, -1)
            {
                if (signatures.size() < 2) return;
                std::map<std::vector<int>, std::vector<int>> classes;
                for (int c = 0; c < static_cast<int>(signatures.size()); ++c)
                    classes[signatures[c]].push_back(c);
                std::vector<int> family_of(signatures.size(), -1);
                for (const auto &[signature, members] : classes)
                    if (members.size() > 1)
                    {
                        for (int c : members) family_of[c] = families.size();
                        families.emplace_back();
                    }
                if (families.empty()) return;
                std::vector<bool> linked(families.size(), false);
                for (int i = 0; i < a.n; ++i)
                {
                    int family = family_of[components.of[i]];
                    if (family < 0) continue;
                    families[family].push_back(i);
                    for (auto [j, e] : a.adj[i])
                        if (family_of[components.of[j]] != family) linked[family] = true;
                }
                for (int i = static_cast<int>(families.size()) - 1; i >= 0; --i)
                    if (linked[i]) families.erase(families.begin() + i);
                if (families.empty()) return;
                left_stereo = StereoRoles(a, options);
                right_stereo = StereoRoles(b, options);
            }

            /// @param a Source electronic graph.
            /// @param b Target electronic graph.
            /// @param mapping Complete topology bijection respecting component signatures and charges.
            /// @param stereo_matches Whether this bijection satisfies the projected stereo constraints.
            void Remember(const Graph &a, const Graph &b, const std::vector<int> &mapping, bool stereo_matches)
            {
                for (auto family = families.begin(); family != families.end();)
                {
                    // A signature identifies possible partners, not chemical identity. Verify every
                    // atom and edge of this whole family before suppressing any alternatives.
                    bool exact = std::all_of(family->begin(), family->end(), [&](int i)
                    {
                        const int j = mapping[i];
                        const auto role = std::max(left_stereo[i], right_stereo[j]);
                        int comparable = 0;
                        if (role == StereoRole::Relative || (role == StereoRole::Local && !stereo_matches) ||
                            a.charge[i] != b.charge[j] || a.radical[i] != b.radical[j] ||
                            !AtomAnnotations(a, b, i, j, comparable)) return false;
                        for (auto [neighbor, e] : a.adj[i])
                        {
                            const auto &x = a.edges[e];
                            const auto &y = b.edges[b.EdgeAt(j, mapping[neighbor])];
                            if (x.order != y.order || x.aromatic != y.aromatic) return false;
                        }
                        return true;
                    });
                    if (!exact) { ++family; continue; }
                    for (int i : *family) fixed[i] = mapping[i];
                    family = families.erase(family);
                }
            }
        };

        /// Enumerate compatible vertex mappings under the mapping budget until a witness is accepted.
        /// @param a First graph in the fixed comparison order.
        /// @param b Second graph in the fixed comparison order.
        /// @param candidates Candidate target vertices for each source vertex.
        /// @param options Graph projection settings and bounds for equivalence search.
        /// @param result Result record carrying the operation's output.
        /// @param stereo_found Tracks whether a stereochemically compatible mapping has been found.
        /// @param fixed Source atoms whose target partners are already proved in independent spectator families.
        /// @param accept Callback deciding whether the current atom mapping supplies a witness.
        /// @return True if the mapping search finds an accepted witness.
        bool MapGraphs(const Graph &a, const Graph &b, std::vector<std::vector<int>> candidates,
                       const Options &options, Result &result, const bool &stereo_found, const std::vector<int> &fixed,
                       const std::function<bool(const std::vector<int> &)> &accept)
        {
            /// Prefer localized identity, but never remove a candidate just for a
            /// charge/radical/bond-order difference. The same search proves both
            /// identity and resonance, without fixing an arbitrary first mapping.
            auto local_labels = [](const Graph &g)
            {
                std::vector<std::vector<int>> labels(g.n);
                for (int i = 0; i < g.n; ++i)
                {
                    for (auto [j, e] : g.adj[i]) labels[i].push_back(EdgeLabel(g, e, i, true));
                    std::sort(labels[i].begin(), labels[i].end());
                }
                return labels;
            };
            std::map<std::tuple<int, int, int, int>, int> inventory;
            for (int i = 0; i < a.n; ++i)
            {
                ++inventory[{a.source.atoms[i].atomic_num, a.source.atoms[i].isotope, a.charge[i], a.radical[i]}];
                --inventory[{b.source.atoms[i].atomic_num, b.source.atoms[i].isotope, b.charge[i], b.radical[i]}];
            }
            bool prefer_exact = std::all_of(inventory.begin(), inventory.end(), [](const auto &item) { return item.second == 0; });
            // A localized-identity heuristic is misleading when the inventories
            // already exclude identity: resonance may move these very labels.
            auto la = prefer_exact ? local_labels(a) : std::vector<std::vector<int>>{};
            auto lb = prefer_exact ? local_labels(b) : std::vector<std::vector<int>>{};
            std::vector<unsigned char> allowed(a.n * b.n, 0), annotation_ok(a.n * b.n, 0);
            std::vector<int> available(a.n), mapping(a.n, -1);
            for (int i = 0; i < a.n; ++i)
            {
                /// Prefer target vertices with compatible electron annotations, charges, radicals and local
                /// bond labels.
                auto score = [&](int j)
                {
                    int count = 0;
                    return 8 * !AtomAnnotations(a, b, i, j, count) +
                        (prefer_exact ? 4 * (a.charge[i] != b.charge[j]) +
                            2 * (a.radical[i] != b.radical[j]) + (la[i] != lb[j]) : 0);
                };
                std::stable_sort(candidates[i].begin(), candidates[i].end(), [&](int x, int y) { return score(x) < score(y); });
                available[i] = candidates[i].size();
                for (int j : candidates[i])
                {
                    allowed[i * b.n + j] = 1;
                    int count = 0;
                    annotation_ok[i * b.n + j] = AtomAnnotations(a, b, i, j, count);
                }
            }
            // Equivalent terminal H permutations cannot change the proof. Stereo
            // references are excluded from this pruning and still get full search.
            std::vector<bool> stereo_ref(b.n, false);
            if (options.use_chirality)
                for (const auto &s : b.source.stereo)
                    if (ActiveStereo(b.source, s, options.ignore_metal_coordination_bonds))
                        for (int ref : s.refs) if (ref) stereo_ref[ref - 1] = true;
            /// Group interchangeable target vertices before branching on symmetry-equivalent mappings.
            auto twin_key = [&](int j)
            {
                const auto &atom = b.source.atoms[j];
                int parent = b.adj[j].empty() ? -1 : b.adj[j][0].first;
                return std::vector<int>{parent, atom.atomic_num, atom.isotope, b.charge[j], b.radical[j],
                    atom.electron_property_presence, atom.radical_num, atom.lone_pair_count, atom.unresolved_two_electron_center};
            };
            std::vector<int> removed;
            /// Extend the partial atom bijection until a witness is accepted or a mapping limit is reached.
            std::function<bool(int)> visit = [&](int depth)
            {
                if (result.mapping_nodes >= options.max_mapping_nodes || result.mappings >= options.max_mappings)
                {
                    result.mapping_limit = true;
                    return false;
                }
                ++result.mapping_nodes;
                if (depth == a.n)
                {
                    ++result.mappings;
                    return accept(mapping);
                }
                int selected = -1, best = INT32_MAX, adjacent_best = -1;
                for (int i = 0; i < a.n; ++i)
                {
                    if (mapping[i] >= 0) continue;
                    int adjacent = 0;
                    for (auto [neighbor, e] : a.adj[i]) if (mapping[neighbor] >= 0) ++adjacent;
                    if (available[i] < best || (available[i] == best && adjacent > adjacent_best))
                        selected = i, best = available[i], adjacent_best = adjacent;
                }
                if (!best) return false;
                std::set<std::vector<int>> twins;
                for (int j : candidates[selected])
                {
                    if ((fixed[selected] >= 0 && fixed[selected] != j) || !allowed[selected * b.n + j]) continue;
                    // First obtain a topology/stereo witness for diagnostics.
                    // Afterwards known annotation conflicts can reject partial
                    // mappings; unknown annotations remain compatible.
                    if (stereo_found && !annotation_ok[selected * b.n + j]) continue;
                    bool twin = !stereo_ref[j] && (b.adj[j].empty() ||
                        (b.source.atoms[j].atomic_num == 1 && b.adj[j].size() == 1));
                    if (twin && !twins.insert(twin_key(j)).second) continue;
                    mapping[selected] = j;
                    std::size_t mark = removed.size();
                    bool feasible = true;
                    for (int i = 0; i < a.n && feasible; ++i)
                    {
                        if (mapping[i] >= 0) continue;
                        int label = EdgeLabel(a, a.EdgeAt(selected, i), selected, false);
                        for (int k : candidates[i])
                        {
                            int index = i * b.n + k;
                            if (allowed[index] && (k == j ||
                                label != EdgeLabel(b, b.EdgeAt(j, k), j, false) ||
                                (stereo_found && !annotation_ok[index])))
                            { allowed[index] = 0; --available[i]; removed.push_back(index); }
                        }
                        if (!available[i]) feasible = false;
                    }
                    if (feasible && visit(depth + 1)) return true;
                    while (removed.size() > mark)
                    {
                        int index = removed.back(); removed.pop_back();
                        allowed[index] = 1; ++available[index / b.n];
                    }
                    mapping[selected] = -1;
                    if (result.mapping_limit) return false;
                }
                return false;
            };
            return visit(0);
        }

        using State = std::vector<std::int16_t>;
        /// Hash a vector of charge, radical and bond-order labels for visited-state lookup.
        struct StateHash
        {
            /// Hash a vector of charge, radical and bond-order labels for visited-state lookup.
            /// @param state Charge, radical and bond-order labels of the regional search state.
            /// @return Hash of the state labels for use in the visited table or proof cache.
            std::size_t operator()(const State &state) const
            {
                std::size_t h = 1469598103934665603ULL;
                for (auto x : state) { h ^= static_cast<std::uint16_t>(x); h *= 1099511628211ULL; }
                return h;
            }
        };
        /// A connected subproblem with local atom labels and bond orders for a resonance proof.
        struct Region
        {
            std::vector<int> atoms, edges, fixed_valence;
            std::vector<std::pair<int, int>> endpoints;
            std::vector<std::vector<std::pair<int, int>>> adj;
            State left, right;
            std::size_t witness_start = 0, witness_end = 0;
            bool proved = false;
        };

        /// Normalize eligible carbene electronic labels in place and record the moves in the witness.
        /// @param g Indexed graph used for equivalence matching.
        /// @param q Formal charges indexed by vertex.
        /// @param r Unpaired-electron counts indexed by vertex.
        /// @param orders Bond orders indexed by edge.
        /// @param witness Electronic moves forming the equivalence proof.
        /// @param side Graph side on which to record the normalization move.
        /// @return True if at least one carbene normalization was applied.
        bool CarbeneNormalize(const Graph &g, std::vector<int> &q, std::vector<int> &r, std::vector<int> &orders,
                              std::vector<Move> &witness, int side)
        {
            std::vector<int> valence(g.n, 0);
            for (int e = 0; e < static_cast<int>(g.edges.size()); ++e)
                if (!g.edges[e].kind) { valence[g.edges[e].u] += orders[e]; valence[g.edges[e].v] += orders[e]; }
            bool changed = false;
            for (int e = 0; e < static_cast<int>(g.edges.size()); ++e)
            {
                auto b = g.edges[e];
                if (b.kind || orders[e] != 2) continue;
                for (auto [u, v] : {std::pair<int, int>{b.u, b.v}, {b.v, b.u}})
                {
                    if (!CarbeneSite(g, u, v, q, r, valence)) continue;
                    --orders[e]; --valence[u]; --valence[v];
                    ++q[u]; --q[v]; r[u] = g.paired_carbene_mode ? 0 : 2; r[v] = 0;
                    witness.push_back({g.paired_carbene_mode ? "paired_carbene_normalization" : "carbene_normalization", u, v, -1, side});
                    changed = true;
                    break;
                }
            }
            return changed;
        }

        /// Bound the number of local electron shifts needed to reach a target label vector.
        /// @param state Charge, radical and bond-order labels of the regional search state.
        /// @param target Desired regional electron and bond-order labels.
        /// @param n Number of local vertices represented in the electronic state.
        /// @return Minimum number of moves implied by the differing charge, radical and bond labels.
        int LowerBound(const State &state, const State &target, int n)
        {
            int q = 0, r = 0, bonds = 0;
            for (int i = 0; i < n; ++i) { q += std::abs(state[i] - target[i]); r += std::abs(state[n + i] - target[n + i]); }
            for (int i = 2 * n; i < static_cast<int>(state.size()); ++i) bonds += std::abs(state[i] - target[i]);
            return std::max({(q + 1) / 2, (r + 1) / 2, (bonds + 1) / 2});
        }

        /// @param region Connected atoms and eligible bonds in one ligand resonance problem.
        /// @return Shortest bond-path lengths for estimating how far charge or radical labels must move.
        std::vector<std::vector<int>> BondDistances(const Region &region)
        {
            const int n = region.atoms.size();
            std::vector<std::vector<int>> distances(n, std::vector<int>(n, n));
            for (int source = 0; source < n; ++source)
            {
                auto &row = distances[source];
                row[source] = 0;
                std::vector<int> queue{source};
                for (std::size_t i = 0; i < queue.size(); ++i)
                    for (auto [next, edge] : region.adj[queue[i]])
                        if (row[next] == n)
                        {
                            row[next] = row[queue[i]] + 1;
                            queue.push_back(next);
                        }
            }
            return distances;
        }

        /// @param state Current formal-charge, radical and bond-order labels.
        /// @param target Corresponding labels at the opposite end of the search.
        /// @param distances Shortest bond paths inside the region.
        /// @return Lower bound on the electron moves needed to transport charge and radicals.
        int TransportBound(const State &state, const State &target,
                           const std::vector<std::vector<int>> &distances)
        {
            const int n = distances.size();
            std::int64_t moves = 0;
            for (int offset : {0, n})
            {
                std::int64_t excess = 0, deficit = 0;
                for (int i = 0; i < n; ++i)
                {
                    int delta = state[offset + i] - target[offset + i];
                    if (!delta) continue;
                    int nearest = n;
                    for (int j = 0; j < n; ++j)
                        if (delta > 0 ? state[offset + j] < target[offset + j]
                                      : state[offset + j] > target[offset + j])
                            nearest = std::min(nearest, distances[i][j]);
                    if (delta > 0) excess += delta * nearest;
                    else deficit -= delta * nearest;
                }
                // A move transports one unit of either charge or radical by at
                // most two bonds. Both source and destination distances are lower bounds.
                moves += (std::max(excess, deficit) + 1) / 2;
            }
            return static_cast<int>(std::min<std::int64_t>(moves, MaxResonanceSteps + 1));
        }

        /// Limits which prevented expansion in this round of ligand resonance search.
        struct ResonanceCutoffs
        {
            int next_depth = INT32_MAX;
            bool state_limit = false;
        };

        /// Search the region's electronic assignments for a path within the configured limits.
        /// @param g Indexed graph used for equivalence matching.
        /// @param region Connected electronic subproblem under the fixed atom mapping.
        /// @param options Graph projection settings and bounds for equivalence search.
        /// @param result Result record carrying the operation's output.
        /// @param cutoffs Smallest excluded path bound and whether the state slice was exhausted.
        /// @param directed Whether to prioritize regional states using their estimated distance to the
        ///     target.
        /// @return True if a path between the electronic assignments was found.
        bool SearchRegion(const Graph &g, const Region &region, const Options &options, Result &result,
                          ResonanceCutoffs &cutoffs, bool directed)
        {
            const int n = region.atoms.size();
            if (region.left == region.right) return true;
            ++result.regions;
            result.resonance_attempted = true;
            // Equal charges/radicals and balanced single/double-bond changes
            // decompose into alternating cycles. Construct that proof directly:
            // different Kekule assignments need no enumeration or aromatic flags.
            if (std::equal(region.left.begin(), region.left.begin() + 2 * n, region.right.begin()))
            {
                std::vector<int> delta(region.edges.size()), balance(n, 0);
                bool supported = true;
                for (int e = 0; e < static_cast<int>(delta.size()); ++e)
                {
                    int x = region.left[2 * n + e], y = region.right[2 * n + e];
                    delta[e] = y - x;
                    if (x != y && !((x == 1 && y == 2) || (x == 2 && y == 1))) supported = false;
                    auto [u, v] = region.endpoints[e];
                    balance[u] += delta[e]; balance[v] += delta[e];
                }
                if (supported && std::all_of(balance.begin(), balance.end(), [](int x) { return x == 0; }))
                {
                    for (int seed = 0; seed < static_cast<int>(delta.size()); ++seed)
                    {
                        if (!delta[seed]) continue;
                        int start = region.endpoints[seed].first, current = start, wanted = delta[seed];
                        Move move{"kekule_cycle", -1, -1, wanted, 0, {region.atoms[start]}};
                        do
                        {
                            int selected = -1, next = -1;
                            for (auto [j, e] : region.adj[current])
                                if (delta[e] == wanted) { selected = e; next = j; break; }
                            if (selected < 0) throw std::logic_error("Unbalanced Kekule trail");
                            delta[selected] = 0; current = next; wanted = -wanted;
                            move.cycle.push_back(region.atoms[current]);
                        } while (current != start || wanted != move.c);
                        result.witness.push_back(std::move(move));
                    }
                    return true;
                }
            }
            const auto distances = directed ? BondDistances(region) : std::vector<std::vector<int>>{};
            const int lower_bound = std::max(LowerBound(region.left, region.right, n),
                                             TransportBound(region.left, region.right, distances));
            if (lower_bound > options.max_resonance_steps)
            {
                cutoffs.next_depth = std::min(cutoffs.next_depth, lower_bound);
                result.resonance_limit = true;
                return false;
            }
            if (result.states_left + result.states_right + 2 > 2 * options.max_resonance)
            { cutoffs.state_limit = true; result.resonance_limit = true; return false; }
            for (int offset : {0, n})
                if (std::accumulate(region.left.begin() + offset, region.left.begin() + offset + n, 0) !=
                    std::accumulate(region.right.begin() + offset, region.right.begin() + offset + n, 0)) return false;

            /// Search state with the parent information needed to reconstruct a witness path.
            struct Node { State state; int parent, depth; Move move; };
            std::vector<Node> nodes[2];
            std::unordered_map<State, int, StateHash> seen[2];
            std::vector<int> frontier[2]{{0}, {0}};
            using Entry = std::tuple<int, int, int>;
            std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> ordered[2];
            if (directed) { ordered[0].emplace(0, 0, 0); ordered[1].emplace(0, 0, 0); }
            nodes[0].push_back({region.left, -1, 0, {}});
            nodes[1].push_back({region.right, -1, 0, {}});
            seen[0].emplace(region.left, 0); seen[1].emplace(region.right, 0);
            ++result.states_left; ++result.states_right;
            int charge_bound = 0;
            for (int i = 0; i < n; ++i) charge_bound += std::max(std::abs(region.left[i]), std::abs(region.right[i]));
            charge_bound += 2;
            /// Recompute regional valences and reject proposed charge, radical or bond labels outside their
            /// limits.
            auto valid = [&](const State &state, std::vector<int> &valence)
            {
                valence = region.fixed_valence;
                for (int e = 0; e < static_cast<int>(region.edges.size()); ++e)
                {
                    int order = state[2 * n + e];
                    if (order < 1 || order > 3) return false;
                    auto [u, v] = region.endpoints[e]; valence[u] += order; valence[v] += order;
                }
                int absolute_charge = 0;
                for (int i = 0; i < n; ++i)
                {
                    int z = g.source.atoms[region.atoms[i]].atomic_num;
                    int q = state[i], r = state[n + i], outer = Outer(z);
                    int nonbonding = outer - q - valence[i];
                    int shell = outer - q + valence[i];
                    int maximum = z <= 10 || z == 14 ? 8 : 12;
                    if (outer < 0 || r < 0 || r > 4 || nonbonding < r || (nonbonding - r) % 2 || shell > maximum ||
                        std::abs(q) > 4 || valence[i] > 6) return false;
                    absolute_charge += std::abs(q);
                }
                return absolute_charge <= charge_bound;
            };
            std::vector<int> check;
            if (!valid(region.left, check) || !valid(region.right, check))
            { result.validation_failed = true; return false; }
            /// Follow parent links from a meeting state on each side to append the electronic moves to the
            /// proof.
            auto witness = [&](int side, int here, int there)
            {
                for (auto [s, id] : {std::pair<int, int>{side, here}, {1 - side, there}})
                {
                    std::vector<Move> path;
                    while (nodes[s][id].parent >= 0)
                    {
                        auto move = nodes[s][id].move; move.side = s; path.push_back(move);
                        id = nodes[s][id].parent;
                    }
                    std::reverse(path.begin(), path.end());
                    result.witness.insert(result.witness.end(), path.begin(), path.end());
                }
            };
            /// Count pending states for this side using its active search queue.
            auto frontier_size = [&](int side) { return directed ? ordered[side].size() : frontier[side].size(); };
            while (frontier_size(0) || frontier_size(1))
            {
                int side = !frontier_size(1) ? 0 : !frontier_size(0) ? 1 :
                    frontier_size(0) <= frontier_size(1) ? 0 : 1;
                const State &target = side ? region.left : region.right;
                std::vector<int> current, next;
                if (directed)
                {
                    current.push_back(std::get<2>(ordered[side].top()));
                    ordered[side].pop();
                }
                else current.swap(frontier[side]);
                for (int parent : current)
                {
                    // Node vectors may grow below: keep one small state copy,
                    // never a reference invalidated by vector reallocation.
                    State state = nodes[side][parent].state;
                    int depth = nodes[side][parent].depth;
                    if (seen[side].at(state) != parent) continue;
                    if (depth >= options.max_resonance_steps)
                    {
                        cutoffs.next_depth = std::min(cutoffs.next_depth, depth + 1);
                        result.resonance_limit = true;
                        continue;
                    }
                    std::vector<int> valence;
                    valid(state, valence);
                    /// A prospective local state ranked by its estimated remaining distance to the target.
                    struct Candidate { State state; Move move; int distance, transport; };
                    std::vector<Candidate> candidates;
                    /// Retain a valid electronic move only when its target-distance bound fits the
                    /// remaining step budget.
                    auto emit = [&](State candidate, const char *kind, int a, int b, int c)
                    {
                        auto previous = seen[side].find(candidate);
                        if (previous != seen[side].end() && nodes[side][previous->second].depth <= depth + 1) return;
                        std::vector<int> unused;
                        if (!valid(candidate, unused)) return;
                        const int transport = TransportBound(candidate, target, distances);
                        const int bound = std::max(LowerBound(candidate, target, n), transport);
                        if (depth + 1 + bound > options.max_resonance_steps)
                        {
                            cutoffs.next_depth = std::min(cutoffs.next_depth, depth + 1 + bound);
                            ++result.pruned_moves; result.resonance_limit = true;
                            return;
                        }
                        candidates.push_back({std::move(candidate), {kind, region.atoms[a], region.atoms[b], c < 0 ? -1 : region.atoms[c], side}, bound, transport});
                    };
                    for (int a = 0; a < n; ++a)
                    {
                        int nonbonding = Outer(g.source.atoms[region.atoms[a]].atomic_num) - state[a] - valence[a];
                        bool pair = nonbonding - state[n + a] >= 2;
                        bool vacancy = Outer(g.source.atoms[region.atoms[a]].atomic_num) - state[a] + valence[a] <= 6;
                        for (auto [b, ab] : region.adj[a])
                        {
                            int order_ab = state[2 * n + ab];
                            // Polarization and its inverse keep the sigma skeleton.
                            if (order_ab >= 2)
                            {
                                State candidate = state;
                                --candidate[2 * n + ab]; ++candidate[a]; --candidate[b];
                                emit(std::move(candidate), "polarization", a, b, -1);
                            }
                            int shell_b = Outer(g.source.atoms[region.atoms[b]].atomic_num) - state[b] + valence[b];
                            if (pair && order_ab < 3 && shell_b <= 6)
                            {
                                State candidate = state;
                                ++candidate[2 * n + ab]; ++candidate[a]; --candidate[b];
                                emit(std::move(candidate), "pair_donation", a, b, -1);
                            }
                            if (order_ab >= 3) continue;
                            for (auto [c, bc] : region.adj[b])
                            {
                                if (c == a || state[2 * n + bc] < 2) continue;
                                for (int rule = 0; rule < 3; ++rule)
                                {
                                    if ((rule == 0 && !pair) || (rule == 1 && !vacancy) || (rule == 2 && !state[n + a])) continue;
                                    State candidate = state;
                                    ++candidate[2 * n + ab]; --candidate[2 * n + bc];
                                    if (rule == 0) { ++candidate[a]; --candidate[c]; }
                                    if (rule == 1) { --candidate[a]; ++candidate[c]; }
                                    if (rule == 2) { --candidate[n + a]; ++candidate[n + c]; }
                                    emit(std::move(candidate), rule == 0 ? "electron_pair_shift" : rule == 1 ? "cation_shift" : "radical_shift", a, b, c);
                                }
                            }
                        }
                    }
                    std::stable_sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) { return a.distance < b.distance; });
                    for (auto &candidate : candidates)
                    {
                        auto previous = seen[side].find(candidate.state);
                        if (previous != seen[side].end() && nodes[side][previous->second].depth <= depth + 1) continue;
                        if (result.states_left + result.states_right >= 2 * options.max_resonance)
                        { cutoffs.state_limit = true; result.resonance_limit = true; return false; }
                        int index = nodes[side].size();
                        // Directed expansion may reach a state by a longer path
                        // first. Reopen it at a shorter depth without changing
                        // the parent chain of an already recorded witness.
                        seen[side].insert_or_assign(candidate.state, index);
                        auto meeting = seen[1 - side].find(candidate.state);
                        nodes[side].push_back({std::move(candidate.state), parent, depth + 1, candidate.move});
                        if (side) ++result.states_right; else ++result.states_left;
                        if (meeting != seen[1 - side].end() && depth + 1 + nodes[1 - side][meeting->second].depth <= options.max_resonance_steps)
                        { witness(side, index, meeting->second); return true; }
                        if (directed)
                        {
                            int distance = 0;
                            for (int k = 0; k < static_cast<int>(target.size()); ++k)
                                distance += std::abs(nodes[side][index].state[k] - target[k]);
                            // This is an expansion preference, not a proof or a
                            // pruning bound. All chemical/step checks above apply.
                            ordered[side].emplace(distance + 2 * candidate.transport, depth + 1, index);
                        }
                        else next.push_back(index);
                    }
                }
                if (!directed) frontier[side].swap(next);
            }
            return false;
        }

        /// Hash the encoded resonance subproblem for memoization.
        struct ProblemHash
        {
            /// Hash the encoded resonance subproblem for memoization.
            /// @param key Lookup key for the recorded state or property.
            /// @return Hash of the state labels for use in the visited table or proof cache.
            std::size_t operator()(const std::vector<int> &key) const
            {
                std::size_t h = 1469598103934665603ULL;
                for (int x : key) { h ^= static_cast<std::uint32_t>(x); h *= 1099511628211ULL; }
                return h;
            }
        };

        /// Scoped to one deepening round: the left graph, stereo constraints, allowed
        /// moves and step bound are fixed. Hash collisions still compare full keys.
        struct ResonanceContext
        {
            ResonanceCutoffs cutoffs;
            bool allow_normalization = true, directed = false;
            bool initialized = false, normalized = false, target_can_normalize = false;
            std::vector<int> charge, radical, orders, valence;
            std::vector<Move> normalization;
            /// Cached match outcome and local electron-move witness.
            struct Proof { bool matched; std::vector<Move> moves; };
            std::unordered_set<std::vector<int>, ProblemHash> failed_targets;
            std::unordered_map<std::vector<int>, Proof, ProblemHash> regions;
            std::size_t payload_bytes = 0;

            /// Initialize the reusable search context for the source and target graphs.
            /// @param g Indexed graph used for equivalence matching.
            /// @param target Target graph used in preparing the resonance context.
            void Initialize(const Graph &g, const Graph &target)
            {
                if (initialized) return;
                charge = g.charge; radical = g.radical;
                orders.reserve(g.edges.size());
                for (const auto &edge : g.edges) orders.push_back(edge.order);
                target_can_normalize = allow_normalization && CanNormalizeCarbene(target);
                if (allow_normalization && CanNormalizeCarbene(g))
                    normalized = CarbeneNormalize(g, charge, radical, orders, normalization, 0);
                valence = g.valence;
                for (const auto &move : normalization) { --valence[move.a]; --valence[move.b]; }
                initialized = true;
            }

            /// @param bytes Payload size requested from the bounded proof cache.
            /// @return True if the requested storage fits within the memoization budget.
            bool Reserve(std::size_t bytes)
            {
                // Bound both payload and container overhead. Saturation merely
                // disables new entries; it never truncates a chemical search.
                if (failed_targets.size() + regions.size() >= 1024 ||
                    bytes > 4 * 1024 * 1024 - payload_bytes) return false;
                payload_bytes += bytes;
                return true;
            }

            /// Cache a region's outcome and the corresponding slice of its witness when space permits.
            /// @param key Lookup key for the recorded state or property.
            /// @param matched Whether this electronic subproblem has a successful witness.
            /// @param witness Electronic moves forming the equivalence proof.
            /// @param start First move of the witness slice to cache.
            /// @param end End of the retained witness range.
            void RememberRegion(std::vector<int> key, bool matched, const std::vector<Move> &witness,
                                std::size_t start, std::size_t end)
            {
                std::size_t bytes = key.size() * sizeof(int);
                if (matched)
                    for (std::size_t i = start; i < end; ++i)
                        bytes += sizeof(Move) + witness[i].kind.size() + witness[i].cycle.size() * sizeof(int);
                if (!Reserve(bytes)) return;
                Proof proof{matched, {}};
                if (matched) proof.moves.assign(witness.begin() + start, witness.begin() + end);
                regions.emplace(std::move(key), std::move(proof));
            }

            /// Cache a failed target to avoid repeating the electronic-state search.
            /// @param key Lookup key for the recorded state or property.
            void RememberFailure(std::vector<int> key)
            {
                if (Reserve(key.size() * sizeof(int))) failed_targets.insert(std::move(key));
            }
        };

        /// Encode the region's connectivity and endpoint states for a cache lookup.
        /// @param region Connected electronic subproblem under the fixed atom mapping.
        /// @return Integer key encoding the local resonance subproblem.
        std::vector<int> RegionKey(const Region &region)
        {
            // Explicit membership/boundaries distinguish equal-looking states
            // in different regions. Keep int indices: edge ids can exceed int16.
            std::vector<int> key{static_cast<int>(region.atoms.size()), static_cast<int>(region.edges.size())};
            for (const auto *part : {&region.atoms, &region.edges, &region.fixed_valence})
                key.insert(key.end(), part->begin(), part->end());
            key.insert(key.end(), region.left.begin(), region.left.end());
            key.insert(key.end(), region.right.begin(), region.right.end());
            return key;
        }

        /// Reuse a cached proof or run bounded resonance search and record its outcome.
        /// @param g Indexed graph used for equivalence matching.
        /// @param region Connected electronic subproblem under the fixed atom mapping.
        /// @param options Graph projection settings and bounds for equivalence search.
        /// @param result Result record carrying the operation's output.
        /// @param context Reusable search context and memoized resonance proofs.
        /// @return True if a path between the electronic assignments was found.
        bool SearchRegionCached(const Graph &g, const Region &region, const Options &options,
                                Result &result, ResonanceContext &context)
        {
            if (region.left == region.right) return true;
            // Successful first mappings need no cache. Their regional proofs
            // are stored lazily only if a later region rejects that mapping.
            if (context.failed_targets.empty()) return SearchRegion(g, region, options, result, context.cutoffs, context.directed);
            auto key = RegionKey(region);
            auto found = context.regions.find(key);
            if (found != context.regions.end())
            {
                ++result.region_cache_hits;
                if (found->second.matched)
                    result.witness.insert(result.witness.end(), found->second.moves.begin(), found->second.moves.end());
                return found->second.matched;
            }
            std::size_t start = result.witness.size();
            bool matched = SearchRegion(g, region, options, result, context.cutoffs, context.directed);
            context.RememberRegion(std::move(key), matched, result.witness, start, result.witness.size());
            // A failed search stays inconclusive. Reuse is safe only within this
            // round; deeper rounds discard all cached negative results.
            return matched;
        }

        /// Test resonance equivalence under a fixed atom mapping and record a successful electron-shift
        /// witness.
        /// @param a First graph in the fixed comparison order.
        /// @param b Second graph in the fixed comparison order.
        /// @param mapping Zero-based image of each source atom in the target graph.
        /// @param options Graph projection settings and bounds for equivalence search.
        /// @param result Result record carrying the operation's output.
        /// @param carbene_only Output indicating whether carbene normalization alone explains the
        ///     difference.
        /// @param context Reusable search context and memoized resonance proofs.
        /// @param raw_context Reusable search context and memoized resonance proofs.
        /// @return True if the two mapped electronic states are connected by the allowed moves.
        bool Resonance(const Graph &a, const Graph &b, const std::vector<int> &mapping,
                       const Options &options, Result &result, bool &carbene_only, ResonanceContext &context,
                       ResonanceContext *raw_context = nullptr)
        {
            const auto start = Clock::now();
            /// Add the elapsed comparison time to the result on every scope exit.
            struct Timer
            {
                Result &r;
                Clock::time_point t;
                /// Add the time spent in this resonance comparison to its result.
                ~Timer() { r.resonance_ms += Milliseconds(t); }
            } timer{result, start};
            context.Initialize(a, b);
            const auto &qa = context.charge, &ra = context.radical, &ba = context.orders, &va = context.valence;
            std::vector<int> qb(a.n), rb(a.n), bb;
            bb.reserve(a.edges.size());
            for (int i = 0; i < a.n; ++i) { qb[i] = b.charge[mapping[i]]; rb[i] = b.radical[mapping[i]]; }
            for (const auto &e : a.edges)
            {
                bb.push_back(b.edges[b.EdgeAt(mapping[e.u], mapping[e.v])].order);
            }
            result.witness = context.normalization;
            bool normalized = context.target_can_normalize && CarbeneNormalize(a, qb, rb, bb, result.witness, 1);
            normalized |= context.normalized;
            if (qa == qb && ra == rb && ba == bb) { carbene_only = normalized; return true; }
            carbene_only = false;
            // Carbene conventions can move a closed-shell resonance problem
            // into a less useful radical bookkeeping space. Before that search,
            // try a bounded proof on the prepared, unnormalized electronic graphs.
            // A failed probe never establishes inequivalence. Its states count
            // toward the same global budget, leaving at least seven eighths for
            // the normalized search on this mapping.
            const int used = result.states_left + result.states_right;
            const int probe_budget = std::min(4096, (2 * options.max_resonance - used) / 16);
            if (raw_context && normalized && probe_budget == 0)
                raw_context->cutoffs.state_limit = true;
            if (raw_context && normalized && probe_budget > 0)
            {
                auto probe_options = options;
                probe_options.max_resonance = used / 2 + probe_budget;
                auto normalization = result.witness;
                bool previous_limit = result.resonance_limit, previous_validation = result.validation_failed;
                double previous_ms = result.resonance_ms;
                bool unused = false;
                bool proved = Resonance(a, b, mapping, probe_options, result, unused, *raw_context);
                // The outer timer already includes the recursive probe.
                result.resonance_ms = previous_ms;
                result.resonance_limit = previous_limit;
                result.validation_failed = previous_validation;
                if (proved) return true;
                result.witness = std::move(normalization);
            }
            std::vector<int> target;
            /// Encode the target labels under this mapping to reuse an earlier resonance-search result.
            auto target_key = [&]()
            {
                target.reserve(2 * a.n + bb.size());
                for (const auto *part : {&qb, &rb, &bb}) target.insert(target.end(), part->begin(), part->end());
            };
            if (!context.failed_targets.empty())
            {
                target_key();
                if (context.failed_targets.count(target)) { ++result.resonance_cache_hits; return false; }
            }
            bool matched = [&]()
            {
                std::vector<int> vb(a.n, 0);
                std::vector<bool> pi(a.n, false);
                for (int e = 0; e < static_cast<int>(a.edges.size()); ++e)
                {
                    const auto &edge = a.edges[e];
                    if (edge.kind) continue;
                    for (int i : {edge.u, edge.v})
                    {
                        vb[i] += bb[e];
                        if (ba[e] > 1 || bb[e] > 1) pi[i] = true;
                    }
                }
                for (int i = 0; i < a.n; ++i)
                {
                    int z = a.source.atoms[i].atomic_num;
                    pi[i] = !a.metal[i] && z != 1 && !a.stereo_fixed[i] && !a.unsupported[i] &&
                        (pi[i] || qa[i] || qb[i] || ra[i] || rb[i] || Outer(z) - qa[i] - va[i] - ra[i] >= 2 || Outer(z) - qb[i] - vb[i] - rb[i] >= 2);
                }
                std::vector<int> region_of(a.n, -1);
                std::vector<Region> regions;
                for (int seed = 0; seed < a.n; ++seed)
                {
                    if (!pi[seed] || region_of[seed] >= 0) continue;
                    int index = regions.size(); regions.emplace_back();
                    std::vector<int> queue{seed}; region_of[seed] = index;
                    for (std::size_t p = 0; p < queue.size(); ++p)
                    {
                        int i = queue[p]; regions.back().atoms.push_back(i);
                        for (auto [j, e] : a.adj[i])
                            if (!a.edges[e].kind && pi[j] && region_of[j] < 0)
                                region_of[j] = index, queue.push_back(j);
                    }
                }
                for (int i = 0; i < a.n; ++i)
                    if (region_of[i] < 0 && (qa[i] != qb[i] || ra[i] != rb[i])) return false;
                for (int e = 0; e < static_cast<int>(a.edges.size()); ++e)
                {
                    auto edge = a.edges[e];
                    int index = region_of[edge.u];
                    if (!edge.kind && index >= 0 && index == region_of[edge.v]) regions[index].edges.push_back(e);
                    else if (ba[e] != bb[e]) return false;
                }
                // Check every region before spending any BFS budget. A later
                // charge/radical contradiction invalidates the whole mapping.
                for (const auto &region : regions)
                {
                    int dq = 0, dr = 0;
                    for (int i : region.atoms) { dq += qa[i] - qb[i]; dr += ra[i] - rb[i]; }
                    if (dq || dr) { ++result.region_constraint_rejections; return false; }
                }
                std::size_t witness_start = result.witness.size();
                for (auto &region : regions)
                {
                    int n = region.atoms.size();
                    region.fixed_valence.resize(n); region.adj.resize(n);
                    region.left.resize(2 * n + region.edges.size()); region.right.resize(region.left.size());
                    std::vector<int> local(a.n, -1);
                    for (int i = 0; i < n; ++i)
                    {
                        int index = region.atoms[i]; local[index] = i;
                        region.left[i] = qa[index]; region.left[n + i] = ra[index];
                        region.right[i] = qb[index]; region.right[n + i] = rb[index];
                        region.fixed_valence[i] = va[index];
                    }
                    for (int i = 0; i < static_cast<int>(region.edges.size()); ++i)
                    {
                        int e = region.edges[i], u = local[a.edges[e].u], v = local[a.edges[e].v];
                        region.endpoints.emplace_back(u, v);
                        region.adj[u].emplace_back(v, i); region.adj[v].emplace_back(u, i);
                        region.fixed_valence[u] -= ba[e]; region.fixed_valence[v] -= ba[e];
                        region.left[2 * n + i] = ba[e]; region.right[2 * n + i] = bb[e];
                    }
                    region.witness_start = result.witness.size();
                    region.proved = SearchRegionCached(a, region, options, result, context);
                    region.witness_end = result.witness.size();
                    if (!region.proved)
                    {
                        if (context.failed_targets.empty())
                        {
                            // Keep already-proved regions without charging their
                            // BFS work again under another complete mapping.
                            for (const auto &done : regions)
                                if (done.proved && done.left != done.right)
                                    context.RememberRegion(RegionKey(done), true, result.witness, done.witness_start, done.witness_end);
                            context.RememberRegion(RegionKey(region), false, result.witness, 0, 0);
                        }
                        result.witness.resize(witness_start);
                        return false;
                    }
                }
                return true;
            }();
            if (!matched)
            {
                if (target.empty()) target_key();
                context.RememberFailure(std::move(target));
            }
            return matched;
        }
    }

    Result Evaluate(const MoleculeData &left, const MoleculeData &right, const Options &options)
    {
        Result result;
        for (const char *name : {"raw_total_formal_charge", "heavy_atom_formula", "explicit_h_formula", "atom_count", "formal_charge", "radical_electrons", "metal_state", "metal_coordination", "nonmetal_connectivity", "component_electrons", "stereochemistry", "electronic_properties"}) result.invariants[name] = 0;
        try
        {
            if (options.max_resonance < 0 || options.max_resonance > 1000000 ||
                options.max_resonance_steps < 0 || options.max_resonance_steps > MaxResonanceSteps ||
                options.max_mappings < 0 || options.max_mapping_nodes < 0)
                throw std::invalid_argument("Invalid native search budget");
            auto started = Clock::now();
            Validate(left); Validate(right);
            auto left_carbenes = MetalCarbeneSites(left), right_carbenes = MetalCarbeneSites(right);
            bool paired_carbene_mode = HasKnownPairedMetalCarbene(left, left_carbenes) ||
                                       HasKnownPairedMetalCarbene(right, right_carbenes);
            auto a = Prepare(left, options, result.left, paired_carbene_mode, left_carbenes);
            auto b = Prepare(right, options, result.right, paired_carbene_mode, right_carbenes);
            result.preparation_ms = Milliseconds(started);
            auto invariant = [&](const char *name, bool same) { result.invariants[name] = same ? 1 : -1; return same; };
            auto fail = [&](const char *reason) { result.decision = "not_equivalent"; result.reason = reason; return result; };
            invariant("formal_charge", result.left.organic_charge == result.right.organic_charge);
            invariant("radical_electrons", result.left.organic_radicals == result.right.organic_radicals);
            if (!invariant("raw_total_formal_charge", result.left.charge == result.right.charge)) return fail("Raw whole-molecule total formal charges differ.");
            if (!invariant("heavy_atom_formula", result.left.heavy_formula == result.right.heavy_formula)) return fail("Heavy-atom formulas differ.");
            if (!invariant("metal_state", result.left.metal_states == result.right.metal_states)) return fail("Standardized metal formal states differ.");
            if (!invariant("explicit_h_formula", result.left.hydrogen_count == result.right.hydrogen_count)) return fail("Explicit-hydrogen counts differ.");
            if (!invariant("atom_count", a.n == b.n)) return fail("Atom counts differ.");
            // Isotopes are fixed graph labels even when chirality is disabled.
            std::multiset<std::pair<int, int>> labels_a, labels_b;
            for (const auto &atom : left.atoms) labels_a.emplace(atom.atomic_num, atom.isotope);
            for (const auto &atom : right.atoms) labels_b.emplace(atom.atomic_num, atom.isotope);
            if (labels_a != labels_b) return fail("Isotope inventories differ.");

            bool topology_found = false, stereo_found = false, stereo_unknown = false, annotation_found = false;
            bool structural_found = false;
            // With no active records on exactly one side, every possible
            // topology mapping lacks stereo information. A single topology
            // witness suffices for INC; still prove hard topology/charge
            // contradictions first. If both sides have records, keep searching:
            // a count mismatch can coexist with a specified contradiction.
            bool stereo_missing = options.use_chirality &&
                (HasActiveStereo(left, options.ignore_metal_coordination_bonds) !=
                 HasActiveStereo(right, options.ignore_metal_coordination_bonds));
            // Mapping enumeration is charged once. Try short electron-move paths as
            // mappings arrive, then revisit compatible mappings in depth rounds.
            // Each attempt gets a slice of the one shared state budget.
            Options search = options;
            int allowance = options.max_resonance;
            if (options.max_mappings > 1 && options.max_resonance > 0)
            {
                search.max_resonance_steps = std::min(2, options.max_resonance_steps);
                allowance = std::min(16, std::max(1, options.max_resonance / 8));
            }
            std::vector<std::vector<int>> pending;
            ResonanceContext resonance_context, raw_resonance_context;
            raw_resonance_context.allow_normalization = false;
            raw_resonance_context.directed = true;
            /// Look for an electronic witness within this mapping's current depth and state allowance.
            auto prove = [&](const std::vector<int> &mapping, bool exact)
            {
                const int used = result.states_left + result.states_right;
                search.max_resonance = std::min(options.max_resonance, used / 2 + allowance);
                bool carbene_only = false;
                if (!exact && !Resonance(a, b, mapping, search, result, carbene_only,
                                         resonance_context, &raw_resonance_context))
                { result.witness.clear(); return false; }
                result.mapping = mapping;
                int comparable = 0;
                Annotations(a, b, mapping, comparable);
                result.invariants["electronic_properties"] = comparable ? 1 : 0;
                result.decision = "equivalent";
                result.method = exact ? "ideal" : carbene_only ? "carbene_zwitterion" : "resonance";
                result.reason = exact ? "Native atom-mapped graph identity." : carbene_only ? "Native carbene/zwitterion normalization witness." : "Native regional resonance witness.";
                result.stop_reason = exact || carbene_only ? "not_attempted" : "target_found";
                return true;
            };
            started = Clock::now();
            auto [ca, cb] = Colors(a, b);
            /// Classify the failed graph mapping using the search limits and stereo evidence recorded so
            /// far.
            auto topology_failure = [&]()
            {
                result.mapping_ms = Milliseconds(started);
                result.invariants[options.ignore_metal_coordination_bonds ? "nonmetal_connectivity" : "metal_coordination"] = -1;
                return fail("No element/isotope-labeled topology mapping exists.");
            };
            if (!SameColors(ca, cb)) return topology_failure();
            Components ac(a, ca), bc(b, cb);
            auto sa = ac.signatures, sb = bc.signatures;
            std::sort(sa.begin(), sa.end()); std::sort(sb.begin(), sb.end());
            if (sa != sb) return topology_failure();
            // Every supported normalization and resonance move preserves charge
            // within a covalent component, including when coordination is kept.
            // These are topology-defined supersets, not mapping-dependent pi regions.
            sa = ac.signatures; sb = bc.signatures;
            for (int i = 0; i < static_cast<int>(sa.size()); ++i) sa[i].push_back(ac.charge[i]);
            for (int i = 0; i < static_cast<int>(sb.size()); ++i) sb[i].push_back(bc.charge[i]);
            auto sorted_a = sa, sorted_b = sb;
            std::sort(sorted_a.begin(), sorted_a.end()); std::sort(sorted_b.begin(), sorted_b.end());
            if (sorted_a != sorted_b)
            {
                result.mapping_ms = Milliseconds(started);
                result.invariants["component_electrons"] = -1;
                return fail("Covalent-component charges cannot be matched under any topology mapping.");
            }
            // Carbene normalization can change radical totals and its choice can
            // depend on the mapping. Only use raw component radicals when no
            // first normalization step is possible in either input, in any order.
            if (!CanNormalizeCarbene(a) && !CanNormalizeCarbene(b))
            {
                auto ra = sa, rb = sb;
                for (int i = 0; i < static_cast<int>(ra.size()); ++i) ra[i].push_back(ac.radicals[i]);
                for (int i = 0; i < static_cast<int>(rb.size()); ++i) rb[i].push_back(bc.radicals[i]);
                std::sort(ra.begin(), ra.end()); std::sort(rb.begin(), rb.end());
                if (ra != rb)
                {
                    result.mapping_ms = Milliseconds(started);
                    result.stop_reason = "rules_exhausted";
                    result.reason = "Covalent-component radicals differ and no supported carbene normalization applies.";
                    return result; // A limitation of the supported rules, not a hard NEQ proof.
                }
            }
            bool component_restricted = false;
            std::vector<std::vector<int>> buckets(b.n), candidates(a.n);
            for (int j = 0; j < b.n; ++j) buckets[cb[j]].push_back(j);
            for (int i = 0; i < a.n; ++i)
                for (int j : buckets[ca[i]])
                {
                    if (sa[ac.of[i]] == sb[bc.of[j]]) candidates[i].push_back(j);
                    else if (ac.signatures[ac.of[i]] == bc.signatures[bc.of[j]]) component_restricted = true;
                }
            auto mapping_options = options;
            // Stereo references cannot distinguish the topology witness in the
            // missing-information case, so terminal-H symmetry pruning applies.
            if (stereo_missing) mapping_options.use_chirality = false;
            SpectatorMappings spectators(a, b, ac, sa, mapping_options);
            /// Check the proposed atom mapping against electronic and stereochemical equivalence
            /// requirements.
            auto accept = [&](const std::vector<int> &mapping, bool exact)
            {
                topology_found = true;
                result.invariants["nonmetal_connectivity"] = 1;
                if (!options.ignore_metal_coordination_bonds) result.invariants["metal_coordination"] = 1;
                if (stereo_missing)
                {
                    result.reason = "Insufficient matching stereochemical information.";
                    result.stop_reason = "stereo_incomplete";
                    return true; // Stop mapping with INC, never an EQ witness.
                }
                int stereo = options.use_chirality ? CompareStereo(left, right, mapping, options.ignore_metal_coordination_bonds) : 1;
                spectators.Remember(a, b, mapping, stereo == 1);
                if (stereo < 0) stereo_unknown = true;
                if (stereo != 1) return false;
                stereo_found = true;
                result.invariants["stereochemistry"] = options.use_chirality ? 1 : 0;
                int comparable = 0;
                bool compatible = Annotations(a, b, mapping, comparable);
                result.comparable_fields = std::max(result.comparable_fields, comparable);
                if (!compatible) { if (exact) structural_found = true; return false; }
                annotation_found = true;
                if (prove(mapping, exact)) return true;
                if (result.states_left + result.states_right < 2 * options.max_resonance &&
                    (search.max_resonance_steps < options.max_resonance_steps || allowance < options.max_resonance))
                    pending.push_back(mapping);
                return false;
            };
            bool matched = MapGraphs(a, b, std::move(candidates), mapping_options, result, stereo_found, spectators.fixed,
                [&](const auto &map) { return accept(map, !stereo_missing && ExactMapping(a, b, map)); });
            while (!matched && !pending.empty() &&
                   result.states_left + result.states_right < 2 * options.max_resonance &&
                   (search.max_resonance_steps < options.max_resonance_steps || allowance < options.max_resonance))
            {
                // Admit the shortest path bound pruned by any mapping. Skipping
                // directly to a looser depth can flood a shallow layer with
                // irrelevant electronic states before other mappings get a turn.
                const int next_depth = std::min(resonance_context.cutoffs.next_depth,
                                                raw_resonance_context.cutoffs.next_depth);
                const bool grow_slice = allowance < options.max_resonance &&
                    (resonance_context.cutoffs.state_limit || raw_resonance_context.cutoffs.state_limit);
                if (next_depth > options.max_resonance_steps && !grow_slice) break;
                if (next_depth <= options.max_resonance_steps) search.max_resonance_steps = next_depth;
                if (grow_slice) allowance = std::min(options.max_resonance, 2 * allowance);
                // A truncated proof under a shallower depth or a smaller slice
                // says nothing about this round. Keep cumulative counters only.
                resonance_context = {};
                // Within the deeper contour, prefer electron moves toward the
                // target labels before spending the slice on wide side branches.
                resonance_context.directed = true;
                raw_resonance_context = {};
                raw_resonance_context.allow_normalization = false;
                raw_resonance_context.directed = true;
                result.resonance_limit = false;
                for (const auto &mapping : pending)
                    if (prove(mapping, false)) { matched = true; break; }
            }
            result.mapping_ms = std::max(0.0, Milliseconds(started) - result.resonance_ms);
            if (matched) return result;
            if (result.mapping_limit)
            { result.reason = "Native atom-mapping budget reached."; result.stop_reason = "mapping_limit"; return result; }
            if (!topology_found)
            {
                if (component_restricted)
                    return fail("No topology mapping satisfies covalent-component charge conservation.");
                result.invariants[options.ignore_metal_coordination_bonds ? "nonmetal_connectivity" : "metal_coordination"] = -1;
                return fail("No element/isotope-labeled topology mapping exists.");
            }
            if (!stereo_found && !stereo_unknown)
            {
                if (component_restricted)
                    return fail("No topology mapping jointly satisfies component charges and specified stereochemistry.");
                result.invariants["stereochemistry"] = -1;
                return fail("Specified stereochemistry differs under every topology mapping.");
            }
            if (structural_found && !annotation_found && !a.rearrangeable && !b.rearrangeable)
            { result.invariants["electronic_properties"] = -1; return fail("Exhaustive mappings have conflicting localized electronic annotations."); }
            result.stop_reason = result.validation_failed ? "validation_failed" : result.resonance_limit ? "limit_reached" : "rules_exhausted";
            result.reason = stereo_unknown && !stereo_found ? "Insufficient matching stereochemical information." :
                !annotation_found ? "No joint graph/electronic annotation witness." : "No witness within the supported native resonance rules and budget.";
        }
        catch (const std::exception &error)
        {
            result.reason = std::string("Native input/check could not be completed: ") + error.what();
            result.stop_reason = "error";
        }
        return result;
    }

    int CompareMappedStereo(const MoleculeData &left, const MoleculeData &right,
                            const std::vector<int> &mapping)
    {
        Validate(left);
        Validate(right);
        if (left.atoms.size() != right.atoms.size() || mapping.size() != left.atoms.size())
            throw std::invalid_argument("Stereo mapping must cover both molecules");
        std::vector<bool> seen(mapping.size(), false);
        for (int index : mapping)
        {
            if (index < 0 || index >= static_cast<int>(mapping.size()) || seen[index])
                throw std::invalid_argument("Stereo mapping must be a bijection");
            seen[index] = true;
        }
        return CompareStereo(left, right, mapping, false);
    }
}

#include "molgr/utils/coordination_seeds.h"
#include "molgr/utils/metals/hydrazido_geometry.h"
#include "molgr/utils/metals/scoring.h"
#include "molgr/utils/electrons.h"
#include "molgr/compat/openbabel_iter.h"
#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/elements.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

namespace molgr::metal::hydrazido_geometry
{
namespace
{
    constexpr auto key = "metal_terminal_hydrazido_geometry_conflict_count";
    using namespace molgr::coordination::hydrazido;

    using Atom = std::tuple<int, int, int, int, int, bool, int, double, double, double>;
    using Bond = std::tuple<int, int, int>;
    // Preserve isotope, spin, lone pairs, coordinates and the surrounding bonds.
    // Only the two N charges and their shared bond order may differ at a recognized site.
    /// Mask only the formal charges and N–N orders at recognized sites when comparing the remaining ligand.
    struct Features
    {
        Sites sites;
        std::vector<Atom> atoms;
        std::vector<Bond> bonds;
    };

    /// Collect the hydrazido sites and spectator labels used for comparing single-metal candidates.
    /// @param candidate One metal assignment, its ligand state and metal-visibility metadata.
    /// @return Hydrazido-site assignments and the surrounding ligand signature.
    Features Describe(const molgr::state::MetalCandidateState &candidate)
    {
        Features result;
        if (candidate.metal_states.size() != 1 || !candidate.no_metal_state) return result;
        auto &mol = const_cast<OpenBabel::OBMol &>(candidate.no_metal_state->Mol());
        std::set<int> masked;
        std::set<Pair> pairs;
        for (const auto &[pair, state] : LocalSites(mol))
            if (SupportedGeometry(*mol.GetAtom(pair.first), *mol.GetAtom(pair.second), candidate.metal_states.front().Site()))
            {
                result.sites[pair] = state;
                masked.insert(pair.first); masked.insert(pair.second);
                pairs.insert(std::minmax(pair.first, pair.second));
            }
        if (result.sites.empty()) return result;
        FOR_ATOMS_OF_MOL(a, mol)
            result.atoms.emplace_back(a->GetAtomicNum(), a->GetIsotope(), masked.count(a->GetIdx()) ? 0 : a->GetFormalCharge(),
                molgr::utils::GetUnpairedElectronCount(*a), molgr::utils::GetLonePairCount(*a),
                molgr::utils::HasUnresolvedTwoElectronCenter(*a), a->GetImplicitHCount(), a->GetX(), a->GetY(), a->GetZ());
        FOR_BONDS_OF_MOL(b, mol)
        {
            const Pair pair = std::minmax(b->GetBeginAtomIdx(), b->GetEndAtomIdx());
            result.bonds.emplace_back(pair.first, pair.second, pairs.count(pair) ? 0 : b->GetBondOrder());
        }
        std::sort(result.bonds.begin(), result.bonds.end());
        return result;
    }
}

void Clear(std::vector<molgr::state::MetalCandidateState> &candidates)
{
    for (auto &candidate : candidates)
    {
        const int previous = scoring::MetadataInt(candidate, key);
        for (const auto *field : {"metal_discordance_count", "metal_discordance_structural_count"})
            candidate.metadata[field] = scoring::MetadataDouble(candidate, field) - previous;
        candidate.metadata[key] = 0;
    }
}

void Annotate(std::vector<molgr::state::MetalCandidateState> &candidates)
{
    Clear(candidates);
    std::vector<Features> features;
    std::vector<double> counts;
    for (const auto &c : candidates)
    {
        features.push_back(Describe(c));
        counts.push_back(scoring::MetadataDouble(c, "metal_discordance_structural_count"));
    }
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
        auto &c = candidates[i]; const auto &f = features[i];
        if (f.sites.empty()) continue;
        std::set<Pair> conflicts;
        for (std::size_t j = 0; j < candidates.size(); ++j)
        {
            const auto &p = candidates[j]; const auto &pf = features[j];
            if (f.atoms != pf.atoms || f.bonds != pf.bonds || f.sites.size() != pf.sites.size() ||
                counts[j] > counts[i] || c.no_metal_radical_target != p.no_metal_radical_target) continue;
            std::set<Pair> changed; bool valid = true;
            for (const auto &[site, electronic] : f.sites)
            {
                const auto other = pf.sites.find(site);
                if (other == pf.sites.end()) { valid = false; break; }
                if (electronic == neutral && other->second == reduced) changed.insert(site);
                else if (electronic != other->second) valid = false;
            }
            if (!valid || changed.empty()) continue;
            const auto &m = c.metal_states[0], &o = p.metal_states[0];
            if (m.idx == o.idx && m.element_idx == o.element_idx && m.radical_num == o.radical_num &&
                m.position_x == o.position_x && m.position_y == o.position_y && m.position_z == o.position_z &&
                o.valence - m.valence == 2 * static_cast<int>(changed.size()))
                conflicts.insert(changed.begin(), changed.end());
        }
        const int count = static_cast<int>(conflicts.size());
        c.metadata[key] = count;
        for (const auto *field : {"metal_discordance_count", "metal_discordance_structural_count"})
            c.metadata[field] = scoring::MetadataDouble(c, field) + count;
    }
}
}

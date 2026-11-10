#include "molgr/utils/equivalence_stereo.h"
#include "molgr/utils/consts.h"
#include "molgr/vendor/stereo_permutations.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>

namespace molgr::equivalence
{
    using namespace utils;
    namespace
    {
        bool Metal(const MoleculeData &d, int index)
        {
            return kNonMetalDict.count(d.atoms.at(index - 1).atomic_num) == 0;
        }

        int SwapPermutation(StereoKind kind, int p, int x, int y)
        {
            if (x == y || !p)
                return p;
            if (x > y)
                std::swap(x, y);
            if (kind == StereoKind::Tetrahedral)
                return 3 - p;
            int n = kind == StereoKind::SquarePlanar ? 4 : kind == StereoKind::TrigonalBipyramidal ? 5 : 6;
            int col = x * (2 * n - x - 1) / 2 + y - x - 1;
            switch (kind)
            {
            case StereoKind::SquarePlanar: return detail::swap_squareplanar_table[p][col];
            case StereoKind::TrigonalBipyramidal: return detail::swap_trigonalbipyramidal_table[p][col];
            case StereoKind::Octahedral: return detail::swap_octahedral_table[p][col];
            default: return 0;
            }
        }

        bool SameConfiguration(StereoKind kind, std::vector<int> from,
                               const std::vector<int> &to, int p, int target)
        {
            if (from.size() != to.size())
                return false;
            // Repeated virtual references denote indistinguishable vacant sites.
            // Consider their permutations, rather than choosing an arbitrary one.
            std::function<bool(int, int)> visit = [&](int depth, int perm)
            {
                if (depth == static_cast<int>(from.size()))
                    return perm == target;
                for (int j = depth; j < static_cast<int>(from.size()); ++j)
                {
                    if (from[j] != to[depth])
                        continue;
                    std::swap(from[depth], from[j]);
                    bool equal = visit(depth + 1, SwapPermutation(kind, perm, depth, j));
                    std::swap(from[depth], from[j]);
                    if (equal)
                        return true;
                }
                return false;
            };
            return visit(0, p);
        }
    }

    bool ActiveStereo(const MoleculeData &d, const StereoData &s, bool ignore)
    {
        if (!ignore) return true;
        if (Metal(d, s.center) || (s.end && Metal(d, s.end))) return false;
        for (int ref : s.refs)
            if (ref && Metal(d, ref)) return false;
        return true;
    }

    bool HasActiveStereo(const MoleculeData &d, bool ignore)
    {
        return std::any_of(d.stereo.begin(), d.stereo.end(),
                          [&](const auto &s) { return ActiveStereo(d, s, ignore); });
    }

    int CompareStereo(const MoleculeData &left, const MoleculeData &right,
                      const std::vector<int> &mapping, bool ignore)
    {
        auto mapped = [&](int i) { return i ? mapping.at(i - 1) + 1 : 0; };
        using GroupKey = std::pair<int, std::vector<int>>;
        std::set<GroupKey> groups_left, groups_right;
        std::map<int, GroupKey> relative_members;
        std::set<int> active_left, active_right;
        for (const auto &s : left.stereo)
            if (!s.end && ActiveStereo(left, s, ignore)) active_left.insert(s.center);
        for (const auto &s : right.stereo)
            if (!s.end && ActiveStereo(right, s, ignore)) active_right.insert(s.center);
        for (const auto &g : left.stereo_groups)
        {
            if (g.kind == StereoGroupKind::Absolute)
                continue;
            std::vector<int> atoms;
            for (int i : g.atoms)
                if (active_left.count(i))
                    atoms.push_back(mapped(i));
            std::sort(atoms.begin(), atoms.end());
            if (atoms.empty())
                continue;
            GroupKey key{static_cast<int>(g.kind), atoms};
            groups_left.insert(key);
            for (int atom : atoms)
                relative_members[atom] = key;
        }
        for (const auto &g : right.stereo_groups)
        {
            if (g.kind == StereoGroupKind::Absolute)
                continue;
            std::vector<int> atoms;
            for (int i : g.atoms)
                if (active_right.count(i))
                    atoms.push_back(i);
            std::sort(atoms.begin(), atoms.end());
            if (!atoms.empty())
                groups_right.emplace(static_cast<int>(g.kind), atoms);
        }
        if (groups_left != groups_right)
            return -1; // Different specified sets, not proof of disjoint configurations.

        std::vector<bool> used(right.stereo.size(), false);
        std::map<GroupKey, int> flips;
        for (const auto &a : left.stereo)
        {
            if (!ActiveStereo(left, a, ignore))
                continue;
            int found = -1;
            for (int j = 0; j < static_cast<int>(right.stereo.size()); ++j)
            {
                const auto &b = right.stereo[j];
                if (used[j] || !ActiveStereo(right, b, ignore) || a.kind != b.kind)
                    continue;
                if ((mapped(a.center) == b.center && mapped(a.end) == b.end) ||
                    (a.kind == StereoKind::DoubleBond && mapped(a.center) == b.end && mapped(a.end) == b.center))
                {
                    found = j;
                    break;
                }
            }
            if (found < 0)
                return -1;
            used[found] = true;
            const auto &b = right.stereo[found];
            if (!a.configuration || !b.configuration)
            {
                if (a.configuration != b.configuration)
                    return -1;
                continue;
            }
            std::vector<int> refs;
            for (int i : a.refs)
                refs.push_back(mapped(i));
            bool match = false;
            if (a.kind == StereoKind::DoubleBond)
            {
                auto other = b.refs;
                if (mapped(a.center) != b.center)
                {
                    std::swap(other[0], other[2]);
                    std::swap(other[1], other[3]);
                }
                int orientation = a.configuration;
                for (int offset : {0, 2})
                {
                    if (refs[offset] == other[offset] && refs[offset + 1] == other[offset + 1])
                        continue;
                    if (refs[offset] != other[offset + 1] || refs[offset + 1] != other[offset])
                        return -1;
                    orientation = 3 - orientation;
                }
                match = orientation == b.configuration;
            }
            else
                match = SameConfiguration(a.kind, refs, b.refs, a.configuration, b.configuration);
            auto relative = relative_members.find(b.center);
            if (relative != relative_members.end() && a.kind == StereoKind::Tetrahedral)
            {
                int flip = match ? 0 : 1;
                auto [where, inserted] = flips.emplace(relative->second, flip);
                if (!inserted && where->second != flip)
                    return 0;
            }
            else if (!match)
                return relative == relative_members.end() ? 0 : -1;
        }
        for (int j = 0; j < static_cast<int>(right.stereo.size()); ++j)
            if (!used[j] && ActiveStereo(right, right.stereo[j], ignore))
                return -1;
        return 1;
    }
}

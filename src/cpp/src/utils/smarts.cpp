/*
 * @Author: TMJ
 * @Date: 2026-05-15 15:36:10
 * @LastEditors: TMJ
 * @LastEditTime: 2026-06-28 19:16:26
 * @Description: 请填写简介
 */
#include "molgr/utils/smarts.h"

#include "molgr/vendor/openbabel_threading.h"

#include <openbabel/parsmart.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>

namespace
{
    using PatternId = molgr::smarts::PatternId;

    constexpr std::size_t kPatternCount = static_cast<std::size_t>(PatternId::COUNT);

    constexpr std::array<const char *, kPatternCount> kSmartsPatterns = {
        "[Nv0,Cv1,Nv3,Clv1,Clv2,Clv3,Brv1,Brv2,Brv3,Iv1,Iv2,Iv3,Asv3,Sev2]",
        "[Hv0,Bv2,Bv3,Cv0,Cv1,Cv2,Cv3,Nv1,Nv2,Ov0,Ov1,Clv0,Siv3,Pv2,Sv0,Sv1,Brv0,Iv0]",
        "[Pv2]", "[Cv0,Cv1,Cv2,Cv3,Nv2]",
        "[Ov1]", "[Cv3]",
        "[#34v3,#53v4,#53v5]", "[Ov0,Ov1]",
        "[#15D5v5]", "[#8v1,#6v3]",
        "[#14D4v4]-[#6D2v3]=[#8]",
        "[#15D4v4]-[#15D4v4]",
        "[Cv5,Nv5,Pv5,Siv5]=,#[*]",
        "[#16,#15,#33,#9,#17,#35,#53]=,#[*]",
        "[#6]1([#6]2)([#6]3)[#7]23[#6]1",
        "[#6]1([#6]2)[#7]2[#6]1",
        "[*+1,*+2,*+3]-[Ov1+0,Nv2+0,Sv1+0]",
        "[Ov1+0]-C=O",
        "[#7v1+0]-[#7v2+0]-[#7v1+0]",
        "[#7v3+0]-[#7v2+0]-[#7v3+0]",
        "[*-1]-,=[N+0,O+0,S+0,P+0]-,=[*]",
        "[Nv3+0]=[Nv2+0]",
        "[#6v3+0,#6v2+0,#1v0+0]",
        "[*]-,=[Nv3,Ov2,Pv3,Sv2]",
        "[#9v0+0]",
        "[#8v0+0]",
        "[#8v1+0]",
        "[#17v0+0]",
        "[#7v0+0]",
        "[#7v1+0]",
        "[#7v2+0]",
        "[#35v0+0]",
        "[#53v0+0]",
        "[#16v0+0]",
        "[#16v1+0]",
        "[#34v0+0]",
        "[#34v1+0]",
        "[#15v0+0]",
        "[#15v1+0]",
        "[#15v2+0]",
        "[#5v0+0]",
        "[#5v1+0]",
        "[#5v2+0]",
        "[#6v3+0]",
        "[#1v0+0]",
        "[#6v2+0,#6v1+0,#6v0+0]",
        "[#6v3+0,#6v4+0]1=[#6v3+0,#6v4+0]-[#6v3+0,#6v4+0]=[#6v3+0,#6v4+0]-[#6v2+0,#6v3+0]-1",
        "[*]-[*]=[*]",
        "[*]-,=[N,O,P,S]-,=[*]",
        "[*]-[*]=[*]-[*]",
        "[*]-[*]=[*]-[*]=[*]-[*]",
        "[*-]-[*]=[*]~[*+]",
        "[*-]=[*+]=[*+0]",
        "[#8]=[#6](-[!-])-[*]=[*]-[#7-,#6-]",
        "[#7v2+]=[*]-[*]=[*]-[#8-]",
        "[#7+,#8+]=[*]-[#6-,#7-,#8-]",
        "[#7+0,#8+0,#16+0]=[#6+0]-[#6-,#7-]",
        "[#6]=[#6]=[#6-,#7-]",
        "[*-]1-,:[*](=,:[*])-,:[*]=,:[*]-,:[*]=,:[*]1",
        "[*-]1-,:[*]=,:[*]-,:[*](=,:[*])-,:[*]=,:[*]1",
        "[*+,*+2,*+3]-,=[*-,*-2,*-3]",
        "[*]-[*]=,#[*]-[*]",
        "[#7v3+0,#8v2+0,#16v2+0]-,=,:[*+1]",
        "[#7v3+0,#8v2+0,#16v2+0]-,:[*]=,:[*]-,:[*+1]",
        "[*-]:[*]=[#7+0,#8+0]",
        "[#15-,#16-,#17-,#35-,#53-]#[#7+1,#8+1,#16+1]",
        "[*-1]-,:[*]=,:[*]-,:[*]=,:[*+1]",
        "[*-]-[*R]=[*R]=[*R]",
        "[*]-[#7+0]=[#7+0]-[#7+0]",
        "[*]~[*+0]=,:[*+0]~[*]",
        "[*]~[*+0](=,:[*+0])~[*]",
        "[*+0]#,=[*+0]",
        "[#7+1,#15+1]=[*+0]",
        "[*+0]:[*+0]",
        "[*]-,=,:[*]=,#,:[*]",
        "[*]=,#,:[*]-,:[*]=,#,:[*]",
    };
    using PatternArray = std::array<std::unique_ptr<OpenBabel::OBSmartsPattern>, kPatternCount>;

    /// @return The current thread's compiled reconstruction SMARTS patterns.
    PatternArray &ThreadLocalPatterns()
    {
        // Open Babel SMARTS destructors are unsafe during external worker
        // thread teardown; keep the compiled patterns process-lived.
        thread_local PatternArray *compiled_patterns = []()
        {
            auto *patterns = new PatternArray{};
            for (std::size_t idx = 0; idx < kPatternCount; ++idx)
            {
                auto pattern = std::make_unique<OpenBabel::OBSmartsPattern>();
                if (!pattern->Init(kSmartsPatterns[idx]))
                {
                    throw std::runtime_error(
                        std::string("Invalid built-in SMARTS pattern: ") + kSmartsPatterns[idx]);
                }
                (*patterns)[idx] = std::move(pattern);
            }
            return patterns;
        }();
        return *compiled_patterns;
    }

}

namespace molgr
{
    namespace smarts
    {
        std::vector<std::vector<int>> FindAll(
            OpenBabel::OBSmartsPattern &pattern,
            OpenBabel::OBMol &mol)
        {
            molgr::vendor::openbabel_threading::PrepareForSmartsMatching(mol);
            std::vector<std::vector<int>> matches;
            pattern.Match(mol, matches, OpenBabel::OBSmartsPattern::AllUnique);
            return matches;
        }

        std::vector<std::vector<int>> FindAll(OpenBabel::OBMol &mol, PatternId pattern_id)
        {
            OpenBabel::OBSmartsPattern &pattern =
                *ThreadLocalPatterns().at(static_cast<std::size_t>(pattern_id));
            return FindAll(pattern, mol);
        }
    }
}

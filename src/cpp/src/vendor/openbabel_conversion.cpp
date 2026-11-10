#include "molgr/vendor/openbabel_conversion.h"
#include "molgr/process_guard.h"
#include "molgr/vendor/openbabel_copy.h"

#include <openbabel/obconversion.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace
{
    OpenBabel::OBConversion &SmilesInputConversion()
    {
        static OpenBabel::OBConversion *conversion = []()
        {
            auto instance = std::make_unique<OpenBabel::OBConversion>();
            if (!instance->SetInFormat("smi"))
            {
                throw std::runtime_error("Open Babel SMILES input format is unavailable");
            }
            return instance.release();
        }();
        return *conversion;
    }

    OpenBabel::OBConversion &SmilesOutputConversion()
    {
        static OpenBabel::OBConversion *conversion = []()
        {
            auto instance = std::make_unique<OpenBabel::OBConversion>();
            if (!instance->SetOutFormat("smi"))
            {
                throw std::runtime_error("Open Babel SMILES output format is unavailable");
            }
            return instance.release();
        }();
        return *conversion;
    }
}

namespace molgr
{
    namespace vendor
    {
        namespace openbabel_conversion
        {
            bool ReadSmiles(const std::string &smiles, OpenBabel::OBMol *mol)
            {
                molgr::EnsureCurrentProcess("molgr.native.smiles_reader");
                if (mol == nullptr)
                {
                    return false;
                }
                std::lock_guard<std::recursive_mutex> guard(
                    molgr::vendor::openbabel_threading::NativeOpenBabelMutex());
                mol->Clear();
                return SmilesInputConversion().ReadString(mol, smiles);
            }

            std::string WriteSmilesFirstToken(const OpenBabel::OBMol &mol)
            {
                molgr::EnsureCurrentProcess("molgr.native.smiles_writer");
                std::lock_guard<std::recursive_mutex> guard(
                    molgr::vendor::openbabel_threading::NativeOpenBabelMutex());
                OpenBabel::OBMol copy = molgr::vendor::openbabel_threading::CopyMol(mol);
                const std::string smiles = SmilesOutputConversion().WriteString(&copy, true);
                std::istringstream input(smiles);
                std::string token;
                input >> token;
                return token;
            }
        }
    }
}

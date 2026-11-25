#include "molgr/utils/xyz.h"
#include "molgr/process_guard.h"

#include "molgr/vendor/openbabel_threading.h"

#include <openbabel/atom.h>
#include <openbabel/elements.h>

#include <algorithm>
#include <cctype>
#include <locale>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    /// Parsed XYZ atom with atomic number and Cartesian coordinates in angstroms.
    struct ParsedXyzAtom
    {
        int atomic_num = 0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    /// Parse the declared positive atom count from the XYZ header.
    /// @param line One line from the XYZ input.
    /// @param atom_count Destination for the declared number of XYZ atoms.
    /// @return True if parsing succeeded and the output was populated.
    bool ParseAtomCountLine(const std::string &line, std::size_t *atom_count)
    {
        std::istringstream stream(line);
        stream.imbue(std::locale::classic());

        std::size_t count = 0;
        stream >> count;
        if (!stream)
        {
            return false;
        }

        std::string trailing;
        if (stream >> trailing)
        {
            return false;
        }

        *atom_count = count;
        return count > 0;
    }

    /// Accept an XYZ element symbol or numeric identifier and return its atomic number through the output.
    /// @param token Token extracted from or supplied by the input line.
    /// @param atomic_num Output atomic number for the parsed element.
    /// @return True if parsing succeeded and the output was populated.
    bool ParseAtomicNumber(std::string token, int *atomic_num)
    {
        if (token.empty())
        {
            return false;
        }

        const bool numeric_token = std::all_of(
            token.begin(),
            token.end(),
            [](unsigned char value)
            {
                return std::isdigit(value) != 0;
            });
        if (numeric_token)
        {
            std::istringstream stream(token);
            int parsed_atomic_num = 0;
            stream.imbue(std::locale::classic());
            stream >> parsed_atomic_num;
            if (!stream || parsed_atomic_num <= 0)
            {
                return false;
            }
            *atomic_num = parsed_atomic_num;
            return true;
        }

        token[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(token[0])));
        for (std::size_t idx = 1; idx < token.size(); ++idx)
        {
            token[idx] = static_cast<char>(std::tolower(static_cast<unsigned char>(token[idx])));
        }

        const int parsed_atomic_num = OpenBabel::OBElements::GetAtomicNum(token.c_str());
        if (parsed_atomic_num <= 0)
        {
            return false;
        }
        *atomic_num = parsed_atomic_num;
        return true;
    }

    /// Parse an element token and three Cartesian coordinates from one XYZ atom line.
    /// @param line One line from the XYZ input.
    /// @param parsed_atom Destination for the element and Cartesian coordinates.
    /// @return True if parsing succeeded and the output was populated.
    bool ParseAtomLine(const std::string &line, ParsedXyzAtom *parsed_atom)
    {
        std::istringstream stream(line);
        stream.imbue(std::locale::classic());

        std::string atom_token;
        stream >> atom_token >> parsed_atom->x >> parsed_atom->y >> parsed_atom->z;
        if (!stream)
        {
            return false;
        }

        return ParseAtomicNumber(atom_token, &parsed_atom->atomic_num);
    }
}

namespace molgr
{
    namespace utils
    {
        bool ParseXyzAtomicNumbers(
            const std::string &xyz_block,
            std::vector<int> *atomic_numbers)
        {
            if (atomic_numbers == nullptr)
            {
                return false;
            }

            std::istringstream input(xyz_block);
            input.imbue(std::locale::classic());

            std::string atom_count_line;
            if (!std::getline(input, atom_count_line))
            {
                return false;
            }

            std::size_t atom_count = 0;
            if (!ParseAtomCountLine(atom_count_line, &atom_count))
            {
                return false;
            }

            std::string title_line;
            if (!std::getline(input, title_line))
            {
                return false;
            }

            std::vector<int> parsed_numbers;
            parsed_numbers.reserve(atom_count);
            std::string atom_line;
            for (std::size_t idx = 0; idx < atom_count; ++idx)
            {
                if (!std::getline(input, atom_line))
                {
                    return false;
                }

                ParsedXyzAtom parsed_atom;
                if (!ParseAtomLine(atom_line, &parsed_atom))
                {
                    return false;
                }
                parsed_numbers.push_back(parsed_atom.atomic_num);
            }

            *atomic_numbers = std::move(parsed_numbers);
            return true;
        }

        bool ReadXyzBlockToMol(const std::string &xyz_block, OpenBabel::OBMol *mol)
        {
            molgr::EnsureCurrentProcess("molgr.native.xyz_reader");
            if (mol == nullptr)
            {
                return false;
            }

            std::istringstream input(xyz_block);
            input.imbue(std::locale::classic());

            std::string atom_count_line;
            if (!std::getline(input, atom_count_line))
            {
                return false;
            }

            std::size_t atom_count = 0;
            if (!ParseAtomCountLine(atom_count_line, &atom_count))
            {
                return false;
            }

            std::string title_line;
            if (!std::getline(input, title_line))
            {
                return false;
            }

            std::vector<ParsedXyzAtom> atoms;
            atoms.reserve(atom_count);
            std::string atom_line;
            for (std::size_t idx = 0; idx < atom_count; ++idx)
            {
                if (!std::getline(input, atom_line))
                {
                    return false;
                }

                ParsedXyzAtom parsed_atom;
                if (!ParseAtomLine(atom_line, &parsed_atom))
                {
                    return false;
                }
                atoms.push_back(parsed_atom);
            }

            mol->Clear();
            mol->BeginModify();
            mol->ReserveAtoms(static_cast<unsigned int>(atom_count));
            mol->SetDimension(3);
            mol->SetTitle(title_line.c_str());
            for (const ParsedXyzAtom &parsed_atom : atoms)
            {
                OpenBabel::OBAtom *atom = mol->NewAtom();
                atom->SetAtomicNum(parsed_atom.atomic_num);
                atom->SetVector(parsed_atom.x, parsed_atom.y, parsed_atom.z);
            }
            molgr::vendor::openbabel_threading::ConnectTheDotsAndPerceiveBondOrders(*mol);
            mol->EndModify();
            return true;
        }

        std::string WriteXyzBlock(const OpenBabel::OBMol &mol)
        {
            std::ostringstream output;
            output.imbue(std::locale::classic());
            output << mol.NumAtoms() << '\n' << mol.GetTitle() << '\n';
            output << std::fixed << std::setprecision(10);

            for (unsigned int index = 1; index <= mol.NumAtoms(); ++index)
            {
                const OpenBabel::OBAtom *atom = mol.GetAtom(index);
                if (atom == nullptr)
                {
                    continue;
                }
                output << OpenBabel::OBElements::GetSymbol(atom->GetAtomicNum()) << ' '
                       << atom->GetX() << ' ' << atom->GetY() << ' ' << atom->GetZ() << '\n';
            }
            return output.str();
        }
    }
}

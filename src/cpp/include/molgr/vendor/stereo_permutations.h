#pragma once
// Standard SMILES permutation swap tables from RDKit NontetrahedralStereo.cpp.
// Copyright (C) 2022 Greg Landrum and other RDKit contributors.
// Source: https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/NontetrahedralStereo.cpp
// Source SHA256: 6ce9712059ad157e106c493e00c748fe880c603f64d86a973e2a22197f7535dd
// BSD-3-Clause; full license retained below.
/*
BSD 3-Clause License

Copyright (c) 2006-2015, Rational Discovery LLC, Greg Landrum, and Julie Penzotti and others
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
namespace molgr::equivalence::detail {
constexpr unsigned char swap_squareplanar_table[4][6] = {
    // 0  0  0  1  1  2
    // 1  2  3  2  3  3
    {0, 0, 0, 0, 0, 0},
    {3, 1, 2, 2, 1, 3},  // SP1
    {2, 3, 1, 1, 3, 2},  // SP2
    {1, 2, 3, 3, 2, 1}   // SP3
};

constexpr unsigned char swap_trigonalbipyramidal_table[21][10] = {
    // 0   0   0   0   1   1   1   2   2   3
    // 1   2   3   4   2   3   4   3   4   4
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {9, 20, 17, 2, 2, 2, 7, 2, 6, 3},        // TB1
    {11, 15, 18, 1, 1, 1, 8, 1, 5, 4},       // TB2
    {10, 19, 4, 18, 4, 8, 4, 5, 4, 1},       // TB3
    {12, 16, 3, 17, 3, 7, 3, 6, 3, 2},       // TB4
    {13, 6, 16, 20, 7, 6, 6, 3, 2, 6},       // TB5
    {14, 5, 19, 15, 8, 5, 5, 4, 1, 5},       // TB6
    {8, 14, 10, 11, 5, 4, 1, 8, 8, 8},       // TB7
    {7, 13, 12, 9, 6, 3, 2, 7, 7, 7},        // TB8
    {1, 11, 11, 8, 15, 18, 11, 11, 14, 10},  // TB9
    {3, 12, 7, 12, 16, 12, 17, 13, 12, 9},   // TB10
    {2, 9, 9, 7, 20, 17, 9, 9, 13, 12},      // TB11
    {4, 10, 8, 10, 19, 10, 18, 14, 10, 11},  // TB12
    {5, 8, 14, 14, 14, 19, 15, 10, 11, 14},  // TB13
    {6, 7, 13, 13, 13, 16, 20, 12, 9, 13},   // TB14
    {20, 2, 20, 6, 9, 20, 13, 17, 20, 16},   // TB15
    {19, 4, 5, 19, 10, 14, 19, 19, 18, 15},  // TB16
    {18, 18, 1, 4, 18, 11, 10, 15, 19, 18},  // TB17
    {17, 17, 2, 3, 17, 9, 12, 20, 16, 17},   // TB18
    {16, 3, 6, 16, 12, 13, 16, 16, 17, 20},  // TB19
    {15, 1, 15, 5, 11, 15, 14, 18, 15, 19}   // TB20
};

constexpr unsigned char swap_octahedral_table[31][15] = {
    // 0   0   0   0   0   1   1   1   1   2   2   2   3   3   4
    // 1   2   3   4   5   2   3   4   5   3   4   5   4   5   5
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {17, 16, 30, 21, 2, 14, 2, 10, 25, 8, 2, 22, 4, 7, 3},       // OH1
    {7, 3, 25, 22, 1, 4, 1, 8, 30, 10, 1, 21, 14, 17, 16},       // OH2
    {18, 2, 29, 16, 22, 15, 16, 26, 11, 9, 21, 16, 6, 5, 1},     // OH3
    {15, 18, 19, 28, 14, 2, 8, 14, 27, 14, 10, 24, 1, 6, 5},     // OH4
    {14, 17, 20, 15, 27, 16, 9, 28, 15, 15, 23, 11, 7, 3, 4},    // OH5
    {16, 14, 18, 26, 24, 17, 29, 18, 13, 19, 12, 18, 3, 4, 7},   // OH6
    {2, 15, 17, 23, 25, 18, 30, 12, 17, 20, 17, 13, 5, 1, 6},    // OH7
    {23, 26, 11, 12, 10, 10, 4, 2, 29, 1, 14, 20, 10, 13, 9},    // OH8
    {24, 25, 10, 11, 13, 11, 5, 30, 16, 3, 19, 15, 12, 11, 8},   // OH9
    {20, 29, 9, 13, 8, 8, 14, 1, 26, 2, 4, 23, 8, 12, 11},       // OH10
    {19, 30, 8, 9, 12, 9, 15, 25, 3, 16, 24, 5, 13, 9, 10},      // OH11
    {22, 27, 13, 8, 11, 13, 28, 7, 18, 21, 6, 17, 9, 10, 13},    // OH12
    {21, 28, 12, 10, 9, 12, 27, 17, 6, 22, 18, 7, 11, 8, 12},    // OH13
    {5, 6, 24, 27, 4, 1, 10, 4, 28, 4, 8, 19, 2, 18, 15},        // OH14
    {4, 7, 23, 5, 28, 3, 11, 27, 5, 5, 20, 9, 17, 16, 14},       // OH15
    {6, 1, 26, 3, 21, 5, 3, 29, 9, 11, 22, 3, 18, 15, 2},        // OH16
    {1, 5, 7, 20, 30, 6, 25, 13, 7, 23, 7, 12, 15, 2, 18},       // OH17
    {3, 4, 6, 29, 19, 7, 26, 6, 12, 24, 13, 6, 16, 14, 17},      // OH18
    {11, 24, 4, 30, 18, 25, 23, 24, 22, 6, 9, 14, 21, 24, 20},   // OH19
    {10, 23, 5, 17, 29, 26, 24, 21, 23, 7, 15, 8, 23, 22, 19},   // OH20
    {13, 22, 28, 1, 16, 27, 22, 20, 24, 12, 3, 2, 19, 23, 22},   // OH21
    {12, 21, 27, 2, 3, 28, 21, 23, 19, 13, 16, 1, 24, 20, 21},   // OH22
    {8, 20, 15, 7, 26, 29, 19, 22, 20, 17, 5, 10, 20, 21, 24},   // OH23
    {9, 19, 14, 25, 6, 30, 20, 19, 21, 18, 11, 4, 22, 19, 23},   // OH24
    {30, 9, 2, 24, 7, 19, 17, 11, 1, 29, 30, 28, 27, 30, 26},    // OH25
    {29, 8, 16, 6, 23, 20, 18, 3, 10, 30, 27, 29, 29, 28, 25},   // OH26
    {28, 12, 22, 14, 5, 21, 13, 15, 4, 28, 26, 30, 25, 29, 28},  // OH27
    {27, 13, 21, 4, 15, 22, 12, 5, 14, 27, 29, 25, 30, 26, 27},  // OH28
    {26, 10, 3, 18, 20, 23, 6, 16, 8, 25, 28, 26, 26, 27, 30},   // OH29
    {25, 11, 1, 19, 17, 24, 7, 9, 2, 26, 25, 27, 28, 25, 29}     // OH30
};

}

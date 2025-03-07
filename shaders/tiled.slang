/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/// M/N/K values filled out at pipeline creation time
// Specialization consts are not compile time in slang, hard-code their values here :(
static const uint lM = 16;
static const uint lN = 16;
static const uint lK = 16;
static const uint TILE_M = 16;
static const uint TILE_N = 16;
[vk::constant_id(5)] const uint TILE_K = 1;
[vk::constant_id(6)] const uint K = 1;
[vk::constant_id(7)] const uint strideA = 1;
[vk::constant_id(8)] const uint strideB = 1;
[vk::constant_id(9)] const uint strideC = 1;
[vk::constant_id(10)] const uint strideD = 1;
[vk::constant_id(11)] const float alpha = 1.0;
[vk::constant_id(12)] const float beta = 1.0;
static const bool BColMajor = false;


// #defines set on command line:
// A_BITS = 8 or 16 or 32 (bits per component)
// A_TYPE = e.g. float or float16_t
// C_BITS = 8 or 16 or 32 (bits per component)
// C_TYPE = e.g. float or float16_t

struct UniformStruct
{
    A_TYPE* inputA;
    A_TYPE* inputB;
    C_TYPE* inputC;
    C_TYPE* outputO;
}

[vk::binding(0, 0)]
ConstantBuffer<UniformStruct, Std430DataLayout> params;

static const uint C_ROWS = TILE_M / lM;
static const uint C_COLS = TILE_N / lN;

uint coordToOffset(uint i, uint j, uint stride, bool colMajor)
{
    return colMajor ? (stride * j + i) : (stride * i + j);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 workGroupID : SV_GroupID)
{
    uint2 tileID = workGroupID.xy;

    A_TYPE* inputA = params.inputA;
    A_TYPE* inputB = params.inputB;
    C_TYPE* inputC = params.inputC;
    C_TYPE* outputO = params.outputO;

    CoopMat<C_TYPE, CoopMatScope::Subgroup, lM, lN, CoopMatMatrixUse::Accumulator> result[C_ROWS][C_COLS];

    // Initialize result to zero
    [[ForceUnroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[ForceUnroll]] for (uint j = 0; j < C_COLS; ++j) {
            result[i][j] = CoopMat<C_TYPE, CoopMatScope::Subgroup, lM, lN, CoopMatMatrixUse::Accumulator>(0.0);
        }
    }

    // On each iteration, load a row of cooperative matrices from matrix A,
    // load a column of cooperative matrices from matrix B, and multiply all
    // pairs of those matrices.
    for (uint chunkK = 0; chunkK < K; chunkK += TILE_K) {
        CoopMat<A_TYPE, CoopMatScope::Subgroup, lM, lK, CoopMatMatrixUse::A> matA[C_ROWS];
        [[ForceUnroll]] for (uint i = 0; i < C_ROWS; ++i) {
            uint gi = TILE_M * tileID.y + lM * i;
            uint gk = chunkK;
            matA[i] =  CoopMat<A_TYPE, CoopMatScope::Subgroup, lM, lK, CoopMatMatrixUse::A>.load(inputA, coordToOffset(gi, gk, strideA, false), strideA, CoopMatMatrixLayout::RowMajor);
        }

        CoopMat<A_TYPE, CoopMatScope::Subgroup, lK, lN, CoopMatMatrixUse::B> matB;
        [[ForceUnroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gj = TILE_N * tileID.x + lN * j;
            uint gk = chunkK;
            matB = CoopMat<A_TYPE, CoopMatScope::Subgroup, lK, lN, CoopMatMatrixUse::B>.load(inputB, coordToOffset(gk, gj, strideB, BColMajor), strideB, BColMajor ? CoopMatMatrixLayout::ColumnMajor : CoopMatMatrixLayout::RowMajor);
            [[ForceUnroll]] for (uint i = 0; i < C_ROWS; ++i) {
                result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j], CoopMatMatrixOperands::None);
            }
        }
    }

    [[ForceUnroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[ForceUnroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = TILE_M * tileID.y + lM * i;
            uint gj = TILE_N * tileID.x + lN * j;

            // Fetch and add C matrix
            let matC = CoopMat<C_TYPE, CoopMatScope::Subgroup, lM, lN, CoopMatMatrixUse::Accumulator>.load(inputC, coordToOffset(gi, gj, strideC, false), strideC, CoopMatMatrixLayout::RowMajor);

            result[i][j] = C_TYPE(alpha) * result[i][j] + C_TYPE(beta) * matC;
            result[i][j].store(outputO, coordToOffset(gi, gj, strideD, false), strideD, CoopMatMatrixLayout::RowMajor);
        }
    }
}

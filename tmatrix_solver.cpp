// Fast C++20 Orbit-Space Circulant T-Matrix Solver (Tabu+GLS & Tight-Band Reheat SA)
// Usage: ./tmatrix_solver <t> <g> <m1> <m2> <m3> <m4> <secs> <out_json> [threads=1] [mode=sa|tabu|hybrid]
//
// Exact row-sum-preserving moves on the (r-1) uniform-size orbits (size = |U|):
//   1. Same-matrix sign-swap: (i in a with +1, j in a with -1) -> swap signs
//   2. Cross-matrix same-sign swap: (i in a, j in b, same sign) -> swap matrices
//   3. Zero-sum pair transfer: (i in a with +1, j in a with -1) -> move both to b != a
//      with signs (+1,-1) or (-1,+1).
//
// Energy E = sum_{k=0}^{K-1} P[k]^2 (exact integer, target E == 0).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX2__
static inline int64_t hsum_epi32_to_64(__m256i v) {
    // Sign-extend lower 4 and upper 4 int32 lanes to int64 and sum
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m256i lo64 = _mm256_cvtepi32_epi64(lo);
    __m256i hi64 = _mm256_cvtepi32_epi64(hi);
    __m256i s = _mm256_add_epi64(lo64, hi64);
    __m128i s_lo = _mm256_castsi256_si128(s);
    __m128i s_hi = _mm256_extracti128_si256(s, 1);
    __m128i s2 = _mm_add_epi64(s_lo, s_hi);
    return _mm_cvtsi128_si64(s2) + _mm_extract_epi64(s2, 1);
}
#endif

struct Instance {
    int t, g, h, r, K;
    int m_target[4]; // m1, m2, m3, m4 (orbit-sum targets; orbit 0 fixed to mat 0, sgn +1)
    std::vector<std::vector<int>> orbits;
    std::vector<int> orb_id;
    std::vector<int> d_reps;
    int K_pad;
    // MT[(i*r + j)*K_pad + k]
    std::vector<int> MT;
    std::vector<int> diag_sum; // sum_i MT[i][i][k]
};

static Instance build_instance(int t, int g, int m1, int m2, int m3, int m4) {
    Instance inst;
    inst.t = t;
    inst.g = g;
    inst.m_target[0] = m1;
    inst.m_target[1] = m2;
    inst.m_target[2] = m3;
    inst.m_target[3] = m4;

    std::vector<bool> seen(t, false);
    inst.orb_id.assign(t, -1);
    for (int x = 0; x < t; ++x) {
        if (!seen[x]) {
            std::vector<int> orb;
            int cur = x;
            while (!seen[cur]) {
                seen[cur] = true;
                inst.orb_id[cur] = (int)inst.orbits.size();
                orb.push_back(cur);
                cur = (int)((1LL * cur * g) % t);
            }
            inst.orbits.push_back(std::move(orb));
        }
    }
    inst.r = (int)inst.orbits.size();
    inst.h = (inst.r > 1) ? (int)inst.orbits[1].size() : 1;

    // Shift representatives of (+/- <g>) on {1..t-1}
    std::vector<bool> seen_d(t, false);
    seen_d[0] = true;
    for (int d = 1; d < t; ++d) {
        if (!seen_d[d]) {
            inst.d_reps.push_back(d);
            int cur = d;
            while (!seen_d[cur]) {
                seen_d[cur] = true;
                seen_d[(t - cur) % t] = true;
                cur = (int)((1LL * cur * g) % t);
            }
        }
    }
    inst.K = (int)inst.d_reps.size();
    inst.K_pad = ((inst.K + 7) / 8) * 8;

    inst.MT.assign(static_cast<size_t>(inst.r) * inst.r * inst.K_pad, 0);
    inst.diag_sum.assign(inst.K_pad, 0);
    for (int k = 0; k < inst.K; ++k) {
        int d = inst.d_reps[k];
        for (int x = 0; x < t; ++x) {
            int i = inst.orb_id[x];
            int j = inst.orb_id[(x + d) % t];
            if (i == j) {
                inst.MT[(i * inst.r + i) * inst.K_pad + k] += 1;
                inst.diag_sum[k] += 1;
            } else {
                inst.MT[(i * inst.r + j) * inst.K_pad + k] += 1;
                inst.MT[(j * inst.r + i) * inst.K_pad + k] += 1;
            }
        }
    }
    return inst;
}

struct State {
    const Instance* inst;
    int r, K, K_pad;
    std::vector<int> mat; // 0..3
    std::vector<int> sgn; // -1, +1
    std::vector<int> P;   // length K_pad
    // C[m * r * K_pad + i * K_pad + k]
    std::vector<int> C;
    int64_t E;

    explicit State(const Instance* in)
        : inst(in), r(in->r), K(in->K), K_pad(in->K_pad),
          mat(in->r, 0), sgn(in->r, 1), P(in->K_pad, 0), C(4 * in->r * in->K_pad, 0), E(0) {}

    void init_random(std::mt19937_64& rng) {
        // Orbit 0 fixed to mat 0, sgn +1
        mat[0] = 0;
        sgn[0] = 1;
        int e = r - 1;
        // Build a valid random assignment of orbits 1..r-1 matching m_target[0..3]
        std::vector<int> perm(e);
        std::iota(perm.begin(), perm.end(), 1);
        std::shuffle(perm.begin(), perm.end(), rng);

        int pos = 0;
        for (int a = 0; a < 4; ++a) {
            int m = inst->m_target[a];
            int s = (m >= 0) ? 1 : -1;
            int cnt = std::abs(m);
            for (int c = 0; c < cnt; ++c) {
                int o = perm[pos++];
                mat[o] = a;
                sgn[o] = s;
            }
        }
        int rem = e - pos;
        // Remaining orbits come in zero-sum (+1, -1) pairs assigned to random matrices
        for (int c = 0; c < rem; c += 2) {
            int a = (int)(rng() & 3ULL);
            int o1 = perm[pos++];
            int o2 = perm[pos++];
            mat[o1] = a; sgn[o1] = 1;
            mat[o2] = a; sgn[o2] = -1;
        }
        recompute();
    }

    void recompute() {
        std::fill(P.begin(), P.end(), 0);
        std::fill(C.begin(), C.end(), 0);
        for (int k = 0; k < K; ++k) P[k] = inst->diag_sum[k];
        for (int i = 0; i < r; ++i) {
            int mi = mat[i], si = sgn[i];
            for (int j = 0; j < r; ++j) {
                if (i == j) continue;
                int mj = mat[j], sj = sgn[j];
                const int* mt = &inst->MT[(i * r + j) * K_pad];
                int* cm = &C[(mj * r + i) * K_pad];
                if (sj == 1) {
                    for (int k = 0; k < K; ++k) cm[k] += mt[k];
                } else {
                    for (int k = 0; k < K; ++k) cm[k] -= mt[k];
                }
                if (i < j && mi == mj) {
                    int prod = si * sj;
                    if (prod == 1) {
                        for (int k = 0; k < K; ++k) P[k] += mt[k];
                    } else {
                        for (int k = 0; k < K; ++k) P[k] -= mt[k];
                    }
                }
            }
        }
        E = 0;
        for (int k = 0; k < K; ++k) E += 1LL * P[k] * P[k];
    }

    // Helper to add (delta_s * MT[o][l][k]) to C[m][l][k] for all l != o
    inline void update_C_orbit(int m, int o, int delta_s) {
        int* __restrict__ C_m = &C[m * r * K_pad];
        const int* __restrict__ MT_o = &inst->MT[o * r * K_pad];
        if (delta_s == 1) {
            for (int l = 0; l < r; ++l) {
                if (l == o) continue;
                int* __restrict__ dst = C_m + l * K_pad;
                const int* __restrict__ src = MT_o + l * K_pad;
                for (int k = 0; k < K_pad; ++k) dst[k] += src[k];
            }
        } else if (delta_s == -1) {
            for (int l = 0; l < r; ++l) {
                if (l == o) continue;
                int* __restrict__ dst = C_m + l * K_pad;
                const int* __restrict__ src = MT_o + l * K_pad;
                for (int k = 0; k < K_pad; ++k) dst[k] -= src[k];
            }
        } else if (delta_s == 2) {
            for (int l = 0; l < r; ++l) {
                if (l == o) continue;
                int* __restrict__ dst = C_m + l * K_pad;
                const int* __restrict__ src = MT_o + l * K_pad;
                for (int k = 0; k < K_pad; ++k) dst[k] += 2 * src[k];
            }
        } else if (delta_s == -2) {
            for (int l = 0; l < r; ++l) {
                if (l == o) continue;
                int* __restrict__ dst = C_m + l * K_pad;
                const int* __restrict__ src = MT_o + l * K_pad;
                for (int k = 0; k < K_pad; ++k) dst[k] -= 2 * src[k];
            }
        }
    }

    // Move 1: Same-matrix sign swap: mat[i]==mat[j]==a, sgn[i]==+1, sgn[j]==-1
    inline void eval_move1_both(int i, int j, int a, const int* __restrict__ W, int64_t& dW_out, int64_t& dE_out) const {
        const int* __restrict__ Cai = &C[(a * r + i) * K_pad];
        const int* __restrict__ Caj = &C[(a * r + j) * K_pad];
        const int* __restrict__ mij = &inst->MT[(i * r + j) * K_pad];
        const int* __restrict__ pptr = P.data();
        int64_t dE = 0, dW = 0;
#ifdef __AVX2__
        __m256i accE = _mm256_setzero_si256();
        __m256i accW = _mm256_setzero_si256();
        for (int k = 0; k < K_pad; k += 8) {
            __m256i vcai = _mm256_loadu_si256((const __m256i*)(Cai + k));
            __m256i vcaj = _mm256_loadu_si256((const __m256i*)(Caj + k));
            __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
            __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
            __m256i vdp = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_sub_epi32(vcaj, vcai), _mm256_slli_epi32(vmij, 1)), 1);
            __m256i vterm = _mm256_mullo_epi32(vdp, _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp));
            accE = _mm256_add_epi32(accE, vterm);
            if (W) {
                __m256i vw = _mm256_loadu_si256((const __m256i*)(W + k));
                accW = _mm256_add_epi32(accW, _mm256_mullo_epi32(vw, vterm));
            }
        }
        dE = hsum_epi32_to_64(accE);
        dW = W ? hsum_epi32_to_64(accW) : dE;
#else
        for (int k = 0; k < K; ++k) {
            int dp = 2 * (Caj[k] - Cai[k] - 2 * mij[k]);
            int64_t term = 1LL * dp * (2 * pptr[k] + dp);
            dE += term;
            if (W) dW += 1LL * W[k] * term;
        }
        if (!W) dW = dE;
#endif
        dE_out = dE;
        dW_out = dW;
    }

    inline int64_t eval_move1(int i, int j, int a, const int* W = nullptr) const {
        int64_t dW, dE;
        eval_move1_both(i, j, a, W, dW, dE);
        return W ? dW : dE;
    }

    void apply_move1(int i, int j, int a) {
        const int* Cai = &C[(a * r + i) * K_pad];
        const int* Caj = &C[(a * r + j) * K_pad];
        const int* mij = &inst->MT[(i * r + j) * K_pad];
        for (int k = 0; k < K; ++k) {
            int dp = 2 * (Caj[k] - Cai[k] - 2 * mij[k]);
            E += 1LL * dp * (2 * P[k] + dp);
            P[k] += dp;
        }
        sgn[i] = -1;
        sgn[j] = 1;
        update_C_orbit(a, i, -2);
        update_C_orbit(a, j, 2);
    }

    // Move 2: Cross-matrix same-sign swap: mat[i]==a, mat[j]==b, a!=b, sgn[i]==sgn[j]==eps
    inline void eval_move2_both(int i, int j, int a, int b, int eps, const int* __restrict__ W, int64_t& dW_out, int64_t& dE_out) const {
        const int* __restrict__ Cai = &C[(a * r + i) * K_pad];
        const int* __restrict__ Cbi = &C[(b * r + i) * K_pad];
        const int* __restrict__ Caj = &C[(a * r + j) * K_pad];
        const int* __restrict__ Cbj = &C[(b * r + j) * K_pad];
        const int* __restrict__ mij = &inst->MT[(i * r + j) * K_pad];
        const int* __restrict__ pptr = P.data();
        int64_t dE = 0, dW = 0;
#ifdef __AVX2__
        __m256i accE = _mm256_setzero_si256();
        __m256i accW = _mm256_setzero_si256();
        for (int k = 0; k < K_pad; k += 8) {
            __m256i vcai = _mm256_loadu_si256((const __m256i*)(Cai + k));
            __m256i vcbi = _mm256_loadu_si256((const __m256i*)(Cbi + k));
            __m256i vcaj = _mm256_loadu_si256((const __m256i*)(Caj + k));
            __m256i vcbj = _mm256_loadu_si256((const __m256i*)(Cbj + k));
            __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
            __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
            __m256i vdiff = _mm256_add_epi32(_mm256_sub_epi32(vcbi, vcai), _mm256_sub_epi32(vcaj, vcbj));
            if (eps < 0) vdiff = _mm256_sub_epi32(_mm256_setzero_si256(), vdiff);
            __m256i vdp = _mm256_sub_epi32(vdiff, _mm256_slli_epi32(vmij, 1));
            __m256i vterm = _mm256_mullo_epi32(vdp, _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp));
            accE = _mm256_add_epi32(accE, vterm);
            if (W) {
                __m256i vw = _mm256_loadu_si256((const __m256i*)(W + k));
                accW = _mm256_add_epi32(accW, _mm256_mullo_epi32(vw, vterm));
            }
        }
        dE = hsum_epi32_to_64(accE);
        dW = W ? hsum_epi32_to_64(accW) : dE;
#else
        for (int k = 0; k < K; ++k) {
            int dp = eps * (Cbi[k] - Cai[k] + Caj[k] - Cbj[k]) - 2 * mij[k];
            int64_t term = 1LL * dp * (2 * pptr[k] + dp);
            dE += term;
            if (W) dW += 1LL * W[k] * term;
        }
        if (!W) dW = dE;
#endif
        dE_out = dE;
        dW_out = dW;
    }

    inline int64_t eval_move2(int i, int j, int a, int b, int eps, const int* W = nullptr) const {
        int64_t dW, dE;
        eval_move2_both(i, j, a, b, eps, W, dW, dE);
        return W ? dW : dE;
    }

    void apply_move2(int i, int j, int a, int b, int eps) {
        const int* Cai = &C[(a * r + i) * K_pad];
        const int* Cbi = &C[(b * r + i) * K_pad];
        const int* Caj = &C[(a * r + j) * K_pad];
        const int* Cbj = &C[(b * r + j) * K_pad];
        const int* mij = &inst->MT[(i * r + j) * K_pad];
        for (int k = 0; k < K; ++k) {
            int dp = eps * (Cbi[k] - Cai[k] + Caj[k] - Cbj[k]) - 2 * mij[k];
            E += 1LL * dp * (2 * P[k] + dp);
            P[k] += dp;
        }
        mat[i] = b;
        mat[j] = a;
        update_C_orbit(a, i, -eps);
        update_C_orbit(b, i, eps);
        update_C_orbit(b, j, -eps);
        update_C_orbit(a, j, eps);
    }

    // Move 3: Zero-sum pair transfer: mat[i]==mat[j]==a, sgn[i]==+1, sgn[j]==-1 -> move both to b!=a
    // flip_b == 0 -> (+1, -1) in b; flip_b == 1 -> (-1, +1) in b
    inline void eval_move3_both(int i, int j, int a, int b, int flip_b, const int* __restrict__ W, int64_t& dW_out, int64_t& dE_out) const {
        const int* __restrict__ Cai = &C[(a * r + i) * K_pad];
        const int* __restrict__ Caj = &C[(a * r + j) * K_pad];
        const int* __restrict__ Cbi = &C[(b * r + i) * K_pad];
        const int* __restrict__ Cbj = &C[(b * r + j) * K_pad];
        const int* __restrict__ mij = &inst->MT[(i * r + j) * K_pad];
        const int* __restrict__ pptr = P.data();
        int64_t dE = 0, dW = 0;
        int fb = flip_b ? -1 : 1;
#ifdef __AVX2__
        __m256i accE = _mm256_setzero_si256();
        __m256i accW = _mm256_setzero_si256();
        for (int k = 0; k < K_pad; k += 8) {
            __m256i vcai = _mm256_loadu_si256((const __m256i*)(Cai + k));
            __m256i vcaj = _mm256_loadu_si256((const __m256i*)(Caj + k));
            __m256i vcbi = _mm256_loadu_si256((const __m256i*)(Cbi + k));
            __m256i vcbj = _mm256_loadu_si256((const __m256i*)(Cbj + k));
            __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
            __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
            __m256i vb = _mm256_sub_epi32(vcbi, vcbj);
            if (fb < 0) vb = _mm256_sub_epi32(_mm256_setzero_si256(), vb);
            __m256i va = _mm256_sub_epi32(_mm256_sub_epi32(vcaj, vcai), _mm256_slli_epi32(vmij, 1));
            __m256i vdp = _mm256_add_epi32(vb, va);
            __m256i vterm = _mm256_mullo_epi32(vdp, _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp));
            accE = _mm256_add_epi32(accE, vterm);
            if (W) {
                __m256i vw = _mm256_loadu_si256((const __m256i*)(W + k));
                accW = _mm256_add_epi32(accW, _mm256_mullo_epi32(vw, vterm));
            }
        }
        dE = hsum_epi32_to_64(accE);
        dW = W ? hsum_epi32_to_64(accW) : dE;
#else
        for (int k = 0; k < K; ++k) {
            int dp = fb * (Cbi[k] - Cbj[k]) + (Caj[k] - Cai[k] - 2 * mij[k]);
            int64_t term = 1LL * dp * (2 * pptr[k] + dp);
            dE += term;
            if (W) dW += 1LL * W[k] * term;
        }
        if (!W) dW = dE;
#endif
        dE_out = dE;
        dW_out = dW;
    }

    inline int64_t eval_move3(int i, int j, int a, int b, int flip_b, const int* W = nullptr) const {
        int64_t dW, dE;
        eval_move3_both(i, j, a, b, flip_b, W, dW, dE);
        return W ? dW : dE;
    }

    // Fused evaluator for Move 1 + all 6 Move 3 variants (b != a, fb in {0, 1}) for pair (i in pos[a], j in neg[a]).
    // Outputs dW1, dE1 for Move 1, and dW3[b][fb], dE3[b][fb] for b != a.
    inline void eval_move1_and_3_all(int i, int j, int a, const int* __restrict__ W,
                                     int64_t& dW1, int64_t& dE1,
                                     int64_t dW3[4][2], int64_t dE3[4][2]) const {
        const int* __restrict__ Cai = &C[(a * r + i) * K_pad];
        const int* __restrict__ Caj = &C[(a * r + j) * K_pad];
        const int* __restrict__ mij = &inst->MT[(i * r + j) * K_pad];
        const int* __restrict__ pptr = P.data();
        int b_list[3];
        int nb = 0;
        for (int b = 0; b < 4; ++b) if (b != a) b_list[nb++] = b;
        const int* __restrict__ Cbi0 = &C[(b_list[0] * r + i) * K_pad];
        const int* __restrict__ Cbj0 = &C[(b_list[0] * r + j) * K_pad];
        const int* __restrict__ Cbi1 = &C[(b_list[1] * r + i) * K_pad];
        const int* __restrict__ Cbj1 = &C[(b_list[1] * r + j) * K_pad];
        const int* __restrict__ Cbi2 = &C[(b_list[2] * r + i) * K_pad];
        const int* __restrict__ Cbj2 = &C[(b_list[2] * r + j) * K_pad];
#ifdef __AVX2__
        __m256i accE1 = _mm256_setzero_si256(), accW1 = _mm256_setzero_si256();
        __m256i accE3[3][2], accW3[3][2];
        for (int u = 0; u < 3; ++u) {
            accE3[u][0] = accE3[u][1] = _mm256_setzero_si256();
            accW3[u][0] = accW3[u][1] = _mm256_setzero_si256();
        }
        for (int k = 0; k < K_pad; k += 8) {
            __m256i vcai = _mm256_loadu_si256((const __m256i*)(Cai + k));
            __m256i vcaj = _mm256_loadu_si256((const __m256i*)(Caj + k));
            __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
            __m256i vp2 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i*)(pptr + k)), 1);
            __m256i va = _mm256_sub_epi32(_mm256_sub_epi32(vcaj, vcai), _mm256_slli_epi32(vmij, 1));
            __m256i vdp1 = _mm256_slli_epi32(va, 1);
            __m256i vt1 = _mm256_mullo_epi32(vdp1, _mm256_add_epi32(vp2, vdp1));
            accE1 = _mm256_add_epi32(accE1, vt1);

            __m256i vb0 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)(Cbi0 + k)), _mm256_loadu_si256((const __m256i*)(Cbj0 + k)));
            __m256i vb1 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)(Cbi1 + k)), _mm256_loadu_si256((const __m256i*)(Cbj1 + k)));
            __m256i vb2 = _mm256_sub_epi32(_mm256_loadu_si256((const __m256i*)(Cbi2 + k)), _mm256_loadu_si256((const __m256i*)(Cbj2 + k)));
            __m256i vbs[3] = {vb0, vb1, vb2};

            if (!W) {
                for (int u = 0; u < 3; ++u) {
                    __m256i dpp = _mm256_add_epi32(va, vbs[u]);
                    __m256i dpm = _mm256_sub_epi32(va, vbs[u]);
                    accE3[u][0] = _mm256_add_epi32(accE3[u][0], _mm256_mullo_epi32(dpp, _mm256_add_epi32(vp2, dpp)));
                    accE3[u][1] = _mm256_add_epi32(accE3[u][1], _mm256_mullo_epi32(dpm, _mm256_add_epi32(vp2, dpm)));
                }
            } else {
                __m256i vw = _mm256_loadu_si256((const __m256i*)(W + k));
                accW1 = _mm256_add_epi32(accW1, _mm256_mullo_epi32(vw, vt1));
                for (int u = 0; u < 3; ++u) {
                    __m256i dpp = _mm256_add_epi32(va, vbs[u]);
                    __m256i dpm = _mm256_sub_epi32(va, vbs[u]);
                    __m256i tp = _mm256_mullo_epi32(dpp, _mm256_add_epi32(vp2, dpp));
                    __m256i tm = _mm256_mullo_epi32(dpm, _mm256_add_epi32(vp2, dpm));
                    accE3[u][0] = _mm256_add_epi32(accE3[u][0], tp);
                    accE3[u][1] = _mm256_add_epi32(accE3[u][1], tm);
                    accW3[u][0] = _mm256_add_epi32(accW3[u][0], _mm256_mullo_epi32(vw, tp));
                    accW3[u][1] = _mm256_add_epi32(accW3[u][1], _mm256_mullo_epi32(vw, tm));
                }
            }
        }
        dE1 = hsum_epi32_to_64(accE1);
        dW1 = W ? hsum_epi32_to_64(accW1) : dE1;
        for (int u = 0; u < 3; ++u) {
            int b = b_list[u];
            dE3[b][0] = hsum_epi32_to_64(accE3[u][0]);
            dE3[b][1] = hsum_epi32_to_64(accE3[u][1]);
            dW3[b][0] = W ? hsum_epi32_to_64(accW3[u][0]) : dE3[b][0];
            dW3[b][1] = W ? hsum_epi32_to_64(accW3[u][1]) : dE3[b][1];
        }
#else
        eval_move1_both(i, j, a, W, dW1, dE1);
        for (int u = 0; u < 3; ++u) {
            int b = b_list[u];
            eval_move3_both(i, j, a, b, 0, W, dW3[b][0], dE3[b][0]);
            eval_move3_both(i, j, a, b, 1, W, dW3[b][1], dE3[b][1]);
        }
#endif
    }

    void apply_move3(int i, int j, int a, int b, int flip_b) {
        const int* Cai = &C[(a * r + i) * K_pad];
        const int* Caj = &C[(a * r + j) * K_pad];
        const int* Cbi = &C[(b * r + i) * K_pad];
        const int* Cbj = &C[(b * r + j) * K_pad];
        const int* mij = &inst->MT[(i * r + j) * K_pad];
        int fb = flip_b ? -1 : 1;
        for (int k = 0; k < K; ++k) {
            int dp = fb * (Cbi[k] - Cbj[k]) + (Caj[k] - Cai[k] - 2 * mij[k]);
            E += 1LL * dp * (2 * P[k] + dp);
            P[k] += dp;
        }
        mat[i] = b; sgn[i] = fb;
        mat[j] = b; sgn[j] = -fb;
        update_C_orbit(a, i, -1);
        update_C_orbit(a, j, 1);
        update_C_orbit(b, i, fb);
        update_C_orbit(b, j, -fb);
    }
};

static void self_check_deltas(const Instance& inst) {
    std::mt19937_64 rng(42);
    State st(&inst);
    st.init_random(rng);
    for (int step = 0; step < 30; ++step) {
        std::vector<int> pos[4], neg[4];
        for (int i = 1; i < inst.r; ++i) {
            if (st.sgn[i] == 1) pos[st.mat[i]].push_back(i);
            else neg[st.mat[i]].push_back(i);
        }
        int mv = step % 4;
        if (mv == 0) {
            for (int a = 0; a < 4; ++a) {
                if (!pos[a].empty() && !neg[a].empty()) {
                    int i = pos[a][0], j = neg[a][0];
                    int64_t pred = st.E + st.eval_move1(i, j, a);
                    st.apply_move1(i, j, a);
                    int64_t inc = st.E;
                    st.recompute();
                    if (pred != st.E || inc != st.E) {
                        std::cerr << "Move 1 delta mismatch! pred=" << pred << " inc=" << inc << " true=" << st.E << "\n";
                        std::exit(2);
                    }
                    break;
                }
            }
        } else if (mv == 1) {
            bool done = false;
            for (int a = 0; a < 4 && !done; ++a)
                for (int b = a + 1; b < 4 && !done; ++b)
                    if (!pos[a].empty() && !pos[b].empty()) {
                        int i = pos[a][0], j = pos[b][0];
                        int64_t pred = st.E + st.eval_move2(i, j, a, b, 1);
                        st.apply_move2(i, j, a, b, 1);
                        int64_t inc = st.E;
                        st.recompute();
                        if (pred != st.E || inc != st.E) {
                            std::cerr << "Move 2 delta mismatch! pred=" << pred << " inc=" << inc << " true=" << st.E << "\n";
                            std::exit(2);
                        }
                        done = true;
                    }
        } else {
            int fb = (mv == 3) ? 1 : 0;
            for (int a = 0; a < 4; ++a) {
                if (!pos[a].empty() && !neg[a].empty()) {
                    int b = (a + 1) % 4;
                    int i = pos[a][0], j = neg[a][0];
                    int64_t pred = st.E + st.eval_move3(i, j, a, b, fb);
                    st.apply_move3(i, j, a, b, fb);
                    int64_t inc = st.E;
                    st.recompute();
                    if (pred != st.E || inc != st.E) {
                        std::cerr << "Move 3 delta mismatch! pred=" << pred << " inc=" << inc << " true=" << st.E << "\n";
                        std::exit(2);
                    }
                    break;
                }
            }
        }
    }
}

static std::atomic<bool> g_found{false};
static std::atomic<int64_t> g_best_E{1LL << 60};
static std::mutex g_io_mtx;
static bool g_witness_failed = false;

static bool save_witness(const std::string& path, const Instance& inst, const State& st) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"t\": " << inst.t << ",\n";
    out << "  \"g\": " << inst.g << ",\n";
    out << "  \"r\": " << inst.r << ",\n";
    out << "  \"K\": " << inst.K << ",\n";
    out << "  \"orbit_mat\": [";
    for (int i = 0; i < inst.r; ++i) out << st.mat[i] << (i + 1 < inst.r ? ", " : "");
    out << "],\n";
    out << "  \"orbit_sign\": [";
    for (int i = 0; i < inst.r; ++i) out << st.sgn[i] << (i + 1 < inst.r ? ", " : "");
    out << "]\n";
    out << "}\n";
    out.close();
    return static_cast<bool>(out);
}

// Check the expanded sequences directly before treating an incremental zero as a witness.
static bool verify_zero(const State& st) {
    const Instance& inst = *st.inst;
    for (int d = 1; d < inst.t; ++d) {
        int paf = 0;
        for (int x = 0; x < inst.t; ++x) {
            int i = inst.orb_id[x], j = inst.orb_id[(x + d) % inst.t];
            if (st.mat[i] == st.mat[j]) paf += st.sgn[i] * st.sgn[j];
        }
        if (paf != 0) return false;
    }
    return true;
}

static void report_best(const State& st, const std::string& out_path, const char* mode,
                        int tid, double elapsed, uint64_t step) {
    int64_t prev = g_best_E.load(std::memory_order_relaxed);
    if (st.E > 0 && st.E >= prev) return;
    std::lock_guard<std::mutex> lock(g_io_mtx);
    if (g_found.load(std::memory_order_relaxed) || st.E >= g_best_E.load(std::memory_order_relaxed)) return;
    if (st.E == 0 && !verify_zero(st)) {
        std::cerr << "Incremental E=0 failed the all-shift autocorrelation check\n";
        g_witness_failed = true;
        g_found.store(true, std::memory_order_relaxed);
        return;
    }
    g_best_E.store(st.E, std::memory_order_relaxed);
    std::cout << "[" << mode << "-T" << tid << " " << elapsed << "s step=" << step
              << "] new best E=" << st.E;
    if (st.E <= 8) {
        std::cout << " P=[";
        for (int k = 0; k < st.K; ++k) std::cout << st.P[k] << (k + 1 < st.K ? "," : "");
        std::cout << "]";
    }
    std::cout << std::endl;
    if (st.E == 0) {
        g_witness_failed = !save_witness(out_path, *st.inst, st);
        if (g_witness_failed)
            std::cerr << "Could not write T-matrix witness to " << out_path << "\n";
        else
            std::cout << "Verified zero autocorrelation for every nonzero shift. Witness saved to "
                      << out_path << std::endl;
        g_found.store(true, std::memory_order_relaxed);
    }
}

struct BestMove {
    int type = 0; // 1, 2, 3
    int i = 0, j = 0, a = 0, b = 0, p = 0;
    int64_t dW = (1LL << 60);
    int64_t dE = (1LL << 60);
};

static inline void apply_move(State& st, const BestMove& m) {
    if (m.type == 1) st.apply_move1(m.i, m.j, m.a);
    else if (m.type == 2) st.apply_move2(m.i, m.j, m.a, m.b, m.p);
    else if (m.type == 3) st.apply_move3(m.i, m.j, m.a, m.b, m.p);
}

// Steepest 1-move descent + exact 2-move lookahead from top-24 1-moves when E is low.
static bool polish_2move(State& st, int top_k = 24) {
    const Instance* inst = st.inst;
    std::vector<int> pos[4], neg[4];
    bool improved_any = false;
    while (st.E > 0) {
        for (int a = 0; a < 4; ++a) { pos[a].clear(); neg[a].clear(); }
        for (int i = 1; i < inst->r; ++i) {
            if (st.sgn[i] == 1) pos[st.mat[i]].push_back(i);
            else neg[st.mat[i]].push_back(i);
        }
        std::vector<BestMove> cands;
        cands.reserve(512);
        BestMove best1;
        for (int a = 0; a < 4; ++a) {
            for (int i : pos[a]) {
                for (int j : neg[a]) {
                    int64_t dE = st.eval_move1(i, j, a);
                    if (dE < best1.dE) best1 = {1, i, j, a, a, 0, dE, dE};
                    cands.push_back({1, i, j, a, a, 0, dE, dE});
                }
            }
        }
        for (int a = 0; a < 4; ++a) {
            for (int b = a + 1; b < 4; ++b) {
                for (int i : pos[a]) {
                    for (int j : pos[b]) {
                        int64_t dE = st.eval_move2(i, j, a, b, 1);
                        if (dE < best1.dE) best1 = {2, i, j, a, b, 1, dE, dE};
                        cands.push_back({2, i, j, a, b, 1, dE, dE});
                    }
                }
                for (int i : neg[a]) {
                    for (int j : neg[b]) {
                        int64_t dE = st.eval_move2(i, j, a, b, -1);
                        if (dE < best1.dE) best1 = {2, i, j, a, b, -1, dE, dE};
                        cands.push_back({2, i, j, a, b, -1, dE, dE});
                    }
                }
            }
        }
        for (int a = 0; a < 4; ++a) {
            for (int i : pos[a]) {
                for (int j : neg[a]) {
                    for (int b = 0; b < 4; ++b) {
                        if (b == a) continue;
                        for (int fb = 0; fb < 2; ++fb) {
                            int64_t dE = st.eval_move3(i, j, a, b, fb);
                            if (dE < best1.dE) best1 = {3, i, j, a, b, fb, dE, dE};
                            cands.push_back({3, i, j, a, b, fb, dE, dE});
                        }
                    }
                }
            }
        }
        if (best1.dE < 0) {
            apply_move(st, best1);
            improved_any = true;
            continue;
        }
        // Try 2-move lookahead from top_k candidates
        int lim = std::min<int>((int)cands.size(), top_k);
        std::partial_sort(cands.begin(), cands.begin() + lim, cands.end(),
                          [](const BestMove& x, const BestMove& y) { return x.dE < y.dE; });
        int64_t best2_E = st.E;
        BestMove best_m1, best_m2;
        std::vector<int> p2[4], n2[4];
        for (int ci = 0; ci < lim; ++ci) {
            State tmp = st;
            apply_move(tmp, cands[ci]);
            for (int a = 0; a < 4; ++a) { p2[a].clear(); n2[a].clear(); }
            for (int i = 1; i < inst->r; ++i) {
                if (tmp.sgn[i] == 1) p2[tmp.mat[i]].push_back(i);
                else n2[tmp.mat[i]].push_back(i);
            }
            auto check2 = [&](int type, int i, int j, int a, int b, int p, int64_t dE2) {
                if (tmp.E + dE2 < best2_E) {
                    best2_E = tmp.E + dE2;
                    best_m1 = cands[ci];
                    best_m2 = {type, i, j, a, b, p, dE2, dE2};
                }
            };
            for (int a = 0; a < 4; ++a) {
                for (int i : p2[a]) {
                    for (int j : n2[a]) {
                        check2(1, i, j, a, a, 0, tmp.eval_move1(i, j, a));
                        for (int b = 0; b < 4; ++b) {
                            if (b == a) continue;
                            check2(3, i, j, a, b, 0, tmp.eval_move3(i, j, a, b, 0));
                            check2(3, i, j, a, b, 1, tmp.eval_move3(i, j, a, b, 1));
                        }
                    }
                }
            }
            for (int a = 0; a < 4; ++a) {
                for (int b = a + 1; b < 4; ++b) {
                    for (int i : p2[a])
                        for (int j : p2[b])
                            check2(2, i, j, a, b, 1, tmp.eval_move2(i, j, a, b, 1));
                    for (int i : n2[a])
                        for (int j : n2[b])
                            check2(2, i, j, a, b, -1, tmp.eval_move2(i, j, a, b, -1));
                }
            }
            if (best2_E == 0) break;
        }
        if (best2_E < st.E) {
            apply_move(st, best_m1);
            apply_move(st, best_m2);
            improved_any = true;
        } else {
            break;
        }
    }
    return improved_any;
}

// 2-Matrix LNS: hold the other 2 T-matrices fixed and run fast Tabu on orbits in {ma, mb}.
static void lns_pair_tabu(State& st, int ma, int mb, int steps, std::mt19937_64& rng) {
    const Instance* inst = st.inst;
    std::vector<int> tabu(inst->r * 8, 0);
    std::vector<int> pos[2], neg[2];
    int mats[2] = {ma, mb};
    State best_st = st;
    int64_t best_E = st.E;
    int tenure = 3;
    for (int step = 1; step <= steps && st.E > 0; ++step) {
        pos[0].clear(); neg[0].clear(); pos[1].clear(); neg[1].clear();
        for (int i = 1; i < inst->r; ++i) {
            int m = st.mat[i];
            if (m == ma) {
                if (st.sgn[i] == 1) pos[0].push_back(i); else neg[0].push_back(i);
            } else if (m == mb) {
                if (st.sgn[i] == 1) pos[1].push_back(i); else neg[1].push_back(i);
            }
        }
        BestMove best;
        uint64_t ties = 0;
        auto consider = [&](int type, int i, int j, int a, int b, int p,
                            int ni_m, int ni_s, int nj_m, int nj_s, int64_t dE) {
            bool is_t = (tabu[i * 8 + ni_m * 2 + (ni_s > 0)] > step) ||
                        (tabu[j * 8 + nj_m * 2 + (nj_s > 0)] > step);
            if (is_t && st.E + dE >= best_E) return;
            if (dE < best.dE || (dE == best.dE && (rng() % (++ties + 1) == 0))) {
                if (dE < best.dE) ties = 0;
                best = {type, i, j, a, b, p, dE, dE};
            }
        };
        for (int u = 0; u < 2; ++u) {
            int a = mats[u], b = mats[1 - u];
            for (int i : pos[u]) {
                for (int j : neg[u]) {
                    consider(1, i, j, a, a, 0, a, -1, a, 1, st.eval_move1(i, j, a));
                    consider(3, i, j, a, b, 0, b, 1, b, -1, st.eval_move3(i, j, a, b, 0));
                    consider(3, i, j, a, b, 1, b, -1, b, 1, st.eval_move3(i, j, a, b, 1));
                }
            }
        }
        for (int i : pos[0])
            for (int j : pos[1])
                consider(2, i, j, ma, mb, 1, mb, 1, ma, 1, st.eval_move2(i, j, ma, mb, 1));
        for (int i : neg[0])
            for (int j : neg[1])
                consider(2, i, j, ma, mb, -1, mb, -1, ma, -1, st.eval_move2(i, j, ma, mb, -1));
        if (best.type == 0) break;
        int omi = st.mat[best.i], osi = st.sgn[best.i], omj = st.mat[best.j], osj = st.sgn[best.j];
        apply_move(st, best);
        int tv = tenure + (int)(rng() % 3);
        tabu[best.i * 8 + omi * 2 + (osi > 0)] = step + tv;
        tabu[best.j * 8 + omj * 2 + (osj > 0)] = step + tv;
        if (st.E < best_E) {
            best_E = st.E;
            best_st = st;
        }
    }
    if (best_E < st.E) st = std::move(best_st);
}

static uint64_t hash_state(const State& st) {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    for (int i = 1; i < st.r; ++i) {
        uint64_t v = (uint64_t)(st.mat[i] * 2 + (st.sgn[i] > 0 ? 1 : 0));
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

static void worker_tabu(int tid, const Instance* inst, double max_secs, const std::string& out_path) {
    auto t_start = std::chrono::steady_clock::now();
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL ^ ((uint64_t)(tid + 1) * 0xbf58476d1ce4e5b9ULL));

    State st(inst);
    st.init_random(rng);
    State best_st = st;
    // Assignment-based tabu: tabu[orbit * 8 + mat * 2 + (sgn > 0 ? 1 : 0)]
    std::vector<int> tabu(inst->r * 8, 0);
    std::vector<int> W(inst->K_pad, 1);
    for (int k = inst->K; k < inst->K_pad; ++k) W[k] = 0;

    int64_t basin_best = st.E;
    report_best(st, out_path, "tabu", tid, 0, 0);
    bool weighted = false;
    int iter = 0;
    int stall = 0;
    int kicks_since_improve = 0;
    const int tenure = std::max(4, inst->r / 8);
    const int kick_limit = std::max(200, inst->r * 6);
    const int64_t polish_thresh = (inst->K <= 16) ? 8 : 32;
    const int64_t deep_polish_thresh = (inst->K <= 16) ? 4 : 16;
    const int64_t diag_lo1 = (inst->K <= 16) ? 4 : 32;
    const int64_t diag_lo2 = (inst->K <= 16) ? 2 : 28;
    std::unordered_set<uint64_t> seen_lo1, seen_lo2;

    std::vector<int> pos[4], neg[4];

    auto is_assign_tabu = [&](int o, int m, int s) -> bool {
        return tabu[o * 8 + m * 2 + (s > 0 ? 1 : 0)] > iter;
    };
    auto set_assign_tabu = [&](int o, int m, int s, int until) {
        tabu[o * 8 + m * 2 + (s > 0 ? 1 : 0)] = until;
    };

    while (!g_found.load(std::memory_order_relaxed)) {
        if ((iter & 255) == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            if (el >= max_secs) break;
        }
        ++iter;

        for (int a = 0; a < 4; ++a) { pos[a].clear(); neg[a].clear(); }
        for (int i = 1; i < inst->r; ++i) {
            if (st.sgn[i] == 1) pos[st.mat[i]].push_back(i);
            else neg[st.mat[i]].push_back(i);
        }

        BestMove best;
        uint64_t tie_cnt = 0;
        const int* wptr = weighted ? W.data() : nullptr;

        auto consider = [&](int type, int i, int j, int a, int b, int p,
                            int ni_m, int ni_s, int nj_m, int nj_s, int64_t dW, int64_t dE) {
            bool is_tabu = is_assign_tabu(i, ni_m, ni_s) || is_assign_tabu(j, nj_m, nj_s);
            if (is_tabu && (st.E + dE >= basin_best)) return;
            if (dW < best.dW || (dW == best.dW && (dE < best.dE || (dE == best.dE && (rng() % (++tie_cnt + 1) == 0))))) {
                if (dW < best.dW || dE < best.dE) tie_cnt = 0;
                best = {type, i, j, a, b, p, dW, dE};
            }
        };

        // Fused Move 1 + Move 3 evaluation
        int64_t dW3[4][2], dE3[4][2];
        for (int a = 0; a < 4; ++a) {
            for (int i : pos[a]) {
                for (int j : neg[a]) {
                    int64_t dW1, dE1;
                    st.eval_move1_and_3_all(i, j, a, wptr, dW1, dE1, dW3, dE3);
                    consider(1, i, j, a, a, 0, a, -1, a, 1, dW1, dE1);
                    for (int b = 0; b < 4; ++b) {
                        if (b == a) continue;
                        consider(3, i, j, a, b, 0, b, 1, b, -1, dW3[b][0], dE3[b][0]);
                        consider(3, i, j, a, b, 1, b, -1, b, 1, dW3[b][1], dE3[b][1]);
                    }
                }
            }
        }
        // Move 2: cross-matrix same-sign swaps
        for (int a = 0; a < 4; ++a) {
            for (int b = a + 1; b < 4; ++b) {
                for (int i : pos[a]) {
                    for (int j : pos[b]) {
                        int64_t dW, dE;
                        st.eval_move2_both(i, j, a, b, 1, wptr, dW, dE);
                        consider(2, i, j, a, b, 1, b, 1, a, 1, dW, dE);
                    }
                }
                for (int i : neg[a]) {
                    for (int j : neg[b]) {
                        int64_t dW, dE;
                        st.eval_move2_both(i, j, a, b, -1, wptr, dW, dE);
                        consider(2, i, j, a, b, -1, b, -1, a, -1, dW, dE);
                    }
                }
            }
        }

        if (best.type == 0) {
            std::fill(tabu.begin(), tabu.end(), 0);
            continue;
        }

        int old_mi = st.mat[best.i], old_si = st.sgn[best.i];
        int old_mj = st.mat[best.j], old_sj = st.sgn[best.j];
        apply_move(st, best);
        int t_val = tenure + (int)(rng() % 4);
        set_assign_tabu(best.i, old_mi, old_si, iter + t_val);
        set_assign_tabu(best.j, old_mj, old_sj, iter + t_val);

        if (st.E <= diag_lo1) seen_lo1.insert(hash_state(st));
        if (st.E <= diag_lo2) seen_lo2.insert(hash_state(st));

        if (st.E < basin_best) {
            if (st.E > 0 && st.E <= polish_thresh) {
                polish_2move(st, st.E <= deep_polish_thresh ? 512 : 24);
                if (st.E <= diag_lo1) seen_lo1.insert(hash_state(st));
                if (st.E <= diag_lo2) seen_lo2.insert(hash_state(st));
            }
            basin_best = st.E;
            stall = 0;
            if (st.E <= best_st.E) {
                if (st.E < best_st.E) kicks_since_improve = 0;
                best_st = st;
            }
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            report_best(st, out_path, "tabu", tid, el, iter);
            if (st.E == 0) return;
        } else {
            ++stall;
            if (stall >= kick_limit) {
                for (int k = 0; k < inst->K; ++k) W[k] = 1;
                weighted = false;
                ++kicks_since_improve;
                if (kicks_since_improve >= 4 || rng() % 2 == 0) {
                    st.init_random(rng);
                    best_st = st;
                    basin_best = st.E;
                    kicks_since_improve = 0;
                } else if (rng() % 2 == 0) {
                    // 2-Matrix LNS escape from best_st: hold 2 matrices fixed, re-optimize the other 2!
                    st = best_st;
                    int ma = (int)(rng() & 3ULL);
                    int mb = (ma + 1 + (int)(rng() % 3)) & 3;
                    lns_pair_tabu(st, ma, mb, 60, rng);
                    if (st.E > 0 && st.E <= deep_polish_thresh) polish_2move(st, 512);
                    if (st.E <= diag_lo1) seen_lo1.insert(hash_state(st));
                    if (st.E <= diag_lo2) seen_lo2.insert(hash_state(st));
                    if (st.E < best_st.E) {
                        best_st = st;
                        kicks_since_improve = 0;
                        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                        report_best(st, out_path, "tabu-lns", tid, el, iter);
                        if (st.E == 0) return;
                    }
                    basin_best = st.E;
                } else {
                    // ILS kick from best_st: 3..6 random valid moves
                    st = best_st;
                    int n_kicks = 3 + (int)(rng() % 4);
                    for (int kk = 0; kk < n_kicks; ++kk) {
                        for (int a = 0; a < 4; ++a) { pos[a].clear(); neg[a].clear(); }
                        for (int i = 1; i < inst->r; ++i) {
                            if (st.sgn[i] == 1) pos[st.mat[i]].push_back(i);
                            else neg[st.mat[i]].push_back(i);
                        }
                        int a = (int)(rng() & 3ULL);
                        int b = (a + 1 + (int)(rng() % 3)) & 3;
                        int rmode = (int)(rng() % 3);
                        if (rmode == 0 && !pos[a].empty() && !pos[b].empty()) {
                            st.apply_move2(pos[a][rng() % pos[a].size()], pos[b][rng() % pos[b].size()], a, b, 1);
                        } else if (rmode == 1 && !neg[a].empty() && !neg[b].empty()) {
                            st.apply_move2(neg[a][rng() % neg[a].size()], neg[b][rng() % neg[b].size()], a, b, -1);
                        } else if (!pos[a].empty() && !neg[a].empty()) {
                            if (rng() & 1) st.apply_move3(pos[a][rng() % pos[a].size()], neg[a][rng() % neg[a].size()], a, b, (int)(rng() & 1));
                            else st.apply_move1(pos[a][rng() % pos[a].size()], neg[a][rng() % neg[a].size()], a);
                        }
                    }
                    basin_best = st.E;
                }
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                report_best(st, out_path, "tabu", tid, el, iter);
                std::fill(tabu.begin(), tabu.end(), 0);
                stall = 0;
            }
        }
    }
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::lock_guard<std::mutex> lock(g_io_mtx);
    std::cout << "[tabu-T" << tid << " stats] iters=" << iter
              << " (" << (el > 0 ? iter / el : 0.0) << " iters/s)"
              << ", distinct E<=" << diag_lo1 << ": " << seen_lo1.size()
              << ", distinct E<=" << diag_lo2 << ": " << seen_lo2.size() << std::endl;
}

static void worker_sa(int tid, const Instance* inst, double max_secs, const std::string& out_path) {
    auto t_start = std::chrono::steady_clock::now();
    std::mt19937_64 rng(0x517cc1b727220a95ULL ^ ((uint64_t)(tid + 1) * 0x94d049bb133111ebULL));
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    State st(inst);
    st.init_random(rng);

    // Maintain live lists pos[4], neg[4] and index-in-list for O(1) random move sampling!
    std::vector<int> pos_list[4], neg_list[4];
    int pos_sz[4], neg_sz[4];
    std::vector<int> idx_in_list(inst->r);
    for (int a = 0; a < 4; ++a) {
        pos_list[a].resize(inst->r);
        neg_list[a].resize(inst->r);
    }

    auto rebuild_lists = [&]() {
        for (int a = 0; a < 4; ++a) pos_sz[a] = neg_sz[a] = 0;
        for (int i = 1; i < inst->r; ++i) {
            int a = st.mat[i];
            if (st.sgn[i] == 1) {
                idx_in_list[i] = pos_sz[a];
                pos_list[a][pos_sz[a]++] = i;
            } else {
                idx_in_list[i] = neg_sz[a];
                neg_list[a][neg_sz[a]++] = i;
            }
        }
    };
    auto remove_pos = [&](int a, int i) {
        int p = idx_in_list[i];
        int last = pos_list[a][--pos_sz[a]];
        pos_list[a][p] = last;
        idx_in_list[last] = p;
    };
    auto add_pos = [&](int a, int i) {
        idx_in_list[i] = pos_sz[a];
        pos_list[a][pos_sz[a]++] = i;
    };
    auto remove_neg = [&](int a, int i) {
        int p = idx_in_list[i];
        int last = neg_list[a][--neg_sz[a]];
        neg_list[a][p] = last;
        idx_in_list[last] = p;
    };
    auto add_neg = [&](int a, int i) {
        idx_in_list[i] = neg_sz[a];
        neg_list[a][neg_sz[a]++] = i;
    };

    rebuild_lists();

    // Tight-band reheat SA parameters scaled to K
    double T_hi = std::max(5.0, 0.45 * inst->K);
    double T_lo = 0.8;
    const uint64_t cycle_len = 600000ULL;
    double alpha = std::pow(T_lo / T_hi, 1.0 / (double)cycle_len);
    double T = T_hi;
    uint64_t step = 0;
    int64_t local_best = st.E;
    const int64_t polish_thresh = (inst->K <= 16) ? 14 : 44;
    report_best(st, out_path, "sa", tid, 0, 0);

    while (!g_found.load(std::memory_order_relaxed)) {
        if ((step & 65535ULL) == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            if (el >= max_secs) break;
        }
        ++step;
        T *= alpha;
        if (T < T_lo) {
            // Reheat! Periodically full-restart, otherwise tight reheat
            if ((step / cycle_len) % 8 == 0) {
                st.init_random(rng);
                rebuild_lists();
                local_best = st.E;
                T = T_hi;
            } else {
                T = T_hi * (0.4 + 0.6 * u01(rng));
            }
        }

        uint64_t rval = rng();
        int mtype = (int)(rval % 10ULL);
        rval >>= 4;

        if (mtype < 3) {
            // Move 1: same-matrix sign swap
            int a = (int)(rval & 3ULL);
            if (pos_sz[a] == 0 || neg_sz[a] == 0) continue;
            int i = pos_list[a][(rval >> 2) % pos_sz[a]];
            int j = neg_list[a][(rval >> 12) % neg_sz[a]];
            int64_t dE = st.eval_move1(i, j, a);
            if (dE <= 0 || u01(rng) < std::exp(-(double)dE / T)) {
                st.apply_move1(i, j, a);
                remove_pos(a, i); add_neg(a, i);
                remove_neg(a, j); add_pos(a, j);
            } else continue;
        } else if (mtype < 7) {
            // Move 2: cross-matrix same-sign swap
            int a = (int)(rval & 3ULL);
            int b = (a + 1 + (int)((rval >> 2) % 3ULL)) & 3;
            int is_pos = (int)((rval >> 4) & 1ULL);
            if (is_pos) {
                if (pos_sz[a] == 0 || pos_sz[b] == 0) continue;
                int i = pos_list[a][(rval >> 5) % pos_sz[a]];
                int j = pos_list[b][(rval >> 15) % pos_sz[b]];
                int64_t dE = st.eval_move2(i, j, a, b, 1);
                if (dE <= 0 || u01(rng) < std::exp(-(double)dE / T)) {
                    st.apply_move2(i, j, a, b, 1);
                    remove_pos(a, i); add_pos(b, i);
                    remove_pos(b, j); add_pos(a, j);
                } else continue;
            } else {
                if (neg_sz[a] == 0 || neg_sz[b] == 0) continue;
                int i = neg_list[a][(rval >> 5) % neg_sz[a]];
                int j = neg_list[b][(rval >> 15) % neg_sz[b]];
                int64_t dE = st.eval_move2(i, j, a, b, -1);
                if (dE <= 0 || u01(rng) < std::exp(-(double)dE / T)) {
                    st.apply_move2(i, j, a, b, -1);
                    remove_neg(a, i); add_neg(b, i);
                    remove_neg(b, j); add_neg(a, j);
                } else continue;
            }
        } else {
            // Move 3: zero-sum pair transfer
            int a = (int)(rval & 3ULL);
            if (pos_sz[a] == 0 || neg_sz[a] == 0) continue;
            int b = (a + 1 + (int)((rval >> 2) % 3ULL)) & 3;
            int fb = (int)((rval >> 4) & 1ULL);
            int i = pos_list[a][(rval >> 5) % pos_sz[a]];
            int j = neg_list[a][(rval >> 15) % neg_sz[a]];
            int64_t dE = st.eval_move3(i, j, a, b, fb);
            if (dE <= 0 || u01(rng) < std::exp(-(double)dE / T)) {
                st.apply_move3(i, j, a, b, fb);
                remove_pos(a, i);
                remove_neg(a, j);
                if (!fb) {
                    add_pos(b, i);
                    add_neg(b, j);
                } else {
                    add_neg(b, i);
                    add_pos(b, j);
                }
            } else continue;
        }

        if (st.E < local_best) {
            if (st.E > 0 && st.E <= polish_thresh) {
                State pol = st;
                if (polish_2move(pol, 20) && pol.E < st.E) {
                    st = std::move(pol);
                    rebuild_lists();
                }
            }
            local_best = st.E;
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            report_best(st, out_path, "sa", tid, el, step);
            if (st.E == 0) return;
        }
    }
}

struct PtReplica {
    State st;
    std::vector<int> pos_list[4], neg_list[4];
    int pos_sz[4], neg_sz[4];
    std::vector<int> idx_in_list;
    uint64_t last_polish_step = 0;

    explicit PtReplica(const Instance* inst) : st(inst), idx_in_list(inst->r, 0) {
        for (int a = 0; a < 4; ++a) {
            pos_list[a].resize(inst->r);
            neg_list[a].resize(inst->r);
            pos_sz[a] = neg_sz[a] = 0;
        }
    }
    void rebuild_lists() {
        for (int a = 0; a < 4; ++a) pos_sz[a] = neg_sz[a] = 0;
        for (int i = 1; i < st.r; ++i) {
            int a = st.mat[i];
            if (st.sgn[i] == 1) {
                idx_in_list[i] = pos_sz[a];
                pos_list[a][pos_sz[a]++] = i;
            } else {
                idx_in_list[i] = neg_sz[a];
                neg_list[a][neg_sz[a]++] = i;
            }
        }
    }
    inline void remove_pos(int a, int i) {
        int p = idx_in_list[i];
        int last = pos_list[a][--pos_sz[a]];
        pos_list[a][p] = last;
        idx_in_list[last] = p;
    }
    inline void add_pos(int a, int i) {
        idx_in_list[i] = pos_sz[a];
        pos_list[a][pos_sz[a]++] = i;
    }
    inline void remove_neg(int a, int i) {
        int p = idx_in_list[i];
        int last = neg_list[a][--neg_sz[a]];
        neg_list[a][p] = last;
        idx_in_list[last] = p;
    }
    inline void add_neg(int a, int i) {
        idx_in_list[i] = neg_sz[a];
        neg_list[a][neg_sz[a]++] = i;
    }
};

static void worker_pt(int tid, const Instance* inst, double max_secs, const std::string& out_path) {
    auto t_start = std::chrono::steady_clock::now();
    std::mt19937_64 rng(0xd1b54a32d192ed03ULL ^ ((uint64_t)(tid + 1) * 0x94d049bb133111ebULL));
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    constexpr int R = 12;
    const double T_min = (inst->K <= 16) ? 1.2 : 2.0;
    const double T_max = (inst->K <= 16) ? 20.0 : 36.0;
    double temps[R];
    for (int r = 0; r < R; ++r) {
        temps[r] = T_min * std::pow(T_max / T_min, (double)r / (double)(R - 1));
    }

    std::vector<PtReplica> reps;
    reps.reserve(R);
    int order[R]; // order[slot] = replica index at temperature temps[slot]
    for (int r = 0; r < R; ++r) {
        reps.emplace_back(inst);
        reps.back().st.init_random(rng);
        reps.back().rebuild_lists();
        order[r] = r;
        report_best(reps.back().st, out_path, "pt", tid, 0, 0);
    }

    const int64_t polish_thresh = (inst->K <= 16) ? 6 : 28;
    constexpr uint64_t sweep_steps = 2048;
    uint64_t total_steps = 0;
    uint64_t sweep = 0;

    while (!g_found.load(std::memory_order_relaxed)) {
        if ((sweep & 7ULL) == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            if (el >= max_secs) break;
        }
        ++sweep;

        for (int slot = 0; slot < R && !g_found.load(std::memory_order_relaxed); ++slot) {
            PtReplica& rep = reps[order[slot]];
            State& st = rep.st;
            const double invT = 1.0 / temps[slot];
            for (uint64_t s = 0; s < sweep_steps; ++s) {
                ++total_steps;
                uint64_t rval = rng();
                int mtype = (int)(rval % 10ULL);
                rval >>= 4;
                if (mtype < 3) {
                    int a = (int)(rval & 3ULL);
                    if (rep.pos_sz[a] == 0 || rep.neg_sz[a] == 0) continue;
                    int i = rep.pos_list[a][(rval >> 2) % rep.pos_sz[a]];
                    int j = rep.neg_list[a][(rval >> 12) % rep.neg_sz[a]];
                    int64_t dE = st.eval_move1(i, j, a);
                    if (dE <= 0 || u01(rng) < std::exp(-(double)dE * invT)) {
                        st.apply_move1(i, j, a);
                        rep.remove_pos(a, i); rep.add_neg(a, i);
                        rep.remove_neg(a, j); rep.add_pos(a, j);
                    } else continue;
                } else if (mtype < 7) {
                    int a = (int)(rval & 3ULL);
                    int b = (a + 1 + (int)((rval >> 2) % 3ULL)) & 3;
                    int is_pos = (int)((rval >> 4) & 1ULL);
                    if (is_pos) {
                        if (rep.pos_sz[a] == 0 || rep.pos_sz[b] == 0) continue;
                        int i = rep.pos_list[a][(rval >> 5) % rep.pos_sz[a]];
                        int j = rep.pos_list[b][(rval >> 15) % rep.pos_sz[b]];
                        int64_t dE = st.eval_move2(i, j, a, b, 1);
                        if (dE <= 0 || u01(rng) < std::exp(-(double)dE * invT)) {
                            st.apply_move2(i, j, a, b, 1);
                            rep.remove_pos(a, i); rep.add_pos(b, i);
                            rep.remove_pos(b, j); rep.add_pos(a, j);
                        } else continue;
                    } else {
                        if (rep.neg_sz[a] == 0 || rep.neg_sz[b] == 0) continue;
                        int i = rep.neg_list[a][(rval >> 5) % rep.neg_sz[a]];
                        int j = rep.neg_list[b][(rval >> 15) % rep.neg_sz[b]];
                        int64_t dE = st.eval_move2(i, j, a, b, -1);
                        if (dE <= 0 || u01(rng) < std::exp(-(double)dE * invT)) {
                            st.apply_move2(i, j, a, b, -1);
                            rep.remove_neg(a, i); rep.add_neg(b, i);
                            rep.remove_neg(b, j); rep.add_neg(a, j);
                        } else continue;
                    }
                } else {
                    int a = (int)(rval & 3ULL);
                    if (rep.pos_sz[a] == 0 || rep.neg_sz[a] == 0) continue;
                    int b = (a + 1 + (int)((rval >> 2) % 3ULL)) & 3;
                    int fb = (int)((rval >> 4) & 1ULL);
                    int i = rep.pos_list[a][(rval >> 5) % rep.pos_sz[a]];
                    int j = rep.neg_list[a][(rval >> 15) % rep.neg_sz[a]];
                    int64_t dE = st.eval_move3(i, j, a, b, fb);
                    if (dE <= 0 || u01(rng) < std::exp(-(double)dE * invT)) {
                        st.apply_move3(i, j, a, b, fb);
                        rep.remove_pos(a, i);
                        rep.remove_neg(a, j);
                        if (!fb) {
                            rep.add_pos(b, i);
                            rep.add_neg(b, j);
                        } else {
                            rep.add_neg(b, i);
                            rep.add_pos(b, j);
                        }
                    } else continue;
                }

                if (st.E < g_best_E.load(std::memory_order_relaxed)) {
                    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                    report_best(st, out_path, "pt", tid, el, total_steps);
                    if (st.E == 0) return;
                }
                if (st.E > 0 && st.E <= polish_thresh && (total_steps - rep.last_polish_step) >= 80000ULL) {
                    rep.last_polish_step = total_steps;
                    State pol = st;
                    if (polish_2move(pol, 24) && pol.E < g_best_E.load(std::memory_order_relaxed)) {
                        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                        report_best(pol, out_path, "pt-2m", tid, el, total_steps);
                        if (pol.E == 0) return;
                    }
                }
            }
        }

        // Metropolis replica exchange on alternating parity pairs
        int parity = (int)(sweep & 1ULL);
        for (int slot = parity; slot + 1 < R; slot += 2) {
            int r1 = order[slot];
            int r2 = order[slot + 1];
            double delta = (double)(reps[r1].st.E - reps[r2].st.E) * (1.0 / temps[slot] - 1.0 / temps[slot + 1]);
            if (delta >= 0.0 || u01(rng) < std::exp(delta)) {
                std::swap(order[slot], order[slot + 1]);
            }
        }
    }
}

int main(int argc, char** argv) try {
    if (argc < 9) {
        std::cerr << "Usage: ./tmatrix_solver <t> <g> <m1> <m2> <m3> <m4> <secs> <out_json> [threads=1] [mode=sa|tabu|pt|hybrid]\n";
        return 1;
    }
    int t = std::stoi(argv[1]);
    int g = std::stoi(argv[2]);
    int m1 = std::stoi(argv[3]);
    int m2 = std::stoi(argv[4]);
    int m3 = std::stoi(argv[5]);
    int m4 = std::stoi(argv[6]);
    double secs = std::stod(argv[7]);
    std::string out_path = argv[8];
    int threads = (argc >= 10) ? std::stoi(argv[9]) : 1;
    std::string mode = (argc >= 11) ? argv[10] : "sa";

    if (t < 1 || g < 0 || g >= t || std::gcd(t, g) != 1) {
        std::cerr << "Expected positive t and a unit g with 0 <= g < t\n";
        return 1;
    }
    if (!std::isfinite(secs) || secs <= 0 || threads <= 0 ||
        (mode != "sa" && mode != "tabu" && mode != "pt" && mode != "hybrid")) {
        std::cerr << "Expected positive secs and threads, and mode sa, tabu, pt, or hybrid\n";
        return 1;
    }
    Instance inst = build_instance(t, g, m1, m2, m3, m4);
    if (!std::all_of(inst.orbits.begin() + 1, inst.orbits.end(),
                     [&](const auto& orbit) { return orbit.size() == static_cast<size_t>(inst.h); })) {
        std::cerr << "All nonzero orbits must have the same size\n";
        return 1;
    }
    int64_t remaining = inst.r - 1;
    for (int m : inst.m_target) remaining -= std::abs(static_cast<int64_t>(m));
    if (remaining < 0 || remaining % 2 != 0) {
        std::cerr << "Orbit-sum targets cannot be achieved with the available orbits\n";
        return 1;
    }
    self_check_deltas(inst);
    std::cout << "tmatrix_solver: t=" << t << " g=" << g << " |U|=" << inst.h
              << " r=" << inst.r << " K=" << inst.K
              << " m=(" << m1 << "," << m2 << "," << m3 << "," << m4 << ")"
              << " threads=" << threads << " secs=" << secs << " mode=" << mode << std::endl;

    std::vector<std::thread> pool;
    for (int tid = 0; tid < threads; ++tid) {
        if (mode == "pt" || (mode == "hybrid" && tid % 2 == 0))
            pool.emplace_back(worker_pt, tid, &inst, secs, out_path);
        else if (mode == "tabu" || (mode == "hybrid" && tid % 2 == 1))
            pool.emplace_back(worker_tabu, tid, &inst, secs, out_path);
        else
            pool.emplace_back(worker_sa, tid, &inst, secs, out_path);
    }
    for (auto& th : pool) th.join();

    if (!g_found.load()) {
        std::cout << "Finished " << secs << "s without E=0. Global best E=" << g_best_E.load() << std::endl;
    }
    return g_witness_failed ? 2 : 0;
} catch (const std::exception& e) {
    std::cerr << "tmatrix_solver: " << e.what() << '\n';
    return 1;
}

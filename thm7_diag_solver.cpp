// thm7_diag_solver.cpp — Standalone C++20 AVX2 Orbit-Space Solver for
// Diagonally-Coupled Theorem 7 (b = 12, k = 4, G = Z_2^2 x Z_s,
// tau(c, x) = (sigma_3(c), g * x mod s) with 3 | ord_s(g)).
//
// Builds exact orbits & shift-correlation tensor MT[i][j][k], enforces exact
// coset sums (a_f, b_f) via exact DP initialization and sum-preserving moves
// (same-size swaps, 3-orbit triples, 1-sequence DP-LNS), and uses AVX2-vectorized
// Tabu + Guided Local Search (GLS) + ILS restarts.
//
// Compile:
//   g++ -std=c++20 -O3 -march=native -Wall -Wextra -pthread thm7_diag_solver.cpp -o thm7_diag_solver
//
// Usage:
//   ./thm7_diag_solver <s> <g> <a0,b0:a1,b1:a2,b2:a3,b3> <seconds> <out_json> <threads> [--seed=N] [--save_best=path]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static constexpr int MAX_K_PAD = 64; // Padded multiple of 8 for AVX2 int32

struct Problem {
    int s = 0;
    int g = 0;
    int n = 0;
    int v = 0;
    int r0 = 0;
    int r1 = 0;
    int r = 0;
    int K_S = 0;
    int K_NS = 0;
    int K = 0;
    int K_pad = 0;
    int a_tgt[4] = {0, 0, 0, 0};
    int b_tgt[4] = {0, 0, 0, 0};

    std::vector<std::vector<int>> orbs; // element indices c*s + x in G
    std::vector<int> orb_coset;         // 0 for coset 0, 1 for nonzero cosets
    std::vector<int> orb_w;             // weight on a single coset (size on coset 0, or size/3 on coset 1)
    std::vector<int> oid;               // length 4*s
    std::vector<std::pair<int, int>> shift_reps; // (cd, xd)
    std::vector<int32_t> target_paf;    // length K_pad (-12 on S\{0}, +4 on G\S, 0 on pad)

    // Flat padded tensor MT[i][j][k] (size r * r * K_pad), 32-byte aligned
    std::vector<int32_t> MT_flat;
    inline const int32_t* mt_ptr(int i, int j) const {
        return &MT_flat[(static_cast<size_t>(i) * r + j) * K_pad];
    }

    // Precomputed same-size swap pairs (u < v) within same coset (0 or 1)
    std::vector<std::pair<int, int>> swap_pairs;
    // Precomputed 3-orbit triples (a, b, c) with w[a] == w[b] + w[c], b < c, same coset
    struct Triple { int a, b, c; };
    std::vector<Triple> triples;

    // Exact DP tables for sampling coset 0 (size r0) and coset 1 (size r1) with given sum
    // dp0[i][sum + offset0] = count (capped at 1e18 for safe sampling)
    int sum_max0 = 0, sum_max1 = 0;
    std::vector<std::vector<uint64_t>> dp0, dp1;

    static constexpr uint64_t DP_CAP = 1000000000000000000ULL;

    void build(int s_in, int g_in, const int a_in[4], const int b_in[4]) {
        s = s_in;
        g = g_in;
        v = 4 * s;
        n = 16 * s + 12;
        for (int f = 0; f < 4; ++f) {
            a_tgt[f] = a_in[f];
            b_tgt[f] = b_in[f];
        }

        // Build orbits of tau(c, x) = (sig3[c], g*x % s)
        static constexpr int sig3[4] = {0, 2, 3, 1};
        oid.assign(v, -1);
        orbs.clear();
        orb_coset.clear();
        orb_w.clear();

        // First coset 0 orbits (0..r0-1)
        for (int x = 0; x < s; ++x) {
            if (oid[x] == -1) {
                int id = static_cast<int>(orbs.size());
                std::vector<int> o;
                int xx = x;
                while (oid[xx] == -1) {
                    oid[xx] = id;
                    o.push_back(xx);
                    xx = static_cast<int>((1LL * xx * g) % s);
                }
                orbs.push_back(o);
                orb_coset.push_back(0);
                orb_w.push_back(static_cast<int>(o.size()));
            }
        }
        r0 = static_cast<int>(orbs.size());

        // Nonzero cosets (r0..r-1)
        for (int c = 1; c < 4; ++c) {
            for (int x = 0; x < s; ++x) {
                int idx = c * s + x;
                if (oid[idx] == -1) {
                    int id = static_cast<int>(orbs.size());
                    std::vector<int> o;
                    int cc = c, xx = x;
                    while (oid[cc * s + xx] == -1) {
                        oid[cc * s + xx] = id;
                        o.push_back(cc * s + xx);
                        cc = sig3[cc];
                        xx = static_cast<int>((1LL * xx * g) % s);
                    }
                    orbs.push_back(o);
                    orb_coset.push_back(1);
                    orb_w.push_back(static_cast<int>(o.size()) / 3);
                }
            }
        }
        r = static_cast<int>(orbs.size());
        r1 = r - r0;

        // Build shift representatives of <tau, -1> on G \ {0}
        shift_reps.clear();
        std::vector<uint8_t> seen(s, 0);
        seen[0] = 1;
        for (int d = 1; d < s; ++d) {
            if (!seen[d]) {
                shift_reps.push_back({0, d});
                int cur = d;
                while (!seen[cur]) {
                    seen[cur] = 1;
                    seen[(s - cur) % s] = 1;
                    cur = static_cast<int>((1LL * cur * g) % s);
                }
            }
        }
        K_S = static_cast<int>(shift_reps.size());

        // Nonzero coset shift reps: (1, 0) plus (1, d) mod <g^3, -1>
        shift_reps.push_back({1, 0});
        int g3 = static_cast<int>((1LL * g * g % s) * g % s);
        std::fill(seen.begin(), seen.end(), 0);
        seen[0] = 1;
        for (int d = 1; d < s; ++d) {
            if (!seen[d]) {
                shift_reps.push_back({1, d});
                int cur = d;
                while (!seen[cur]) {
                    seen[cur] = 1;
                    seen[(s - cur) % s] = 1;
                    cur = static_cast<int>((1LL * cur * g3) % s);
                }
            }
        }
        K = static_cast<int>(shift_reps.size());
        K_NS = K - K_S;
        K_pad = ((K + 7) / 8) * 8;
        if (K_pad > MAX_K_PAD) {
            std::cerr << "Error: K_pad=" << K_pad << " exceeds MAX_K_PAD=" << MAX_K_PAD << "\n";
            std::exit(1);
        }

        target_paf.assign(K_pad, 0);
        for (int k = 0; k < K_S; ++k) target_paf[k] = -12;
        for (int k = K_S; k < K; ++k) target_paf[k] = 4;

        // Build MT[i][j][k]
        MT_flat.assign(static_cast<size_t>(r) * r * K_pad, 0);
        for (int k = 0; k < K; ++k) {
            int cd = shift_reps[k].first;
            int xd = shift_reps[k].second;
            for (int c = 0; c < 4; ++c) {
                int c2 = c ^ cd;
                for (int x = 0; x < s; ++x) {
                    int x2 = x + xd;
                    if (x2 >= s) x2 -= s;
                    int i = oid[c * s + x];
                    int j = oid[c2 * s + x2];
                    MT_flat[(static_cast<size_t>(i) * r + j) * K_pad + k] += 1;
                }
            }
        }
        // Symmetrize off-diagonal: MT[i][j][k] = C[i][j][k] + C[j][i][k]
        for (int i = 0; i < r; ++i) {
            for (int j = i + 1; j < r; ++j) {
                for (int k = 0; k < K; ++k) {
                    int32_t cij = MT_flat[(static_cast<size_t>(i) * r + j) * K_pad + k];
                    int32_t cji = MT_flat[(static_cast<size_t>(j) * r + i) * K_pad + k];
                    int32_t sum = cij + cji;
                    MT_flat[(static_cast<size_t>(i) * r + j) * K_pad + k] = sum;
                    MT_flat[(static_cast<size_t>(j) * r + i) * K_pad + k] = sum;
                }
            }
        }

        // Precompute same-size swap pairs and 3-orbit triples
        swap_pairs.clear();
        triples.clear();
        for (int i = 0; i < r; ++i) {
            for (int j = i + 1; j < r; ++j) {
                if (orb_coset[i] == orb_coset[j] && orb_w[i] == orb_w[j]) {
                    swap_pairs.push_back({i, j});
                }
            }
        }
        for (int a = 0; a < r; ++a) {
            for (int b = 0; b < r; ++b) {
                if (b == a || orb_coset[b] != orb_coset[a]) continue;
                for (int c = b + 1; c < r; ++c) {
                    if (c == a || orb_coset[c] != orb_coset[a]) continue;
                    if (orb_w[a] == orb_w[b] + orb_w[c]) {
                        triples.push_back({a, b, c});
                    }
                }
            }
        }

        // Build exact DP tables for coset 0 and coset 1
        sum_max0 = s;
        dp0.assign(r0 + 1, std::vector<uint64_t>(2 * sum_max0 + 1, 0));
        dp0[0][sum_max0] = 1;
        for (int i = 0; i < r0; ++i) {
            int w = orb_w[i];
            for (int sv = -sum_max0; sv <= sum_max0; ++sv) {
                uint64_t cnt = dp0[i][sv + sum_max0];
                if (!cnt) continue;
                if (sv + w <= sum_max0) {
                    uint64_t& dst = dp0[i + 1][sv + w + sum_max0];
                    dst = (dst > DP_CAP - cnt) ? DP_CAP : dst + cnt;
                }
                if (sv - w >= -sum_max0) {
                    uint64_t& dst = dp0[i + 1][sv - w + sum_max0];
                    dst = (dst > DP_CAP - cnt) ? DP_CAP : dst + cnt;
                }
            }
        }

        sum_max1 = s;
        dp1.assign(r1 + 1, std::vector<uint64_t>(2 * sum_max1 + 1, 0));
        dp1[0][sum_max1] = 1;
        for (int i = 0; i < r1; ++i) {
            int w = orb_w[r0 + i];
            for (int sv = -sum_max1; sv <= sum_max1; ++sv) {
                uint64_t cnt = dp1[i][sv + sum_max1];
                if (!cnt) continue;
                if (sv + w <= sum_max1) {
                    uint64_t& dst = dp1[i + 1][sv + w + sum_max1];
                    dst = (dst > DP_CAP - cnt) ? DP_CAP : dst + cnt;
                }
                if (sv - w >= -sum_max1) {
                    uint64_t& dst = dp1[i + 1][sv - w + sum_max1];
                    dst = (dst > DP_CAP - cnt) ? DP_CAP : dst + cnt;
                }
            }
        }

        for (int f = 0; f < 4; ++f) {
            if (std::abs(a_tgt[f]) > sum_max0 || dp0[r0][a_tgt[f] + sum_max0] == 0) {
                std::cerr << "Error: unreachable a_tgt[" << f << "]=" << a_tgt[f] << "\n";
                std::exit(1);
            }
            if (std::abs(b_tgt[f]) > sum_max1 || dp1[r1][b_tgt[f] + sum_max1] == 0) {
                std::cerr << "Error: unreachable b_tgt[" << f << "]=" << b_tgt[f] << "\n";
                std::exit(1);
            }
        }
    }

    void sample_coset0(int target, int8_t* out, std::mt19937_64& rng) const {
        int cur = target;
        for (int i = r0 - 1; i >= 0; --i) {
            int w = orb_w[i];
            uint64_t cp = (cur - w >= -sum_max0 && cur - w <= sum_max0) ? dp0[i][cur - w + sum_max0] : 0;
            uint64_t cm = (cur + w >= -sum_max0 && cur + w <= sum_max0) ? dp0[i][cur + w + sum_max0] : 0;
            if (cp == 0) {
                out[i] = -1;
                cur += w;
            } else if (cm == 0) {
                out[i] = 1;
                cur -= w;
            } else {
                uint64_t tot = (cp > DP_CAP - cm) ? DP_CAP : (cp + cm);
                uint64_t rval = rng() % tot;
                if (rval < cp) {
                    out[i] = 1;
                    cur -= w;
                } else {
                    out[i] = -1;
                    cur += w;
                }
            }
        }
    }

    void sample_coset1(int target, int8_t* out, std::mt19937_64& rng) const {
        int cur = target;
        for (int i = r1 - 1; i >= 0; --i) {
            int w = orb_w[r0 + i];
            uint64_t cp = (cur - w >= -sum_max1 && cur - w <= sum_max1) ? dp1[i][cur - w + sum_max1] : 0;
            uint64_t cm = (cur + w >= -sum_max1 && cur + w <= sum_max1) ? dp1[i][cur + w + sum_max1] : 0;
            if (cp == 0) {
                out[r0 + i] = -1;
                cur += w;
            } else if (cm == 0) {
                out[r0 + i] = 1;
                cur -= w;
            } else {
                uint64_t tot = (cp > DP_CAP - cm) ? DP_CAP : (cp + cm);
                uint64_t rval = rng() % tot;
                if (rval < cp) {
                    out[r0 + i] = 1;
                    cur -= w;
                } else {
                    out[r0 + i] = -1;
                    cur += w;
                }
            }
        }
    }
};

struct State {
    const Problem* P = nullptr;
    std::vector<int8_t> x;     // 4 * r
    alignas(32) int32_t R[MAX_K_PAD];      // residual P_sum[k] - target[k]
    std::vector<int32_t> dot;  // 4 * r * K_pad
    int64_t E = 0;

    inline int8_t* x_seq(int f) { return &x[static_cast<size_t>(f) * P->r]; }
    inline const int8_t* x_seq(int f) const { return &x[static_cast<size_t>(f) * P->r]; }
    inline int32_t* dot_ptr(int f, int i) { return &dot[(static_cast<size_t>(f) * P->r + i) * P->K_pad]; }
    inline const int32_t* dot_ptr(int f, int i) const { return &dot[(static_cast<size_t>(f) * P->r + i) * P->K_pad]; }

    void init(const Problem* prob, std::mt19937_64& rng) {
        P = prob;
        x.assign(4 * P->r, 1);
        dot.assign(static_cast<size_t>(4) * P->r * P->K_pad, 0);
        for (int f = 0; f < 4; ++f) {
            P->sample_coset0(P->a_tgt[f], x_seq(f), rng);
            P->sample_coset1(P->b_tgt[f], x_seq(f), rng);
        }
        recompute_all();
    }

    void recompute_all() {
        int r = P->r, K = P->K, Kp = P->K_pad;
        std::fill(dot.begin(), dot.end(), 0);
        std::memset(R, 0, sizeof(R));
        for (int k = 0; k < K; ++k) R[k] = -P->target_paf[k];

        for (int f = 0; f < 4; ++f) {
            const int8_t* xf = x_seq(f);
            for (int i = 0; i < r; ++i) {
                int32_t* di = dot_ptr(f, i);
                // Diagonal term in R[k]
                const int32_t* mii = P->mt_ptr(i, i);
                for (int k = 0; k < K; ++k) R[k] += mii[k];
                // Off-diagonal terms
                for (int j = 0; j < r; ++j) {
                    if (j == i) continue;
                    const int32_t* mij = P->mt_ptr(i, j);
                    int32_t xj = xf[j];
                    for (int k = 0; k < K; ++k) di[k] += xj * mij[k];
                }
            }
            for (int i = 0; i < r; ++i) {
                int32_t xi = xf[i];
                for (int j = i + 1; j < r; ++j) {
                    int32_t prod = xi * xf[j];
                    const int32_t* mij = P->mt_ptr(i, j);
                    for (int k = 0; k < K; ++k) R[k] += prod * mij[k];
                }
            }
        }
        E = 0;
        for (int k = 0; k < K; ++k) E += static_cast<int64_t>(R[k]) * R[k];
        (void)Kp;
    }

    // AVX2 evaluation of swapping u and v in sequence f (assumes x[u] != x[v])
    // Returns delta_E (exact), and optional GLS-weighted delta_E_gls if w_gls != nullptr.
    inline int64_t eval_swap(int f, int u, int v, const int32_t* w_gls, int64_t& delta_gls) const {
        const int8_t* xf = x_seq(f);
        // sgn = +1 if x[u] == +1, x[v] == -1; -1 if x[u] == -1, x[v] == +1
        // Formula for (x[u]=+1, x[v]=-1): D[k] = 2 * (dot[v][k] - dot[u][k] - 2*MT[u][v][k])
        // For (x[u]=-1, x[v]=+1): swap roles -> D[k] = 2 * (dot[u][k] - dot[v][k] - 2*MT[u][v][k])
        const int32_t* du = dot_ptr(f, u);
        const int32_t* dv = dot_ptr(f, v);
        const int32_t* muv = P->mt_ptr(u, v);
        if (xf[u] < 0) std::swap(du, dv);

        int Kp = P->K_pad;
        __m256i v_de_lo = _mm256_setzero_si256();
        __m256i v_de_hi = _mm256_setzero_si256();
        __m256i v_dg_lo = _mm256_setzero_si256();
        __m256i v_dg_hi = _mm256_setzero_si256();

        for (int k = 0; k < Kp; k += 8) {
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dv + k));
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(du + k));
            __m256i m = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(muv + k));
            __m256i rvec = _mm256_load_si256(reinterpret_cast<const __m256i*>(R + k));

            // diff = dv - du - 2*m
            __m256i diff = _mm256_sub_epi32(_mm256_sub_epi32(a, b), _mm256_slli_epi32(m, 1));
            // D = 2 * diff
            __m256i D = _mm256_slli_epi32(diff, 1);
            // term = D * (2*R + D)
            __m256i twoR_D = _mm256_add_epi32(_mm256_slli_epi32(rvec, 1), D);
            __m256i prod = _mm256_mullo_epi32(D, twoR_D);

            // Accumulate as 64-bit
            __m256i p_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod));
            __m256i p_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod, 1));
            v_de_lo = _mm256_add_epi64(v_de_lo, p_lo);
            v_de_hi = _mm256_add_epi64(v_de_hi, p_hi);

            if (w_gls) {
                __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(w_gls + k));
                __m256i w_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(w));
                __m256i w_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(w, 1));
                // Note: prod fits in 32-bit, w in [1..16], so w * prod fits in 32-bit!
                __m256i wp = _mm256_mullo_epi32(prod, w);
                (void)w_lo; (void)w_hi;
                v_dg_lo = _mm256_add_epi64(v_dg_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(wp)));
                v_dg_hi = _mm256_add_epi64(v_dg_hi, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(wp, 1)));
            }
        }
        auto hsum64 = [](__m256i v1, __m256i v2) -> int64_t {
            __m256i s = _mm256_add_epi64(v1, v2);
            alignas(32) int64_t tmp[4];
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), s);
            return tmp[0] + tmp[1] + tmp[2] + tmp[3];
        };
        int64_t de = hsum64(v_de_lo, v_de_hi);
        delta_gls = w_gls ? hsum64(v_dg_lo, v_dg_hi) : de;
        return de;
    }

    // Apply swap(u, v) in sequence f (assumes x[u] != x[v])
    void apply_swap(int f, int u, int v) {
        int8_t* xf = x_seq(f);
        int8_t xu_old = xf[u];
        int8_t xv_old = xf[v];
        const int32_t* du = dot_ptr(f, u);
        const int32_t* dv = dot_ptr(f, v);
        const int32_t* muv = P->mt_ptr(u, v);
        int r = P->r, K = P->K, Kp = P->K_pad;

        // Update R[k] and E
        int32_t sgn = (xu_old > 0) ? 1 : -1;
        int64_t newE = 0;
        for (int k = 0; k < K; ++k) {
            int32_t dval = 2 * sgn * (dv[k] - du[k]) - 4 * muv[k];
            R[k] += dval;
            newE += static_cast<int64_t>(R[k]) * R[k];
        }
        E = newE;

        // Flip signs
        xf[u] = -xu_old;
        xf[v] = -xv_old;

        // Update dot[f][j][k] for all j
        // delta_xu = -2 * xu_old, delta_xv = -2 * xv_old = +2 * xu_old
        int32_t d_u = -2 * xu_old;
        int32_t d_v = -2 * xv_old;
        for (int j = 0; j < r; ++j) {
            int32_t* dj = dot_ptr(f, j);
            if (j == u) {
                for (int k = 0; k < K; ++k) dj[k] += d_v * muv[k];
            } else if (j == v) {
                for (int k = 0; k < K; ++k) dj[k] += d_u * muv[k];
            } else {
                const int32_t* mju = P->mt_ptr(j, u);
                const int32_t* mjv = P->mt_ptr(j, v);
                if (d_u > 0) {
                    for (int k = 0; k < Kp; k += 8) {
                        __m256i cur = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dj + k));
                        __m256i mu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mju + k));
                        __m256i mv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mjv + k));
                        __m256i upd = _mm256_add_epi32(cur, _mm256_slli_epi32(_mm256_sub_epi32(mu, mv), 1));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dj + k), upd);
                    }
                } else {
                    for (int k = 0; k < Kp; k += 8) {
                        __m256i cur = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dj + k));
                        __m256i mu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mju + k));
                        __m256i mv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mjv + k));
                        __m256i upd = _mm256_add_epi32(cur, _mm256_slli_epi32(_mm256_sub_epi32(mv, mu), 1));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dj + k), upd);
                    }
                }
            }
        }
    }

    // Evaluate 3-orbit sum-preserving triple (a, b, c) where x[a] == -x[b] == -x[c]
    inline int64_t eval_triple(int f, int a, int b, int c, const int32_t* w_gls, int64_t& delta_gls) const {
        const int8_t* xf = x_seq(f);
        int32_t s = xf[a];
        const int32_t* da = dot_ptr(f, a);
        const int32_t* db = dot_ptr(f, b);
        const int32_t* dc = dot_ptr(f, c);
        const int32_t* mab = P->mt_ptr(a, b);
        const int32_t* mac = P->mt_ptr(a, c);
        const int32_t* mbc = P->mt_ptr(b, c);
        int K = P->K;
        int64_t de = 0, dg = 0;
        for (int k = 0; k < K; ++k) {
            int32_t D = 2 * s * (db[k] + dc[k] - da[k]) - 4 * (mab[k] + mac[k] - mbc[k]);
            int64_t term = static_cast<int64_t>(D) * (2 * R[k] + D);
            de += term;
            if (w_gls) dg += term * w_gls[k];
        }
        delta_gls = w_gls ? dg : de;
        return de;
    }

    void apply_triple(int f, int a, int b, int c) {
        int8_t* xf = x_seq(f);
        int32_t s = xf[a];
        const int32_t* da = dot_ptr(f, a);
        const int32_t* db = dot_ptr(f, b);
        const int32_t* dc = dot_ptr(f, c);
        const int32_t* mab = P->mt_ptr(a, b);
        const int32_t* mac = P->mt_ptr(a, c);
        const int32_t* mbc = P->mt_ptr(b, c);
        int r = P->r, K = P->K;

        int64_t newE = 0;
        for (int k = 0; k < K; ++k) {
            int32_t D = 2 * s * (db[k] + dc[k] - da[k]) - 4 * (mab[k] + mac[k] - mbc[k]);
            R[k] += D;
            newE += static_cast<int64_t>(R[k]) * R[k];
        }
        E = newE;
        xf[a] = -xf[a];
        xf[b] = -xf[b];
        xf[c] = -xf[c];

        int32_t d_a = -2 * s, d_b = 2 * s, d_c = 2 * s;
        for (int j = 0; j < r; ++j) {
            int32_t* dj = dot_ptr(f, j);
            if (j != a) {
                const int32_t* m = P->mt_ptr(j, a);
                for (int k = 0; k < K; ++k) dj[k] += d_a * m[k];
            }
            if (j != b) {
                const int32_t* m = P->mt_ptr(j, b);
                for (int k = 0; k < K; ++k) dj[k] += d_b * m[k];
            }
            if (j != c) {
                const int32_t* m = P->mt_ptr(j, c);
                for (int k = 0; k < K; ++k) dj[k] += d_c * m[k];
            }
        }
    }
};

static void self_test_state(const Problem& prob) {
    std::mt19937_64 rng(12345);
    State st;
    st.init(&prob, rng);
    // Verify coset sums
    for (int f = 0; f < 4; ++f) {
        int s0 = 0, s1 = 0;
        for (int i = 0; i < prob.r0; ++i) s0 += st.x_seq(f)[i] * prob.orb_w[i];
        for (int i = prob.r0; i < prob.r; ++i) s1 += st.x_seq(f)[i] * prob.orb_w[i];
        if (s0 != prob.a_tgt[f] || s1 != prob.b_tgt[f]) {
            std::cerr << "SELF-TEST FAIL: coset sum mismatch!\n";
            std::exit(1);
        }
    }
    // Verify 10 random swaps against exact recompute
    for (int t = 0; t < 10 && !prob.swap_pairs.empty(); ++t) {
        int f = t % 4;
        for (auto [u, v] : prob.swap_pairs) {
            if (st.x_seq(f)[u] != st.x_seq(f)[v]) {
                int64_t dg = 0;
                int64_t de = st.eval_swap(f, u, v, nullptr, dg);
                int64_t E_before = st.E;
                st.apply_swap(f, u, v);
                int64_t E_after = st.E;
                st.recompute_all();
                if (E_after != E_before + de || st.E != E_after) {
                    std::cerr << "SELF-TEST FAIL: swap mismatch! E_before=" << E_before
                              << " de=" << de << " E_after=" << E_after << " exact=" << st.E << "\n";
                    std::exit(1);
                }
                break;
            }
        }
    }
    // Verify 5 random triples against exact recompute
    for (int t = 0; t < 5 && !prob.triples.empty(); ++t) {
        int f = t % 4;
        for (auto tr : prob.triples) {
            int8_t* xf = st.x_seq(f);
            if (xf[tr.a] == -xf[tr.b] && xf[tr.a] == -xf[tr.c]) {
                int64_t dg = 0;
                int64_t de = st.eval_triple(f, tr.a, tr.b, tr.c, nullptr, dg);
                int64_t E_before = st.E;
                st.apply_triple(f, tr.a, tr.b, tr.c);
                int64_t E_after = st.E;
                st.recompute_all();
                if (E_after != E_before + de || st.E != E_after) {
                    std::cerr << "SELF-TEST FAIL: triple mismatch!\n";
                    std::exit(1);
                }
                break;
            }
        }
    }
}

static void save_json(const std::string& path, const Problem& prob, const std::vector<int8_t>& x_best, int64_t E_best) {
    std::ofstream ofs(path);
    ofs << "{\n";
    ofs << "  \"n\": " << prob.n << ",\n";
    ofs << "  \"s\": " << prob.s << ",\n";
    ofs << "  \"g\": " << prob.g << ",\n";
    ofs << "  \"r\": " << prob.r << ",\n";
    ofs << "  \"K\": " << prob.K << ",\n";
    ofs << "  \"E\": " << E_best << ",\n";
    ofs << "  \"orbit_signs\": [\n";
    for (int f = 0; f < 4; ++f) {
        ofs << "    [";
        for (int i = 0; i < prob.r; ++i) {
            ofs << static_cast<int>(x_best[static_cast<size_t>(f) * prob.r + i]);
            if (i + 1 < prob.r) ofs << ", ";
        }
        ofs << "]" << (f < 3 ? "," : "") << "\n";
    }
    ofs << "  ],\n";
    // Also expand full 4*s sequences on G = Z_2^2 x Z_s for direct verification by verify_thm7.py
    ofs << "  \"seqs\": [\n";
    for (int f = 0; f < 4; ++f) {
        ofs << "    [";
        for (int idx = 0; idx < prob.v; ++idx) {
            int o = prob.oid[idx];
            ofs << static_cast<int>(x_best[static_cast<size_t>(f) * prob.r + o]);
            if (idx + 1 < prob.v) ofs << ",";
        }
        ofs << "]" << (f < 3 ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <s> <g> <a0,b0:a1,b1:a2,b2:a3,b3> <seconds> <out_json> <threads> [--seed=N] [--save_best=path]\n";
        return 1;
    }
    int s = std::atoi(argv[1]);
    int g = std::atoi(argv[2]);
    std::string ab_str = argv[3];
    double max_secs = std::atof(argv[4]);
    std::string out_json = argv[5];
    int n_threads = std::atoi(argv[6]);
    uint64_t base_seed = 20261124ULL;
    std::string save_best_path;

    for (int i = 7; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--seed=", 0) == 0) {
            base_seed = std::strtoull(arg.c_str() + 7, nullptr, 10);
        } else if (arg.rfind("--save_best=", 0) == 0) {
            save_best_path = arg.substr(12);
        }
    }

    int a_tgt[4], b_tgt[4];
    {
        std::stringstream ss(ab_str);
        std::string tok;
        int idx = 0;
        while (std::getline(ss, tok, ':') && idx < 4) {
            auto comma = tok.find(',');
            a_tgt[idx] = std::atoi(tok.substr(0, comma).c_str());
            b_tgt[idx] = std::atoi(tok.substr(comma + 1).c_str());
            ++idx;
        }
        if (idx != 4) {
            std::cerr << "Error parsing a,b targets: " << ab_str << "\n";
            return 1;
        }
    }

    Problem prob;
    prob.build(s, g, a_tgt, b_tgt);
    self_test_state(prob);

    std::printf("thm7_diag_solver: n=%d (s=%d, g=%d) r=%d (r0=%d, r1=%d) K=%d (KS=%d, KNS=%d) swaps=%zu triples=%zu threads=%d seed=%llu\n",
                prob.n, prob.s, prob.g, prob.r, prob.r0, prob.r1, prob.K, prob.K_S, prob.K_NS,
                prob.swap_pairs.size(), prob.triples.size(), n_threads,
                static_cast<unsigned long long>(base_seed));
    std::fflush(stdout);

    std::atomic<bool> found{false};
    std::atomic<int64_t> global_best_E{1LL << 60};
    std::atomic<uint64_t> total_steps{0};
    std::mutex best_mtx;
    std::vector<int8_t> global_best_x;
    std::vector<int32_t> global_best_R(prob.K, 0);

    auto t_start = std::chrono::steady_clock::now();

    auto worker = [&](int tid) {
        std::mt19937_64 rng(base_seed + 0x9e3779b97f4a7c15ULL * (tid + 1));
        State st;
        st.init(&prob, rng);

        // Orbit-level tabu tenure: tabu_until[f * r + i]
        std::vector<uint64_t> tabu_until(4 * prob.r, 0);
        alignas(32) int32_t w_gls[MAX_K_PAD];
        for (int k = 0; k < MAX_K_PAD; ++k) w_gls[k] = (k < prob.K) ? 1 : 0;

        int64_t local_best_E = st.E;
        std::vector<int8_t> local_best_x = st.x;
        uint64_t step = 0;
        uint64_t last_improve = 0;
        uint64_t gls_active_until = 0;

        auto update_global = [&](const State& cur) {
            int64_t prev = global_best_E.load(std::memory_order_relaxed);
            if (cur.E < prev) {
                std::lock_guard<std::mutex> lk(best_mtx);
                if (cur.E < global_best_E.load()) {
                    global_best_E.store(cur.E);
                    global_best_x = cur.x;
                    for (int k = 0; k < prob.K; ++k) global_best_R[k] = cur.R[k];
                    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                    std::printf("[t=%6.2fs T%d] NEW BEST E=%lld (step=%llu)\n",
                                el, tid, static_cast<long long>(cur.E), static_cast<unsigned long long>(step));
                    std::fflush(stdout);
                    if (cur.E == 0) {
                        save_json(out_json, prob, cur.x, 0);
                        found.store(true);
                    } else if (!save_best_path.empty()) {
                        save_json(save_best_path, prob, cur.x, cur.E);
                    }
                }
            }
        };
        update_global(st);

        int tenure_base = std::max(5, prob.r / 4);
        uint64_t stall_limit = std::max<uint64_t>(4000, 150ULL * prob.r);
        uint64_t restart_limit = stall_limit * 5;
        uint64_t last_restart = 0;

        while (!found.load(std::memory_order_relaxed)) {
            if ((step & 1023ULL) == 0) {
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                if (el >= max_secs) break;
            }
            ++step;

            const int32_t* use_w = (step < gls_active_until) ? w_gls : nullptr;

            int best_type = -1; // 0 = swap, 1 = triple
            int best_f = -1, best_a = -1, best_b = -1, best_c = -1;
            int64_t best_score = (1LL << 60);
            int64_t best_de = (1LL << 60);
            uint32_t ties = 0;

            for (int f = 0; f < 4; ++f) {
                const int8_t* xf = st.x_seq(f);
                const uint64_t* tf = &tabu_until[static_cast<size_t>(f) * prob.r];
                for (const auto& pr : prob.swap_pairs) {
                    int u = pr.first, v = pr.second;
                    if (xf[u] == xf[v]) continue;
                    int64_t dg = 0;
                    int64_t de = st.eval_swap(f, u, v, use_w, dg);
                    bool is_tabu = (tf[u] > step || tf[v] > step);
                    if (is_tabu && st.E + de >= local_best_E) continue;
                    if (dg < best_score) {
                        best_score = dg;
                        best_de = de;
                        best_type = 0;
                        best_f = f; best_a = u; best_b = v;
                        ties = 1;
                    } else if (dg == best_score) {
                        ++ties;
                        if (rng() % ties == 0) {
                            best_de = de;
                            best_type = 0;
                            best_f = f; best_a = u; best_b = v;
                        }
                    }
                }
                for (const auto& tr : prob.triples) {
                    if (xf[tr.a] != -xf[tr.b] || xf[tr.a] != -xf[tr.c]) continue;
                    int64_t dg = 0;
                    int64_t de = st.eval_triple(f, tr.a, tr.b, tr.c, use_w, dg);
                    bool is_tabu = (tf[tr.a] > step || tf[tr.b] > step || tf[tr.c] > step);
                    if (is_tabu && st.E + de >= local_best_E) continue;
                    if (dg < best_score) {
                        best_score = dg;
                        best_de = de;
                        best_type = 1;
                        best_f = f; best_a = tr.a; best_b = tr.b; best_c = tr.c;
                        ties = 1;
                    } else if (dg == best_score) {
                        ++ties;
                        if (rng() % ties == 0) {
                            best_de = de;
                            best_type = 1;
                            best_f = f; best_a = tr.a; best_b = tr.b; best_c = tr.c;
                        }
                    }
                }
            }

            if (best_type == 0) {
                st.apply_swap(best_f, best_a, best_b);
                uint64_t ten = static_cast<uint64_t>(tenure_base + (rng() % (tenure_base + 1)));
                tabu_until[static_cast<size_t>(best_f) * prob.r + best_a] = step + ten;
                tabu_until[static_cast<size_t>(best_f) * prob.r + best_b] = step + ten;
            } else if (best_type == 1) {
                st.apply_triple(best_f, best_a, best_b, best_c);
                uint64_t ten = static_cast<uint64_t>(tenure_base + (rng() % (tenure_base + 1)));
                tabu_until[static_cast<size_t>(best_f) * prob.r + best_a] = step + ten;
                tabu_until[static_cast<size_t>(best_f) * prob.r + best_b] = step + ten;
                tabu_until[static_cast<size_t>(best_f) * prob.r + best_c] = step + ten;
            } else {
                // Clear tabu if all moves were tabu
                std::fill(tabu_until.begin(), tabu_until.end(), 0);
            }

            if (st.E < local_best_E) {
                local_best_E = st.E;
                local_best_x = st.x;
                last_improve = step;
                gls_active_until = 0; // Turn off GLS on new local best
                update_global(st);
                if (st.E == 0) break;
            } else if (step - last_improve > stall_limit) {
                if (step - last_restart > restart_limit) {
                    // Full random restart via exact DP
                    st.init(&prob, rng);
                    local_best_E = st.E;
                    local_best_x = st.x;
                    std::fill(tabu_until.begin(), tabu_until.end(), 0);
                    for (int k = 0; k < prob.K; ++k) w_gls[k] = 1;
                    gls_active_until = 0;
                    last_improve = step;
                    last_restart = step;
                } else {
                    // Alternate between GLS kick and 1-sequence DP-LNS perturbation from local_best
                    if ((rng() & 1ULL) == 0) {
                        // GLS: boost weights of the 3 worst residual shifts
                        for (int k = 0; k < prob.K; ++k) w_gls[k] = 1;
                        std::vector<int> idx(prob.K);
                        std::iota(idx.begin(), idx.end(), 0);
                        std::sort(idx.begin(), idx.end(), [&](int i, int j) {
                            return std::abs(st.R[i]) > std::abs(st.R[j]);
                        });
                        for (int t = 0; t < std::min(4, prob.K); ++t) w_gls[idx[t]] = 3;
                        gls_active_until = step + stall_limit / 2;
                        last_improve = step - stall_limit / 2;
                    } else {
                        // 1-sequence DP-LNS: restore local_best and resample 1 coset (0 or 1) of 1 sequence via exact DP!
                        st.x = local_best_x;
                        int f_pick = static_cast<int>(rng() % 4);
                        if ((rng() & 1ULL) == 0) {
                            prob.sample_coset0(prob.a_tgt[f_pick], st.x_seq(f_pick), rng);
                        } else {
                            prob.sample_coset1(prob.b_tgt[f_pick], st.x_seq(f_pick), rng);
                        }
                        st.recompute_all();
                        std::fill(tabu_until.begin(), tabu_until.end(), 0);
                        gls_active_until = 0;
                        last_improve = step;
                    }
                }
            }
            (void)best_de;
        }
        total_steps.fetch_add(step, std::memory_order_relaxed);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    uint64_t steps = total_steps.load();
    int64_t bestE = global_best_E.load();
    std::printf("DONE in %.2fs | total_steps=%llu (%.1f ksteps/s) | best_E=%lld\n",
                elapsed, static_cast<unsigned long long>(steps),
                steps / (1000.0 * std::max(1e-6, elapsed)),
                static_cast<long long>(bestE));
    if (bestE > 0) {
        std::printf("Best residual R[0..%d]: [", prob.K - 1);
        for (int k = 0; k < prob.K; ++k) {
            std::printf("%d%s", global_best_R[k], (k + 1 < prob.K ? ", " : ""));
        }
        std::printf("]\n");
    }
    return (bestE == 0) ? 0 : 2;
}

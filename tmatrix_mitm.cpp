// Partitioned 2+2 Flat-Pair MITM Solver for U-invariant Circulant T-Matrices
// Usage: ./tmatrix_mitm <t> <g> <m0> <m1> <m2> <m3> <nA> <pool_size> <secs> <flat_max_A> <flat_max_B> <out_json> [qr_split=0|1] [seed=42]
//
// Partitions the (r-1) nonzero orbits into S_A (size nA, assigned to {T_0, T_1} with orbit 0 in T_0)
// and S_B (size r-1-nA, assigned to {T_2, T_3}).
// Because supports are disjoint, PAF_total = P_A + P_B with ZERO cross terms!
// Minimizes F_A = sum_k P_A[k]^2 on S_A and F_B = sum_k P_B[k]^2 on S_B independently,
// harvesting distinct flat half-states and matching P_A == -P_B via 64-bit hash table.
// When qr_split=1 (for prime t == 3 mod 4 with S_A = QR orbits, S_B = NQR orbits),
// QR/U acts cyclically on all K shifts while preserving S_A and S_B, giving a K-fold
// rotation multiplier via canonical shift rotation!

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX2__
static inline int64_t hsum_epi32_to_64(__m256i v) {
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
    int t, g, h, r, K, K_pad;
    int m_target[4];
    std::vector<std::vector<int>> orbits;
    std::vector<int> orb_id;
    std::vector<int> d_reps;
    int shift_gen = 1; // generator of QR/U acting cyclically on d_reps (if qr_split)
    std::vector<int> qr_powers;
    std::vector<int> MT; // [(i*r + j)*K_pad + k]
    std::vector<int> orb_diag; // [i*K_pad + k] = MT[i][i][k]
};

static int mod_pow(int64_t a, int64_t e, int m) {
    int64_t res = 1;
    a %= m;
    while (e > 0) {
        if (e & 1) res = (res * a) % m;
        a = (a * a) % m;
        e >>= 1;
    }
    return (int)res;
}

static Instance build_instance(int t, int g, int m0, int m1, int m2, int m3, bool qr_split) {
    Instance inst;
    inst.t = t;
    inst.g = g;
    inst.m_target[0] = m0;
    inst.m_target[1] = m1;
    inst.m_target[2] = m2;
    inst.m_target[3] = m3;

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

    // Build shift representatives; if qr_split, order d_reps cyclically by a QR generator!
    std::vector<bool> in_stab(t, false);
    {
        int cur = 1;
        do {
            in_stab[cur] = true;
            in_stab[(t - cur) % t] = true;
            cur = (int)((1LL * cur * g) % t);
        } while (cur != 1);
    }
    int stab_sz = 0;
    for (bool b : in_stab) if (b) ++stab_sz;
    inst.K = (t - 1) / stab_sz;

    if (qr_split) {
        bool found_gen = false;
        for (int gen = 2; gen < t; ++gen) {
            if (mod_pow(gen, (t - 1) / 2, t) != 1) continue; // must be a QR!
            std::vector<int> sh;
            int cur = 1;
            do {
                sh.push_back(cur);
                cur = (int)((1LL * cur * gen) % t);
            } while (!in_stab[cur]);
            if ((int)sh.size() == inst.K) {
                inst.d_reps = std::move(sh);
                inst.shift_gen = gen;
                found_gen = true;
                break;
            }
        }
        if (!found_gen) {
            std::cerr << "Could not find cyclic QR shift generator for qr_split=1\n";
            std::exit(1);
        }
        inst.qr_powers.assign(inst.K, 1);
        for (int k = 1; k < inst.K; ++k) {
            inst.qr_powers[k] = (int)((1LL * inst.qr_powers[k - 1] * inst.shift_gen) % t);
        }
    } else {
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
    }
    inst.K_pad = ((inst.K + 7) / 8) * 8;
    inst.MT.assign(static_cast<size_t>(inst.r) * inst.r * inst.K_pad, 0);
    inst.orb_diag.assign(static_cast<size_t>(inst.r) * inst.K_pad, 0);
    for (int k = 0; k < inst.K; ++k) {
        int d = inst.d_reps[k];
        for (int x = 0; x < t; ++x) {
            int i = inst.orb_id[x];
            int j = inst.orb_id[(x + d) % t];
            if (i == j) {
                inst.MT[(i * inst.r + i) * inst.K_pad + k] += 1;
                inst.orb_diag[i * inst.K_pad + k] += 1;
            } else {
                inst.MT[(i * inst.r + j) * inst.K_pad + k] += 1;
                inst.MT[(j * inst.r + i) * inst.K_pad + k] += 1;
            }
        }
    }
    return inst;
}

// 2-matrix subproblem on a subset of orbits (local indices 0..n-1; if has_zero, local 0 is orbit 0 fixed to mat 0, sgn +1)
struct HalfState {
    const Instance* inst;
    int n, K, K_pad;
    bool has_zero;
    int m_target[2];
    std::vector<int> orig_orb; // local idx -> global orbit id
    std::vector<int> MT_sub;   // [(i*n + j)*K_pad + k]
    std::vector<int> diag_sum; // sum_{i=0}^{n-1} orb_diag[orig_orb[i]][k]

    std::vector<int> mat; // 0 or 1
    std::vector<int> sgn; // -1 or +1
    std::vector<int> P;   // length K_pad
    std::vector<int> C;   // [2 * n * K_pad]
    int64_t F;            // sum_k P[k]^2

    HalfState(const Instance* in, const std::vector<int>& orbs, bool has0, int m0, int m1)
        : inst(in), n((int)orbs.size()), K(in->K), K_pad(in->K_pad), has_zero(has0),
          orig_orb(orbs), MT_sub((size_t)n * n * K_pad, 0), diag_sum(K_pad, 0),
          mat(n, 0), sgn(n, 1), P(K_pad, 0), C(2 * n * K_pad, 0), F(0) {
        m_target[0] = m0;
        m_target[1] = m1;
        for (int i = 0; i < n; ++i) {
            int oi = orig_orb[i];
            for (int k = 0; k < K; ++k) diag_sum[k] += in->orb_diag[oi * K_pad + k];
            for (int j = 0; j < n; ++j) {
                int oj = orig_orb[j];
                const int* src = &in->MT[(oi * in->r + oj) * K_pad];
                int* dst = &MT_sub[(i * n + j) * K_pad];
                for (int k = 0; k < K_pad; ++k) dst[k] = src[k];
            }
        }
    }

    void init_random(std::mt19937_64& rng) {
        int start_idx = has_zero ? 1 : 0;
        if (has_zero) { mat[0] = 0; sgn[0] = 1; }
        int free_n = n - start_idx;
        std::vector<int> perm(free_n);
        std::iota(perm.begin(), perm.end(), start_idx);
        std::shuffle(perm.begin(), perm.end(), rng);
        int pos = 0;
        for (int a = 0; a < 2; ++a) {
            int s = (m_target[a] >= 0) ? 1 : -1;
            int cnt = std::abs(m_target[a]);
            for (int c = 0; c < cnt; ++c) {
                int idx = perm[pos++];
                mat[idx] = a; sgn[idx] = s;
            }
        }
        int rem = free_n - pos;
        for (int c = 0; c < rem; c += 2) {
            int a = (int)(rng() & 1ULL);
            int i1 = perm[pos++], i2 = perm[pos++];
            mat[i1] = a; sgn[i1] = 1;
            mat[i2] = a; sgn[i2] = -1;
        }
        recompute();
    }

    void recompute() {
        std::fill(P.begin(), P.end(), 0);
        std::fill(C.begin(), C.end(), 0);
        for (int k = 0; k < K; ++k) P[k] = diag_sum[k];
        for (int i = 0; i < n; ++i) {
            int mi = mat[i], si = sgn[i];
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                int mj = mat[j], sj = sgn[j];
                const int* mt = &MT_sub[(i * n + j) * K_pad];
                int* cm = &C[(mj * n + i) * K_pad];
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
        F = 0;
        for (int k = 0; k < K; ++k) F += 1LL * P[k] * P[k];
    }

    inline void update_C_orbit(int m, int o, int delta_s) {
        int* __restrict__ C_m = &C[m * n * K_pad];
        const int* __restrict__ MT_o = &MT_sub[o * n * K_pad];
        if (delta_s == 1) {
            for (int l = 0; l < n; ++l) {
                if (l == o) continue;
                int* __restrict__ dst = C_m + l * K_pad;
                const int* __restrict__ src = MT_o + l * K_pad;
                for (int k = 0; k < K_pad; ++k) dst[k] += src[k];
            }
        } else if (delta_s == -1) {
            for (int l = 0; l < n; ++l) {
                if (l == o) continue;
                int* __restrict__ dst = C_m + l * K_pad;
                const int* __restrict__ src = MT_o + l * K_pad;
                for (int k = 0; k < K_pad; ++k) dst[k] -= src[k];
            }
        } else if (delta_s == 2) {
            for (int l = 0; l < n; ++l) {
                if (l == o) continue;
                int* __restrict__ dst = C_m + l * K_pad;
                const int* __restrict__ src = MT_o + l * K_pad;
                for (int k = 0; k < K_pad; ++k) dst[k] += 2 * src[k];
            }
        } else if (delta_s == -2) {
            for (int l = 0; l < n; ++l) {
                if (l == o) continue;
                int* __restrict__ dst = C_m + l * K_pad;
                const int* __restrict__ src = MT_o + l * K_pad;
                for (int k = 0; k < K_pad; ++k) dst[k] -= 2 * src[k];
            }
        }
    }

    inline int64_t eval_move1(int i, int j, int a) const {
        const int* __restrict__ Cai = &C[(a * n + i) * K_pad];
        const int* __restrict__ Caj = &C[(a * n + j) * K_pad];
        const int* __restrict__ mij = &MT_sub[(i * n + j) * K_pad];
        const int* __restrict__ pptr = P.data();
#ifdef __AVX2__
        __m256i acc = _mm256_setzero_si256();
        for (int k = 0; k < K_pad; k += 8) {
            __m256i vcai = _mm256_loadu_si256((const __m256i*)(Cai + k));
            __m256i vcaj = _mm256_loadu_si256((const __m256i*)(Caj + k));
            __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
            __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
            __m256i vdp = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_sub_epi32(vcaj, vcai), _mm256_slli_epi32(vmij, 1)), 1);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vdp, _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp)));
        }
        return hsum_epi32_to_64(acc);
#else
        int64_t dF = 0;
        for (int k = 0; k < K; ++k) {
            int dp = 2 * (Caj[k] - Cai[k] - 2 * mij[k]);
            dF += 1LL * dp * (2 * pptr[k] + dp);
        }
        return dF;
#endif
    }

    void apply_move1(int i, int j, int a) {
        const int* Cai = &C[(a * n + i) * K_pad];
        const int* Caj = &C[(a * n + j) * K_pad];
        const int* mij = &MT_sub[(i * n + j) * K_pad];
        for (int k = 0; k < K; ++k) {
            int dp = 2 * (Caj[k] - Cai[k] - 2 * mij[k]);
            F += 1LL * dp * (2 * P[k] + dp);
            P[k] += dp;
        }
        sgn[i] = -1; sgn[j] = 1;
        update_C_orbit(a, i, -2);
        update_C_orbit(a, j, 2);
    }

    inline int64_t eval_move2(int i, int j, int eps) const {
        // mat[i]==0, mat[j]==1, sgn[i]==sgn[j]==eps
        const int* __restrict__ C0i = &C[i * K_pad];
        const int* __restrict__ C1i = &C[(n + i) * K_pad];
        const int* __restrict__ C0j = &C[j * K_pad];
        const int* __restrict__ C1j = &C[(n + j) * K_pad];
        const int* __restrict__ mij = &MT_sub[(i * n + j) * K_pad];
        const int* __restrict__ pptr = P.data();
#ifdef __AVX2__
        __m256i acc = _mm256_setzero_si256();
        for (int k = 0; k < K_pad; k += 8) {
            __m256i v0i = _mm256_loadu_si256((const __m256i*)(C0i + k));
            __m256i v1i = _mm256_loadu_si256((const __m256i*)(C1i + k));
            __m256i v0j = _mm256_loadu_si256((const __m256i*)(C0j + k));
            __m256i v1j = _mm256_loadu_si256((const __m256i*)(C1j + k));
            __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
            __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
            __m256i vdiff = _mm256_add_epi32(_mm256_sub_epi32(v1i, v0i), _mm256_sub_epi32(v0j, v1j));
            if (eps < 0) vdiff = _mm256_sub_epi32(_mm256_setzero_si256(), vdiff);
            __m256i vdp = _mm256_sub_epi32(vdiff, _mm256_slli_epi32(vmij, 1));
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vdp, _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp)));
        }
        return hsum_epi32_to_64(acc);
#else
        int64_t dF = 0;
        for (int k = 0; k < K; ++k) {
            int dp = eps * (C1i[k] - C0i[k] + C0j[k] - C1j[k]) - 2 * mij[k];
            dF += 1LL * dp * (2 * pptr[k] + dp);
        }
        return dF;
#endif
    }

    void apply_move2(int i, int j, int eps) {
        const int* C0i = &C[i * K_pad];
        const int* C1i = &C[(n + i) * K_pad];
        const int* C0j = &C[j * K_pad];
        const int* C1j = &C[(n + j) * K_pad];
        const int* mij = &MT_sub[(i * n + j) * K_pad];
        for (int k = 0; k < K; ++k) {
            int dp = eps * (C1i[k] - C0i[k] + C0j[k] - C1j[k]) - 2 * mij[k];
            F += 1LL * dp * (2 * P[k] + dp);
            P[k] += dp;
        }
        mat[i] = 1; mat[j] = 0;
        update_C_orbit(0, i, -eps);
        update_C_orbit(1, i, eps);
        update_C_orbit(1, j, -eps);
        update_C_orbit(0, j, eps);
    }

    inline int64_t eval_move3(int i, int j, int a, int flip_b) const {
        int b = 1 - a;
        const int* __restrict__ Cai = &C[(a * n + i) * K_pad];
        const int* __restrict__ Caj = &C[(a * n + j) * K_pad];
        const int* __restrict__ Cbi = &C[(b * n + i) * K_pad];
        const int* __restrict__ Cbj = &C[(b * n + j) * K_pad];
        const int* __restrict__ mij = &MT_sub[(i * n + j) * K_pad];
        const int* __restrict__ pptr = P.data();
        int fb = flip_b ? -1 : 1;
#ifdef __AVX2__
        __m256i acc = _mm256_setzero_si256();
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
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vdp, _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp)));
        }
        return hsum_epi32_to_64(acc);
#else
        int64_t dF = 0;
        for (int k = 0; k < K; ++k) {
            int dp = fb * (Cbi[k] - Cbj[k]) + (Caj[k] - Cai[k] - 2 * mij[k]);
            dF += 1LL * dp * (2 * pptr[k] + dp);
        }
        return dF;
#endif
    }

    void apply_move3(int i, int j, int a, int flip_b) {
        int b = 1 - a;
        const int* Cai = &C[(a * n + i) * K_pad];
        const int* Caj = &C[(a * n + j) * K_pad];
        const int* Cbi = &C[(b * n + i) * K_pad];
        const int* Cbj = &C[(b * n + j) * K_pad];
        const int* mij = &MT_sub[(i * n + j) * K_pad];
        int fb = flip_b ? -1 : 1;
        for (int k = 0; k < K; ++k) {
            int dp = fb * (Cbi[k] - Cbj[k]) + (Caj[k] - Cai[k] - 2 * mij[k]);
            F += 1LL * dp * (2 * P[k] + dp);
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

struct TableEntry {
    uint64_t hash;
    uint32_t idx;
    int32_t rot; // canonical rotation shift
};

struct StoredHalf {
    std::vector<int8_t> mat;
    std::vector<int8_t> sgn;
    std::vector<int16_t> paf; // canonicalized (or raw if !qr_split)
};

static int canonical_shift(const int* p, int K, int sign) {
    int best = 0;
    for (int k = 1; k < K; ++k) {
        for (int j = 0; j < K; ++j) {
            int a = sign * p[(k + j) % K];
            int b = sign * p[(best + j) % K];
            if (a != b) {
                if (a < b) best = k;
                break;
            }
        }
    }
    return best;
}

static uint64_t hash_paf(const int* p, int K, int rot, int sign, const std::vector<uint64_t>& coeffs) {
    uint64_t h = 0;
    for (int j = 0; j < K; ++j) {
        int64_t v = (int64_t)sign * p[(rot + j) % K];
        h += coeffs[j] * (uint64_t)v;
    }
    // final mix
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

static bool verify_full_and_save(const Instance& inst, const std::vector<int>& mat, const std::vector<int>& sgn, const std::string& out_path) {
    for (int d = 1; d < inst.t; ++d) {
        int paf = 0;
        for (int x = 0; x < inst.t; ++x) {
            int i = inst.orb_id[x], j = inst.orb_id[(x + d) % inst.t];
            if (mat[i] == mat[j]) paf += sgn[i] * sgn[j];
        }
        if (paf != 0) return false;
    }
    std::ofstream out(out_path);
    out << "{\n";
    out << "  \"t\": " << inst.t << ",\n";
    out << "  \"g\": " << inst.g << ",\n";
    out << "  \"r\": " << inst.r << ",\n";
    out << "  \"K\": " << inst.K << ",\n";
    out << "  \"orbit_mat\": [";
    for (int i = 0; i < inst.r; ++i) out << mat[i] << (i + 1 < inst.r ? ", " : "");
    out << "],\n";
    out << "  \"orbit_sign\": [";
    for (int i = 0; i < inst.r; ++i) out << sgn[i] << (i + 1 < inst.r ? ", " : "");
    out << "]\n}\n";
    return static_cast<bool>(out);
}

int main(int argc, char** argv) {
    if (argc < 13) {
        std::cerr << "Usage: ./tmatrix_mitm <t> <g> <m0> <m1> <m2> <m3> <nA> <pool_size> <secs> <flat_max_A> <flat_max_B> <out_json> [qr_split=0|1] [seed=42]\n";
        return 1;
    }
    int t = std::stoi(argv[1]);
    int g = std::stoi(argv[2]);
    int m0 = std::stoi(argv[3]);
    int m1 = std::stoi(argv[4]);
    int m2 = std::stoi(argv[5]);
    int m3 = std::stoi(argv[6]);
    int nA = std::stoi(argv[7]);
    size_t pool_size = std::stoull(argv[8]);
    double secs = std::stod(argv[9]);
    int64_t flat_max_A = std::stoll(argv[10]);
    int64_t flat_max_B = std::stoll(argv[11]);
    std::string out_path = argv[12];
    bool qr_split = (argc >= 14) ? (std::stoi(argv[13]) != 0) : false;
    uint64_t seed = (argc >= 15) ? std::stoull(argv[14]) : 42ULL;

    auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };

    Instance inst = build_instance(t, g, m0, m1, m2, m3, qr_split);
    int e = inst.r - 1;
    int nB = e - nA;
    if (nA < std::abs(m0) + std::abs(m1) || ((nA - std::abs(m0) - std::abs(m1)) % 2 != 0) ||
        nB < std::abs(m2) + std::abs(m3) || ((nB - std::abs(m2) - std::abs(m3)) % 2 != 0)) {
        std::cerr << "Invalid partition size nA=" << nA << " (nB=" << nB << ") for targets ("
                  << m0 << "," << m1 << "," << m2 << "," << m3 << ")\n";
        return 1;
    }

    std::mt19937_64 rng(seed);
    std::vector<int> orbsA, orbsB;
    orbsA.push_back(0); // orbit 0 always in S_A (fixed to T_0, +1)
    if (qr_split) {
        for (int o = 1; o < inst.r; ++o) {
            int rep = inst.orbits[o][0];
            if (mod_pow(rep, (t - 1) / 2, t) == 1) orbsA.push_back(o);
            else orbsB.push_back(o);
        }
        if ((int)orbsA.size() - 1 != nA || (int)orbsB.size() != nB) {
            std::cerr << "qr_split sizes mismatch: |QR|=" << orbsA.size() - 1 << ", |NQR|=" << orbsB.size() << "\n";
            return 1;
        }
    } else {
        std::vector<int> perm(e);
        std::iota(perm.begin(), perm.end(), 1);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < nA; ++i) orbsA.push_back(perm[i]);
        for (int i = nA; i < e; ++i) orbsB.push_back(perm[i]);
    }

    HalfState stA(&inst, orbsA, true, m0, m1);
    HalfState stB(&inst, orbsB, false, m2, m3);

    // Self-check half deltas
    stA.init_random(rng);
    int64_t sumA = std::accumulate(stA.P.begin(), stA.P.begin() + inst.K, int64_t{0});
    stB.init_random(rng);
    int64_t sumB = std::accumulate(stB.P.begin(), stB.P.begin() + inst.K, int64_t{0});
    std::cout << "tmatrix_mitm: t=" << t << " g=" << g << " |U|=" << inst.h << " r=" << inst.r
              << " K=" << inst.K << " m=(" << m0 << "," << m1 << "," << m2 << "," << m3 << ")"
              << " nA=" << nA << " nB=" << nB << " qr_split=" << (qr_split ? 1 : 0)
              << " sum_PA=" << sumA << " sum_PB=" << sumB
              << " pool=" << pool_size << " flat_max=(" << flat_max_A << "," << flat_max_B << ")" << std::endl;

    std::mt19937_64 h_rng(0x24c9bdb9c329ab77ULL);
    std::vector<uint64_t> coeffs(inst.K);
    for (auto& c : coeffs) c = h_rng();

    // Helper: run fast Tabu+ILS walk on HalfState, harvesting distinct states with F <= flat_max
    auto harvest_A = [&](std::vector<StoredHalf>& out_rows, std::vector<TableEntry>& table, double time_budget) {
        std::unordered_set<uint64_t> seen_hashes;
        seen_hashes.reserve(pool_size * 2);
        std::vector<int> tabu(stA.n * 4, 0);
        int start_idx = 1;
        int iter = 0, stall = 0;
        int tenure = std::max(3, stA.n / 5);
        int64_t min_F = INT64_MAX;
        int64_t sum_F = 0;
        std::vector<int> pos[2], neg[2];
        stA.init_random(rng);
        int64_t basin_best = stA.F;

        auto maybe_add = [&]() {
            if (stA.F < min_F) min_F = stA.F;
            if (stA.F > flat_max_A) return;
            int rot = qr_split ? canonical_shift(stA.P.data(), inst.K, 1) : 0;
            uint64_t h = hash_paf(stA.P.data(), inst.K, rot, 1, coeffs);
            if (!seen_hashes.insert(h).second) return;
            uint32_t idx = (uint32_t)out_rows.size();
            StoredHalf sh;
            sh.mat.resize(stA.n); sh.sgn.resize(stA.n);
            for (int i = 0; i < stA.n; ++i) { sh.mat[i] = (int8_t)stA.mat[i]; sh.sgn[i] = (int8_t)stA.sgn[i]; }
            sh.paf.resize(inst.K);
            for (int j = 0; j < inst.K; ++j) sh.paf[j] = (int16_t)stA.P[(rot + j) % inst.K];
            out_rows.push_back(std::move(sh));
            table.push_back({h, idx, rot});
            sum_F += stA.F;
        };

        while (out_rows.size() < pool_size) {
            if ((iter & 255) == 0 && elapsed() >= time_budget) break;
            ++iter;
            for (int a = 0; a < 2; ++a) { pos[a].clear(); neg[a].clear(); }
            for (int i = start_idx; i < stA.n; ++i) {
                if (stA.sgn[i] == 1) pos[stA.mat[i]].push_back(i);
                else neg[stA.mat[i]].push_back(i);
            }
            int btype = 0, bi = 0, bj = 0, ba = 0, bp = 0;
            int64_t best_d = (1LL << 60);
            uint64_t ties = 0;
            auto consider = [&](int type, int i, int j, int a, int p, int ni_m, int ni_s, int nj_m, int nj_s, int64_t d) {
                bool is_tabu = (tabu[i * 4 + ni_m * 2 + (ni_s > 0)] > iter) ||
                               (tabu[j * 4 + nj_m * 2 + (nj_s > 0)] > iter);
                if (is_tabu && stA.F + d >= basin_best) return;
                if (d < best_d || (d == best_d && (rng() % (++ties + 1) == 0))) {
                    if (d < best_d) ties = 0;
                    best_d = d; btype = type; bi = i; bj = j; ba = a; bp = p;
                }
            };
            for (int a = 0; a < 2; ++a) {
                for (int i : pos[a]) {
                    for (int j : neg[a]) {
                        consider(1, i, j, a, 0, a, -1, a, 1, stA.eval_move1(i, j, a));
                        consider(3, i, j, a, 0, 1 - a, 1, 1 - a, -1, stA.eval_move3(i, j, a, 0));
                        consider(3, i, j, a, 1, 1 - a, -1, 1 - a, 1, stA.eval_move3(i, j, a, 1));
                    }
                }
            }
            for (int i : pos[0])
                for (int j : pos[1])
                    consider(2, i, j, 0, 1, 1, 1, 0, 1, stA.eval_move2(i, j, 1));
            for (int i : neg[0])
                for (int j : neg[1])
                    consider(2, i, j, 0, -1, 1, -1, 0, -1, stA.eval_move2(i, j, -1));

            if (btype == 0) { std::fill(tabu.begin(), tabu.end(), 0); continue; }
            int omi = stA.mat[bi], osi = stA.sgn[bi], omj = stA.mat[bj], osj = stA.sgn[bj];
            if (btype == 1) stA.apply_move1(bi, bj, ba);
            else if (btype == 2) stA.apply_move2(bi, bj, bp);
            else stA.apply_move3(bi, bj, ba, bp);
            int tv = tenure + (int)(rng() % 4);
            tabu[bi * 4 + omi * 2 + (osi > 0)] = iter + tv;
            tabu[bj * 4 + omj * 2 + (osj > 0)] = iter + tv;

            maybe_add();
            if (stA.F < basin_best) { basin_best = stA.F; stall = 0; }
            else if (++stall >= 200) {
                stA.init_random(rng);
                basin_best = stA.F;
                std::fill(tabu.begin(), tabu.end(), 0);
                stall = 0;
            }
        }
        double dt = elapsed();
        std::cout << "Half A: collected=" << out_rows.size() << ", iters=" << iter
                  << ", time=" << dt << "s (" << (dt > 0 ? out_rows.size() / dt : 0.0) << " uniq/s)"
                  << ", min_F=" << min_F
                  << ", mean_F=" << (out_rows.empty() ? 0.0 : (double)sum_F / out_rows.size()) << std::endl;
    };

    std::vector<StoredHalf> rowsA;
    std::vector<TableEntry> table;
    rowsA.reserve(pool_size);
    table.reserve(pool_size);
    harvest_A(rowsA, table, secs * 0.5);
    if (table.empty()) {
        std::cout << "No Half A states collected under flat_max_A=" << flat_max_A << "\n";
        return 2;
    }

    double sort_t0 = elapsed();
    std::sort(table.begin(), table.end(), [](const TableEntry& a, const TableEntry& b) {
        return a.hash < b.hash;
    });
    constexpr unsigned pbits = 18;
    constexpr size_t nbuckets = size_t{1} << pbits;
    std::vector<size_t> buckets(nbuckets + 1);
    size_t pos_e = 0;
    for (size_t b = 0; b < nbuckets; ++b) {
        buckets[b] = pos_e;
        while (pos_e < table.size() && (table[pos_e].hash >> (64 - pbits)) == b) ++pos_e;
    }
    buckets.back() = table.size();
    std::cout << "Table built: entries=" << table.size() << ", sort_s=" << (elapsed() - sort_t0) << std::endl;

    // Now stream Half B states and probe directly!
    std::unordered_set<uint64_t> seenB;
    seenB.reserve(pool_size * 2);
    std::vector<int> tabuB(stB.n * 4, 0);
    int iterB = 0, stallB = 0;
    int tenureB = std::max(3, stB.n / 5);
    int64_t min_FB = INT64_MAX;
    int64_t sum_FB = 0;
    uint64_t uniqB = 0, hash_hits = 0;
    std::vector<int> pos[2], neg[2];
    stB.init_random(rng);
    int64_t basin_bestB = stB.F;
    double probe_t0 = elapsed();

    auto rotate_orb = [&](int o, int k) -> int {
        if (!qr_split || o == 0 || k == 0) return o;
        int rep = inst.orbits[o][0];
        return inst.orb_id[(1LL * rep * inst.qr_powers[k]) % inst.t];
    };

    auto probe_B = [&]() -> bool {
        if (stB.F < min_FB) min_FB = stB.F;
        if (stB.F > flat_max_B) return false;
        int rotB = qr_split ? canonical_shift(stB.P.data(), inst.K, -1) : 0;
        uint64_t key = hash_paf(stB.P.data(), inst.K, rotB, -1, coeffs);
        if (!seenB.insert(key).second) return false;
        ++uniqB;
        sum_FB += stB.F;

        size_t b = key >> (64 - pbits);
        auto it = std::lower_bound(table.begin() + buckets[b], table.begin() + buckets[b + 1], key,
            [](const TableEntry& e, uint64_t v) { return e.hash < v; });
        auto end = table.begin() + buckets[b + 1];
        for (; it != end && it->hash == key; ++it) {
            ++hash_hits;
            const auto& rA = rowsA[it->idx];
            bool exact = true;
            for (int j = 0; j < inst.K; ++j) {
                if ((int)rA.paf[j] + stB.P[(rotB + j) % inst.K] != 0) {
                    exact = false;
                    break;
                }
            }
            if (!exact) continue;
            // Reconstruct full T-matrix!
            // If qr_split: rotating sequence by qr_powers[rot] shifts PAF(d_k) -> PAF(d_{(k+rot)%K})?
            // Let's test all K relative rotations between A and B (only K=15 or 21 checks!) to avoid sign/direction ambiguity!
            int k_lim = qr_split ? inst.K : 1;
            for (int rel = 0; rel < k_lim; ++rel) {
                std::vector<int> full_mat(inst.r, 0), full_sgn(inst.r, 1);
                for (int i = 0; i < stA.n; ++i) {
                    int oi = stA.orig_orb[i];
                    full_mat[oi] = rA.mat[i];
                    full_sgn[oi] = rA.sgn[i];
                }
                for (int i = 0; i < stB.n; ++i) {
                    int oi = rotate_orb(stB.orig_orb[i], rel);
                    full_mat[oi] = 2 + stB.mat[i];
                    full_sgn[oi] = stB.sgn[i];
                }
                if (verify_full_and_save(inst, full_mat, full_sgn, out_path)) {
                    std::cout << "FOUND EXACT VERIFIED T-MATRIX! time=" << elapsed()
                              << "s, uniqB=" << uniqB << ", hash_hits=" << hash_hits
                              << ", saved=" << out_path << std::endl;
                    return true;
                }
            }
        }
        return false;
    };

    bool found = false;
    while (uniqB < pool_size) {
        if ((iterB & 255) == 0 && elapsed() >= secs) break;
        ++iterB;
        for (int a = 0; a < 2; ++a) { pos[a].clear(); neg[a].clear(); }
        for (int i = 0; i < stB.n; ++i) {
            if (stB.sgn[i] == 1) pos[stB.mat[i]].push_back(i);
            else neg[stB.mat[i]].push_back(i);
        }
        int btype = 0, bi = 0, bj = 0, ba = 0, bp = 0;
        int64_t best_d = (1LL << 60);
        uint64_t ties = 0;
        auto consider = [&](int type, int i, int j, int a, int p, int ni_m, int ni_s, int nj_m, int nj_s, int64_t d) {
            bool is_tabu = (tabuB[i * 4 + ni_m * 2 + (ni_s > 0)] > iterB) ||
                           (tabuB[j * 4 + nj_m * 2 + (nj_s > 0)] > iterB);
            if (is_tabu && stB.F + d >= basin_bestB) return;
            if (d < best_d || (d == best_d && (rng() % (++ties + 1) == 0))) {
                if (d < best_d) ties = 0;
                best_d = d; btype = type; bi = i; bj = j; ba = a; bp = p;
            }
        };
        for (int a = 0; a < 2; ++a) {
            for (int i : pos[a]) {
                for (int j : neg[a]) {
                    consider(1, i, j, a, 0, a, -1, a, 1, stB.eval_move1(i, j, a));
                    consider(3, i, j, a, 0, 1 - a, 1, 1 - a, -1, stB.eval_move3(i, j, a, 0));
                    consider(3, i, j, a, 1, 1 - a, -1, 1 - a, 1, stB.eval_move3(i, j, a, 1));
                }
            }
        }
        for (int i : pos[0])
            for (int j : pos[1])
                consider(2, i, j, 0, 1, 1, 1, 0, 1, stB.eval_move2(i, j, 1));
        for (int i : neg[0])
            for (int j : neg[1])
                consider(2, i, j, 0, -1, 1, -1, 0, -1, stB.eval_move2(i, j, -1));

        if (btype == 0) { std::fill(tabuB.begin(), tabuB.end(), 0); continue; }
        int omi = stB.mat[bi], osi = stB.sgn[bi], omj = stB.mat[bj], osj = stB.sgn[bj];
        if (btype == 1) stB.apply_move1(bi, bj, ba);
        else if (btype == 2) stB.apply_move2(bi, bj, bp);
        else stB.apply_move3(bi, bj, ba, bp);
        int tv = tenureB + (int)(rng() % 4);
        tabuB[bi * 4 + omi * 2 + (osi > 0)] = iterB + tv;
        tabuB[bj * 4 + omj * 2 + (osj > 0)] = iterB + tv;

        if (probe_B()) { found = true; break; }
        if (stB.F < basin_bestB) { basin_bestB = stB.F; stallB = 0; }
        else if (++stallB >= 200) {
            stB.init_random(rng);
            basin_bestB = stB.F;
            std::fill(tabuB.begin(), tabuB.end(), 0);
            stallB = 0;
        }
    }
    double dtB = elapsed() - probe_t0;
    std::cout << "Half B: probed=" << uniqB << ", iters=" << iterB
              << ", time=" << dtB << "s (" << (dtB > 0 ? uniqB / dtB : 0.0) << " uniq/s)"
              << ", min_F=" << min_FB
              << ", mean_F=" << (uniqB == 0 ? 0.0 : (double)sum_FB / uniqB)
              << ", hash_hits=" << hash_hits
              << ", pairs_tested=" << (uint64_t)table.size() * uniqB * (qr_split ? inst.K : 1)
              << std::endl;
    return found ? 0 : 2;
}

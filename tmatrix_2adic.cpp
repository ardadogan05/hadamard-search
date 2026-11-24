// Standalone C++20 2-adic (mod 2 + mod 4) GF(2)-reduced T-matrix solver.
// Exploits the exact 2-stage GF(2) linear reduction for prime-order U-invariant T-matrices:
//   Level 1 (mod 2): A(X)B(X^-1) + B(X)A(X^-1) = 1 + J (mod 2) + 2 support-parity eqs
//                    => rank-(K+2) linear system in B given A (cuts B by 2^{K+2}).
//   Level 2 (mod 4): PAF(T, d_k) == 0 (mod 4) + 4 row-sum parity eqs
//                    => rank-(K+3) linear system in sign-flip mask N given support (S0..S3)
//                    => reduces 2^e signs to 2^{e-K-3} (2^12=4096 at t=151, 2^18=262144 at t=463).
//
// Modes:
//   --verify-163 : exact self-test verifying tmat_163.json is recovered with E=0
//   --mode=scan  : ultra-fast random Level-1 support sampling + exhaustive Level-2 Gray-code E=0 scan
//   --mode=ils   : guided outer search on Level-1 (A, B) manifold + inner Level-2 sign solver
//
// Build: g++ -std=c++20 -O3 -march=native tmatrix_2adic.cpp -o tmatrix_2adic

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

struct Edge {
    uint8_t a, b;
    int16_t w;
};

struct Problem {
    int t, g, h, r, e, K;
    int m[4];
    vector<vector<int>> orbits;
    vector<int> orb_id;
    vector<int> d_reps;
    vector<vector<vector<int>>> C; // [K][r][r]
    vector<int> diag_const;        // [K]
    // For Level 1: M_mod2[k][a] = bitmask over b in 0..e-1 where (C[k][a+1][b+1] + C[k][b+1][a+1]) is odd
    vector<vector<uint64_t>> M_mod2; // [K][e]
    // Precomputed unordered nonzero edges for all pairs (a, b) with 0 <= a < b < r
    vector<vector<Edge>> all_edges; // [K]
    uint64_t all_mask;
};

static Problem build_problem(int t, int g, int m0, int m1, int m2, int m3) {
    Problem P;
    P.t = t;
    P.g = g;
    P.m[0] = m0; P.m[1] = m1; P.m[2] = m2; P.m[3] = m3;

    vector<bool> seen(t, false);
    for (int x = 0; x < t; ++x) {
        if (!seen[x]) {
            vector<int> orb;
            int c = x;
            while (!seen[c]) {
                seen[c] = true;
                orb.push_back(c);
                c = (int)((1LL * c * g) % t);
            }
            P.orbits.push_back(move(orb));
        }
    }
    P.r = (int)P.orbits.size();
    P.e = P.r - 1;
    P.h = (int)P.orbits[1].size();
    if (P.r > 63) {
        fprintf(stderr, "ERROR: r=%d > 63 not supported by uint64_t masks\n", P.r);
        exit(1);
    }
    P.all_mask = (1ULL << P.e) - 1ULL;

    P.orb_id.assign(t, 0);
    for (int i = 0; i < P.r; ++i)
        for (int x : P.orbits[i]) P.orb_id[x] = i;

    vector<bool> seen_s(P.r, false);
    seen_s[0] = true;
    for (int i = 1; i < P.r; ++i) {
        int d = P.orbits[i][0];
        if (seen_s[P.orb_id[d]]) continue;
        seen_s[P.orb_id[d]] = true;
        seen_s[P.orb_id[(t - d) % t]] = true;
        P.d_reps.push_back(d);
    }
    P.K = (int)P.d_reps.size();
    P.C.assign(P.K, vector<vector<int>>(P.r, vector<int>(P.r, 0)));
    P.diag_const.assign(P.K, 0);
    P.M_mod2.assign(P.K, vector<uint64_t>(P.e, 0ULL));
    P.all_edges.resize(P.K);

    for (int k = 0; k < P.K; ++k) {
        int d = P.d_reps[k];
        for (int x = 0; x < t; ++x) {
            P.C[k][P.orb_id[x]][P.orb_id[(x + d) % t]]++;
        }
        for (int a = 0; a < P.r; ++a) P.diag_const[k] += P.C[k][a][a];
        for (int a = 0; a < P.r; ++a) {
            for (int b = a + 1; b < P.r; ++b) {
                int w = P.C[k][a][b] + P.C[k][b][a];
                if (w > 0) {
                    P.all_edges[k].push_back({(uint8_t)a, (uint8_t)b, (int16_t)w});
                    if (a >= 1 && b >= 1 && (w & 1)) {
                        P.M_mod2[k][a - 1] |= (1ULL << (b - 1));
                        P.M_mod2[k][b - 1] |= (1ULL << (a - 1));
                    }
                }
            }
        }
    }
    return P;
}

struct GF2Sol {
    bool ok = false;
    int rank = 0;
    uint64_t part = 0;
    int null_dim = 0;
    uint64_t null_basis[64];
};

// Fast 64-bit GF(2) Gaussian elimination on n_eq equations in n_vars variables (n_vars <= 62).
static inline GF2Sol solve_gf2(const uint64_t* rows, const uint8_t* rhs, int n_eq, int n_vars) {
    GF2Sol sol;
    uint64_t basis_m[64] = {};
    uint8_t basis_b[64] = {};
    uint64_t piv_mask = 0;

    for (int i = 0; i < n_eq; ++i) {
        uint64_t m = rows[i];
        uint8_t b = rhs[i] & 1;
        while (m) {
            int p = 63 - __builtin_clzll(m);
            if ((piv_mask >> p) & 1ULL) {
                m ^= basis_m[p];
                b ^= basis_b[p];
            } else {
                basis_m[p] = m;
                basis_b[p] = b;
                piv_mask |= (1ULL << p);
                sol.rank++;
                break;
            }
        }
        if (m == 0 && b) {
            sol.ok = false;
            return sol;
        }
    }

    // Back-substitute to RREF
    for (int p = 0; p < n_vars; ++p) {
        if (!((piv_mask >> p) & 1ULL)) continue;
        uint64_t pm = basis_m[p];
        uint8_t pb = basis_b[p];
        for (int p2 = p + 1; p2 < n_vars; ++p2) {
            if (((piv_mask >> p2) & 1ULL) && ((basis_m[p2] >> p) & 1ULL)) {
                basis_m[p2] ^= pm;
                basis_b[p2] ^= pb;
            }
        }
    }

    uint64_t part = 0;
    for (int p = 0; p < n_vars; ++p) {
        if (((piv_mask >> p) & 1ULL) && basis_b[p]) part |= (1ULL << p);
    }
    sol.part = part;
    sol.null_dim = 0;
    for (int fv = 0; fv < n_vars; ++fv) {
        if ((piv_mask >> fv) & 1ULL) continue;
        uint64_t vec = (1ULL << fv);
        for (int p = 0; p < n_vars; ++p) {
            if (((piv_mask >> p) & 1ULL) && ((basis_m[p] >> fv) & 1ULL)) {
                vec |= (1ULL << p);
            }
        }
        sol.null_basis[sol.null_dim++] = vec;
    }
    sol.ok = true;
    return sol;
}

// Level 1: given A (e-bit mask, bit j = orbit j+1), solve for B (e-bit mask)
// satisfying A(X)B(X^-1) + B(X)A(X^-1) = 1 + J (mod 2) + 2 support-parity eqs.
static inline GF2Sol solve_level1(const Problem& P, uint64_t A) {
    uint64_t rows[64];
    uint8_t rhs[64];
    int n_eq = P.K + 2;
    for (int k = 0; k < P.K; ++k) {
        uint64_t rmask = 0;
        uint64_t a_bits = A;
        const auto& Mk = P.M_mod2[k];
        while (a_bits) {
            int a = __builtin_ctzll(a_bits);
            rmask ^= Mk[a];
            a_bits &= a_bits - 1;
        }
        rows[k] = rmask;
        rhs[k] = 1;
    }
    // Support parity eq 1: popcount(B) == (m2 + m3) mod 2
    rows[P.K] = P.all_mask;
    rhs[P.K] = (uint8_t)((abs(P.m[2]) + abs(P.m[3])) & 1);
    // Support parity eq 2: popcount(A & B) == m3 mod 2
    rows[P.K + 1] = A;
    rhs[P.K + 1] = (uint8_t)(abs(P.m[3]) & 1);

    return solve_gf2(rows, rhs, n_eq, P.e);
}

struct SupportData {
    uint64_t A, B;
    uint64_t S[4];      // e-bit masks (bits 0..e-1 = orbits 1..e)
    int n[4];           // popcount(S[i])
    int target_neg[4];  // (n[i] - m[i]) / 2
    uint8_t mat[64];    // mat[0] = 0, mat[1..e] in {0,1,2,3}
    int paf_S[64];      // sum_i PAF(S_i, d_k) (guaranteed EVEN by Level 1!)
    GF2Sol l2;
    // Same-matrix edges per shift k (using 1-based orbit indices 0..r-1; bit 0 of N_full is always 0!)
    int n_edges[64];
    Edge same_edges[64][256];
};

static inline bool build_support_and_level2(const Problem& P, uint64_t A, uint64_t B, SupportData& sd) {
    sd.A = A;
    sd.B = B;
    sd.S[0] = (~A & ~B) & P.all_mask;
    sd.S[1] = ( A & ~B) & P.all_mask;
    sd.S[2] = (~A &  B) & P.all_mask;
    sd.S[3] = ( A &  B) & P.all_mask;
    for (int i = 0; i < 4; ++i) {
        sd.n[i] = __builtin_popcountll(sd.S[i]);
        if (sd.n[i] < abs(P.m[i]) || ((sd.n[i] - P.m[i]) & 1) != 0) return false;
        sd.target_neg[i] = (sd.n[i] - P.m[i]) >> 1;
    }
    sd.mat[0] = 0;
    for (int j = 0; j < P.e; ++j) {
        sd.mat[j + 1] = (uint8_t)(((A >> j) & 1ULL) | (((B >> j) & 1ULL) << 1));
    }

    uint64_t rows[64];
    uint8_t rhs[64];
    for (int k = 0; k < P.K; ++k) {
        int paf_s = P.diag_const[k];
        uint64_t rmask = 0;
        int ne = 0;
        for (const auto& ed : P.all_edges[k]) {
            if (sd.mat[ed.a] == sd.mat[ed.b]) {
                paf_s += ed.w;
                sd.same_edges[k][ne++] = ed;
                if (ed.w & 1) {
                    if (ed.a >= 1) rmask ^= (1ULL << (ed.a - 1));
                    if (ed.b >= 1) rmask ^= (1ULL << (ed.b - 1));
                }
            }
        }
        // Sanity: Level 1 guarantees paf_s is EVEN!
        if (paf_s & 1) return false;
        sd.paf_S[k] = paf_s;
        sd.n_edges[k] = ne;
        rows[k] = rmask;
        rhs[k] = (uint8_t)((paf_s >> 1) & 1);
    }
    for (int i = 0; i < 4; ++i) {
        rows[P.K + i] = sd.S[i];
        rhs[P.K + i] = (uint8_t)(sd.target_neg[i] & 1);
    }
    sd.l2 = solve_gf2(rows, rhs, P.K + 4, P.e);
    return sd.l2.ok;
}

// Evaluate exact PAF(d_k) for shift k given N (e-bit mask; shifted left by 1 so bit 0 = orbit 0 = 0).
static inline int eval_paf_k(const SupportData& sd, int k, uint64_t N_full) {
    int val = sd.paf_S[k];
    int ne = sd.n_edges[k];
    const Edge* eds = sd.same_edges[k];
    int neg_sum = 0;
    for (int i = 0; i < ne; ++i) {
        uint64_t diff = ((N_full >> eds[i].a) ^ (N_full >> eds[i].b)) & 1ULL;
        if (diff) neg_sum += eds[i].w;
    }
    return val - 2 * neg_sum;
}

// Compute full E_paf = sum_{k=0}^{K-1} PAF(d_k)^2 (with optional early abort at cutoff).
static inline int64_t eval_E_paf(const Problem& P, const SupportData& sd, uint64_t N, int64_t cutoff = (1LL << 60)) {
    uint64_t N_full = N << 1;
    int64_t E = 0;
    for (int k = 0; k < P.K; ++k) {
        int pk = eval_paf_k(sd, k, N_full);
        E += (int64_t)pk * pk;
        if (E >= cutoff) return E;
    }
    return E;
}

static inline bool check_exact_row_sums(const SupportData& sd, uint64_t N) {
    return (__builtin_popcountll(N & sd.S[0]) == sd.target_neg[0]) &&
           (__builtin_popcountll(N & sd.S[1]) == sd.target_neg[1]) &&
           (__builtin_popcountll(N & sd.S[2]) == sd.target_neg[2]) &&
           (__builtin_popcountll(N & sd.S[3]) == sd.target_neg[3]);
}

static inline int eval_row_err(const SupportData& sd, uint64_t N) {
    int e = 0;
    for (int i = 0; i < 4; ++i) {
        int d = __builtin_popcountll(N & sd.S[i]) - sd.target_neg[i];
        e += d * d;
    }
    return e;
}

// Exhaustive Gray-code scan of ALL 2^{null_dim} Level-2 sign states!
// Returns best E_paf among exact-row-sum states (or 1e18 if none), and writes best_N.
static int64_t exhaustive_level2_scan(
    const Problem& P, const SupportData& sd, uint64_t& best_N, uint64_t& n_row_ok, bool stop_on_zero = true, int64_t cutoff_E = (1LL << 60)
) {
    int d = sd.l2.null_dim;
    uint64_t total = 1ULL << d;
    uint64_t N = sd.l2.part;
    int64_t best_E = cutoff_E;
    best_N = N;
    n_row_ok = 0;

    for (uint64_t step = 0; step < total; ++step) {
        if (step > 0) {
            int bit = __builtin_ctzll(step);
            N ^= sd.l2.null_basis[bit];
        }
        if (check_exact_row_sums(sd, N)) {
            n_row_ok++;
            int64_t E = eval_E_paf(P, sd, N, best_E);
            if (E < best_E) {
                best_E = E;
                best_N = N;
                if (E == 0 && stop_on_zero) return 0;
            }
        }
    }
    return best_E;
}

// Fast inner local search on the d-dimensional ker_N space (for t=463 where d=18).
// Minimizes E_paf + w_row * E_row using steepest descent + random restarts.
static int64_t inner_level2_search(
    const Problem& P, const SupportData& sd, mt19937_64& rng,
    uint64_t& best_N, int64_t& best_exact_E, int n_starts = 4, int max_steps = 40
) {
    int d = sd.l2.null_dim;
    int64_t best_obj = (1LL << 60);
    best_exact_E = (1LL << 60);
    best_N = sd.l2.part;
    const int64_t W_ROW = 64;

    for (int st = 0; st < n_starts; ++st) {
        uint64_t N = sd.l2.part;
        for (int j = 0; j < d; ++j) {
            if (rng() & 1ULL) N ^= sd.l2.null_basis[j];
        }
        int re = eval_row_err(sd, N);
        int64_t pe = eval_E_paf(P, sd, N);
        int64_t obj = pe + W_ROW * re;
        if (re == 0 && pe < best_exact_E) { best_exact_E = pe; best_N = N; }
        if (obj < best_obj) { best_obj = obj; if (best_exact_E == (1LL << 60)) best_N = N; }
        if (re == 0 && pe == 0) return 0;

        for (int iter = 0; iter < max_steps; ++iter) {
            int best_j = -1;
            int64_t best_cand_obj = obj;
            int64_t best_cand_pe = pe;
            int best_cand_re = re;
            for (int j = 0; j < d; ++j) {
                uint64_t N2 = N ^ sd.l2.null_basis[j];
                int re2 = eval_row_err(sd, N2);
                int64_t max_pe = best_cand_obj - W_ROW * re2;
                if (max_pe <= 0) continue;
                int64_t pe2 = eval_E_paf(P, sd, N2, max_pe);
                int64_t obj2 = pe2 + W_ROW * re2;
                if (re2 == 0 && pe2 < best_exact_E) {
                    best_exact_E = pe2;
                    best_N = N2;
                    if (pe2 == 0) return 0;
                }
                if (obj2 < best_cand_obj) {
                    best_cand_obj = obj2;
                    best_cand_pe = pe2;
                    best_cand_re = re2;
                    best_j = j;
                }
            }
            if (best_j < 0) break;
            N ^= sd.l2.null_basis[best_j];
            obj = best_cand_obj;
            pe = best_cand_pe;
            re = best_cand_re;
            if (obj < best_obj) best_obj = obj;
        }
    }
    return best_obj;
}

static void save_witness_json(const string& path, const Problem& P, const SupportData& sd, uint64_t N) {
    ofstream out(path);
    out << "{\n";
    out << "  \"t\": " << P.t << ",\n";
    out << "  \"g\": " << P.g << ",\n";
    out << "  \"r\": " << P.r << ",\n";
    out << "  \"K\": " << P.K << ",\n";
    out << "  \"orbit_mat\": [";
    for (int i = 0; i < P.r; ++i) out << (i ? ", " : "") << (int)sd.mat[i];
    out << "],\n  \"orbit_sign\": [";
    for (int i = 0; i < P.r; ++i) {
        int s = (i == 0) ? 1 : (((N >> (i - 1)) & 1ULL) ? -1 : 1);
        out << (i ? ", " : "") << s;
    }
    out << "]\n}\n";
    out.close();
}

static bool self_test_163() {
    // Known witness from tmat_163.json: t=163, g=38, m=(0, 1, 1, 0)
    Problem P = build_problem(163, 38, 0, 1, 1, 0);
    int om[19] = {0, 2, 2, 0, 3, 0, 3, 3, 0, 0, 1, 0, 3, 0, 1, 1, 0, 0, 2};
    int os[19] = {1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1};
    uint64_t A = 0, B = 0, N = 0;
    for (int i = 1; i < 19; ++i) {
        if (om[i] & 1) A |= (1ULL << (i - 1));
        if (om[i] & 2) B |= (1ULL << (i - 1));
        if (os[i] == -1) N |= (1ULL << (i - 1));
    }
    GF2Sol l1 = solve_level1(P, A);
    if (!l1.ok) { fprintf(stderr, "[FAIL] Level 1 failed on tmat_163\n"); return false; }
    SupportData sd;
    if (!build_support_and_level2(P, A, B, sd)) {
        fprintf(stderr, "[FAIL] Level 2 failed on tmat_163\n"); return false;
    }
    uint64_t found_N = 0, n_row_ok = 0;
    int64_t E = exhaustive_level2_scan(P, sd, found_N, n_row_ok, true);
    if (E != 0) {
        fprintf(stderr, "[FAIL] Exhaustive Level-2 scan did not recover E=0 on tmat_163 (got E=%lld)\n", (long long)E);
        return false;
    }
    printf("[SELF-TEST PASSED] tmat_163.json recovered with E=0! (L1 rank=%d/%d, L2 rank=%d/%d, null_dim=%d, row_ok=%llu)\n",
           l1.rank, P.K + 2, sd.l2.rank, P.K + 4, sd.l2.null_dim, (unsigned long long)n_row_ok);
    return true;
}

static inline uint64_t random_A_with_parity(const Problem& P, mt19937_64& rng) {
    uint64_t A = rng() & P.all_mask;
    int need_par = (abs(P.m[1]) + abs(P.m[3])) & 1;
    if ((__builtin_popcountll(A) & 1) != need_par) {
        A ^= (1ULL << (rng() % P.e));
    }
    return A;
}

static inline uint64_t sample_B_from_l1(const GF2Sol& l1, mt19937_64& rng) {
    uint64_t B = l1.part;
    uint64_t bits = rng();
    for (int j = 0; j < l1.null_dim; ++j) {
        if ((bits >> j) & 1ULL) B ^= l1.null_basis[j];
    }
    return B;
}

int main(int argc, char** argv) {
    int t = 151, g = 8, m0 = 0, m1 = 2, m2 = 1, m3 = 1;
    double time_limit = 30.0;
    string out_path = "/tmp/tmatrix_2adic_out.json";
    string mode = "scan";
    uint64_t seed = 2026;
    bool verify_only = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--verify-163") verify_only = true;
        else if (arg == "--t" && i + 1 < argc) t = atoi(argv[++i]);
        else if (arg == "--g" && i + 1 < argc) g = atoi(argv[++i]);
        else if (arg == "--m" && i + 4 < argc) {
            m0 = atoi(argv[++i]); m1 = atoi(argv[++i]); m2 = atoi(argv[++i]); m3 = atoi(argv[++i]);
        }
        else if (arg == "--time" && i + 1 < argc) time_limit = atof(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
    }

    if (!self_test_163()) return 1;
    if (verify_only) return 0;

    Problem P = build_problem(t, g, m0, m1, m2, m3);
    printf("Starting tmatrix_2adic: t=%d g=%d |U|=%d r=%d e=%d K=%d m=(%d,%d,%d,%d) mode=%s time=%.1fs seed=%llu\n",
           P.t, P.g, P.h, P.r, P.e, P.K, m0, m1, m2, m3, mode.c_str(), time_limit, (unsigned long long)seed);
    fflush(stdout);

    mt19937_64 rng(seed);
    auto t0 = chrono::steady_clock::now();
    SupportData sd;

    uint64_t l1_solves = 0, supports_tested = 0, total_row_ok = 0;
    int64_t global_best_E = (1LL << 60);
    bool found = false;

    if (mode == "scan") {
        // Ultra-fast random Level-1 support sampling + exhaustive Level-2 Gray-code scan (for t=163, t=151)
        while (true) {
            if ((supports_tested & 255ULL) == 0) {
                double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                if (el >= time_limit) break;
            }
            uint64_t A = random_A_with_parity(P, rng);
            GF2Sol l1 = solve_level1(P, A);
            l1_solves++;
            if (!l1.ok) continue;

            // Sample several B's per A (amortizing Level-1 solve cost!)
            int b_batch = (P.e <= 30) ? 16 : 4;
            for (int bi = 0; bi < b_batch; ++bi) {
                uint64_t B = sample_B_from_l1(l1, rng);
                if (!build_support_and_level2(P, A, B, sd)) continue;
                supports_tested++;
                uint64_t best_N = 0, n_row_ok = 0;
                // Pass cutoff = 1 for pure E=0 speed when we already have a small global_best_E,
                // or global_best_E to track best E_paf!
                int64_t E = exhaustive_level2_scan(P, sd, best_N, n_row_ok, true, global_best_E);
                total_row_ok += n_row_ok;
                if (E < global_best_E) {
                    global_best_E = E;
                    double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                    printf("  [%.3fs] NEW BEST E=%lld (supports=%llu, rate=%.0f/s, null_dim=%d)\n",
                           el, (long long)global_best_E, (unsigned long long)supports_tested,
                           supports_tested / max(el, 1e-6), sd.l2.null_dim);
                    fflush(stdout);
                    if (E == 0) {
                        save_witness_json(out_path, P, sd, best_N);
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;
        }
    } else if (mode == "ils") {
        // Guided outer search on Level-1 (A, B) manifold + inner Level-2 sign solver (for t=463 and t=151)
        uint64_t cur_A = 0, cur_B = 0;
        GF2Sol cur_l1;
        int64_t cur_score = (1LL << 60);
        int stag = 0;

        auto init_random = [&]() {
            while (true) {
                cur_A = random_A_with_parity(P, rng);
                cur_l1 = solve_level1(P, cur_A);
                l1_solves++;
                if (!cur_l1.ok) continue;
                cur_B = sample_B_from_l1(cur_l1, rng);
                if (!build_support_and_level2(P, cur_A, cur_B, sd)) continue;
                supports_tested++;
                uint64_t bn = 0;
                int64_t ex_E = (1LL << 60);
                if (sd.l2.null_dim <= 12) {
                    uint64_t nr = 0;
                    ex_E = exhaustive_level2_scan(P, sd, bn, nr, true);
                    cur_score = ex_E;
                } else {
                    cur_score = inner_level2_search(P, sd, rng, bn, ex_E, 4, 30);
                }
                if (ex_E < global_best_E) {
                    global_best_E = ex_E;
                    if (ex_E == 0) { save_witness_json(out_path, P, sd, bn); found = true; }
                }
                stag = 0;
                break;
            }
        };

        init_random();
        while (!found) {
            if ((supports_tested & 63ULL) == 0) {
                double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                if (el >= time_limit) break;
            }

            // Propose move on Level-1 manifold:
            // 70%: flip 1 kernel vector in B (keeps A fixed, exact Level-1!)
            // 30%: flip 2 bits in A (preserves parity), re-solve L1, pick B in new subspace
            bool move_A = ((rng() % 10) < 3);
            uint64_t cand_A = cur_A;
            uint64_t cand_B = cur_B;
            GF2Sol cand_l1 = cur_l1;

            if (!move_A && cur_l1.null_dim > 0) {
                int j = (int)(rng() % cur_l1.null_dim);
                cand_B = cur_B ^ cur_l1.null_basis[j];
            } else {
                int b1 = (int)(rng() % P.e);
                int b2 = (int)(rng() % P.e);
                while (b2 == b1) b2 = (int)(rng() % P.e);
                cand_A = cur_A ^ (1ULL << b1) ^ (1ULL << b2);
                cand_l1 = solve_level1(P, cand_A);
                l1_solves++;
                if (!cand_l1.ok) continue;
                cand_B = sample_B_from_l1(cand_l1, rng);
            }

            if (!build_support_and_level2(P, cand_A, cand_B, sd)) continue;
            supports_tested++;

            uint64_t bn = 0;
            int64_t ex_E = (1LL << 60);
            int64_t score = (1LL << 60);
            if (sd.l2.null_dim <= 12) {
                uint64_t nr = 0;
                ex_E = exhaustive_level2_scan(P, sd, bn, nr, true);
                total_row_ok += nr;
                score = ex_E;
            } else {
                score = inner_level2_search(P, sd, rng, bn, ex_E, 4, 30);
                // If inner search gets very close, run exhaustive 2^18 Level-2 sweep!
                if (ex_E > 0 && ex_E <= 128 && sd.l2.null_dim <= 18) {
                    uint64_t nr = 0, bn2 = 0;
                    int64_t ex_E2 = exhaustive_level2_scan(P, sd, bn2, nr, true, ex_E);
                    total_row_ok += nr;
                    if (ex_E2 < ex_E) { ex_E = ex_E2; bn = bn2; }
                }
            }

            if (ex_E < global_best_E) {
                global_best_E = ex_E;
                double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
                printf("  [%.3fs] NEW BEST E=%lld (supports=%llu, rate=%.0f/s, null_dim=%d)\n",
                       el, (long long)global_best_E, (unsigned long long)supports_tested,
                       supports_tested / max(el, 1e-6), sd.l2.null_dim);
                fflush(stdout);
                if (ex_E == 0) {
                    save_witness_json(out_path, P, sd, bn);
                    found = true;
                    break;
                }
            }

            // Accept if better or small-temperature SA acceptance
            if (score <= cur_score || (score - cur_score <= 64 && (rng() % 8 == 0))) {
                cur_A = cand_A;
                cur_B = cand_B;
                cur_l1 = cand_l1;
                cur_score = score;
                stag = 0;
            } else {
                stag++;
                if (stag >= 150) init_random();
            }
        }
    }

    double elapsed = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    printf("DONE in %.3fs: best_E=%lld, supports=%llu (%.0f/s), l1_solves=%llu, row_ok=%llu%s\n",
           elapsed, (long long)global_best_E, (unsigned long long)supports_tested,
           supports_tested / max(elapsed, 1e-6), (unsigned long long)l1_solves,
           (unsigned long long)total_row_ok, found ? " [FOUND E=0!]" : "");
    return found ? 0 : 2;
}

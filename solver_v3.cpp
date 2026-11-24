#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

static constexpr int64_t BASE_ROW_WEIGHT = 4;

struct SubsetSumDP {
    std::vector<int> orb_ids;
    std::vector<int> vals;
    int total_sum = 0;
    std::vector<uint64_t> dp;
    bool built = false;

    static constexpr uint64_t CAP = (1ULL << 60);

    void build(const std::vector<int>& ids, const std::vector<int>& sizes) {
        orb_ids = ids;
        vals = sizes;
        total_sum = 0;
        for (int v : vals) total_sum += v;
        int n = (int)vals.size();
        dp.assign((size_t)(n + 1) * (total_sum + 1), 0);
        dp[0] = 1;
        for (int i = 0; i < n; ++i) {
            int v = vals[i];
            const uint64_t* prev = &dp[(size_t)i * (total_sum + 1)];
            uint64_t* cur = &dp[(size_t)(i + 1) * (total_sum + 1)];
            for (int s = 0; s <= total_sum; ++s) {
                uint64_t c = prev[s];
                if (s >= v) {
                    uint64_t a = prev[s - v];
                    c = (c > CAP - a) ? CAP : (c + a);
                }
                cur[s] = c;
            }
        }
        built = true;
    }

    bool can_hit(int T) const {
        if (!built) return false;
        int num = total_sum + T;
        if (num < 0 || (num & 1) != 0) return false;
        int half = num >> 1;
        if (half > total_sum) return false;
        return dp[(size_t)vals.size() * (total_sum + 1) + half] > 0;
    }

    bool sample(int T, std::vector<int>& seq, std::mt19937_64& rng) const {
        if (!can_hit(T)) return false;
        int s = (total_sum + T) >> 1;
        int n = (int)vals.size();
        for (int i = n - 1; i >= 0; --i) {
            int v = vals[i];
            const uint64_t* prev = &dp[(size_t)i * (total_sum + 1)];
            uint64_t c_minus = prev[s];
            uint64_t c_plus = (s >= v) ? prev[s - v] : 0;
            uint64_t tot = c_minus + c_plus;
            bool pick_plus = false;
            if (c_plus == 0) {
                pick_plus = false;
            } else if (c_minus == 0) {
                pick_plus = true;
            } else {
                uint64_t r = rng() % tot;
                pick_plus = (r < c_plus);
            }
            if (pick_plus) {
                seq[orb_ids[i]] = 1;
                s -= v;
            } else {
                seq[orb_ids[i]] = -1;
            }
        }
        return true;
    }
};

struct AltTuple {
    int a[4];
};

struct OrbitTriple {
    int i, j, k; // orb_size[i] == orb_size[j] + orb_size[k], j < k
};

struct OrbitProblem {
    int n = 0, m = 0, b = 0;
    std::vector<int> gens;
    std::vector<int> U;
    int r = 0, K = 0, Kpad = 0;
    bool use_alt = false;
    bool use_target_alt = false;
    int target_alt[4] = {0, 0, 0, 0};
    int64_t w_row = 64;
    int64_t w_alt = 16;
    bool lock_row = false;
    bool flat_init = false;
    int max_orb_size = 1;

    bool infeasible = false;
    std::string infeasible_reason;

    std::vector<std::vector<int>> orbits;
    std::vector<int> orb_size;
    std::vector<int> orb_alt;
    std::vector<int> elem_to_orb;
    std::vector<int> shift_reps;
    std::vector<int> shift_weights;
    std::vector<int> MT; // MT[(i * r + j) * Kpad + k]
    std::vector<int> diag_const;
    int target_ac = 0;
    int target_s[4] = {2, 0, 0, 0};
    std::vector<std::vector<int>> size_groups; // groups of orbits with identical size (len >= 2)
    std::vector<std::pair<int, int>> same_size_pairs;
    std::vector<std::pair<int, int>> same_size_same_alt_pairs;
    std::vector<OrbitTriple> row_preserving_triples;

    // Exact manifold / row-sum DP
    bool pure_parity_orbits = false;
    SubsetSumDP even_dp;
    SubsetSumDP odd_dp;
    SubsetSumDP full_dp;
    std::vector<AltTuple> valid_alt_tuples;
};

static std::vector<int> close_subgroup(int m, const std::vector<int>& gens) {
    std::vector<uint8_t> in_U(m, 0);
    std::vector<int> U;
    in_U[1 % m] = 1;
    U.push_back(1 % m);
    std::vector<int> clean_gens;
    for (int g : gens) {
        int gg = ((g % m) + m) % m;
        if (gg == 0 || std::gcd(gg, m) != 1) {
            std::cerr << "WARNING: generator " << g << " is not coprime to m=" << m << " (skipped)\n";
            continue;
        }
        clean_gens.push_back(gg);
    }
    for (size_t idx = 0; idx < U.size(); ++idx) {
        int u = U[idx];
        for (int g : clean_gens) {
            int v = (int)((1LL * u * g) % m);
            if (!in_U[v]) {
                in_U[v] = 1;
                U.push_back(v);
            }
        }
    }
    std::sort(U.begin(), U.end());
    return U;
}

OrbitProblem build_problem(int m, const std::vector<int>& gens, int b, const int ts[4],
                           int64_t w_row, int64_t w_alt, bool lock_row, bool flat_init,
                           bool use_talt, const int talt[4]) {
    OrbitProblem prob;
    prob.m = m;
    prob.gens = gens;
    prob.b = b;
    prob.n = 4 * m + b;
    prob.target_ac = -b;
    prob.use_alt = (m % 2 == 0) && (b == 4 || b == 0);
    prob.use_target_alt = prob.use_alt && use_talt;
    prob.w_row = w_row;
    prob.w_alt = prob.use_alt ? w_alt : 0;
    prob.lock_row = lock_row;
    prob.flat_init = flat_init;
    prob.max_orb_size = 1;
    for (int s = 0; s < 4; ++s) {
        prob.target_s[s] = ts[s];
        prob.target_alt[s] = talt[s];
    }

    prob.U = close_subgroup(m, gens);

    prob.elem_to_orb.assign(m, -1);
    for (int x = 0; x < m; ++x) {
        if (prob.elem_to_orb[x] == -1) {
            int oid = (int)prob.orbits.size();
            std::vector<int> orb;
            int alt_c = 0;
            for (int u : prob.U) {
                int y = (int)((1LL * x * u) % m);
                if (prob.elem_to_orb[y] == -1) {
                    prob.elem_to_orb[y] = oid;
                    orb.push_back(y);
                    alt_c += ((y & 1) == 0) ? 1 : -1;
                }
            }
            prob.orb_size.push_back((int)orb.size());
            if ((int)orb.size() > prob.max_orb_size) prob.max_orb_size = (int)orb.size();
            prob.orb_alt.push_back(alt_c);
            prob.orbits.push_back(std::move(orb));
        }
    }
    prob.r = (int)prob.orbits.size();

    for (int i = 0; i < prob.r; ++i) {
        for (int j = i + 1; j < prob.r; ++j) {
            if (prob.orb_size[i] == prob.orb_size[j]) {
                prob.same_size_pairs.push_back({i, j});
                if (prob.orb_alt[i] == prob.orb_alt[j]) {
                    prob.same_size_same_alt_pairs.push_back({i, j});
                }
            }
        }
    }

    // Group orbits by size (for branchless pos x neg swap iteration)
    {
        std::vector<int> seen_sz;
        for (int i = 0; i < prob.r; ++i) {
            int sz = prob.orb_size[i];
            if (std::find(seen_sz.begin(), seen_sz.end(), sz) == seen_sz.end()) {
                seen_sz.push_back(sz);
                std::vector<int> grp;
                for (int j = 0; j < prob.r; ++j) {
                    if (prob.orb_size[j] == sz) grp.push_back(j);
                }
                if (grp.size() >= 2) prob.size_groups.push_back(std::move(grp));
            }
        }
    }

    // Build row-sum-preserving 3-orbit triples: orb_size[i] == orb_size[j] + orb_size[k]
    std::vector<OrbitTriple> all_triples;
    for (int i = 0; i < prob.r; ++i) {
        int si = prob.orb_size[i];
        for (int j = 0; j < prob.r; ++j) {
            if (j == i) continue;
            int sj = prob.orb_size[j];
            if (sj >= si) continue;
            int need = si - sj;
            for (int k = j + 1; k < prob.r; ++k) {
                if (k == i) continue;
                if (prob.orb_size[k] == need) {
                    all_triples.push_back({i, j, k});
                }
            }
        }
    }
    const size_t MAX_TRIPLES = 384;
    if (all_triples.size() <= MAX_TRIPLES) {
        prob.row_preserving_triples = std::move(all_triples);
    } else {
        prob.row_preserving_triples.reserve(MAX_TRIPLES);
        for (size_t idx = 0; idx < MAX_TRIPLES; ++idx) {
            size_t pos = (idx * all_triples.size()) / MAX_TRIPLES;
            prob.row_preserving_triples.push_back(all_triples[pos]);
        }
    }

    std::vector<uint8_t> seen_d(m, 0);
    seen_d[0] = 1;
    for (int d = 1; d < m; ++d) {
        if (!seen_d[d]) {
            prob.shift_reps.push_back(d);
            int w = 0;
            for (int u : prob.U) {
                int y = (int)((1LL * d * u) % m);
                if (!seen_d[y]) {
                    seen_d[y] = 1;
                    ++w;
                }
                int neg = (m - y) % m;
                if (!seen_d[neg]) {
                    seen_d[neg] = 1;
                    ++w;
                }
            }
            prob.shift_weights.push_back(w);
        }
    }
    prob.K = (int)prob.shift_reps.size();
    prob.Kpad = (prob.K + 7) & ~7;

    prob.MT.assign((size_t)prob.r * prob.r * prob.Kpad, 0);
    prob.diag_const.assign(prob.Kpad, 0);

    for (int k = 0; k < prob.K; ++k) {
        int d = prob.shift_reps[k];
        std::vector<int> C((size_t)prob.r * prob.r, 0);
        for (int x = 0; x < m; ++x) {
            int i = prob.elem_to_orb[x];
            int j = prob.elem_to_orb[(x + d) % m];
            C[(size_t)i * prob.r + j]++;
        }
        int dc = 0;
        for (int i = 0; i < prob.r; ++i) {
            dc += C[(size_t)i * prob.r + i];
            for (int j = 0; j < prob.r; ++j) {
                if (i != j) {
                    prob.MT[((size_t)i * prob.r + j) * prob.Kpad + k] =
                        C[(size_t)i * prob.r + j] + C[(size_t)j * prob.r + i];
                }
            }
        }
        prob.diag_const[k] = dc;

        int64_t lb = 4LL * (2LL * dc - m);
        if (lb > prob.target_ac) {
            prob.infeasible = true;
            std::ostringstream oss;
            oss << "Diagonal-Shift Obstruction at d=" << d
                << ": diag_const=" << dc << "/" << m
                << " forces sum(PAF) >= " << lb << " > target " << prob.target_ac;
            prob.infeasible_reason = oss.str();
            return prob;
        }
    }

    // Build exact DP for row-sum (and 2-compression when m is even)
    if (m <= 20000) {
        std::vector<int> all_ids(prob.r);
        std::iota(all_ids.begin(), all_ids.end(), 0);
        prob.full_dp.build(all_ids, prob.orb_size);
        for (int s = 0; s < 4; ++s) {
            if (!prob.full_dp.can_hit(prob.target_s[s])) {
                prob.infeasible = true;
                std::ostringstream oss;
                oss << "Row-Sum Obstruction: target_s[" << s << "]=" << prob.target_s[s]
                    << " not achievable by orbit sizes";
                prob.infeasible_reason = oss.str();
                return prob;
            }
        }

        if (prob.use_alt) {
            std::vector<int> even_ids, even_sz, odd_ids, odd_sz;
            bool pure = true;
            for (int i = 0; i < prob.r; ++i) {
                if (prob.orb_alt[i] == prob.orb_size[i]) {
                    even_ids.push_back(i);
                    even_sz.push_back(prob.orb_size[i]);
                } else if (prob.orb_alt[i] == -prob.orb_size[i]) {
                    odd_ids.push_back(i);
                    odd_sz.push_back(prob.orb_size[i]);
                } else {
                    pure = false;
                    break;
                }
            }
            if (pure) {
                prob.pure_parity_orbits = true;
                prob.even_dp.build(even_ids, even_sz);
                prob.odd_dp.build(odd_ids, odd_sz);

                auto can_hit_alt = [&](int s_idx, int a) -> bool {
                    int rs = prob.target_s[s_idx];
                    if (((rs + a) & 1) != 0) return false;
                    int te = (rs + a) >> 1;
                    int to = (rs - a) >> 1;
                    return prob.even_dp.can_hit(te) && prob.odd_dp.can_hit(to);
                };

                if (prob.use_target_alt) {
                    std::vector<int> opts[4];
                    for (int s = 0; s < 4; ++s) {
                        int ta = std::abs(prob.target_alt[s]);
                        if (can_hit_alt(s, ta)) opts[s].push_back(ta);
                        if (ta != 0 && can_hit_alt(s, -ta)) opts[s].push_back(-ta);
                    }
                    for (int a0 : opts[0])
                        for (int a1 : opts[1])
                            for (int a2 : opts[2])
                                for (int a3 : opts[3])
                                    prob.valid_alt_tuples.push_back({{a0, a1, a2, a3}});
                } else {
                    int lim = (int)std::sqrt((double)prob.n);
                    std::vector<int> valid_a[4];
                    for (int s = 0; s < 4; ++s) {
                        for (int a = -lim; a <= lim; ++a) {
                            if (can_hit_alt(s, a)) valid_a[s].push_back(a);
                        }
                    }
                    std::vector<std::vector<int>> sq3(prob.n + 1);
                    for (int a3 : valid_a[3]) {
                        int q = a3 * a3;
                        if (q <= prob.n) sq3[q].push_back(a3);
                    }
                    for (int a0 : valid_a[0]) {
                        int rem0 = prob.n - a0 * a0;
                        if (rem0 < 0) continue;
                        for (int a1 : valid_a[1]) {
                            int rem1 = rem0 - a1 * a1;
                            if (rem1 < 0) continue;
                            for (int a2 : valid_a[2]) {
                                int rem2 = rem1 - a2 * a2;
                                if (rem2 < 0) continue;
                                for (int a3 : sq3[rem2]) {
                                    prob.valid_alt_tuples.push_back({{a0, a1, a2, a3}});
                                    if (prob.valid_alt_tuples.size() >= 65536) break;
                                }
                                if (prob.valid_alt_tuples.size() >= 65536) break;
                            }
                            if (prob.valid_alt_tuples.size() >= 65536) break;
                        }
                        if (prob.valid_alt_tuples.size() >= 65536) break;
                    }
                    if (prob.valid_alt_tuples.empty()) {
                        prob.infeasible = true;
                        prob.infeasible_reason = "2-Compression Manifold Obstruction: no achievable (a0,a1,a2,a3) with sum(a_s^2) == n";
                        return prob;
                    }
                }
            }
        }
    }

    return prob;
}

bool exact_verify_sequences(const OrbitProblem& prob, const std::vector<int> seq[4]) {
    int m = prob.m;
    std::vector<int> full[4];
    for (int s = 0; s < 4; ++s) {
        full[s].resize(m);
        int rsum = 0;
        for (int x = 0; x < m; ++x) {
            int v = seq[s][prob.elem_to_orb[x]];
            full[s][x] = v;
            rsum += v;
        }
        if (std::abs(rsum) != std::abs(prob.target_s[s])) return false;
    }
    for (int d = 1; d < m; ++d) {
        int ac = 0;
        for (int s = 0; s < 4; ++s) {
            int p = 0;
            for (int x = 0; x < m; ++x) p += full[s][x] * full[s][(x + d) % m];
            ac += p;
        }
        if (ac != prob.target_ac) return false;
    }
    return true;
}

std::atomic<bool> g_found{false};
std::atomic<uint64_t> g_pt_steps{0};
std::atomic<uint64_t> g_tabu_steps{0};
std::atomic<uint64_t> g_two_opt_hits{0};
std::atomic<int64_t> g_best_true_E{INT64_MAX};
std::atomic<int64_t> g_best_rsum0_E{INT64_MAX};
std::atomic<int64_t> g_best_manifold_E{INT64_MAX};
std::mutex g_io_mtx;
std::vector<int> g_best_P;
int g_best_rsum[4] = {0, 0, 0, 0};
std::vector<int> g_best_rsum0_P;
std::string g_save_best_path;
int64_t g_two_opt_thresh = 2000;

struct EliteEntry {
    int64_t E;
    std::vector<int> seq[4];
};
std::mutex g_elite_mtx;
std::vector<EliteEntry> g_elite;
static constexpr size_t MAX_ELITE = 8;

void elite_try_insert(int64_t E, const std::vector<int> seq[4]) {
    std::lock_guard<std::mutex> lk(g_elite_mtx);
    if (g_elite.size() >= MAX_ELITE && E >= g_elite.back().E) return;
    for (const auto& e : g_elite) {
        if (e.E == E) {
            bool same = true;
            for (int s = 0; s < 4 && same; ++s) {
                if (e.seq[s] != seq[s]) same = false;
            }
            if (same) return;
        }
    }
    EliteEntry ent;
    ent.E = E;
    for (int s = 0; s < 4; ++s) ent.seq[s] = seq[s];
    g_elite.push_back(std::move(ent));
    std::sort(g_elite.begin(), g_elite.end(), [](const EliteEntry& a, const EliteEntry& b) {
        return a.E < b.E;
    });
    if (g_elite.size() > MAX_ELITE) g_elite.resize(MAX_ELITE);
}

void save_witness(const std::string& path, const OrbitProblem& prob, const std::vector<int> seq[4], int64_t E_val = 0) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"n\": " << prob.n << ",\n";
    out << "  \"m\": " << prob.m << ",\n";
    out << "  \"g\": " << (prob.gens.empty() ? 1 : prob.gens[0]) << ",\n";
    out << "  \"gens\": [";
    for (size_t i = 0; i < prob.gens.size(); ++i) {
        if (i) out << ",";
        out << prob.gens[i];
    }
    out << "],\n";
    out << "  \"U_size\": " << prob.U.size() << ",\n";
    out << "  \"b\": " << prob.b << ",\n";
    out << "  \"r\": " << prob.r << ",\n";
    out << "  \"K\": " << prob.K << ",\n";
    out << "  \"E_ac\": " << E_val << ",\n";
    out << "  \"orbit_signs\": [\n";
    for (int s = 0; s < 4; ++s) {
        out << "    [";
        for (int i = 0; i < prob.r; ++i) {
            if (i) out << ",";
            out << seq[s][i];
        }
        out << "]" << (s < 3 ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

#ifdef __AVX2__
static inline int64_t hsum_epi64(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi64(lo, hi);
    __m128i shuf = _mm_unpackhi_epi64(sum, sum);
    sum = _mm_add_epi64(sum, shuf);
    return _mm_cvtsi128_si64(sum);
}
#endif

struct SearchState {
    const OrbitProblem* prob;
    int r, K, Kpad;
    std::vector<int> seq[4];
    std::vector<int> P;
    std::vector<int> P_seq[4];
    std::vector<int> grad;
    std::vector<int> shift_w;
    bool gls_active = false;
    int rsum[4];
    int alt[4];
    int64_t S_alt;
    int64_t E_ac;
    int64_t E_flat;
    int64_t E_row; // raw sum (rsum[s] - target_s[s])^2
    int64_t E_true; // E_ac + BASE_ROW_WEIGHT * E_row
    int64_t E_gls;  // sum shift_w[k]*P[k]^2 + prob->w_row * E_row
    int64_t E_alt;
    int64_t E_search;

    SearchState()
        : prob(nullptr), r(0), K(0), Kpad(0), S_alt(0),
          E_ac(0), E_flat(0), E_row(0), E_true(0), E_gls(0), E_alt(0), E_search(0) {}

    explicit SearchState(const OrbitProblem& p)
        : prob(&p), r(p.r), K(p.K), Kpad(p.Kpad),
          P(p.Kpad, 0), grad((size_t)4 * p.r * p.Kpad, 0), shift_w(p.Kpad, 0) {
        for (int k = 0; k < K; ++k) shift_w[k] = 1;
        for (int s = 0; s < 4; ++s) {
            seq[s].assign(r, 1);
            P_seq[s].assign(p.Kpad, 0);
            rsum[s] = 0;
            alt[s] = 0;
        }
        S_alt = 0;
        E_ac = E_flat = E_row = E_true = E_gls = E_alt = E_search = 0;
    }

    void reset_gls_weights() {
        for (int k = 0; k < K; ++k) shift_w[k] = 1;
        for (int k = K; k < Kpad; ++k) shift_w[k] = 0;
        gls_active = false;
        E_gls = E_ac + prob->w_row * E_row;
        E_search = E_gls + E_alt;
    }

    void recompute_E_gls() {
        int64_t eg = 0;
        for (int k = 0; k < K; ++k) eg += (int64_t)shift_w[k] * (int64_t)P[k] * (int64_t)P[k];
        E_gls = eg + prob->w_row * E_row;
        E_search = E_gls + E_alt;
    }

    inline int64_t calc_E_alt_from_S(int64_t s_alt) const {
        if (!prob->use_alt || prob->w_alt == 0) return 0;
        int64_t d = std::abs(s_alt - prob->n) >> 2;
        int64_t base = (d <= 8) ? (d * d) : (16 * d - 64);
        return (prob->w_alt * base) >> 2;
    }

    inline int64_t calc_E_alt_full(int64_t s_alt, const int a[4]) const {
        if (!prob->use_alt || prob->w_alt == 0) return 0;
        if (prob->use_target_alt) {
            int64_t e = 0;
            for (int s = 0; s < 4; ++s) {
                int64_t da = std::abs(a[s]) - std::abs(prob->target_alt[s]);
                e += da * da;
            }
            return prob->w_alt * e;
        }
        return calc_E_alt_from_S(s_alt);
    }

    void recompute() {
        S_alt = 0;
        E_row = 0;
        for (int s = 0; s < 4; ++s) {
            int rs = 0, al = 0;
            for (int i = 0; i < r; ++i) {
                rs += seq[s][i] * prob->orb_size[i];
                al += seq[s][i] * prob->orb_alt[i];
            }
            rsum[s] = rs;
            alt[s] = al;
            S_alt += (int64_t)al * al;
            int64_t dr = rs - prob->target_s[s];
            E_row += dr * dr;
            std::fill(P_seq[s].begin(), P_seq[s].end(), 0);
        }
        std::fill(P.begin(), P.end(), 0);
        std::fill(grad.begin(), grad.end(), 0);
        int64_t ef = 0;
        for (int k = 0; k < K; ++k) {
            int64_t pk = 4LL * prob->diag_const[k] - prob->target_ac;
            for (int s = 0; s < 4; ++s) {
                int64_t acc = 0;
                for (int i = 0; i < r; ++i) {
                    int64_t gval = 0;
                    const int* rowM = &prob->MT[((size_t)i * r) * Kpad];
                    for (int j = 0; j < r; ++j) {
                        gval += (int64_t)rowM[(size_t)j * Kpad + k] * seq[s][j];
                    }
                    grad[((size_t)s * r + i) * Kpad + k] = (int)gval;
                    acc += (int64_t)seq[s][i] * gval;
                }
                int64_t psk = prob->diag_const[k] + acc / 2;
                P_seq[s][k] = (int)psk;
                ef += psk * psk;
                pk += acc / 2;
            }
            P[k] = (int)pk;
        }
        E_flat = ef;
        int64_t et = 0, eg = 0;
        for (int k = 0; k < K; ++k) {
            int64_t p2 = (int64_t)P[k] * P[k];
            et += p2;
            eg += (int64_t)shift_w[k] * p2;
        }
        E_ac = et;
        E_true = E_ac + BASE_ROW_WEIGHT * E_row;
        E_gls = eg + prob->w_row * E_row;
        E_alt = calc_E_alt_full(S_alt, alt);
        E_search = E_gls + E_alt;
    }

    void randomize_uniform(std::mt19937_64& rng) {
        for (int s = 0; s < 4; ++s) {
            for (int i = 0; i < r; ++i) {
                seq[s][i] = (rng() & 1) ? 1 : -1;
            }
        }
        recompute();
    }

    // Fast single-sequence flat-spectrum descent minimizing sum_k PAF_s(d_k)^2 (preserves rsum[s])
    void flat_seed_single(int s, std::mt19937_64& rng, int max_steps = 120) {
        if (prob->size_groups.empty()) return;
        std::vector<int> Ps(Kpad, 0);
        std::vector<int> gs((size_t)r * Kpad, 0);
        for (int k = 0; k < K; ++k) {
            int64_t pk = prob->diag_const[k];
            int64_t acc = 0;
            for (int i = 0; i < r; ++i) {
                int64_t gval = 0;
                const int* rowM = &prob->MT[((size_t)i * r) * Kpad];
                for (int j = 0; j < r; ++j) {
                    gval += (int64_t)rowM[(size_t)j * Kpad + k] * seq[s][j];
                }
                gs[(size_t)i * Kpad + k] = (int)gval;
                acc += (int64_t)seq[s][i] * gval;
            }
            pk += acc / 2;
            Ps[k] = (int)pk;
        }

        std::vector<int> pos, neg;
        pos.reserve(r);
        neg.reserve(r);
        for (int step = 0; step < max_steps; ++step) {
            int best_i = -1, best_j = -1;
            int64_t best_d = 0;
            int n_ties = 0;
            const int* __restrict__ pptr = Ps.data();
            for (const auto& grp : prob->size_groups) {
                pos.clear();
                neg.clear();
                for (int idx : grp) {
                    if (seq[s][idx] > 0) pos.push_back(idx);
                    else neg.push_back(idx);
                }
                for (int i : pos) {
                    const int* __restrict__ gi = &gs[(size_t)i * Kpad];
                    for (int j : neg) {
                        const int* __restrict__ gj = &gs[(size_t)j * Kpad];
                        const int* __restrict__ mij = &prob->MT[((size_t)i * r + j) * Kpad];
                        int64_t d = 0;
#ifdef __AVX2__
                        __m256i acc = _mm256_setzero_si256();
                        for (int k = 0; k < Kpad; k += 8) {
                            __m256i vgi = _mm256_loadu_si256((const __m256i*)(gi + k));
                            __m256i vgj = _mm256_loadu_si256((const __m256i*)(gj + k));
                            __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
                            __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
                            __m256i vdp = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_sub_epi32(vgj, vgi), 1),
                                                           _mm256_slli_epi32(vmij, 2));
                            __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                            __m256i ev = _mm256_mul_epi32(vdp, vsum);
                            __m256i od = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), _mm256_srli_epi64(vsum, 32));
                            acc = _mm256_add_epi64(acc, _mm256_add_epi64(ev, od));
                        }
                        d = hsum_epi64(acc);
#else
                        for (int k = 0; k < Kpad; ++k) {
                            int64_t dp = 2LL * (gj[k] - gi[k]) - 4LL * mij[k];
                            d += dp * (2LL * pptr[k] + dp);
                        }
#endif
                        if (d < best_d) {
                            best_d = d; best_i = i; best_j = j; n_ties = 1;
                        } else if (d == best_d && best_i != -1 && (int)(rng() % (++n_ties)) == 0) {
                            best_i = i; best_j = j;
                        }
                    }
                }
            }
            if (best_i == -1) break;
            // Apply swap (best_i: +1 -> -1, best_j: -1 -> +1)
            seq[s][best_i] = -1;
            seq[s][best_j] = 1;
            // Flip best_i (factor = -2)
            {
                const int* gi = &gs[(size_t)best_i * Kpad];
                for (int k = 0; k < Kpad; ++k) Ps[k] -= 2 * gi[k];
                const int* mbase = &prob->MT[((size_t)best_i * r) * Kpad];
                for (int u = 0; u < r; ++u) {
                    if (u == best_i) continue;
                    int* gu = &gs[(size_t)u * Kpad];
                    const int* miu = &mbase[(size_t)u * Kpad];
                    for (int k = 0; k < Kpad; ++k) gu[k] -= 2 * miu[k];
                }
            }
            // Flip best_j (factor = +2)
            {
                const int* gj = &gs[(size_t)best_j * Kpad];
                for (int k = 0; k < Kpad; ++k) Ps[k] += 2 * gj[k];
                const int* mbase = &prob->MT[((size_t)best_j * r) * Kpad];
                for (int u = 0; u < r; ++u) {
                    if (u == best_j) continue;
                    int* gu = &gs[(size_t)u * Kpad];
                    const int* mju = &mbase[(size_t)u * Kpad];
                    for (int k = 0; k < Kpad; ++k) gu[k] += 2 * mju[k];
                }
            }
        }
    }

    // Single-row Tabu flatness optimizer minimizing sum_k PAF_s(d_k)^2 (preserves rsum[s]).
    void flat_tabu_single(int s, std::mt19937_64& rng, int steps, int tabu_len = 8) {
        if (prob->size_groups.empty() || steps <= 0) return;
        std::vector<int> Ps(Kpad, 0);
        std::vector<int> gs((size_t)r * Kpad, 0);
        for (int k = 0; k < K; ++k) {
            int64_t pk = prob->diag_const[k];
            int64_t acc = 0;
            for (int i = 0; i < r; ++i) {
                int64_t gval = 0;
                const int* rowM = &prob->MT[((size_t)i * r) * Kpad];
                for (int j = 0; j < r; ++j) {
                    gval += (int64_t)rowM[(size_t)j * Kpad + k] * seq[s][j];
                }
                gs[(size_t)i * Kpad + k] = (int)gval;
                acc += (int64_t)seq[s][i] * gval;
            }
            pk += acc / 2;
            Ps[k] = (int)pk;
        }
        int64_t cur_flat = 0;
        for (int k = 0; k < K; ++k) cur_flat += (int64_t)Ps[k] * Ps[k];
        int64_t best_flat = cur_flat;
        std::vector<int> best_seq = seq[s];
        std::vector<int> tabu(r, 0);
        std::vector<int> pos, neg;
        pos.reserve(r);
        neg.reserve(r);

        for (int step = 1; step <= steps; ++step) {
            int best_i = -1, best_j = -1;
            int64_t best_d = INT64_MAX;
            int n_ties = 0;
            const int* __restrict__ pptr = Ps.data();
            for (const auto& grp : prob->size_groups) {
                pos.clear();
                neg.clear();
                for (int idx : grp) {
                    if (seq[s][idx] > 0) pos.push_back(idx);
                    else neg.push_back(idx);
                }
                for (int i : pos) {
                    const int* __restrict__ gi = &gs[(size_t)i * Kpad];
                    bool ti = (tabu[i] > step);
                    for (int j : neg) {
                        bool is_tabu = ti || (tabu[j] > step);
                        const int* __restrict__ gj = &gs[(size_t)j * Kpad];
                        const int* __restrict__ mij = &prob->MT[((size_t)i * r + j) * Kpad];
                        int64_t d = 0;
#ifdef __AVX2__
                        __m256i acc = _mm256_setzero_si256();
                        for (int k = 0; k < Kpad; k += 8) {
                            __m256i vgi = _mm256_loadu_si256((const __m256i*)(gi + k));
                            __m256i vgj = _mm256_loadu_si256((const __m256i*)(gj + k));
                            __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
                            __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
                            __m256i vdp = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_sub_epi32(vgj, vgi), 1),
                                                           _mm256_slli_epi32(vmij, 2));
                            __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                            __m256i ev = _mm256_mul_epi32(vdp, vsum);
                            __m256i od = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), _mm256_srli_epi64(vsum, 32));
                            acc = _mm256_add_epi64(acc, _mm256_add_epi64(ev, od));
                        }
                        d = hsum_epi64(acc);
#else
                        for (int k = 0; k < Kpad; ++k) {
                            int64_t dp = 2LL * (gj[k] - gi[k]) - 4LL * mij[k];
                            d += dp * (2LL * pptr[k] + dp);
                        }
#endif
                        if (is_tabu && cur_flat + d >= best_flat) continue;
                        if (d < best_d) {
                            best_d = d; best_i = i; best_j = j; n_ties = 1;
                        } else if (d == best_d && best_i != -1 && (int)(rng() % (++n_ties)) == 0) {
                            best_i = i; best_j = j;
                        }
                    }
                }
            }
            if (best_i == -1) break;
            seq[s][best_i] = -1;
            seq[s][best_j] = 1;
            cur_flat += best_d;
            tabu[best_i] = step + tabu_len + (int)(rng() % 5);
            tabu[best_j] = step + tabu_len + (int)(rng() % 5);
            {
                const int* gi = &gs[(size_t)best_i * Kpad];
                for (int k = 0; k < Kpad; ++k) Ps[k] -= 2 * gi[k];
                const int* mbase = &prob->MT[((size_t)best_i * r) * Kpad];
                for (int u = 0; u < r; ++u) {
                    if (u == best_i) continue;
                    int* gu = &gs[(size_t)u * Kpad];
                    const int* miu = &mbase[(size_t)u * Kpad];
                    for (int k = 0; k < Kpad; ++k) gu[k] -= 2 * miu[k];
                }
            }
            {
                const int* gj = &gs[(size_t)best_j * Kpad];
                for (int k = 0; k < Kpad; ++k) Ps[k] += 2 * gj[k];
                const int* mbase = &prob->MT[((size_t)best_j * r) * Kpad];
                for (int u = 0; u < r; ++u) {
                    if (u == best_j) continue;
                    int* gu = &gs[(size_t)u * Kpad];
                    const int* mju = &mbase[(size_t)u * Kpad];
                    for (int k = 0; k < Kpad; ++k) gu[k] += 2 * mju[k];
                }
            }
            if (cur_flat < best_flat) {
                best_flat = cur_flat;
                best_seq = seq[s];
            }
        }
        seq[s] = std::move(best_seq);
    }

    void resample_single_seq(int s, std::mt19937_64& rng) {
        if (prob->full_dp.built) {
            prob->full_dp.sample(prob->target_s[s], seq[s], rng);
        } else {
            for (int i = 0; i < r; ++i) seq[s][i] = (rng() & 1) ? 1 : -1;
        }
        if (prob->flat_init) flat_seed_single(s, rng, 80);
    }

    void randomize_manifold(std::mt19937_64& rng) {
        if (prob->pure_parity_orbits && !prob->valid_alt_tuples.empty()) {
            const AltTuple& tup = prob->valid_alt_tuples[rng() % prob->valid_alt_tuples.size()];
            bool ok = true;
            for (int s = 0; s < 4; ++s) {
                int te = (prob->target_s[s] + tup.a[s]) >> 1;
                int to = (prob->target_s[s] - tup.a[s]) >> 1;
                if (!prob->even_dp.sample(te, seq[s], rng) ||
                    !prob->odd_dp.sample(to, seq[s], rng)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                if (prob->flat_init) {
                    for (int s = 0; s < 4; ++s) flat_seed_single(s, rng, 100);
                }
                recompute();
                return;
            }
        }
        if (prob->full_dp.built) {
            bool ok = true;
            for (int s = 0; s < 4; ++s) {
                if (!prob->full_dp.sample(prob->target_s[s], seq[s], rng)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                if (prob->flat_init) {
                    for (int s = 0; s < 4; ++s) flat_seed_single(s, rng, 120);
                }
                recompute();
                return;
            }
        }
        randomize_uniform(rng);
    }

    // Fast AVX2 swap assuming seq[s][i_pos] == +1 and seq[s][j_neg] == -1 (m_i = -2 constant!)
    inline void eval_swap_pos_neg(int s, int i, int j, int64_t& d_ac, int64_t& d_gls, int64_t& d_alt) const {
        const int* __restrict__ gi = &grad[((size_t)s * r + i) * Kpad];
        const int* __restrict__ gj = &grad[((size_t)s * r + j) * Kpad];
        const int* __restrict__ mij = &prob->MT[((size_t)i * r + j) * Kpad];
        const int* __restrict__ pptr = P.data();

        int64_t dt_ac = 0;
        int64_t dg_ac = 0;

#ifdef __AVX2__
        __m256i acc_t = _mm256_setzero_si256();
        if (!gls_active) {
            for (int k = 0; k < Kpad; k += 8) {
                __m256i vgi = _mm256_loadu_si256((const __m256i*)(gi + k));
                __m256i vgj = _mm256_loadu_si256((const __m256i*)(gj + k));
                __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
                __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
                __m256i vdp = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_sub_epi32(vgj, vgi), 1),
                                               _mm256_slli_epi32(vmij, 2));
                __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                __m256i ev = _mm256_mul_epi32(vdp, vsum);
                __m256i od = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), _mm256_srli_epi64(vsum, 32));
                acc_t = _mm256_add_epi64(acc_t, _mm256_add_epi64(ev, od));
            }
            dt_ac = hsum_epi64(acc_t);
            dg_ac = dt_ac;
        } else {
            const int* __restrict__ wptr = shift_w.data();
            __m256i acc_g = _mm256_setzero_si256();
            for (int k = 0; k < Kpad; k += 8) {
                __m256i vgi = _mm256_loadu_si256((const __m256i*)(gi + k));
                __m256i vgj = _mm256_loadu_si256((const __m256i*)(gj + k));
                __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
                __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
                __m256i vw = _mm256_loadu_si256((const __m256i*)(wptr + k));
                __m256i vdp = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_sub_epi32(vgj, vgi), 1),
                                               _mm256_slli_epi32(vmij, 2));
                __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                __m256i vsum_od = _mm256_srli_epi64(vsum, 32);

                __m256i ev_t = _mm256_mul_epi32(vdp, vsum);
                __m256i od_t = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), vsum_od);
                acc_t = _mm256_add_epi64(acc_t, _mm256_add_epi64(ev_t, od_t));

                __m256i vwdp = _mm256_mullo_epi32(vw, vdp);
                __m256i ev_g = _mm256_mul_epi32(vwdp, vsum);
                __m256i od_g = _mm256_mul_epi32(_mm256_srli_epi64(vwdp, 32), vsum_od);
                acc_g = _mm256_add_epi64(acc_g, _mm256_add_epi64(ev_g, od_g));
            }
            dt_ac = hsum_epi64(acc_t);
            dg_ac = hsum_epi64(acc_g);
        }
#else
        if (!gls_active) {
            for (int k = 0; k < Kpad; ++k) {
                int64_t dp = 2LL * (gj[k] - gi[k]) - 4LL * mij[k];
                dt_ac += dp * (2LL * pptr[k] + dp);
            }
            dg_ac = dt_ac;
        } else {
            const int* __restrict__ wptr = shift_w.data();
            for (int k = 0; k < Kpad; ++k) {
                int64_t dp = 2LL * (gj[k] - gi[k]) - 4LL * mij[k];
                int64_t term = dp * (2LL * pptr[k] + dp);
                dt_ac += term;
                dg_ac += (int64_t)wptr[k] * term;
            }
        }
#endif

        d_ac = dt_ac;
        d_gls = dg_ac;

        if (prob->w_alt != 0) {
            int oa_i = prob->orb_alt[i];
            int oa_j = prob->orb_alt[j];
            if (oa_i == oa_j) {
                d_alt = 0;
            } else {
                int64_t al_old = alt[s];
                int64_t al_new = al_old - 2LL * oa_i + 2LL * oa_j;
                if (prob->use_target_alt) {
                    int64_t ta = std::abs(prob->target_alt[s]);
                    int64_t d_old = std::abs(al_old) - ta;
                    int64_t d_new = std::abs(al_new) - ta;
                    d_alt = prob->w_alt * (d_new * d_new - d_old * d_old);
                } else {
                    int64_t S_new = S_alt + (al_new * al_new - al_old * al_old);
                    d_alt = calc_E_alt_from_S(S_new) - E_alt;
                }
            }
        } else {
            d_alt = 0;
        }
    }

    // Evaluate delta of single flip (s, i): returns {d_ac, d_row_raw, d_gls, d_alt}
    inline void eval_flip(int s, int i, int64_t& d_ac, int64_t& d_row_raw, int64_t& d_gls, int64_t& d_alt) const {
        int si = seq[s][i];
        int64_t dr_old = rsum[s] - prob->target_s[s];
        int64_t dr_new = dr_old - 2LL * si * prob->orb_size[i];
        d_row_raw = dr_new * dr_new - dr_old * dr_old;

        const int* __restrict__ gptr = &grad[((size_t)s * r + i) * Kpad];
        const int* __restrict__ pptr = P.data();
        const int mult = -2 * si;

        int64_t dt_ac = 0;
        int64_t dg_ac = 0;

#ifdef __AVX2__
        __m256i v_mult = _mm256_set1_epi32(mult);
        __m256i acc_t = _mm256_setzero_si256();
        if (!gls_active) {
            for (int k = 0; k < Kpad; k += 8) {
                __m256i vg = _mm256_loadu_si256((const __m256i*)(gptr + k));
                __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
                __m256i vdp = _mm256_mullo_epi32(v_mult, vg);
                __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                __m256i ev = _mm256_mul_epi32(vdp, vsum);
                __m256i od = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), _mm256_srli_epi64(vsum, 32));
                acc_t = _mm256_add_epi64(acc_t, _mm256_add_epi64(ev, od));
            }
            dt_ac = hsum_epi64(acc_t);
            dg_ac = dt_ac;
        } else {
            const int* __restrict__ wptr = shift_w.data();
            __m256i acc_g = _mm256_setzero_si256();
            for (int k = 0; k < Kpad; k += 8) {
                __m256i vg = _mm256_loadu_si256((const __m256i*)(gptr + k));
                __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
                __m256i vw = _mm256_loadu_si256((const __m256i*)(wptr + k));
                __m256i vdp = _mm256_mullo_epi32(v_mult, vg);
                __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                __m256i vsum_od = _mm256_srli_epi64(vsum, 32);

                __m256i ev_t = _mm256_mul_epi32(vdp, vsum);
                __m256i od_t = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), vsum_od);
                acc_t = _mm256_add_epi64(acc_t, _mm256_add_epi64(ev_t, od_t));

                __m256i vwdp = _mm256_mullo_epi32(vw, vdp);
                __m256i ev_g = _mm256_mul_epi32(vwdp, vsum);
                __m256i od_g = _mm256_mul_epi32(_mm256_srli_epi64(vwdp, 32), vsum_od);
                acc_g = _mm256_add_epi64(acc_g, _mm256_add_epi64(ev_g, od_g));
            }
            dt_ac = hsum_epi64(acc_t);
            dg_ac = hsum_epi64(acc_g);
        }
#else
        if (!gls_active) {
            for (int k = 0; k < Kpad; ++k) {
                int64_t dp = (int64_t)mult * gptr[k];
                dt_ac += dp * (2LL * pptr[k] + dp);
            }
            dg_ac = dt_ac;
        } else {
            const int* __restrict__ wptr = shift_w.data();
            for (int k = 0; k < Kpad; ++k) {
                int64_t dp = (int64_t)mult * gptr[k];
                int64_t term = dp * (2LL * pptr[k] + dp);
                dt_ac += term;
                dg_ac += (int64_t)wptr[k] * term;
            }
        }
#endif

        d_ac = dt_ac;
        d_gls = dg_ac + prob->w_row * d_row_raw;

        if (prob->w_alt != 0) {
            int64_t al_old = alt[s];
            int64_t al_new = al_old - 2LL * si * prob->orb_alt[i];
            if (prob->use_target_alt) {
                int64_t ta = std::abs(prob->target_alt[s]);
                int64_t d_old = std::abs(al_old) - ta;
                int64_t d_new = std::abs(al_new) - ta;
                d_alt = prob->w_alt * (d_new * d_new - d_old * d_old);
            } else {
                int64_t S_new = S_alt + (al_new * al_new - al_old * al_old);
                d_alt = calc_E_alt_from_S(S_new) - E_alt;
            }
        } else {
            d_alt = 0;
        }
    }

    // Evaluate delta of same-size swap (s, i, j) with seq[s][i] != seq[s][j] (d_row_raw == 0)
    inline void eval_swap(int s, int i, int j, int64_t& d_ac, int64_t& d_gls, int64_t& d_alt) const {
        int si = seq[s][i];
        int sj = seq[s][j]; // == -si
        const int* __restrict__ gi = &grad[((size_t)s * r + i) * Kpad];
        const int* __restrict__ gj = &grad[((size_t)s * r + j) * Kpad];
        const int* __restrict__ mij = &prob->MT[((size_t)i * r + j) * Kpad];
        const int* __restrict__ pptr = P.data();
        const int m_i = -2 * si;

        int64_t dt_ac = 0;
        int64_t dg_ac = 0;

#ifdef __AVX2__
        __m256i v_mi = _mm256_set1_epi32(m_i);
        __m256i acc_t = _mm256_setzero_si256();
        if (!gls_active) {
            for (int k = 0; k < Kpad; k += 8) {
                __m256i vgi = _mm256_loadu_si256((const __m256i*)(gi + k));
                __m256i vgj = _mm256_loadu_si256((const __m256i*)(gj + k));
                __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
                __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
                __m256i vdp = _mm256_sub_epi32(_mm256_mullo_epi32(v_mi, _mm256_sub_epi32(vgi, vgj)),
                                               _mm256_slli_epi32(vmij, 2));
                __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                __m256i ev = _mm256_mul_epi32(vdp, vsum);
                __m256i od = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), _mm256_srli_epi64(vsum, 32));
                acc_t = _mm256_add_epi64(acc_t, _mm256_add_epi64(ev, od));
            }
            dt_ac = hsum_epi64(acc_t);
            dg_ac = dt_ac;
        } else {
            const int* __restrict__ wptr = shift_w.data();
            __m256i acc_g = _mm256_setzero_si256();
            for (int k = 0; k < Kpad; k += 8) {
                __m256i vgi = _mm256_loadu_si256((const __m256i*)(gi + k));
                __m256i vgj = _mm256_loadu_si256((const __m256i*)(gj + k));
                __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
                __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
                __m256i vw = _mm256_loadu_si256((const __m256i*)(wptr + k));
                __m256i vdp = _mm256_sub_epi32(_mm256_mullo_epi32(v_mi, _mm256_sub_epi32(vgi, vgj)),
                                               _mm256_slli_epi32(vmij, 2));
                __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                __m256i vsum_od = _mm256_srli_epi64(vsum, 32);

                __m256i ev_t = _mm256_mul_epi32(vdp, vsum);
                __m256i od_t = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), vsum_od);
                acc_t = _mm256_add_epi64(acc_t, _mm256_add_epi64(ev_t, od_t));

                __m256i vwdp = _mm256_mullo_epi32(vw, vdp);
                __m256i ev_g = _mm256_mul_epi32(vwdp, vsum);
                __m256i od_g = _mm256_mul_epi32(_mm256_srli_epi64(vwdp, 32), vsum_od);
                acc_g = _mm256_add_epi64(acc_g, _mm256_add_epi64(ev_g, od_g));
            }
            dt_ac = hsum_epi64(acc_t);
            dg_ac = hsum_epi64(acc_g);
        }
#else
        if (!gls_active) {
            for (int k = 0; k < Kpad; ++k) {
                int64_t dp = (int64_t)m_i * (gi[k] - gj[k]) - 4LL * mij[k];
                dt_ac += dp * (2LL * pptr[k] + dp);
            }
            dg_ac = dt_ac;
        } else {
            const int* __restrict__ wptr = shift_w.data();
            for (int k = 0; k < Kpad; ++k) {
                int64_t dp = (int64_t)m_i * (gi[k] - gj[k]) - 4LL * mij[k];
                int64_t term = dp * (2LL * pptr[k] + dp);
                dt_ac += term;
                dg_ac += (int64_t)wptr[k] * term;
            }
        }
#endif

        d_ac = dt_ac;
        d_gls = dg_ac;

        if (prob->w_alt != 0) {
            int oa_i = prob->orb_alt[i];
            int oa_j = prob->orb_alt[j];
            if (oa_i == oa_j) {
                d_alt = 0;
            } else {
                int64_t al_old = alt[s];
                int64_t al_new = al_old - 2LL * si * oa_i - 2LL * sj * oa_j;
                if (prob->use_target_alt) {
                    int64_t ta = std::abs(prob->target_alt[s]);
                    int64_t d_old = std::abs(al_old) - ta;
                    int64_t d_new = std::abs(al_new) - ta;
                    d_alt = prob->w_alt * (d_new * d_new - d_old * d_old);
                } else {
                    int64_t S_new = S_alt + (al_new * al_new - al_old * al_old);
                    d_alt = calc_E_alt_from_S(S_new) - E_alt;
                }
            }
        } else {
            d_alt = 0;
        }
    }

    // Evaluate delta of row-sum-preserving 3-orbit flip (s, i, j, k):
    // orb_size[i] == orb_size[j] + orb_size[k], seq[s][j] == seq[s][k] == -seq[s][i]
    inline void eval_triple(int s, int i, int j, int k_orb, int64_t& d_ac, int64_t& d_gls, int64_t& d_alt) const {
        int si = seq[s][i];
        const int* __restrict__ gi = &grad[((size_t)s * r + i) * Kpad];
        const int* __restrict__ gj = &grad[((size_t)s * r + j) * Kpad];
        const int* __restrict__ gk = &grad[((size_t)s * r + k_orb) * Kpad];
        const int* __restrict__ mij = &prob->MT[((size_t)i * r + j) * Kpad];
        const int* __restrict__ mik = &prob->MT[((size_t)i * r + k_orb) * Kpad];
        const int* __restrict__ mjk = &prob->MT[((size_t)j * r + k_orb) * Kpad];
        const int* __restrict__ pptr = P.data();
        const int m_i = -2 * si;

        int64_t dt_ac = 0;
        int64_t dg_ac = 0;

#ifdef __AVX2__
        __m256i v_mi = _mm256_set1_epi32(m_i);
        __m256i acc_t = _mm256_setzero_si256();
        if (!gls_active) {
            for (int kk = 0; kk < Kpad; kk += 8) {
                __m256i vgi = _mm256_loadu_si256((const __m256i*)(gi + kk));
                __m256i vgj = _mm256_loadu_si256((const __m256i*)(gj + kk));
                __m256i vgk = _mm256_loadu_si256((const __m256i*)(gk + kk));
                __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + kk));
                __m256i vmik = _mm256_loadu_si256((const __m256i*)(mik + kk));
                __m256i vmjk = _mm256_loadu_si256((const __m256i*)(mjk + kk));
                __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + kk));

                __m256i gdiff = _mm256_sub_epi32(vgi, _mm256_add_epi32(vgj, vgk));
                __m256i mcorr = _mm256_sub_epi32(vmjk, _mm256_add_epi32(vmij, vmik));
                __m256i vdp = _mm256_add_epi32(_mm256_mullo_epi32(v_mi, gdiff),
                                               _mm256_slli_epi32(mcorr, 2));
                __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                __m256i ev = _mm256_mul_epi32(vdp, vsum);
                __m256i od = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), _mm256_srli_epi64(vsum, 32));
                acc_t = _mm256_add_epi64(acc_t, _mm256_add_epi64(ev, od));
            }
            dt_ac = hsum_epi64(acc_t);
            dg_ac = dt_ac;
        } else {
            const int* __restrict__ wptr = shift_w.data();
            __m256i acc_g = _mm256_setzero_si256();
            for (int kk = 0; kk < Kpad; kk += 8) {
                __m256i vgi = _mm256_loadu_si256((const __m256i*)(gi + kk));
                __m256i vgj = _mm256_loadu_si256((const __m256i*)(gj + kk));
                __m256i vgk = _mm256_loadu_si256((const __m256i*)(gk + kk));
                __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + kk));
                __m256i vmik = _mm256_loadu_si256((const __m256i*)(mik + kk));
                __m256i vmjk = _mm256_loadu_si256((const __m256i*)(mjk + kk));
                __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + kk));
                __m256i vw = _mm256_loadu_si256((const __m256i*)(wptr + kk));

                __m256i gdiff = _mm256_sub_epi32(vgi, _mm256_add_epi32(vgj, vgk));
                __m256i mcorr = _mm256_sub_epi32(vmjk, _mm256_add_epi32(vmij, vmik));
                __m256i vdp = _mm256_add_epi32(_mm256_mullo_epi32(v_mi, gdiff),
                                               _mm256_slli_epi32(mcorr, 2));
                __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
                __m256i vsum_od = _mm256_srli_epi64(vsum, 32);

                __m256i ev_t = _mm256_mul_epi32(vdp, vsum);
                __m256i od_t = _mm256_mul_epi32(_mm256_srli_epi64(vdp, 32), vsum_od);
                acc_t = _mm256_add_epi64(acc_t, _mm256_add_epi64(ev_t, od_t));

                __m256i vwdp = _mm256_mullo_epi32(vw, vdp);
                __m256i ev_g = _mm256_mul_epi32(vwdp, vsum);
                __m256i od_g = _mm256_mul_epi32(_mm256_srli_epi64(vwdp, 32), vsum_od);
                acc_g = _mm256_add_epi64(acc_g, _mm256_add_epi64(ev_g, od_g));
            }
            dt_ac = hsum_epi64(acc_t);
            dg_ac = hsum_epi64(acc_g);
        }
#else
        if (!gls_active) {
            for (int kk = 0; kk < Kpad; ++kk) {
                int64_t dp = (int64_t)m_i * (gi[kk] - gj[kk] - gk[kk]) + 4LL * (mjk[kk] - mij[kk] - mik[kk]);
                dt_ac += dp * (2LL * pptr[kk] + dp);
            }
            dg_ac = dt_ac;
        } else {
            const int* __restrict__ wptr = shift_w.data();
            for (int kk = 0; kk < Kpad; ++kk) {
                int64_t dp = (int64_t)m_i * (gi[kk] - gj[kk] - gk[kk]) + 4LL * (mjk[kk] - mij[kk] - mik[kk]);
                int64_t term = dp * (2LL * pptr[kk] + dp);
                dt_ac += term;
                dg_ac += (int64_t)wptr[kk] * term;
            }
        }
#endif

        d_ac = dt_ac;
        d_gls = dg_ac;

        if (prob->w_alt != 0) {
            int64_t d_al = (int64_t)m_i * (prob->orb_alt[i] - prob->orb_alt[j] - prob->orb_alt[k_orb]);
            if (d_al == 0) {
                d_alt = 0;
            } else {
                int64_t al_old = alt[s];
                int64_t al_new = al_old + d_al;
                if (prob->use_target_alt) {
                    int64_t ta = std::abs(prob->target_alt[s]);
                    int64_t d_old = std::abs(al_old) - ta;
                    int64_t d_new = std::abs(al_new) - ta;
                    d_alt = prob->w_alt * (d_new * d_new - d_old * d_old);
                } else {
                    int64_t S_new = S_alt + (al_new * al_new - al_old * al_old);
                    d_alt = calc_E_alt_from_S(S_new) - E_alt;
                }
            }
        } else {
            d_alt = 0;
        }
    }

    // Fast AVX2 swap assuming seq[s][i]==+1, seq[s][j]==-1, returning both d_ac and d_flat
    inline void eval_swap_pos_neg_flat(int s, int i, int j, int64_t& d_ac, int64_t& d_flat) const {
        const int* __restrict__ gi = &grad[((size_t)s * r + i) * Kpad];
        const int* __restrict__ gj = &grad[((size_t)s * r + j) * Kpad];
        const int* __restrict__ mij = &prob->MT[((size_t)i * r + j) * Kpad];
        const int* __restrict__ pptr = P.data();
        const int* __restrict__ psptr = P_seq[s].data();
#ifdef __AVX2__
        __m256i acc_t = _mm256_setzero_si256();
        __m256i acc_f = _mm256_setzero_si256();
        for (int k = 0; k < Kpad; k += 8) {
            __m256i vgi = _mm256_loadu_si256((const __m256i*)(gi + k));
            __m256i vgj = _mm256_loadu_si256((const __m256i*)(gj + k));
            __m256i vmij = _mm256_loadu_si256((const __m256i*)(mij + k));
            __m256i vp = _mm256_loadu_si256((const __m256i*)(pptr + k));
            __m256i vps = _mm256_loadu_si256((const __m256i*)(psptr + k));
            __m256i vdp = _mm256_sub_epi32(_mm256_slli_epi32(_mm256_sub_epi32(vgj, vgi), 1),
                                           _mm256_slli_epi32(vmij, 2));
            __m256i vdp_od = _mm256_srli_epi64(vdp, 32);

            __m256i vsum = _mm256_add_epi32(_mm256_slli_epi32(vp, 1), vdp);
            __m256i ev_t = _mm256_mul_epi32(vdp, vsum);
            __m256i od_t = _mm256_mul_epi32(vdp_od, _mm256_srli_epi64(vsum, 32));
            acc_t = _mm256_add_epi64(acc_t, _mm256_add_epi64(ev_t, od_t));

            __m256i vfsum = _mm256_add_epi32(_mm256_slli_epi32(vps, 1), vdp);
            __m256i ev_f = _mm256_mul_epi32(vdp, vfsum);
            __m256i od_f = _mm256_mul_epi32(vdp_od, _mm256_srli_epi64(vfsum, 32));
            acc_f = _mm256_add_epi64(acc_f, _mm256_add_epi64(ev_f, od_f));
        }
        d_ac = hsum_epi64(acc_t);
        d_flat = hsum_epi64(acc_f);
#else
        int64_t dt = 0, df = 0;
        for (int k = 0; k < Kpad; ++k) {
            int64_t dp = 2LL * (gj[k] - gi[k]) - 4LL * mij[k];
            dt += dp * (2LL * pptr[k] + dp);
            df += dp * (2LL * psptr[k] + dp);
        }
        d_ac = dt;
        d_flat = df;
#endif
    }

    inline void apply_flip_core(int s, int i) {
        int si = seq[s][i];
        seq[s][i] = -si;
        rsum[s] -= 2 * si * prob->orb_size[i];
        int64_t al_old = alt[s];
        int64_t al_new = al_old - 2LL * si * prob->orb_alt[i];
        alt[s] = (int)al_new;
        S_alt += (al_new * al_new - al_old * al_old);

        const int factor = -2 * si;
        const int* __restrict__ gptr = &grad[((size_t)s * r + i) * Kpad];
        int* __restrict__ pptr = P.data();
        int* __restrict__ psptr = P_seq[s].data();
        for (int k = 0; k < Kpad; ++k) {
            int delta = factor * gptr[k];
            pptr[k] += delta;
            psptr[k] += delta;
        }
        const int* __restrict__ mptr_base = &prob->MT[((size_t)i * r) * Kpad];
        int* __restrict__ gbase = &grad[((size_t)s * r) * Kpad];
        for (int j = 0; j < r; ++j) {
            if (j == i) continue;
            int* __restrict__ gj = &gbase[(size_t)j * Kpad];
            const int* __restrict__ mij = &mptr_base[(size_t)j * Kpad];
            for (int k = 0; k < Kpad; ++k) {
                gj[k] += factor * mij[k];
            }
        }
    }

    inline void apply_flip(int s, int i, int64_t d_ac, int64_t d_row_raw, int64_t d_gls, int64_t d_alt) {
        apply_flip_core(s, i);
        E_ac += d_ac;
        E_row += d_row_raw;
        E_true = E_ac + BASE_ROW_WEIGHT * E_row;
        E_gls += d_gls;
        E_alt += d_alt;
        E_search = E_gls + E_alt;
    }

    inline void apply_swap(int s, int i, int j, int64_t d_ac, int64_t d_gls, int64_t d_alt) {
        apply_flip_core(s, i);
        apply_flip_core(s, j);
        E_ac += d_ac;
        E_true = E_ac + BASE_ROW_WEIGHT * E_row;
        E_gls += d_gls;
        E_alt += d_alt;
        E_search = E_gls + E_alt;
    }

    inline void apply_triple(int s, int i, int j, int k_orb, int64_t d_ac, int64_t d_gls, int64_t d_alt) {
        apply_flip_core(s, i);
        apply_flip_core(s, j);
        apply_flip_core(s, k_orb);
        E_ac += d_ac;
        E_true = E_ac + BASE_ROW_WEIGHT * E_row;
        E_gls += d_gls;
        E_alt += d_alt;
        E_search = E_gls + E_alt;
    }
};

void record_if_best(int tid, const char* tag, const OrbitProblem& prob, const SearchState& st,
                    const std::chrono::steady_clock::time_point& t0, const std::string& out_path) {
    if (st.E_row == 0) {
        int64_t cur_r0 = g_best_rsum0_E.load(std::memory_order_relaxed);
        if (st.E_ac < cur_r0) {
            std::lock_guard<std::mutex> lk(g_io_mtx);
            if (st.E_ac < g_best_rsum0_E.load(std::memory_order_relaxed)) {
                g_best_rsum0_E.store(st.E_ac, std::memory_order_relaxed);
                g_best_rsum0_P.assign(st.P.begin(), st.P.begin() + prob.K);
                if (!g_save_best_path.empty()) {
                    save_witness(g_save_best_path, prob, st.seq, st.E_ac);
                }
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::cout << "[t=" << (int)el << "s, T" << tid << " " << tag << "] new best rsum0 E_ac=" << st.E_ac
                          << " (E_alt=" << st.E_alt << ", S_alt=" << st.S_alt << "/" << prob.n << ")" << std::endl;
            }
        }
        if (prob.use_alt && st.S_alt == prob.n) {
            int64_t cur_m = g_best_manifold_E.load(std::memory_order_relaxed);
            while (st.E_ac < cur_m &&
                   !g_best_manifold_E.compare_exchange_weak(cur_m, st.E_ac, std::memory_order_relaxed)) {}
        }
    }
    int64_t cur_best = g_best_true_E.load(std::memory_order_relaxed);
    if (st.E_true < cur_best) {
        std::lock_guard<std::mutex> lk(g_io_mtx);
        if (st.E_true < g_best_true_E.load(std::memory_order_relaxed)) {
            g_best_true_E.store(st.E_true, std::memory_order_relaxed);
            g_best_P.assign(st.P.begin(), st.P.begin() + prob.K);
            for (int s = 0; s < 4; ++s) g_best_rsum[s] = st.rsum[s];
            if (st.E_row != 0) {
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::cout << "[t=" << (int)el << "s, T" << tid << " " << tag << "] new best E=" << st.E_true
                          << " (E_row=" << st.E_row << ", E_alt=" << st.E_alt << ")" << std::endl;
            }
        }
    }
    if (st.E_true == 0) {
        if (exact_verify_sequences(prob, st.seq)) {
            if (!g_found.exchange(true)) {
                save_witness(out_path, prob, st.seq, 0);
                std::lock_guard<std::mutex> lk(g_io_mtx);
                std::cout << "FOUND EXACT VERIFIED SOLUTION! Saved to " << out_path << std::endl;
            }
        }
    }
}

struct SwapCand {
    int64_t d;
    int s, i, j;
};

// Focused 2-Swap Lookahead (2-opt polish) at low-energy rsum0 local minima
bool polish_2opt(int tid, SearchState& st, const std::chrono::steady_clock::time_point& t0,
                 double max_seconds, const std::string& out_path, int max_cands = 64) {
    const OrbitProblem& prob = *st.prob;
    if (st.E_row != 0 || prob.size_groups.empty()) return false;
    bool gls_was = st.gls_active;
    st.gls_active = false;
    bool any_improved = false;

    std::vector<int> pos_buf, neg_buf;
    pos_buf.reserve(prob.r);
    neg_buf.reserve(prob.r);
    std::vector<SwapCand> cands;
    cands.reserve(4096);

    while (!g_found.load(std::memory_order_relaxed)) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= max_seconds) break;

        // Pass 1: collect all 1-swaps; if any strictly improves E_ac, take best 1-swap immediately
        cands.clear();
        int64_t best_d1 = 0;
        int bs = -1, bi = -1, bj = -1;
        for (int s = 0; s < 4; ++s) {
            for (const auto& grp : prob.size_groups) {
                pos_buf.clear();
                neg_buf.clear();
                for (int idx : grp) {
                    if (st.seq[s][idx] > 0) pos_buf.push_back(idx);
                    else neg_buf.push_back(idx);
                }
                for (int i : pos_buf) {
                    for (int j : neg_buf) {
                        int64_t dac, dg, da;
                        st.eval_swap_pos_neg(s, i, j, dac, dg, da);
                        if (dac < best_d1) {
                            best_d1 = dac; bs = s; bi = i; bj = j;
                        }
                        cands.push_back({dac, s, i, j});
                    }
                }
            }
        }
        if (bs != -1) {
            st.apply_swap(bs, bi, bj, best_d1, best_d1, 0);
            any_improved = true;
            record_if_best(tid, "2opt-1", prob, st, t0, out_path);
            if (g_found.load(std::memory_order_relaxed)) break;
            continue;
        }

        // At a 1-swap local minimum: take top max_cands first swaps and lookahead for best 2nd swap
        int M = std::min((int)cands.size(), max_cands);
        std::nth_element(cands.begin(), cands.begin() + M, cands.end(),
                         [](const SwapCand& a, const SwapCand& b) { return a.d < b.d; });
        std::sort(cands.begin(), cands.begin() + M,
                  [](const SwapCand& a, const SwapCand& b) { return a.d < b.d; });

        bool found_2swap = false;
        for (int c = 0; c < M; ++c) {
            const auto& c1 = cands[c];
            // Tentatively apply swap c1 (exact involution!)
            st.apply_swap(c1.s, c1.i, c1.j, c1.d, c1.d, 0);

            int64_t best_d2 = -c1.d; // need d1 + d2 < 0 => d2 < -d1
            int b2s = -1, b2i = -1, b2j = -1;
            for (int s = 0; s < 4; ++s) {
                for (const auto& grp : prob.size_groups) {
                    pos_buf.clear();
                    neg_buf.clear();
                    for (int idx : grp) {
                        if (st.seq[s][idx] > 0) pos_buf.push_back(idx);
                        else neg_buf.push_back(idx);
                    }
                    for (int i : pos_buf) {
                        for (int j : neg_buf) {
                            int64_t dac, dg, da;
                            st.eval_swap_pos_neg(s, i, j, dac, dg, da);
                            if (dac < best_d2) {
                                best_d2 = dac; b2s = s; b2i = i; b2j = j;
                            }
                        }
                    }
                }
            }
            if (b2s != -1) {
                st.apply_swap(b2s, b2i, b2j, best_d2, best_d2, 0);
                g_two_opt_hits.fetch_add(1, std::memory_order_relaxed);
                any_improved = true;
                found_2swap = true;
                record_if_best(tid, "2opt", prob, st, t0, out_path);
                break;
            } else {
                // Exact undo of c1
                st.apply_swap(c1.s, c1.i, c1.j, -c1.d, -c1.d, 0);
            }
        }
        if (!found_2swap) break;
    }

    st.gls_active = gls_was;
    st.recompute_E_gls();
    return any_improved;
}

// Fast Tabu polish with optional GLS, strict row-sum lock, branchless pos/neg swaps, and 1-seq LNS
bool tabu_polish(int tid, const char* tag, SearchState& st, int max_steps, int max_no_imp,
                 bool use_gls, bool lock_row, std::mt19937_64& rng,
                 const std::chrono::steady_clock::time_point& t0,
                 double max_seconds, const std::string& out_path, int only_s = -1) {
    const OrbitProblem& prob = *st.prob;
    int r = prob.r;
    std::vector<uint64_t> tabu((size_t)4 * r, 0);
    int base_tenure = std::max(6, r / 4);
    int tabu_tenure = base_tenure + (int)(rng() % (base_tenure + 1));
    int s_lo = (only_s >= 0) ? only_s : 0;
    int s_hi = (only_s >= 0) ? (only_s + 1) : 4;

    int64_t start_E_true = st.E_true;
    int64_t best_E_search = st.E_search;
    int64_t best_E_true = st.E_true;
    std::vector<int> best_seq[4];
    for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];

    std::vector<int> pos_buf, neg_buf;
    pos_buf.reserve(r);
    neg_buf.reserve(r);

    int no_imp_search = 0;
    int no_imp_true = 0;
    const int gls_patience = 220;
    int step = 0;

    for (step = 1; step <= max_steps && no_imp_true < max_no_imp; ++step) {
        if ((step & 255) == 0) {
            g_tabu_steps.fetch_add(256, std::memory_order_relaxed);
            if (g_found.load(std::memory_order_relaxed)) return false;
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (el >= max_seconds) break;
        }

        int best_type = -1, best_s = -1, best_i = -1, best_j = -1, best_k = -1;
        int64_t best_d_search = INT64_MAX;
        int64_t best_d_ac = 0, best_d_row = 0, best_d_gls = 0, best_d_alt = 0;
        int n_ties = 0;

        // 0. Single flips (only when not strictly locked on E_row == 0)
        if (!lock_row || st.E_row != 0) {
            for (int s = s_lo; s < s_hi; ++s) {
                for (int i = 0; i < r; ++i) {
                    int64_t dac, drow, dg, da;
                    st.eval_flip(s, i, dac, drow, dg, da);
                    int64_t ds = dg + da;
                    int64_t dt = dac + BASE_ROW_WEIGHT * drow;
                    bool is_tabu = (tabu[s * r + i] > (uint64_t)step) &&
                                   (st.E_search + ds >= best_E_search) &&
                                   (st.E_true + dt >= best_E_true);
                    if (!is_tabu) {
                        if (ds < best_d_search) {
                            best_d_search = ds; best_d_ac = dac; best_d_row = drow; best_d_gls = dg; best_d_alt = da;
                            best_type = 0; best_s = s; best_i = i; n_ties = 1;
                        } else if (ds == best_d_search) {
                            if ((int)(rng() % (++n_ties)) == 0) {
                                best_d_ac = dac; best_d_row = drow; best_d_gls = dg; best_d_alt = da;
                                best_type = 0; best_s = s; best_i = i;
                            }
                        }
                    }
                }
            }
        }

        // 1. Same-size swaps via branchless pos x neg split lists (exact row-sum preserving)
        for (int s = s_lo; s < s_hi; ++s) {
            const uint64_t* t_s = &tabu[(size_t)s * r];
            for (const auto& grp : prob.size_groups) {
                pos_buf.clear();
                neg_buf.clear();
                for (int idx : grp) {
                    if (st.seq[s][idx] > 0) pos_buf.push_back(idx);
                    else neg_buf.push_back(idx);
                }
                for (int i : pos_buf) {
                    bool ti = (t_s[i] > (uint64_t)step);
                    for (int j : neg_buf) {
                        int64_t dac, dg, da;
                        st.eval_swap_pos_neg(s, i, j, dac, dg, da);
                        int64_t ds = dg + da;
                        bool is_tabu = (ti || t_s[j] > (uint64_t)step) &&
                                       (st.E_search + ds >= best_E_search) &&
                                       (st.E_true + dac >= best_E_true);
                        if (!is_tabu) {
                            if (ds < best_d_search) {
                                best_d_search = ds; best_d_ac = dac; best_d_row = 0; best_d_gls = dg; best_d_alt = da;
                                best_type = 1; best_s = s; best_i = i; best_j = j; n_ties = 1;
                            } else if (ds == best_d_search) {
                                if ((int)(rng() % (++n_ties)) == 0) {
                                    best_d_ac = dac; best_d_row = 0; best_d_gls = dg; best_d_alt = da;
                                    best_type = 1; best_s = s; best_i = i; best_j = j;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Row-sum-preserving 3-orbit triples: size[i] == size[j] + size[k]
        if (!prob.row_preserving_triples.empty()) {
            for (int s = s_lo; s < s_hi; ++s) {
                for (const auto& tr : prob.row_preserving_triples) {
                    int i = tr.i, j = tr.j, k_orb = tr.k;
                    int sj = st.seq[s][j];
                    if (sj != st.seq[s][k_orb] || st.seq[s][i] == sj) continue;
                    int64_t dac, dg, da;
                    st.eval_triple(s, i, j, k_orb, dac, dg, da);
                    int64_t ds = dg + da;
                    bool is_tabu = (tabu[s * r + i] > (uint64_t)step ||
                                    tabu[s * r + j] > (uint64_t)step ||
                                    tabu[s * r + k_orb] > (uint64_t)step) &&
                                   (st.E_search + ds >= best_E_search) &&
                                   (st.E_true + dac >= best_E_true);
                    if (!is_tabu) {
                        if (ds < best_d_search) {
                            best_d_search = ds; best_d_ac = dac; best_d_row = 0; best_d_gls = dg; best_d_alt = da;
                            best_type = 2; best_s = s; best_i = i; best_j = j; best_k = k_orb; n_ties = 1;
                        } else if (ds == best_d_search) {
                            if ((int)(rng() % (++n_ties)) == 0) {
                                best_d_ac = dac; best_d_row = 0; best_d_gls = dg; best_d_alt = da;
                                best_type = 2; best_s = s; best_i = i; best_j = j; best_k = k_orb;
                            }
                        }
                    }
                }
            }
        }

        if (best_type == -1) break;

        if (best_type == 0) {
            st.apply_flip(best_s, best_i, best_d_ac, best_d_row, best_d_gls, best_d_alt);
            tabu[best_s * r + best_i] = step + tabu_tenure + (rng() % 5);
        } else if (best_type == 1) {
            st.apply_swap(best_s, best_i, best_j, best_d_ac, best_d_gls, best_d_alt);
            tabu[best_s * r + best_i] = step + tabu_tenure + (rng() % 5);
            tabu[best_s * r + best_j] = step + tabu_tenure + (rng() % 5);
        } else {
            st.apply_triple(best_s, best_i, best_j, best_k, best_d_ac, best_d_gls, best_d_alt);
            tabu[best_s * r + best_i] = step + tabu_tenure + (rng() % 5);
            tabu[best_s * r + best_j] = step + tabu_tenure + (rng() % 5);
            tabu[best_s * r + best_k] = step + tabu_tenure + (rng() % 5);
        }

        if (st.E_true < best_E_true) {
            best_E_true = st.E_true;
            for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];
            no_imp_true = 0;
            record_if_best(tid, tag, prob, st, t0, out_path);
            if (g_found.load(std::memory_order_relaxed)) return true;
        } else {
            ++no_imp_true;
            if (st.E_row == 0 && st.E_ac < g_best_rsum0_E.load(std::memory_order_relaxed)) {
                record_if_best(tid, tag, prob, st, t0, out_path);
                if (g_found.load(std::memory_order_relaxed)) return true;
            }
        }

        if (st.E_search < best_E_search) {
            best_E_search = st.E_search;
            no_imp_search = 0;
        } else {
            ++no_imp_search;
        }

        if (use_gls && no_imp_search >= gls_patience) {
            int best_k_shift = -1;
            double best_u = -1.0;
            for (int k = 0; k < prob.K; ++k) {
                if (st.P[k] == 0) continue;
                double u = (double)((int64_t)st.P[k] * st.P[k]) / (double)st.shift_w[k];
                if (u > best_u) {
                    best_u = u;
                    best_k_shift = k;
                }
            }
            if (best_k_shift >= 0) {
                st.shift_w[best_k_shift] += 1;
                st.gls_active = true;
                if (st.shift_w[best_k_shift] > 16) {
                    for (int k = 0; k < prob.K; ++k) {
                        st.shift_w[k] = std::max(1, (st.shift_w[k] + 1) >> 1);
                    }
                }
                st.recompute_E_gls();
                best_E_search = st.E_search;
                no_imp_search = 0;
            }
        }
    }

    g_tabu_steps.fetch_add((uint64_t)(step & 255), std::memory_order_relaxed);
    if (st.gls_active) st.reset_gls_weights();
    if (best_E_true < st.E_true) {
        for (int s = 0; s < 4; ++s) st.seq[s] = best_seq[s];
        st.recompute();
    }
    return best_E_true < start_E_true;
}

// Worker A: Parallel Tempering (12 replicas) + Manifold Init + Triples + Periodic Tabu Polish
void worker_pt(int tid, const OrbitProblem& prob, double max_seconds,
               const std::string& out_path, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto t0 = std::chrono::steady_clock::now();

    const int R = 12;
    std::vector<double> temps(R);
    double scale = (double)prob.max_orb_size * (double)prob.max_orb_size * (double)prob.K;
    double t_min = std::max(4.0, 0.00005 * scale);
    double t_max = std::max(4000.0, 0.5 * scale);
    for (int i = 0; i < R; ++i) {
        temps[i] = t_min * std::pow(t_max / t_min, (double)i / (R - 1));
    }

    std::vector<SearchState> reps;
    reps.reserve(R);
    for (int i = 0; i < R; ++i) {
        reps.emplace_back(prob);
        if (prob.lock_row || i < (R * 3) / 4) {
            reps.back().randomize_manifold(rng);
        } else {
            reps.back().randomize_uniform(rng);
        }
        if (i < 2) {
            tabu_polish(tid, "PT-init", reps.back(), 200, 60, false, prob.lock_row, rng, t0, max_seconds, out_path);
            record_if_best(tid, "PT-init", prob, reps.back(), t0, out_path);
            if (g_found.load(std::memory_order_relaxed)) return;
        }
    }

    const int SWEEP_STEPS = 512;
    const int n_pairs = (int)prob.same_size_pairs.size();
    const int n_same_alt_pairs = (int)prob.same_size_same_alt_pairs.size();
    const int n_triples = (int)prob.row_preserving_triples.size();
    uint64_t round = 0;
    int64_t last_polished_E = INT64_MAX;
    int rounds_since_polish = 0;

    while (!g_found.load(std::memory_order_relaxed)) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= max_seconds) break;

        for (int rep = 0; rep < R; ++rep) {
            SearchState& st = reps[rep];
            double invT = 1.0 / temps[rep];
            int flip_pct = prob.lock_row ? 0 : (5 + (rep * 30) / (R - 1)); // 5% .. 35%
            int triple_pct = (n_triples > 0) ? 25 : 0;

            for (int step = 0; step < SWEEP_STEPS; ++step) {
                int roll = (int)(rng() % 100);
                if (roll < flip_pct || (n_pairs == 0 && n_triples == 0)) {
                    int s = (int)(rng() & 3);
                    int i = (int)(rng() % prob.r);
                    int64_t dac, drow, dg, da;
                    st.eval_flip(s, i, dac, drow, dg, da);
                    int64_t ds = dg + da;
                    if (ds <= 0 || u01(rng) < std::exp(-(double)ds * invT)) {
                        st.apply_flip(s, i, dac, drow, dg, da);
                        if (st.E_true < g_best_true_E.load(std::memory_order_relaxed) ||
                            (st.E_row == 0 && st.E_ac < g_best_rsum0_E.load(std::memory_order_relaxed))) {
                            record_if_best(tid, "PT", prob, st, t0, out_path);
                            if (g_found.load(std::memory_order_relaxed)) return;
                        }
                    }
                } else if (roll < flip_pct + triple_pct && n_triples > 0) {
                    int s = (int)(rng() & 3);
                    const auto& tr = prob.row_preserving_triples[rng() % n_triples];
                    int sj = st.seq[s][tr.j];
                    if (sj != st.seq[s][tr.k] || st.seq[s][tr.i] == sj) continue;
                    int64_t dac, dg, da;
                    st.eval_triple(s, tr.i, tr.j, tr.k, dac, dg, da);
                    int64_t ds = dg + da;
                    if (ds <= 0 || u01(rng) < std::exp(-(double)ds * invT)) {
                        st.apply_triple(s, tr.i, tr.j, tr.k, dac, dg, da);
                        if (st.E_true < g_best_true_E.load(std::memory_order_relaxed) ||
                            (st.E_row == 0 && st.E_ac < g_best_rsum0_E.load(std::memory_order_relaxed))) {
                            record_if_best(tid, "PT", prob, st, t0, out_path);
                            if (g_found.load(std::memory_order_relaxed)) return;
                        }
                    }
                } else if (n_pairs > 0) {
                    int s = (int)(rng() & 3);
                    std::pair<int, int> pr;
                    if (prob.use_alt && n_same_alt_pairs > 0 && (rng() & 1)) {
                        pr = prob.same_size_same_alt_pairs[rng() % n_same_alt_pairs];
                    } else {
                        pr = prob.same_size_pairs[rng() % n_pairs];
                    }
                    int i = pr.first, j = pr.second;
                    if (st.seq[s][i] == st.seq[s][j]) continue;
                    int64_t dac, dg, da;
                    st.eval_swap(s, i, j, dac, dg, da);
                    int64_t ds = dg + da;
                    if (ds <= 0 || u01(rng) < std::exp(-(double)ds * invT)) {
                        st.apply_swap(s, i, j, dac, dg, da);
                        if (st.E_true < g_best_true_E.load(std::memory_order_relaxed) ||
                            (st.E_row == 0 && st.E_ac < g_best_rsum0_E.load(std::memory_order_relaxed))) {
                            record_if_best(tid, "PT", prob, st, t0, out_path);
                            if (g_found.load(std::memory_order_relaxed)) return;
                        }
                    }
                }
            }
        }
        g_pt_steps.fetch_add((uint64_t)R * SWEEP_STEPS, std::memory_order_relaxed);

        int start_idx = (int)(round & 1);
        for (int i = start_idx; i + 1 < R; i += 2) {
            double beta_diff = (1.0 / temps[i]) - (1.0 / temps[i + 1]);
            double e_diff = (double)(reps[i].E_search - reps[i + 1].E_search);
            double log_acc = beta_diff * e_diff;
            if (log_acc >= 0.0 || u01(rng) < std::exp(log_acc)) {
                std::swap(reps[i], reps[i + 1]);
            }
        }

        ++rounds_since_polish;
        if (rounds_since_polish >= 80 || reps[0].E_search < last_polished_E) {
            rounds_since_polish = 0;
            tabu_polish(tid, "PT-pol", reps[0], 500, 160, false, prob.lock_row, rng, t0, max_seconds, out_path);
            last_polished_E = reps[0].E_search;
            record_if_best(tid, "PT-pol", prob, reps[0], t0, out_path);
            if (g_found.load(std::memory_order_relaxed)) return;
        }
        ++round;
    }
}

// Worker B: Manifold-Initialized Tabu + Guided Local Search (GLS) + ILS + 1-Seq LNS
void worker_gls_ils(int tid, const OrbitProblem& prob, double max_seconds,
                    bool use_gls, const std::string& out_path, uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto t0 = std::chrono::steady_clock::now();

    SearchState st(prob);
    std::vector<int> best_seq[4];
    for (int s = 0; s < 4; ++s) best_seq[s].resize(prob.r);

    while (!g_found.load(std::memory_order_relaxed)) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= max_seconds) break;

        st.reset_gls_weights();
        st.randomize_manifold(rng);
        record_if_best(tid, "ILS-init", prob, st, t0, out_path);
        if (g_found.load(std::memory_order_relaxed)) return;

        int64_t best_E_true = st.E_true;
        for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];

        int kicks_without_improve = 0;
        const int max_kicks = 20;

        while (kicks_without_improve < max_kicks && !g_found.load(std::memory_order_relaxed)) {
            double el0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (el0 >= max_seconds) return;

            int max_no_imp = 2500 + (int)(rng() % 2000);
            tabu_polish(tid, use_gls ? "GLS" : "ILS", st, 25000, max_no_imp, use_gls, prob.lock_row,
                        rng, t0, max_seconds, out_path);
            if (g_found.load(std::memory_order_relaxed)) return;

            // Focused 2-Swap Lookahead (2-opt) at low-energy rsum0 local minima
            if (st.E_row == 0 && st.E_ac <= g_two_opt_thresh) {
                polish_2opt(tid, st, t0, max_seconds, out_path, 64);
                if (g_found.load(std::memory_order_relaxed)) return;
            }

            if (st.E_row == 0) {
                elite_try_insert(st.E_ac, st.seq);
            }

            if (st.E_true < best_E_true) {
                best_E_true = st.E_true;
                for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];
                kicks_without_improve = 0;
            } else {
                ++kicks_without_improve;
            }

            // Alternate between Elite Crossover (20%), 1-Sequence LNS (35%), and small kicks (45%)
            for (int s = 0; s < 4; ++s) st.seq[s] = best_seq[s];
            int kick_roll = (int)(rng() % 100);
            bool did_crossover = false;
            if (kick_roll < 20) {
                std::vector<int> eA[4], eB[4];
                {
                    std::lock_guard<std::mutex> lk(g_elite_mtx);
                    if (g_elite.size() >= 2) {
                        size_t iA = rng() % g_elite.size();
                        size_t iB = (iA + 1 + (rng() % (g_elite.size() - 1))) % g_elite.size();
                        for (int s = 0; s < 4; ++s) {
                            eA[s] = g_elite[iA].seq[s];
                            eB[s] = g_elite[iB].seq[s];
                        }
                        did_crossover = true;
                    }
                }
                if (did_crossover) {
                    // Take 2 or 3 sequences from elite A and remaining from elite B (preserving target_s!)
                    int mask = 1 + (int)(rng() % 14); // non-trivial subset of {0,1,2,3}
                    for (int s = 0; s < 4; ++s) {
                        st.seq[s] = ((mask >> s) & 1) ? eA[s] : eB[s];
                    }
                    st.reset_gls_weights();
                    st.recompute();
                    int s_pol = (int)(rng() & 3);
                    tabu_polish(tid, "Xover-1s", st, 800, 200, false, prob.lock_row,
                                rng, t0, max_seconds, out_path, s_pol);
                    if (g_found.load(std::memory_order_relaxed)) return;
                }
            }
            if (!did_crossover && kick_roll < 55) {
                // 1-Sequence Large Neighborhood Search: resample 1 sequence & polish it against fixed 3
                int s_lns = (int)(rng() & 3);
                st.resample_single_seq(s_lns, rng);
                st.reset_gls_weights();
                st.recompute();
                tabu_polish(tid, "LNS-1s", st, 1200, 300, false, prob.lock_row,
                            rng, t0, max_seconds, out_path, s_lns);
                if (g_found.load(std::memory_order_relaxed)) return;
            } else if (!did_crossover) {
                int n_kicks = 3 + (int)(rng() % 6);
                for (int kc = 0; kc < n_kicks; ++kc) {
                    int s = (int)(rng() & 3);
                    if (prob.lock_row) {
                        bool done = false;
                        for (int tries = 0; tries < 20 && !done; ++tries) {
                            if (!prob.row_preserving_triples.empty() && (rng() % 3) == 0) {
                                const auto& tr = prob.row_preserving_triples[rng() % prob.row_preserving_triples.size()];
                                int sj = st.seq[s][tr.j];
                                if (sj == st.seq[s][tr.k] && st.seq[s][tr.i] != sj) {
                                    st.seq[s][tr.i] = -st.seq[s][tr.i];
                                    st.seq[s][tr.j] = -st.seq[s][tr.j];
                                    st.seq[s][tr.k] = -st.seq[s][tr.k];
                                    done = true;
                                }
                            } else if (!prob.same_size_pairs.empty()) {
                                const auto& pr = prob.same_size_pairs[rng() % prob.same_size_pairs.size()];
                                if (st.seq[s][pr.first] != st.seq[s][pr.second]) {
                                    st.seq[s][pr.first] = -st.seq[s][pr.first];
                                    st.seq[s][pr.second] = -st.seq[s][pr.second];
                                    done = true;
                                }
                            }
                        }
                    } else {
                        int mode = (int)(rng() % 6);
                        if (mode < 2 && prob.use_alt && !prob.same_size_same_alt_pairs.empty()) {
                            const auto& pr = prob.same_size_same_alt_pairs[rng() % prob.same_size_same_alt_pairs.size()];
                            if (st.seq[s][pr.first] != st.seq[s][pr.second]) {
                                st.seq[s][pr.first] = -st.seq[s][pr.first];
                                st.seq[s][pr.second] = -st.seq[s][pr.second];
                            }
                        } else if (mode < 4 && !prob.same_size_pairs.empty()) {
                            const auto& pr = prob.same_size_pairs[rng() % prob.same_size_pairs.size()];
                            if (st.seq[s][pr.first] != st.seq[s][pr.second]) {
                                st.seq[s][pr.first] = -st.seq[s][pr.first];
                                st.seq[s][pr.second] = -st.seq[s][pr.second];
                            }
                        } else if (mode == 4 && !prob.row_preserving_triples.empty()) {
                            const auto& tr = prob.row_preserving_triples[rng() % prob.row_preserving_triples.size()];
                            int sj = st.seq[s][tr.j];
                            if (sj == st.seq[s][tr.k] && st.seq[s][tr.i] != sj) {
                                st.seq[s][tr.i] = -st.seq[s][tr.i];
                                st.seq[s][tr.j] = -st.seq[s][tr.j];
                                st.seq[s][tr.k] = -st.seq[s][tr.k];
                            }
                        } else {
                            int i = (int)(rng() % prob.r);
                            st.seq[s][i] = -st.seq[s][i];
                        }
                    }
                }
                st.reset_gls_weights();
                st.recompute();
            }
        }
    }
}

static inline uint64_t fast_rng(uint64_t& s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

static inline double fast_u01(uint64_t& s) {
    return (double)(fast_rng(s) >> 11) * (1.0 / 9007199254740992.0);
}

static void build_pos_neg_lists(const SearchState& st,
                                std::vector<std::vector<int>> pos_g[4],
                                std::vector<std::vector<int>> neg_g[4]) {
    const OrbitProblem& prob = *st.prob;
    size_t G = prob.size_groups.size();
    for (int s = 0; s < 4; ++s) {
        pos_g[s].assign(G, {});
        neg_g[s].assign(G, {});
        for (size_t g = 0; g < G; ++g) {
            for (int idx : prob.size_groups[g]) {
                if (st.seq[s][idx] > 0) pos_g[s][g].push_back(idx);
                else neg_g[s][g].push_back(idx);
            }
        }
    }
}

// Worker C: Tight-Band Reheat Simulated Annealing (O(1) pos/neg sampling + optional active flatness)
void worker_sa(int tid, const OrbitProblem& prob, double max_seconds,
               double w_flat, double t_hi, double t_lo,
               const std::string& out_path, uint64_t seed) {
    std::mt19937_64 mt_rng(seed);
    uint64_t rng_s = seed ^ 0x9e3779b97f4a7c15ULL;
    if (rng_s == 0) rng_s = 1;
    auto t0 = std::chrono::steady_clock::now();

    SearchState st(prob);
    st.randomize_manifold(mt_rng);
    record_if_best(tid, "SA-init", prob, st, t0, out_path);
    if (g_found.load(std::memory_order_relaxed)) return;

    int64_t best_E = st.E_ac;
    std::vector<int> best_seq[4];
    for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];

    std::vector<std::vector<int>> pos_g[4], neg_g[4];
    size_t G = prob.size_groups.size();
    if (G == 0) return;
    build_pos_neg_lists(st, pos_g, neg_g);

    const uint64_t CYCLE_STEPS = 8000000ULL;
    const uint64_t BLOCK = 4096ULL;
    uint64_t cycle = 0;

    while (!g_found.load(std::memory_order_relaxed)) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= max_seconds) break;

        double t_start = t_hi;
        if (cycle > 0) {
            if ((fast_rng(rng_s) % 10) < 6) {
                // Reheat from best state to mid-high band
                for (int s = 0; s < 4; ++s) st.seq[s] = best_seq[s];
                st.recompute();
                t_start = t_lo + (t_hi - t_lo) * (0.4 + 0.6 * fast_u01(rng_s));
            } else {
                // Fresh flat-seeded restart
                st.randomize_manifold(mt_rng);
                t_start = t_hi;
            }
            build_pos_neg_lists(st, pos_g, neg_g);
        }

        double log_ratio = std::log(t_lo / t_start);
        for (uint64_t step_base = 0; step_base < CYCLE_STEPS; step_base += BLOCK) {
            if (g_found.load(std::memory_order_relaxed)) return;
            double elc = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (elc >= max_seconds) return;

            double frac = (double)step_base / (double)CYCLE_STEPS;
            double T = t_start * std::exp(log_ratio * frac);
            double invT = 1.0 / T;
            double wf = (w_flat > 0.0 && st.E_ac > 160) ? (w_flat * (1.0 - 0.85 * frac)) : 0.0;

            for (uint64_t b = 0; b < BLOCK; ++b) {
                int s = (int)(fast_rng(rng_s) & 3);
                size_t g = (G == 1) ? 0 : (size_t)(fast_rng(rng_s) % G);
                auto& pvec = pos_g[s][g];
                auto& nvec = neg_g[s][g];
                if (pvec.empty() || nvec.empty()) continue;
                size_t ip = (size_t)(fast_rng(rng_s) % pvec.size());
                size_t jn = (size_t)(fast_rng(rng_s) % nvec.size());
                int i = pvec[ip];
                int j = nvec[jn];

                int64_t dac = 0, dflat = 0;
                double deff = 0.0;
                if (wf > 0.0) {
                    st.eval_swap_pos_neg_flat(s, i, j, dac, dflat);
                    deff = (double)dac + wf * (double)dflat;
                } else {
                    int64_t dg, da;
                    st.eval_swap_pos_neg(s, i, j, dac, dg, da);
                    deff = (double)dac;
                }

                bool accept = (deff <= 0.0);
                if (!accept) {
                    double x = deff * invT;
                    if (x < 18.0 && fast_u01(rng_s) < std::exp(-x)) accept = true;
                }
                if (accept) {
                    st.apply_swap(s, i, j, dac, dac, 0);
                    st.E_flat += dflat;
                    pvec[ip] = j;
                    nvec[jn] = i;
                    if (st.E_ac < best_E) {
                        best_E = st.E_ac;
                        for (int q = 0; q < 4; ++q) best_seq[q] = st.seq[q];
                        record_if_best(tid, "SA", prob, st, t0, out_path);
                        if (g_found.load(std::memory_order_relaxed)) return;
                    }
                }
            }
            g_pt_steps.fetch_add(BLOCK, std::memory_order_relaxed);
        }

        // Quick greedy 1-swap polish at T_lo
        if (st.E_ac <= g_two_opt_thresh) {
            polish_2opt(tid, st, t0, max_seconds, out_path, 32);
            if (g_found.load(std::memory_order_relaxed)) return;
            if (st.E_ac < best_E) {
                best_E = st.E_ac;
                for (int q = 0; q < 4; ++q) best_seq[q] = st.seq[q];
            }
        }
        ++cycle;
    }
}

// Record-to-record travel with an optional flatness term in the record itself.
void worker_rrt(int tid, const OrbitProblem& prob, double max_seconds, double w_flat,
                uint64_t stage_steps, const std::string& out_path, uint64_t seed) {
    std::mt19937_64 mt_rng(seed);
    uint64_t rng_s = seed ^ 0xd1b54a32d192ed03ULL;
    if (rng_s == 0) rng_s = 1;
    auto t0 = std::chrono::steady_clock::now();
    auto expired = [&] {
        return g_found.load(std::memory_order_relaxed) ||
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= max_seconds;
    };

    SearchState st(prob);
    std::vector<std::vector<int>> pos_g[4], neg_g[4];
    std::vector<std::pair<int, size_t>> active_groups;
    const int64_t slacks[] = {256, 192, 160, 128, 96, 64, 32, 0};
    constexpr uint64_t block = 4096;
    int64_t best_E = INT64_MAX;
    std::vector<int> best_seq[4];

    while (!expired()) {
        if (best_E != INT64_MAX && fast_rng(rng_s) % 2 == 0) {
            for (int s = 0; s < 4; ++s) st.seq[s] = best_seq[s];
            st.recompute();
        } else {
            st.randomize_manifold(mt_rng);
        }
        if (st.E_ac < best_E) {
            best_E = st.E_ac;
            for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];
        }
        record_if_best(tid, "RRT-init", prob, st, t0, out_path);

        build_pos_neg_lists(st, pos_g, neg_g);
        active_groups.clear();
        for (int s = 0; s < 4; ++s) {
            for (size_t g = 0; g < prob.size_groups.size(); ++g) {
                if (!pos_g[s][g].empty() && !neg_g[s][g].empty()) {
                    active_groups.emplace_back(s, g);
                }
            }
        }
        if (active_groups.empty()) continue;

        // Keep the flatness term through the final stage. Otherwise the search
        // drops the hard-won spectral structure as the threshold tightens.
        double record_score = st.E_ac + w_flat * st.E_flat;
        for (int64_t slack : slacks) {
            for (uint64_t step = 0; step < stage_steps;) {
                if (expired()) return;
                uint64_t steps = std::min(block, stage_steps - step);
                for (uint64_t b = 0; b < steps; ++b) {
                    auto [s, g] = active_groups[fast_rng(rng_s) % active_groups.size()];
                    auto& positives = pos_g[s][g];
                    auto& negatives = neg_g[s][g];
                    size_t ip = fast_rng(rng_s) % positives.size();
                    size_t jn = fast_rng(rng_s) % negatives.size();
                    int i = positives[ip];
                    int j = negatives[jn];

                    int64_t dac, dflat = 0, dg = 0, da = 0;
                    if (w_flat > 0.0) {
                        st.eval_swap_pos_neg_flat(s, i, j, dac, dflat);
                    } else {
                        st.eval_swap_pos_neg(s, i, j, dac, dg, da);
                    }
                    double next_score = (st.E_ac + dac) + w_flat * (st.E_flat + dflat);
                    if (next_score > record_score + slack) continue;

                    if (w_flat > 0.0 && prob.use_alt) {
                        st.eval_swap_pos_neg(s, i, j, dac, dg, da);
                    }
                    st.apply_swap(s, i, j, dac, dac, da);
                    st.E_flat += dflat;
                    positives[ip] = j;
                    negatives[jn] = i;
                    record_score = std::min(record_score, next_score);
                    if (st.E_ac < best_E) {
                        best_E = st.E_ac;
                        for (int q = 0; q < 4; ++q) best_seq[q] = st.seq[q];
                        record_if_best(tid, "RRT", prob, st, t0, out_path);
                        if (g_found.load(std::memory_order_relaxed)) {
                            g_pt_steps.fetch_add(b + 1, std::memory_order_relaxed);
                            return;
                        }
                    }
                }
                step += steps;
                g_pt_steps.fetch_add(steps, std::memory_order_relaxed);
            }
        }
        if (st.E_ac <= g_two_opt_thresh) {
            polish_2opt(tid, st, t0, max_seconds, out_path, 32);
            if (st.E_ac < best_E) {
                best_E = st.E_ac;
                for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];
            }
        }
    }
}

struct MitmOptions {
    uint32_t pool = 128;
    int64_t flat_max = INT64_MAX;
    int64_t pair_flat_max = INT64_MAX;
    int tabu_steps = 0;
    std::string inject;
};

struct MitmRow {
    std::vector<int> signs;
    std::vector<int> paf;
    std::vector<uint64_t> fingerprints;
};

struct MitmPair {
    uint64_t fingerprint;
    uint32_t left_packed;   // lower 24 bits: row index, upper 8 bits: rotation
    uint32_t right_packed;  // lower 24 bits: row index, upper 8 bits: rotation

    static MitmPair make(uint64_t fp, uint32_t l, uint32_t r, int lr, int rr) {
        return {fp, (l & 0xFFFFFFu) | ((uint32_t)(lr & 0xFF) << 24),
                (r & 0xFFFFFFu) | ((uint32_t)(rr & 0xFF) << 24)};
    }
    uint32_t left() const { return left_packed & 0xFFFFFFu; }
    uint32_t right() const { return right_packed & 0xFFFFFFu; }
    int left_rotation() const { return (int)(left_packed >> 24); }
    int right_rotation() const { return (int)(right_packed >> 24); }
};
static_assert(sizeof(MitmPair) == 16, "MitmPair should stay 16 bytes");

static inline int64_t pair_flatness_sum(const int* __restrict__ a, const int* __restrict__ b, int K) {
    int64_t sum = 0;
    int k = 0;
#ifdef __AVX2__
    __m256i acc = _mm256_setzero_si256();
    for (; k + 8 <= K; k += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + k));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + k));
        __m256i vs = _mm256_add_epi32(va, vb);
        __m256i ev = _mm256_mul_epi32(vs, vs);
        __m256i od_s = _mm256_srli_epi64(vs, 32);
        __m256i od = _mm256_mul_epi32(od_s, od_s);
        acc = _mm256_add_epi64(acc, _mm256_add_epi64(ev, od));
    }
    sum = hsum_epi64(acc);
#endif
    for (; k < K; ++k) {
        int64_t s = (int64_t)a[k] + b[k];
        sum += s * s;
    }
    return sum;
}

// The multiplicative quotient by <U,-1> permutes the nonzero PAF shifts.
static std::vector<int> cyclic_shifts(const OrbitProblem& prob, int& generator) {
    int m = prob.m;
    if (prob.b != 0 || m < 3 || m % 2 == 0) return {};
    for (int d = 3; d <= m / d; d += 2) {
        if (m % d == 0) return {};
    }
    std::vector<bool> in_shift_stabilizer(m);
    for (int u : prob.U) {
        in_shift_stabilizer[u] = in_shift_stabilizer[m - u] = true;
    }
    for (generator = 1; generator < m; ++generator) {
        std::vector<int> shifts;
        int x = 1;
        do {
            shifts.push_back(x);
            x = (int)(1LL * x * generator % m);
        } while (!in_shift_stabilizer[x]);
        if (shifts.size() == (size_t)prob.K) return shifts;
    }
    return {};
}

static int run_flat_mitm(const OrbitProblem& prob, const MitmOptions& options, double seconds,
                         const std::string& out_path, uint64_t seed) {
    auto start = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
    int generator;
    const auto shifts = cyclic_shifts(prob, generator);
    if (shifts.empty()) {
        std::cerr << "MITM requires odd prime m and b=0.\n";
        return 1;
    }
    const int K = prob.K;
    constexpr uint64_t max_table_entries = 20000000;
    if (!(seconds > 0.0) || !std::isfinite(seconds) || options.flat_max < 0 ||
        options.pair_flat_max < 0 || options.tabu_steps < 0) {
        std::cerr << "MITM requires finite positive seconds and nonnegative flatness cutoffs/tabu steps.\n";
        return 1;
    }
    if (K > 255 || options.pool == 0 || options.pool > (1u << 24)) {
        std::cerr << "MITM requires K <= 255 and 1 <= pool <= 2^24.\n";
        return 1;
    }
    if (options.pair_flat_max == INT64_MAX && options.pool > max_table_entries / K / options.pool) {
        std::cerr << "MITM requires K * pool^2 <= " << max_table_entries
                  << " unless --mitm_pair_flat_max is set.\n";
        return 1;
    }

    std::mt19937_64 rng(seed);
    std::mt19937_64 fingerprint_rng(0x24c9bdb9c329ab77ULL);
    std::vector<uint64_t> coefficients(K);
    for (auto& coefficient : coefficients) coefficient = fingerprint_rng();
    std::vector<int> powers(K, 1);
    for (int k = 1; k < K; ++k) powers[k] = (int)(1LL * powers[k - 1] * generator % prob.m);

    std::vector<int> orbit_reps;
    for (const auto& orbit : prob.orbits) orbit_reps.push_back(orbit.front());
    auto rotate_signs = [&](const std::vector<int>& signs, int k) {
        std::vector<int> rotated;
        rotated.reserve(prob.r);
        for (int x : orbit_reps) {
            rotated.push_back(signs[prob.elem_to_orb[1LL * x * powers[k] % prob.m]]);
        }
        return rotated;
    };

    auto pair_anchor = [K](const MitmRow& a, const MitmRow& b, int relative, int sign) {
        const int* first = a.paf.data();
        const int* second = b.paf.data() + relative;
        int best = 0;
        for (int k = 1; k < K; ++k) {
            int difference = first[k] + second[k] - first[best] - second[best];
            for (int j = 1; difference == 0 && j < K; ++j) {
                difference = first[k + j] + second[k + j] -
                             first[best + j] - second[best + j];
            }
            if (sign * difference < 0) best = k;
        }
        return best;
    };

    std::vector<MitmRow> pools[4];
    std::set<std::vector<int>> seen[4];
    uint64_t attempts[4] = {};
    auto add_row = [&](int s, const std::vector<int>& signs, bool injected) {
        std::vector<int> seq(prob.m);
        for (int x = 0; x < prob.m; ++x) seq[x] = signs[prob.elem_to_orb[x]];
        if (std::accumulate(seq.begin(), seq.end(), 0) != prob.target_s[s]) return false;
        std::vector<int> paf;
        paf.reserve(3 * K);
        int64_t flat = 0;
        for (int shift : shifts) {
            int value = 0;
            for (int x = 0; x < prob.m; ++x) value += seq[x] * seq[(x + shift) % prob.m];
            paf.push_back(value);
            flat += (int64_t)value * value;
        }
        if (!injected && flat > options.flat_max) return false;

        // Keep one row for each PAF vector up to multiplicative rotation.
        // Reversal has the same PAF, so this also removes reversal duplicates.
        int canonical = 0;
        for (int k = 1; k < K; ++k) {
            for (int j = 0; j < K; ++j) {
                int left = paf[(k + j) % K], right = paf[(canonical + j) % K];
                if (left == right) continue;
                if (left < right) canonical = k;
                break;
            }
        }
        std::rotate(paf.begin(), paf.begin() + canonical, paf.end());
        if (!seen[s].insert(paf).second) return false;

        std::vector<uint64_t> fingerprints(K);
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < K; ++j) {
                fingerprints[k] += coefficients[j] * (uint64_t)paf[(k + j) % K];
            }
        }
        // Three copies cover every contiguous window of length K at any
        // relative shift and joint rotation without modulo in comparisons.
        paf.resize(3 * K);
        std::copy_n(paf.begin(), K, paf.begin() + K);
        std::copy_n(paf.begin(), K, paf.begin() + 2 * K);
        pools[s].push_back({rotate_signs(signs, canonical), std::move(paf), std::move(fingerprints)});
        return true;
    };

    if (!options.inject.empty()) {
        std::ifstream input(options.inject);
        for (int s = 0; s < 4; ++s) {
            std::string text;
            input >> text;
            if ((int)text.size() != prob.r || text.find_first_not_of("+-") != std::string::npos) {
                std::cerr << "MITM injection requires one orbit sign string per row.\n";
                return 1;
            }
            std::vector<int> signs;
            for (char sign : text) signs.push_back(sign == '+' ? 1 : -1);
            if (!add_row(s, signs, true)) {
                std::cerr << "MITM injected row has the wrong row sum.\n";
                return 1;
            }
        }
        std::string extra;
        if (input >> extra) {
            std::cerr << "MITM injection requires exactly four rows.\n";
            return 1;
        }
    }

    SearchState st(prob);
    bool progress;
    do {
        progress = false;
        for (int s = 0; s < 4 && elapsed() < seconds; ++s) {
            if (pools[s].size() == options.pool || attempts[s] >= 32ULL * options.pool) continue;
            ++attempts[s];
            progress = true;
            if (!prob.full_dp.sample(prob.target_s[s], st.seq[s], rng)) {
                std::cerr << "MITM could not sample the requested row sum.\n";
                return 1;
            }
            if (options.tabu_steps > 0) st.flat_tabu_single(s, rng, options.tabu_steps);
            else st.flat_seed_single(s, rng);
            add_row(s, st.seq[s], false);
        }
    } while (progress);

    double gen_seconds = elapsed();
    uint64_t total_rows = pools[0].size() + pools[1].size() + pools[2].size() + pools[3].size();
    std::cout << "MITM generation: generator=" << generator
              << ", injected=" << !options.inject.empty()
              << ", tabu_steps=" << options.tabu_steps
              << ", flat_max=" << options.flat_max
              << ", pair_flat_max=" << options.pair_flat_max
              << ", seconds=" << gen_seconds
              << ", rows_per_sec=" << (gen_seconds > 0 ? total_rows / gen_seconds : 0.0) << std::endl;
    for (int s = 0; s < 4; ++s) {
        std::vector<int64_t> flats;
        flats.reserve(pools[s].size());
        for (const auto& row : pools[s]) {
            flats.push_back(std::inner_product(row.paf.begin(), row.paf.begin() + K, row.paf.begin(), int64_t{0}));
        }
        std::sort(flats.begin(), flats.end());
        std::cout << "MITM row " << s << ": count=" << pools[s].size() << ", attempts=" << attempts[s];
        if (!flats.empty()) {
            double denom = (double)K * prob.m;
            int64_t med = flats[flats.size() / 2];
            std::cout << ", flatness=[" << flats.front() << "," << med << "," << flats.back() << "]"
                      << ", Phi_f=[" << flats.front() / denom << "," << med / denom << "," << flats.back() / denom << "]";
        }
        std::cout << std::endl;
    }
    for (const auto& pool : pools) {
        if (pool.empty()) return 2;
    }

    double match_start = elapsed();
    if (match_start >= seconds) {
        std::cout << "MITM time limit reached during generation.\n";
        return 2;
    }
    std::vector<MitmPair> table;
    uint64_t max_possible = (uint64_t)pools[0].size() * pools[1].size() * K;
    const bool filter_pairs = (options.pair_flat_max < INT64_MAX);
    uint64_t init_reserve = filter_pairs ? std::min<uint64_t>(max_possible, 1000000)
                                         : std::min<uint64_t>(max_possible, max_table_entries);
    table.reserve((size_t)init_reserve);
    uint64_t build_scanned = 0;
    bool table_capped = false;
    for (uint32_t a = 0; a < pools[0].size() && elapsed() < seconds && !table_capped; ++a) {
        const auto& left = pools[0][a];
        const int* lpaf = left.paf.data();
        for (uint32_t b = 0; b < pools[1].size(); ++b) {
            const auto& right = pools[1][b];
            const int* rpaf = right.paf.data();
            for (int relative = 0; relative < K; ++relative) {
                ++build_scanned;
                if (filter_pairs && pair_flatness_sum(lpaf, rpaf + relative, K) > options.pair_flat_max) {
                    continue;
                }
                if (table.size() >= max_table_entries) {
                    table_capped = true;
                    break;
                }
                int kl = pair_anchor(left, right, relative, 1);
                int kr = (kl + relative) % K;
                table.push_back(MitmPair::make(left.fingerprints[kl] + right.fingerprints[kr], a, b, kl, kr));
            }
            if (table_capped) break;
        }
    }
    if (elapsed() >= seconds) {
        std::cout << "MITM time limit reached during table construction.\n";
        return 2;
    }
    std::sort(table.begin(), table.end(), [](const MitmPair& a, const MitmPair& b) {
        return a.fingerprint < b.fingerprint;
    });
    // Fingerprint prefixes index the sorted table, leaving only a tiny range
    // for binary search. Equal fingerprints always share a bucket.
    constexpr unsigned prefix_bits = 20;
    constexpr size_t bucket_count = size_t{1} << prefix_bits;
    std::vector<size_t> buckets(bucket_count + 1);
    size_t entry = 0;
    for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
        buckets[bucket] = entry;
        while (entry < table.size() && table[entry].fingerprint >> (64 - prefix_bits) == bucket) ++entry;
    }
    buckets.back() = table.size();
    double build_seconds = elapsed() - match_start;
    uint64_t probes = 0;
    uint64_t probe_scanned = 0;
    auto report = [&] {
        double probe_seconds = elapsed() - match_start - build_seconds;
        std::cout << "MITM matching: build_scanned=" << build_scanned
                  << ", entries=" << table.size()
                  << (table_capped ? " (CAPPED)" : "")
                  << ", table_bytes=" << table.capacity() * sizeof(MitmPair)
                  << ", build_seconds=" << build_seconds
                  << ", probe_scanned=" << probe_scanned
                  << ", probes=" << probes
                  << ", probe_seconds=" << probe_seconds
                  << ", scan_pairs_per_sec="
                  << ((build_seconds + probe_seconds) > 0
                          ? (build_scanned + probe_scanned) / (build_seconds + probe_seconds)
                          : 0.0)
                  << std::endl;
    };

    // Joint cyclic rotations act on both members of a pair. Canonicalize
    // each pair under this action while enumerating all relative rotations.
    // On the probing side canonicalize the negated sum, so complement
    // vectors yield the same canonical form even when tied at a minimum.
    for (const auto& c : pools[2]) {
        const int* cpaf = c.paf.data();
        for (const auto& d : pools[3]) {
            if (elapsed() >= seconds) {
                report();
                return 2;
            }
            const int* dpaf = d.paf.data();
            for (int relative = 0; relative < K; ++relative) {
                ++probe_scanned;
                if (filter_pairs && pair_flatness_sum(cpaf, dpaf + relative, K) > options.pair_flat_max) {
                    continue;
                }
                ++probes;
                int kc = pair_anchor(c, d, relative, -1);
                int kd = (kc + relative) % K;
                uint64_t key = 0 - (c.fingerprints[kc] + d.fingerprints[kd]);
                size_t bucket = key >> (64 - prefix_bits);
                auto end = table.begin() + buckets[bucket + 1];
                auto it = std::lower_bound(table.begin() + buckets[bucket], end, key,
                    [](const MitmPair& pair, uint64_t value) { return pair.fingerprint < value; });
                for (; it != end && it->fingerprint == key; ++it) {
                    const auto& a = pools[0][it->left()];
                    const auto& b = pools[1][it->right()];
                    int ka = it->left_rotation(), kb = it->right_rotation();
                    bool exact = true;
                    for (int j = 0; j < K; ++j) {
                        if (a.paf[ka + j] + b.paf[kb + j] + c.paf[kc + j] + d.paf[kd + j] != 0) {
                            exact = false;
                            break;
                        }
                    }
                    if (!exact) continue;
                    st.seq[0] = rotate_signs(a.signs, ka);
                    st.seq[1] = rotate_signs(b.signs, kb);
                    st.seq[2] = rotate_signs(c.signs, kc);
                    st.seq[3] = rotate_signs(d.signs, kd);
                    st.recompute();
                    record_if_best(0, "MITM", prob, st, start, out_path);
                    if (g_found.load(std::memory_order_relaxed)) {
                        report();
                        return 0;
                    }
                }
            }
        }
    }
    report();
    return 2;
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> res;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) res.push_back(std::stoi(s.substr(p, q - p)));
        p = q + 1;
    }
    return res;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <m> <g|g1,g2,...> <b> <seconds> <out.json> [threads=2] [s0 s1 s2 s3]"
                  << " [--gens=g1,g2,...] [--ts=s0,s1,s2,s3] [--w_row=64] [--w_alt=16]"
                  << " [--lock_row=0|1] [--flat_init=0|1] [--w_flat=0.0] [--sa_thi=220] [--sa_tlo=10]"
                  << " [--target_alt=a0,a1,a2,a3] [--mode=hybrid|ils|pt|sa|rrt|mitm] [--gls=1|0]"
                  << " [--save_best=<path>] [--two_opt_thresh=2000] [--seed=<uint64>] [--rrt_stage_steps=800000]"
                  << " [--mitm_pool=128] [--mitm_flat_max=<energy>] [--mitm_pair_flat_max=<energy>] [--mitm_tabu_steps=0] [--mitm_inject=<signs.txt>]\n";
        return 1;
    }
    int m = std::stoi(argv[1]);
    std::vector<int> gens = parse_int_list(argv[2]);
    int b = std::stoi(argv[3]);
    double secs = std::stod(argv[4]);
    std::string out_path = argv[5];
    int threads = 2;
    int ts[4] = {2, 0, 0, 0};
    int talt[4] = {0, 0, 0, 0};
    bool use_talt = false;
    int64_t w_row = 64;
    int64_t w_alt = 16;
    bool lock_row = false;
    bool flat_init = false;
    double w_flat = 0.0;
    double sa_thi = 220.0;
    double sa_tlo = 10.0;
    std::string mode = "hybrid";
    bool use_gls = true;
    uint64_t base_seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    uint64_t rrt_stage_steps = 800000;
    MitmOptions mitm;

    std::vector<std::string> pos;
    for (int i = 6; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--w_row=", 0) == 0) {
            w_row = std::stoll(a.substr(8));
        } else if (a.rfind("--w_alt=", 0) == 0) {
            w_alt = std::stoll(a.substr(8));
        } else if (a.rfind("--w_flat=", 0) == 0) {
            w_flat = std::stod(a.substr(9));
        } else if (a.rfind("--sa_thi=", 0) == 0) {
            sa_thi = std::stod(a.substr(9));
        } else if (a.rfind("--sa_tlo=", 0) == 0) {
            sa_tlo = std::stod(a.substr(9));
        } else if (a.rfind("--lock_row=", 0) == 0) {
            lock_row = (std::stoi(a.substr(11)) != 0);
        } else if (a.rfind("--flat_init=", 0) == 0) {
            flat_init = (std::stoi(a.substr(12)) != 0);
        } else if (a.rfind("--save_best=", 0) == 0) {
            g_save_best_path = a.substr(12);
        } else if (a.rfind("--two_opt_thresh=", 0) == 0) {
            g_two_opt_thresh = std::stoll(a.substr(17));
        } else if (a.rfind("--ts=", 0) == 0) {
            auto vals = parse_int_list(a.substr(5));
            if (vals.size() == 4) {
                for (int k = 0; k < 4; ++k) ts[k] = vals[k];
            }
        } else if (a.rfind("--gens=", 0) == 0) {
            gens = parse_int_list(a.substr(7));
        } else if (a == "--gens" && i + 1 < argc) {
            gens = parse_int_list(argv[++i]);
        } else if (a == "-g" && i + 1 < argc) {
            gens = parse_int_list(argv[++i]);
        } else if (a.rfind("--seed=", 0) == 0) {
            base_seed = std::stoull(a.substr(7));
        } else if (a.rfind("--rrt_stage_steps=", 0) == 0) {
            rrt_stage_steps = std::stoull(a.substr(18));
        } else if (a.rfind("--mitm_pool=", 0) == 0) {
            mitm.pool = std::stoi(a.substr(12));
        } else if (a.rfind("--mitm_flat_max=", 0) == 0) {
            mitm.flat_max = std::stoll(a.substr(16));
        } else if (a.rfind("--mitm_pair_flat_max=", 0) == 0) {
            mitm.pair_flat_max = std::stoll(a.substr(21));
        } else if (a.rfind("--mitm_tabu_steps=", 0) == 0) {
            mitm.tabu_steps = std::stoi(a.substr(18));
        } else if (a.rfind("--mitm_inject=", 0) == 0) {
            mitm.inject = a.substr(14);
        } else if (a.rfind("--mode=", 0) == 0) {
            mode = a.substr(7);
        } else if (a.rfind("--gls=", 0) == 0) {
            use_gls = (std::stoi(a.substr(6)) != 0);
        } else if (a.rfind("--target_alt=", 0) == 0) {
            auto vals = parse_int_list(a.substr(13));
            if (vals.size() == 4) {
                for (int k = 0; k < 4; ++k) talt[k] = vals[k];
                use_talt = true;
            }
        } else {
            pos.push_back(a);
        }
    }
    if (!pos.empty()) threads = std::stoi(pos[0]);
    if (pos.size() >= 5) {
        for (int i = 0; i < 4; ++i) ts[i] = std::stoi(pos[1 + i]);
    }

    if (rrt_stage_steps == 0 || !std::isfinite(w_flat) || w_flat < 0) {
        std::cerr << "Expected positive RRT stage steps and a finite, nonnegative flatness weight.\n";
        return 1;
    }

    OrbitProblem prob = build_problem(m, gens, b, ts, w_row, w_alt, lock_row, flat_init, use_talt, talt);
    std::cout << "Hadamard v3 n=" << prob.n << " (m=" << m << ", |U|=" << prob.U.size()
              << ", gens=[";
    for (size_t i = 0; i < prob.gens.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << prob.gens[i];
    }
    std::cout << "], b=" << b << "): r=" << prob.r << " orbits (max_sz=" << prob.max_orb_size
              << "), K=" << prob.K << " (pad=" << prob.Kpad
              << "), pairs=" << prob.same_size_pairs.size()
              << " (same_alt=" << prob.same_size_same_alt_pairs.size()
              << "), triples=" << prob.row_preserving_triples.size()
              << ", valid_2comp=" << prob.valid_alt_tuples.size()
              << ", w_row=" << prob.w_row << ", w_alt=" << prob.w_alt
              << ", lock_row=" << (prob.lock_row ? 1 : 0)
              << ", flat_init=" << (prob.flat_init ? 1 : 0)
              << ", w_flat=" << w_flat
              << ", 2opt_th=" << g_two_opt_thresh
              << ", mode=" << mode << ", gls=" << (use_gls ? 1 : 0);
#ifdef __AVX2__
    std::cout << ", simd=AVX2";
#else
    std::cout << ", simd=scalar";
#endif
    std::cout << ", threads=" << (mode == "mitm" ? 1 : threads) << ", seed=" << base_seed;
    if (mode == "rrt") std::cout << ", rrt_stage_steps=" << rrt_stage_steps;
    std::cout << std::endl;

    if (prob.infeasible) {
        std::cout << "INFEASIBLE (pre-check): " << prob.infeasible_reason << std::endl;
        return 3;
    }

    if (mode == "mitm") return run_flat_mitm(prob, mitm, secs, out_path, base_seed);

    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        uint64_t seed = base_seed + 1000003ULL * t;
        if (mode == "sa") {
            pool.emplace_back(worker_sa, t, std::cref(prob), secs, w_flat, sa_thi, sa_tlo, out_path, seed);
        } else if (mode == "rrt") {
            pool.emplace_back(worker_rrt, t, std::cref(prob), secs, w_flat, rrt_stage_steps, out_path, seed);
        } else if (mode == "pt" || (mode == "hybrid" && (t & 1) == 1)) {
            pool.emplace_back(worker_pt, t, std::cref(prob), secs, out_path, seed);
        } else {
            pool.emplace_back(worker_gls_ils, t, std::cref(prob), secs, use_gls, out_path, seed);
        }
    }
    for (auto& th : pool) th.join();

    std::cout << "Summary: tabu_steps=" << g_tabu_steps.load()
              << ", pt_steps=" << g_pt_steps.load()
              << ", two_opt_hits=" << g_two_opt_hits.load()
              << ", best_E=" << g_best_true_E.load()
              << ", best_rsum0_E=" << (g_best_rsum0_E.load() == INT64_MAX ? -1 : g_best_rsum0_E.load())
              << ", best_manifold_E=" << (g_best_manifold_E.load() == INT64_MAX ? -1 : g_best_manifold_E.load())
              << ", found=" << g_found.load() << std::endl;
    if (!g_best_rsum0_P.empty()) {
        std::cout << "Best rsum0 nonzero P[d]:";
        for (int k = 0; k < prob.K; ++k) {
            if (g_best_rsum0_P[k] != 0) {
                std::cout << " d=" << prob.shift_reps[k] << ":" << g_best_rsum0_P[k];
            }
        }
        std::cout << std::endl;
    } else if (!g_best_P.empty()) {
        std::cout << "Best rsum=(" << g_best_rsum[0] << "," << g_best_rsum[1] << ","
                  << g_best_rsum[2] << "," << g_best_rsum[3] << "), nonzero P[d]:";
        for (int k = 0; k < prob.K; ++k) {
            if (g_best_P[k] != 0) {
                std::cout << " d=" << prob.shift_reps[k] << ":" << g_best_P[k];
            }
        }
        std::cout << std::endl;
    }
    return g_found.load() ? 0 : 2;
}

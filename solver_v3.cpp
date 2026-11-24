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
#include <sstream>
#include <string>
#include <thread>
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
                           int64_t w_row, int64_t w_alt, bool lock_row,
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
std::atomic<int64_t> g_best_true_E{INT64_MAX};
std::atomic<int64_t> g_best_rsum0_E{INT64_MAX};
std::atomic<int64_t> g_best_manifold_E{INT64_MAX};
std::mutex g_io_mtx;
std::vector<int> g_best_P;
int g_best_rsum[4] = {0, 0, 0, 0};
std::vector<int> g_best_rsum0_P;

void save_witness(const std::string& path, const OrbitProblem& prob, const std::vector<int> seq[4]) {
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
    std::vector<int> grad;
    std::vector<int> shift_w;
    bool gls_active = false;
    int rsum[4];
    int alt[4];
    int64_t S_alt;
    int64_t E_ac;
    int64_t E_row; // raw sum (rsum[s] - target_s[s])^2
    int64_t E_true; // E_ac + BASE_ROW_WEIGHT * E_row
    int64_t E_gls;  // sum shift_w[k]*P[k]^2 + prob->w_row * E_row
    int64_t E_alt;
    int64_t E_search;

    SearchState()
        : prob(nullptr), r(0), K(0), Kpad(0), S_alt(0),
          E_ac(0), E_row(0), E_true(0), E_gls(0), E_alt(0), E_search(0) {}

    explicit SearchState(const OrbitProblem& p)
        : prob(&p), r(p.r), K(p.K), Kpad(p.Kpad),
          P(p.Kpad, 0), grad((size_t)4 * p.r * p.Kpad, 0), shift_w(p.Kpad, 0) {
        for (int k = 0; k < K; ++k) shift_w[k] = 1;
        for (int s = 0; s < 4; ++s) {
            seq[s].assign(r, 1);
            rsum[s] = 0;
            alt[s] = 0;
        }
        S_alt = 0;
        E_ac = E_row = E_true = E_gls = E_alt = E_search = 0;
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
        }
        std::fill(P.begin(), P.end(), 0);
        std::fill(grad.begin(), grad.end(), 0);
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
                pk += acc / 2;
            }
            P[k] = (int)pk;
        }
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
                recompute();
                return;
            }
        }
        randomize_uniform(rng);
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
        for (int k = 0; k < Kpad; ++k) {
            pptr[k] += factor * gptr[k];
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
                save_witness(out_path, prob, st.seq);
                std::lock_guard<std::mutex> lk(g_io_mtx);
                std::cout << "FOUND EXACT VERIFIED SOLUTION! Saved to " << out_path << std::endl;
            }
        }
    }
}

// Fast Tabu polish with optional GLS and optional strict row-sum lock
bool tabu_polish(int tid, const char* tag, SearchState& st, int max_steps, int max_no_imp,
                 bool use_gls, bool lock_row, std::mt19937_64& rng,
                 const std::chrono::steady_clock::time_point& t0,
                 double max_seconds, const std::string& out_path) {
    const OrbitProblem& prob = *st.prob;
    int r = prob.r;
    std::vector<uint64_t> tabu((size_t)4 * r, 0);
    int tabu_tenure = 5 + (int)(rng() % 10);

    int64_t start_E_true = st.E_true;
    int64_t best_E_search = st.E_search;
    int64_t best_E_true = st.E_true;
    std::vector<int> best_seq[4];
    for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];

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
            for (int s = 0; s < 4; ++s) {
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

        // 1. Same-size swaps (exact row-sum preserving)
        for (int s = 0; s < 4; ++s) {
            for (const auto& pr : prob.same_size_pairs) {
                int i = pr.first, j = pr.second;
                if (st.seq[s][i] == st.seq[s][j]) continue;
                int64_t dac, dg, da;
                st.eval_swap(s, i, j, dac, dg, da);
                int64_t ds = dg + da;
                bool is_tabu = (tabu[s * r + i] > (uint64_t)step || tabu[s * r + j] > (uint64_t)step) &&
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

        // 2. Row-sum-preserving 3-orbit triples: size[i] == size[j] + size[k]
        for (int s = 0; s < 4; ++s) {
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
                    if (n_same_alt_pairs > 0 && (rng() & 1)) {
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

// Worker B: Manifold-Initialized Tabu + Guided Local Search (GLS) + ILS
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

            if (st.E_true < best_E_true) {
                best_E_true = st.E_true;
                for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];
                kicks_without_improve = 0;
            } else {
                ++kicks_without_improve;
            }

            // Structure-preserving ILS kick from best_seq
            for (int s = 0; s < 4; ++s) st.seq[s] = best_seq[s];
            int n_kicks = 2 + (int)(rng() % 5);
            for (int kc = 0; kc < n_kicks; ++kc) {
                int s = (int)(rng() & 3);
                if (prob.lock_row) {
                    // Strictly row-sum preserving kick (valid swap or valid triple)
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
                    if (mode < 2 && !prob.same_size_same_alt_pairs.empty()) {
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
                  << " [--gens=g1,g2,...] [--w_row=64] [--w_alt=16] [--lock_row=0|1]"
                  << " [--target_alt=a0,a1,a2,a3] [--mode=hybrid|ils|pt] [--gls=1|0]\n";
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
    std::string mode = "hybrid";
    bool use_gls = true;

    std::vector<std::string> pos;
    for (int i = 6; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--w_row=", 0) == 0) {
            w_row = std::stoll(a.substr(8));
        } else if (a.rfind("--w_alt=", 0) == 0) {
            w_alt = std::stoll(a.substr(8));
        } else if (a.rfind("--lock_row=", 0) == 0) {
            lock_row = (std::stoi(a.substr(11)) != 0);
        } else if (a.rfind("--gens=", 0) == 0) {
            gens = parse_int_list(a.substr(7));
        } else if (a == "--gens" && i + 1 < argc) {
            gens = parse_int_list(argv[++i]);
        } else if (a == "-g" && i + 1 < argc) {
            gens = parse_int_list(argv[++i]);
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

    OrbitProblem prob = build_problem(m, gens, b, ts, w_row, w_alt, lock_row, use_talt, talt);
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
              << ", mode=" << mode << ", gls=" << (use_gls ? 1 : 0);
#ifdef __AVX2__
    std::cout << ", simd=AVX2";
#else
    std::cout << ", simd=scalar";
#endif
    std::cout << ", threads=" << threads << std::endl;

    if (prob.infeasible) {
        std::cout << "INFEASIBLE (pre-check): " << prob.infeasible_reason << std::endl;
        return 3;
    }

    std::vector<std::thread> pool;
    uint64_t base_seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (int t = 0; t < threads; ++t) {
        uint64_t seed = base_seed + 1000003ULL * t;
        bool run_pt = (mode == "pt") || (mode == "hybrid" && (t & 1) == 1);
        if (run_pt) {
            pool.emplace_back(worker_pt, t, std::cref(prob), secs, out_path, seed);
        } else {
            pool.emplace_back(worker_gls_ils, t, std::cref(prob), secs, use_gls, out_path, seed);
        }
    }
    for (auto& th : pool) th.join();

    std::cout << "Summary: tabu_steps=" << g_tabu_steps.load()
              << ", pt_steps=" << g_pt_steps.load()
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

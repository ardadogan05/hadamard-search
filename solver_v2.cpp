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
#include <string>
#include <thread>
#include <vector>

struct OrbitProblem {
    int n, m, g, b;
    int r, K, Kpad;
    bool use_alt;
    bool use_target_alt;
    int target_alt[4];
    int64_t w_alt;
    int max_orb_size;
    std::vector<std::vector<int>> orbits;
    std::vector<int> orb_size;
    std::vector<int> orb_alt; // (# even elems) - (# odd elems) in orbit
    std::vector<int> elem_to_orb;
    std::vector<int> shift_reps;
    std::vector<int> shift_weights;
    std::vector<int> MT; // MT[(i * r + j) * Kpad + k], padded to multiple of 8
    std::vector<int> diag_const;
    int target_ac;
    int target_s[4];
    std::vector<std::pair<int, int>> same_size_pairs;
    std::vector<std::pair<int, int>> same_size_same_alt_pairs;
};

OrbitProblem build_problem(int m, int g, int b, const int ts[4], int64_t w_alt,
                           bool use_talt, const int talt[4]) {
    OrbitProblem prob;
    prob.m = m;
    prob.g = g;
    prob.b = b;
    prob.n = 4 * m + b;
    prob.target_ac = -b;
    prob.use_alt = (m % 2 == 0) && (b == 4 || b == 0);
    prob.use_target_alt = prob.use_alt && use_talt;
    prob.w_alt = prob.use_alt ? w_alt : 0;
    prob.max_orb_size = 1;
    for (int s = 0; s < 4; ++s) {
        prob.target_s[s] = ts[s];
        prob.target_alt[s] = talt[s];
    }

    prob.elem_to_orb.assign(m, -1);
    for (int x = 0; x < m; ++x) {
        if (prob.elem_to_orb[x] == -1) {
            int oid = (int)prob.orbits.size();
            std::vector<int> orb;
            long long cur = x;
            int alt_c = 0;
            while (prob.elem_to_orb[cur] == -1) {
                prob.elem_to_orb[cur] = oid;
                orb.push_back((int)cur);
                alt_c += ((cur & 1) == 0) ? 1 : -1;
                cur = (cur * g) % m;
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

    std::vector<bool> seen_d(m, false);
    seen_d[0] = true;
    for (int d = 1; d < m; ++d) {
        if (!seen_d[d]) {
            prob.shift_reps.push_back(d);
            int w = 0;
            long long cur = d;
            while (!seen_d[cur]) {
                seen_d[cur] = true;
                ++w;
                int neg = (m - (int)cur) % m;
                if (!seen_d[neg]) {
                    seen_d[neg] = true;
                    ++w;
                }
                cur = (cur * g) % m;
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
std::atomic<uint64_t> g_steps{0};
std::atomic<int64_t> g_best_true_E{INT64_MAX};
std::atomic<int64_t> g_best_manifold_E{INT64_MAX};
std::mutex g_io_mtx;

void save_witness(const std::string& path, const OrbitProblem& prob, const std::vector<int> seq[4]) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"n\": " << prob.n << ",\n";
    out << "  \"m\": " << prob.m << ",\n";
    out << "  \"g\": " << prob.g << ",\n";
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

static constexpr int64_t ROW_WEIGHT = 4;

struct SearchState {
    const OrbitProblem* prob;
    int r, K, Kpad;
    std::vector<int> seq[4];
    std::vector<int> P;
    std::vector<int> grad;
    int rsum[4];
    int alt[4];
    int64_t S_alt;
    int64_t E_true;
    int64_t E_alt;
    int64_t E_search;

    SearchState() : prob(nullptr), r(0), K(0), Kpad(0), S_alt(0), E_true(0), E_alt(0), E_search(0) {}

    explicit SearchState(const OrbitProblem& p)
        : prob(&p), r(p.r), K(p.K), Kpad(p.Kpad), P(p.Kpad, 0), grad((size_t)4 * p.r * p.Kpad, 0) {
        for (int s = 0; s < 4; ++s) {
            seq[s].assign(r, 1);
            rsum[s] = 0;
            alt[s] = 0;
        }
        S_alt = 0;
        E_true = E_alt = E_search = 0;
    }

    // Huber-scaled 2-compression energy:
    // Let d = |S_alt - n| / 4 (always exact integer for even m).
    // For d <= 8 (|S_alt - n| <= 32): exact quadratic d^2 (matches P[k]^2 scale: d=4 -> 16).
    // For d > 8: smooth C^1 linear tail (16*d - 64) so large orbit swaps do not hit a 10^7 quartic wall.
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
        for (int s = 0; s < 4; ++s) {
            int rs = 0, al = 0;
            for (int i = 0; i < r; ++i) {
                rs += seq[s][i] * prob->orb_size[i];
                al += seq[s][i] * prob->orb_alt[i];
            }
            rsum[s] = rs;
            alt[s] = al;
            S_alt += (int64_t)al * al;
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
        E_true = 0;
        for (int k = 0; k < K; ++k) E_true += (int64_t)P[k] * P[k];
        for (int s = 0; s < 4; ++s) {
            int64_t dr = rsum[s] - prob->target_s[s];
            E_true += ROW_WEIGHT * dr * dr;
        }
        E_alt = calc_E_alt_full(S_alt, alt);
        E_search = E_true + E_alt;
    }

    void randomize(std::mt19937_64& rng) {
        for (int s = 0; s < 4; ++s) {
            for (int i = 0; i < r; ++i) {
                seq[s][i] = (rng() & 1) ? 1 : -1;
            }
        }
        recompute();
    }

    // Evaluate delta of single flip (s, i): returns {delta_E_true, delta_E_alt}
    inline void eval_flip(int s, int i, int64_t& d_true, int64_t& d_alt) const {
        int si = seq[s][i];
        int64_t dr_old = rsum[s] - prob->target_s[s];
        int64_t dr_new = dr_old - 2LL * si * prob->orb_size[i];
        int64_t dt = ROW_WEIGHT * (dr_new * dr_new - dr_old * dr_old);

        const int* __restrict__ gptr = &grad[((size_t)s * r + i) * Kpad];
        const int* __restrict__ pptr = P.data();
        const int mult = -2 * si;
        for (int k = 0; k < Kpad; ++k) {
            int64_t dp = (int64_t)mult * gptr[k];
            dt += dp * (2LL * pptr[k] + dp);
        }
        d_true = dt;

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

    // Evaluate delta of same-size swap (s, i, j) with seq[s][i] != seq[s][j]
    inline void eval_swap(int s, int i, int j, int64_t& d_true, int64_t& d_alt) const {
        int si = seq[s][i];
        int sj = seq[s][j]; // == -si
        const int* __restrict__ gi = &grad[((size_t)s * r + i) * Kpad];
        const int* __restrict__ gj = &grad[((size_t)s * r + j) * Kpad];
        const int* __restrict__ mij = &prob->MT[((size_t)i * r + j) * Kpad];
        const int* __restrict__ pptr = P.data();
        const int m_i = -2 * si;
        const int m_j = -2 * sj;
        int64_t dt = 0;
        for (int k = 0; k < Kpad; ++k) {
            int64_t dp = (int64_t)m_i * gi[k] + (int64_t)m_j * gj[k] - 4LL * mij[k];
            dt += dp * (2LL * pptr[k] + dp);
        }
        d_true = dt;

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

    inline void apply_flip(int s, int i, int64_t d_true, int64_t d_alt) {
        apply_flip_core(s, i);
        E_true += d_true;
        E_alt += d_alt;
        E_search = E_true + E_alt;
    }

    inline void apply_swap(int s, int i, int j, int64_t d_true, int64_t d_alt) {
        apply_flip_core(s, i);
        apply_flip_core(s, j);
        E_true += d_true;
        E_alt += d_alt;
        E_search = E_true + E_alt;
    }
};

void record_if_best(int tid, const char* tag, const OrbitProblem& prob, const SearchState& st,
                    const std::chrono::steady_clock::time_point& t0, const std::string& out_path) {
    if (prob.use_alt && st.S_alt == prob.n) {
        int64_t cur_m = g_best_manifold_E.load(std::memory_order_relaxed);
        while (st.E_true < cur_m &&
               !g_best_manifold_E.compare_exchange_weak(cur_m, st.E_true, std::memory_order_relaxed)) {}
    }
    int64_t cur_best = g_best_true_E.load(std::memory_order_relaxed);
    if (st.E_true < cur_best) {
        std::lock_guard<std::mutex> lk(g_io_mtx);
        if (st.E_true < g_best_true_E.load(std::memory_order_relaxed)) {
            g_best_true_E.store(st.E_true, std::memory_order_relaxed);
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "[t=" << (int)el << "s, T" << tid << " " << tag << "] new best E=" << st.E_true
                      << " (E_alt=" << st.E_alt << ", S_alt=" << st.S_alt << "/" << prob.n << ")" << std::endl;
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

// Fast Tabu polish on a SearchState (returns true if E_search improved)
bool tabu_polish(int tid, const char* tag, SearchState& st, int max_steps, int max_no_imp,
                 std::mt19937_64& rng, const std::chrono::steady_clock::time_point& t0,
                 double max_seconds, const std::string& out_path) {
    const OrbitProblem& prob = *st.prob;
    int r = prob.r;
    std::vector<uint64_t> tabu((size_t)4 * r, 0);
    int tabu_tenure = 5 + (int)(rng() % 10);

    int64_t start_E_search = st.E_search;
    int64_t best_E_search = st.E_search;
    int64_t best_E_true = st.E_true;
    std::vector<int> best_seq[4];
    for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];

    int no_imp = 0;
    for (int step = 1; step <= max_steps && no_imp < max_no_imp; ++step) {
        if ((step & 255) == 0) {
            g_steps.fetch_add(256, std::memory_order_relaxed);
            if (g_found.load(std::memory_order_relaxed)) return false;
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (el >= max_seconds) break;
        }

        int best_type = -1, best_s = -1, best_i = -1, best_j = -1;
        int64_t best_d_search = INT64_MAX;
        int64_t best_d_true = 0, best_d_alt = 0;
        int n_ties = 0;

        for (int s = 0; s < 4; ++s) {
            for (int i = 0; i < r; ++i) {
                int64_t dt, da;
                st.eval_flip(s, i, dt, da);
                int64_t ds = dt + da;
                bool is_tabu = (tabu[s * r + i] > (uint64_t)step) &&
                               (st.E_search + ds >= best_E_search) &&
                               (st.E_true + dt >= best_E_true);
                if (!is_tabu) {
                    if (ds < best_d_search) {
                        best_d_search = ds; best_d_true = dt; best_d_alt = da;
                        best_type = 0; best_s = s; best_i = i; n_ties = 1;
                    } else if (ds == best_d_search) {
                        if ((int)(rng() % (++n_ties)) == 0) {
                            best_d_true = dt; best_d_alt = da;
                            best_type = 0; best_s = s; best_i = i;
                        }
                    }
                }
            }
        }

        for (int s = 0; s < 4; ++s) {
            for (const auto& pr : prob.same_size_pairs) {
                int i = pr.first, j = pr.second;
                if (st.seq[s][i] == st.seq[s][j]) continue;
                int64_t dt, da;
                st.eval_swap(s, i, j, dt, da);
                int64_t ds = dt + da;
                bool is_tabu = (tabu[s * r + i] > (uint64_t)step || tabu[s * r + j] > (uint64_t)step) &&
                               (st.E_search + ds >= best_E_search) &&
                               (st.E_true + dt >= best_E_true);
                if (!is_tabu) {
                    if (ds < best_d_search) {
                        best_d_search = ds; best_d_true = dt; best_d_alt = da;
                        best_type = 1; best_s = s; best_i = i; best_j = j; n_ties = 1;
                    } else if (ds == best_d_search) {
                        if ((int)(rng() % (++n_ties)) == 0) {
                            best_d_true = dt; best_d_alt = da;
                            best_type = 1; best_s = s; best_i = i; best_j = j;
                        }
                    }
                }
            }
        }

        if (best_type == -1) break;

        if (best_type == 0) {
            st.apply_flip(best_s, best_i, best_d_true, best_d_alt);
            tabu[best_s * r + best_i] = step + tabu_tenure + (rng() % 5);
        } else {
            st.apply_swap(best_s, best_i, best_j, best_d_true, best_d_alt);
            tabu[best_s * r + best_i] = step + tabu_tenure + (rng() % 5);
            tabu[best_s * r + best_j] = step + tabu_tenure + (rng() % 5);
        }

        if (st.E_true < best_E_true) {
            best_E_true = st.E_true;
            record_if_best(tid, tag, prob, st, t0, out_path);
            if (g_found.load(std::memory_order_relaxed)) return true;
        }
        if (st.E_search < best_E_search) {
            best_E_search = st.E_search;
            for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];
            no_imp = 0;
        } else {
            ++no_imp;
        }
    }

    if (best_E_search < st.E_search) {
        for (int s = 0; s < 4; ++s) st.seq[s] = best_seq[s];
        st.recompute();
    }
    return best_E_search < start_E_search;
}

// Worker A: Parallel Tempering (8 replicas) + Periodic Tabu Polish
void worker_pt(int tid, const OrbitProblem& prob, double max_seconds,
               const std::string& out_path, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto t0 = std::chrono::steady_clock::now();

    const int R = 10;
    std::vector<double> temps(R);
    // Auto-scale T_max to orbit energy scale (~max_orb_size^2 * K)
    double scale = (double)prob.max_orb_size * (double)prob.max_orb_size * (double)prob.K;
    double t_min = std::max(12.0, 0.0005 * scale);
    double t_max = std::max(3000.0, 0.8 * scale);
    for (int i = 0; i < R; ++i) {
        temps[i] = t_min * std::pow(t_max / t_min, (double)i / (R - 1));
    }

    std::vector<SearchState> reps;
    reps.reserve(R);
    for (int i = 0; i < R; ++i) {
        reps.emplace_back(prob);
        reps.back().randomize(rng);
        // Quick initial descent for cold replicas
        if (i < 2) {
            tabu_polish(tid, "PT-init", reps.back(), 150, 40, rng, t0, max_seconds, out_path);
            record_if_best(tid, "PT-init", prob, reps.back(), t0, out_path);
            if (g_found.load(std::memory_order_relaxed)) return;
        }
    }

    const int SWEEP_STEPS = 512;
    const int n_pairs = (int)prob.same_size_pairs.size();
    const int n_same_alt_pairs = (int)prob.same_size_same_alt_pairs.size();
    uint64_t round = 0;
    int64_t last_polished_E = INT64_MAX;
    int rounds_since_polish = 0;

    while (!g_found.load(std::memory_order_relaxed)) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= max_seconds) break;

        for (int rep = 0; rep < R; ++rep) {
            SearchState& st = reps[rep];
            double invT = 1.0 / temps[rep];
            // Cold replicas favor structure-preserving swaps; hot replicas use more flips
            int flip_pct = 15 + (rep * 30) / (R - 1); // 15% .. 45%

            for (int step = 0; step < SWEEP_STEPS; ++step) {
                int roll = (int)(rng() % 100);
                if (roll < flip_pct || n_pairs == 0) {
                    int s = (int)(rng() & 3);
                    int i = (int)(rng() % prob.r);
                    int64_t dt, da;
                    st.eval_flip(s, i, dt, da);
                    int64_t ds = dt + da;
                    if (ds <= 0 || u01(rng) < std::exp(-(double)ds * invT)) {
                        st.apply_flip(s, i, dt, da);
                        if (st.E_true < g_best_true_E.load(std::memory_order_relaxed)) {
                            record_if_best(tid, "PT", prob, st, t0, out_path);
                            if (g_found.load(std::memory_order_relaxed)) return;
                        }
                    }
                } else {
                    int s = (int)(rng() & 3);
                    std::pair<int, int> pr;
                    if (n_same_alt_pairs > 0 && roll < flip_pct + (100 - flip_pct) / 2) {
                        pr = prob.same_size_same_alt_pairs[rng() % n_same_alt_pairs];
                    } else {
                        pr = prob.same_size_pairs[rng() % n_pairs];
                    }
                    int i = pr.first, j = pr.second;
                    if (st.seq[s][i] == st.seq[s][j]) continue;
                    int64_t dt, da;
                    st.eval_swap(s, i, j, dt, da);
                    int64_t ds = dt + da;
                    if (ds <= 0 || u01(rng) < std::exp(-(double)ds * invT)) {
                        st.apply_swap(s, i, j, dt, da);
                        if (st.E_true < g_best_true_E.load(std::memory_order_relaxed)) {
                            record_if_best(tid, "PT", prob, st, t0, out_path);
                            if (g_found.load(std::memory_order_relaxed)) return;
                        }
                    }
                }
            }
        }
        g_steps.fetch_add((uint64_t)R * SWEEP_STEPS, std::memory_order_relaxed);

        // Replica exchange (alternating even/odd pairs)
        int start_idx = (int)(round & 1);
        for (int i = start_idx; i + 1 < R; i += 2) {
            double beta_diff = (1.0 / temps[i]) - (1.0 / temps[i + 1]);
            double e_diff = (double)(reps[i].E_search - reps[i + 1].E_search);
            double log_acc = beta_diff * e_diff;
            if (log_acc >= 0.0 || u01(rng) < std::exp(log_acc)) {
                std::swap(reps[i], reps[i + 1]);
            }
        }

        // Periodic Tabu polish on coldest replica
        ++rounds_since_polish;
        if (rounds_since_polish >= 100 || reps[0].E_search < last_polished_E) {
            rounds_since_polish = 0;
            tabu_polish(tid, "PT-pol", reps[0], 600, 180, rng, t0, max_seconds, out_path);
            last_polished_E = reps[0].E_search;
            record_if_best(tid, "PT-pol", prob, reps[0], t0, out_path);
            if (g_found.load(std::memory_order_relaxed)) return;
        }
        ++round;
    }
}

// Worker B: 2-Compression Tabu + ILS (with smart parity/size-preserving kicks)
void worker_tabu_ils(int tid, const OrbitProblem& prob, double max_seconds,
                     const std::string& out_path, uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto t0 = std::chrono::steady_clock::now();

    SearchState st(prob);
    std::vector<int> best_seq[4];
    for (int s = 0; s < 4; ++s) best_seq[s].resize(prob.r);

    while (!g_found.load(std::memory_order_relaxed)) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= max_seconds) break;

        st.randomize(rng);
        int64_t best_E_search = st.E_search;
        for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];

        int kicks_without_improve = 0;
        const int max_kicks = 25;

        while (kicks_without_improve < max_kicks && !g_found.load(std::memory_order_relaxed)) {
            double el0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (el0 >= max_seconds) return;

            int max_no_imp = 2500 + (int)(rng() % 2500);
            tabu_polish(tid, "ILS", st, 25000, max_no_imp, rng, t0, max_seconds, out_path);
            if (g_found.load(std::memory_order_relaxed)) return;

            if (st.E_search < best_E_search) {
                best_E_search = st.E_search;
                for (int s = 0; s < 4; ++s) best_seq[s] = st.seq[s];
                kicks_without_improve = 0;
            } else {
                ++kicks_without_improve;
            }

            // ILS Kick from best_seq: mix of same-parity swaps (stay on 2-comp manifold)
            // and opposite-parity swaps / flips (jump to new 4-squares decomposition)
            for (int s = 0; s < 4; ++s) st.seq[s] = best_seq[s];
            int n_kicks = 2 + (int)(rng() % 5);
            for (int kc = 0; kc < n_kicks; ++kc) {
                int s = (int)(rng() & 3);
                int mode = (int)(rng() % 6);
                if (mode < 3 && !prob.same_size_same_alt_pairs.empty()) {
                    const auto& pr = prob.same_size_same_alt_pairs[rng() % prob.same_size_same_alt_pairs.size()];
                    st.seq[s][pr.first] = -st.seq[s][pr.first];
                    st.seq[s][pr.second] = -st.seq[s][pr.second];
                } else if (mode < 5 && !prob.same_size_pairs.empty()) {
                    const auto& pr = prob.same_size_pairs[rng() % prob.same_size_pairs.size()];
                    st.seq[s][pr.first] = -st.seq[s][pr.first];
                    st.seq[s][pr.second] = -st.seq[s][pr.second];
                } else {
                    int i = (int)(rng() % prob.r);
                    st.seq[s][i] = -st.seq[s][i];
                }
            }
            st.recompute();
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <m> <g> <b> <seconds> <out.json> [threads=2] [s0 s1 s2 s3] [--w_alt=1]\n";
        return 1;
    }
    int m = std::stoi(argv[1]);
    int g = std::stoi(argv[2]);
    int b = std::stoi(argv[3]);
    double secs = std::stod(argv[4]);
    std::string out_path = argv[5];
    int threads = 2;
    int ts[4] = {2, 0, 0, 0};
    int talt[4] = {0, 0, 0, 0};
    bool use_talt = false;
    int64_t w_alt = 1;

    std::vector<std::string> pos;
    for (int i = 6; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--w_alt=", 0) == 0) {
            w_alt = std::stoll(a.substr(8));
        } else if (a.rfind("--target_alt=", 0) == 0) {
            std::string vals = a.substr(13);
            int idx = 0;
            size_t p = 0;
            while (idx < 4 && p < vals.size()) {
                size_t q = vals.find(',', p);
                if (q == std::string::npos) q = vals.size();
                talt[idx++] = std::stoi(vals.substr(p, q - p));
                p = q + 1;
            }
            if (idx == 4) use_talt = true;
        } else {
            pos.push_back(a);
        }
    }
    if (!pos.empty()) threads = std::stoi(pos[0]);
    if (pos.size() >= 5) {
        for (int i = 0; i < 4; ++i) ts[i] = std::stoi(pos[1 + i]);
    }

    OrbitProblem prob = build_problem(m, g, b, ts, w_alt, use_talt, talt);
    std::cout << "Hadamard v2 n=" << prob.n << " (m=" << m << ", g=" << g << ", b=" << b
              << "): r=" << prob.r << " orbits (max_sz=" << prob.max_orb_size
              << "), K=" << prob.K << " (pad=" << prob.Kpad
              << "), pairs=" << prob.same_size_pairs.size()
              << " (same_alt=" << prob.same_size_same_alt_pairs.size()
              << "), w_alt=" << prob.w_alt;
    if (prob.use_target_alt) {
        std::cout << ", target_alt=(" << talt[0] << "," << talt[1] << "," << talt[2] << "," << talt[3] << ")";
    }
    std::cout << ", threads=" << threads << std::endl;

    std::vector<std::thread> pool;
    uint64_t base_seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (int t = 0; t < threads; ++t) {
        if ((t & 1) == 0) {
            pool.emplace_back(worker_pt, t, std::cref(prob), secs, out_path, base_seed + 1000003ULL * t);
        } else {
            pool.emplace_back(worker_tabu_ils, t, std::cref(prob), secs, out_path, base_seed + 1000003ULL * t);
        }
    }
    for (auto& th : pool) th.join();

    std::cout << "Total steps: " << g_steps.load()
              << ", best_E=" << g_best_true_E.load()
              << ", best_manifold_E=" << (g_best_manifold_E.load() == INT64_MAX ? -1 : g_best_manifold_E.load())
              << ", found=" << g_found.load() << std::endl;
    return g_found.load() ? 0 : 2;
}

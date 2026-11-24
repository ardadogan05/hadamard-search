#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct OrbitProblem {
    int n, m, g, b;
    int r, K;
    std::vector<std::vector<int>> orbits;
    std::vector<int> orb_size;
    std::vector<int> elem_to_orb;
    std::vector<int> shift_reps;
    std::vector<int> shift_weights;
    std::vector<int> M; // M[k * r * r + i * r + j]
    std::vector<int> diag_const;
    int target_ac;
    int target_s[4];
    std::vector<std::pair<int, int>> same_size_pairs;
};

OrbitProblem build_problem(int m, int g, int b, const int ts[4]) {
    OrbitProblem prob;
    prob.m = m;
    prob.g = g;
    prob.b = b;
    prob.n = 4 * m + b;
    prob.target_ac = -b;
    for (int s = 0; s < 4; ++s) prob.target_s[s] = ts[s];

    prob.elem_to_orb.assign(m, -1);
    for (int x = 0; x < m; ++x) {
        if (prob.elem_to_orb[x] == -1) {
            int oid = (int)prob.orbits.size();
            std::vector<int> orb;
            long long cur = x;
            while (prob.elem_to_orb[cur] == -1) {
                prob.elem_to_orb[cur] = oid;
                orb.push_back((int)cur);
                cur = (cur * g) % m;
            }
            prob.orb_size.push_back((int)orb.size());
            prob.orbits.push_back(std::move(orb));
        }
    }
    prob.r = (int)prob.orbits.size();

    for (int i = 0; i < prob.r; ++i)
        for (int j = i + 1; j < prob.r; ++j)
            if (prob.orb_size[i] == prob.orb_size[j])
                prob.same_size_pairs.push_back({i, j});

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

    prob.M.assign((size_t)prob.K * prob.r * prob.r, 0);
    prob.diag_const.assign(prob.K, 0);

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
                    prob.M[((size_t)k * prob.r + i) * prob.r + j] =
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

void worker(int tid, const OrbitProblem& prob, double max_seconds, const std::string& out_path, uint64_t seed) {
    int r = prob.r;
    int K = prob.K;
    std::mt19937_64 rng(seed);
    auto t0 = std::chrono::steady_clock::now();

    std::vector<int> MT((size_t)r * r * K);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < r; ++j)
            for (int k = 0; k < K; ++k)
                MT[((size_t)i * r + j) * K + k] = prob.M[((size_t)k * r + i) * r + j];

    std::vector<int> seq[4];
    for (int s = 0; s < 4; ++s) seq[s].resize(r);
    std::vector<int> P(K);
    std::vector<int> grad(4 * r * K);
    int rsum[4];
    std::vector<uint64_t> tabu(4 * r, 0);

    int64_t global_best = 1e18;
    int64_t row_weight = 4;

    auto apply_flip = [&](int s, int i) {
        int si = seq[s][i];
        seq[s][i] = -si;
        rsum[s] -= 2 * si * prob.orb_size[i];
        const int* gptr = &grad[((size_t)s * r + i) * K];
        for (int k = 0; k < K; ++k) P[k] -= 2 * si * gptr[k];
        const int* mptr_base = &MT[((size_t)i * r) * K];
        int factor = -2 * si;
        for (int j = 0; j < r; ++j) {
            if (j == i) continue;
            int* gj = &grad[((size_t)s * r + j) * K];
            const int* mij = &mptr_base[(size_t)j * K];
            for (int k = 0; k < K; ++k) gj[k] += factor * mij[k];
        }
    };

    auto init_state = [&]() {
        for (int s = 0; s < 4; ++s) {
            rsum[s] = 0;
            for (int i = 0; i < r; ++i) {
                seq[s][i] = (rng() & 1) ? 1 : -1;
                rsum[s] += seq[s][i] * prob.orb_size[i];
            }
        }
        for (int k = 0; k < K; ++k) {
            P[k] = (int)(4LL * prob.diag_const[k] - prob.target_ac);
            for (int s = 0; s < 4; ++s) {
                int64_t acc = 0;
                for (int i = 0; i < r; ++i) {
                    int64_t gval = 0;
                    const int* rowM = &MT[((size_t)i * r) * K];
                    for (int j = 0; j < r; ++j) gval += (int64_t)rowM[(size_t)j * K + k] * seq[s][j];
                    grad[((size_t)s * r + i) * K + k] = (int)gval;
                    acc += (int64_t)seq[s][i] * gval;
                }
                P[k] += (int)(acc / 2);
            }
        }
        std::fill(tabu.begin(), tabu.end(), 0);
    };

    auto calc_energy = [&]() -> int64_t {
        int64_t e = 0;
        for (int k = 0; k < K; ++k) e += (int64_t)P[k] * P[k];
        for (int s = 0; s < 4; ++s) {
            int64_t dr = rsum[s] - prob.target_s[s];
            e += row_weight * dr * dr;
        }
        return e;
    };

    std::vector<int> best_seq[4];
    for (int s = 0; s < 4; ++s) best_seq[s].resize(r);

    auto recompute_from_seq = [&]() {
        for (int s = 0; s < 4; ++s) {
            rsum[s] = 0;
            for (int i = 0; i < r; ++i) rsum[s] += seq[s][i] * prob.orb_size[i];
        }
        for (int k = 0; k < K; ++k) {
            P[k] = (int)(4LL * prob.diag_const[k] - prob.target_ac);
            for (int s = 0; s < 4; ++s) {
                int64_t acc = 0;
                for (int i = 0; i < r; ++i) {
                    int64_t gval = 0;
                    const int* rowM = &MT[((size_t)i * r) * K];
                    for (int j = 0; j < r; ++j) gval += (int64_t)rowM[(size_t)j * K + k] * seq[s][j];
                    grad[((size_t)s * r + i) * K + k] = (int)gval;
                    acc += (int64_t)seq[s][i] * gval;
                }
                P[k] += (int)(acc / 2);
            }
        }
        std::fill(tabu.begin(), tabu.end(), 0);
    };

    int restart = 0;
    while (!g_found.load(std::memory_order_relaxed)) {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= max_seconds) break;

        init_state();
        int64_t curE = calc_energy();
        int64_t bestE = curE;
        for (int s = 0; s < 4; ++s) best_seq[s] = seq[s];
        int kicks_without_improve = 0;
        int max_kicks = 30;

        while (kicks_without_improve < max_kicks && !g_found.load(std::memory_order_relaxed)) {
            double el0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (el0 >= max_seconds) return;

            uint64_t no_improve = 0;
            int tabu_tenure = 5 + (int)(rng() % 12);
            uint64_t max_no_improve = 4000 + (rng() % 4000);
            bool improved_this_kick = false;

            for (uint64_t step = 1; no_improve < max_no_improve; ++step) {
                if ((step & 1023) == 0) {
                    g_steps.fetch_add(1024, std::memory_order_relaxed);
                    if (g_found.load(std::memory_order_relaxed)) return;
                    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (el >= max_seconds) return;
                }

                int best_type = -1;
                int best_s = -1, best_i = -1, best_j = -1;
                int64_t best_delta = 1e18;
                int n_ties = 0;

                for (int s = 0; s < 4; ++s) {
                    int64_t dr_old = rsum[s] - prob.target_s[s];
                    for (int i = 0; i < r; ++i) {
                        int si = seq[s][i];
                        int64_t dr_new = dr_old - 2LL * si * prob.orb_size[i];
                        int64_t delta = row_weight * (dr_new * dr_new - dr_old * dr_old);
                        const int* gptr = &grad[((size_t)s * r + i) * K];
                        for (int k = 0; k < K; ++k) {
                            int64_t dp = -2LL * si * gptr[k];
                            delta += dp * (2LL * P[k] + dp);
                        }
                        bool is_tabu = (tabu[s * r + i] > step) && (curE + delta >= bestE);
                        if (!is_tabu) {
                            if (delta < best_delta) {
                                best_delta = delta; best_type = 0; best_s = s; best_i = i; n_ties = 1;
                            } else if (delta == best_delta) {
                                if ((int)(rng() % (++n_ties)) == 0) { best_type = 0; best_s = s; best_i = i; }
                            }
                        }
                    }
                }

                for (int s = 0; s < 4; ++s) {
                    for (const auto& pr : prob.same_size_pairs) {
                        int i = pr.first, j = pr.second;
                        int si = seq[s][i], sj = seq[s][j];
                        if (si == sj) continue;
                        bool is_tabu_pair = (tabu[s * r + i] > step || tabu[s * r + j] > step);
                        const int* gi = &grad[((size_t)s * r + i) * K];
                        const int* gj = &grad[((size_t)s * r + j) * K];
                        const int* mij = &MT[((size_t)i * r + j) * K];
                        int64_t delta = 0;
                        for (int k = 0; k < K; ++k) {
                            int64_t dp = -2LL * si * gi[k] - 2LL * sj * gj[k] - 4LL * mij[k];
                            delta += dp * (2LL * P[k] + dp);
                        }
                        if (is_tabu_pair && curE + delta >= bestE) continue;
                        if (delta < best_delta) {
                            best_delta = delta; best_type = 1; best_s = s; best_i = i; best_j = j; n_ties = 1;
                        } else if (delta == best_delta) {
                            if ((int)(rng() % (++n_ties)) == 0) {
                                best_type = 1; best_s = s; best_i = i; best_j = j;
                            }
                        }
                    }
                }

                if (best_type == -1) break;

                if (best_type == 0) {
                    apply_flip(best_s, best_i);
                    tabu[best_s * r + best_i] = step + tabu_tenure + (rng() % 5);
                } else {
                    apply_flip(best_s, best_i);
                    apply_flip(best_s, best_j);
                    tabu[best_s * r + best_i] = step + tabu_tenure + (rng() % 5);
                    tabu[best_s * r + best_j] = step + tabu_tenure + (rng() % 5);
                }

                curE += best_delta;

                if (curE < bestE) {
                    bestE = curE;
                    for (int s = 0; s < 4; ++s) best_seq[s] = seq[s];
                    no_improve = 0;
                    improved_this_kick = true;
                    if (bestE < global_best) {
                        global_best = bestE;
                        if (tid == 0) {
                            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                            std::cout << "[t=" << (int)el << "s, r=" << restart << "] new best E=" << global_best << std::endl;
                        }
                    }
                    if (bestE == 0) {
                        if (exact_verify_sequences(prob, seq)) {
                            if (!g_found.exchange(true)) {
                                save_witness(out_path, prob, seq);
                                std::cout << "FOUND EXACT VERIFIED SOLUTION! Saved to " << out_path << std::endl;
                            }
                            return;
                        }
                    }
                } else {
                    ++no_improve;
                }
            }

            if (improved_this_kick) {
                kicks_without_improve = 0;
            } else {
                ++kicks_without_improve;
            }

            // ILS Kick from best_seq: perform 2..5 random swaps/flips
            for (int s = 0; s < 4; ++s) seq[s] = best_seq[s];
            int n_kicks = 2 + (int)(rng() % 4);
            for (int kc = 0; kc < n_kicks; ++kc) {
                int s = (int)(rng() % 4);
                if (!prob.same_size_pairs.empty() && (rng() % 3 != 0)) {
                    const auto& pr = prob.same_size_pairs[rng() % prob.same_size_pairs.size()];
                    seq[s][pr.first] = -seq[s][pr.first];
                    seq[s][pr.second] = -seq[s][pr.second];
                } else {
                    int i = (int)(rng() % r);
                    seq[s][i] = -seq[s][i];
                }
            }
            recompute_from_seq();
            curE = calc_energy();
        }
        ++restart;
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <m> <g> <b> <seconds> <out.json> [threads=2] [s0 s1 s2 s3]\n";
        return 1;
    }
    int m = std::stoi(argv[1]);
    int g = std::stoi(argv[2]);
    int b = std::stoi(argv[3]);
    double secs = std::stod(argv[4]);
    std::string out_path = argv[5];
    int threads = (argc >= 7) ? std::stoi(argv[6]) : 2;
    int ts[4] = {2, 0, 0, 0};
    if (argc >= 11) {
        for (int i = 0; i < 4; ++i) ts[i] = std::stoi(argv[7 + i]);
    }

    OrbitProblem prob = build_problem(m, g, b, ts);
    std::cout << "Hadamard n=" << prob.n << " (m=" << m << ", g=" << g << ", b=" << b
              << "): r=" << prob.r << " orbits, K=" << prob.K << " eqs, pairs="
              << prob.same_size_pairs.size() << ", threads=" << threads << std::endl;

    std::vector<std::thread> pool;
    uint64_t base_seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back(worker, t, std::cref(prob), secs, out_path, base_seed + 1000003ULL * t);
    }
    for (auto& th : pool) th.join();

    std::cout << "Total steps: " << g_steps.load() << ", found=" << g_found.load() << std::endl;
    return g_found.load() ? 0 : 2;
}

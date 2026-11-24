// Fast C++20 Orbit-Space Circulant T-Matrix Solver (Tabu+GLS & Tight-Band Reheat SA)
// Usage: ./tmatrix_solver <t> <g> <m1> <m2> <m3> <m4> <secs> <out_json> [threads=1] [mode=tabu|sa]
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
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

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

    inst.MT.assign(inst.r * inst.r * inst.K_pad, 0);
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
        int* C_m = &C[m * r * K_pad];
        const int* MT_o = &inst->MT[o * r * K_pad];
        if (delta_s == 1) {
            for (int l = 0; l < r; ++l) {
                if (l == o) continue;
                int* dst = C_m + l * K_pad;
                const int* src = MT_o + l * K_pad;
                for (int k = 0; k < K; ++k) dst[k] += src[k];
            }
        } else if (delta_s == -1) {
            for (int l = 0; l < r; ++l) {
                if (l == o) continue;
                int* dst = C_m + l * K_pad;
                const int* src = MT_o + l * K_pad;
                for (int k = 0; k < K; ++k) dst[k] -= src[k];
            }
        } else if (delta_s == 2) {
            for (int l = 0; l < r; ++l) {
                if (l == o) continue;
                int* dst = C_m + l * K_pad;
                const int* src = MT_o + l * K_pad;
                for (int k = 0; k < K; ++k) dst[k] += 2 * src[k];
            }
        } else if (delta_s == -2) {
            for (int l = 0; l < r; ++l) {
                if (l == o) continue;
                int* dst = C_m + l * K_pad;
                const int* src = MT_o + l * K_pad;
                for (int k = 0; k < K; ++k) dst[k] -= 2 * src[k];
            }
        }
    }

    // Move 1: Same-matrix sign swap: mat[i]==mat[j]==a, sgn[i]==+1, sgn[j]==-1
    inline int64_t eval_move1(int i, int j, int a, const int* W = nullptr) const {
        const int* Cai = &C[(a * r + i) * K_pad];
        const int* Caj = &C[(a * r + j) * K_pad];
        const int* mij = &inst->MT[(i * r + j) * K_pad];
        int64_t dE = 0;
        if (!W) {
            for (int k = 0; k < K; ++k) {
                int dp = 2 * (Caj[k] - Cai[k] - 2 * mij[k]);
                dE += 1LL * dp * (2 * P[k] + dp);
            }
        } else {
            for (int k = 0; k < K; ++k) {
                int dp = 2 * (Caj[k] - Cai[k] - 2 * mij[k]);
                dE += 1LL * W[k] * dp * (2 * P[k] + dp);
            }
        }
        return dE;
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
    inline int64_t eval_move2(int i, int j, int a, int b, int eps, const int* W = nullptr) const {
        const int* Cai = &C[(a * r + i) * K_pad];
        const int* Cbi = &C[(b * r + i) * K_pad];
        const int* Caj = &C[(a * r + j) * K_pad];
        const int* Cbj = &C[(b * r + j) * K_pad];
        const int* mij = &inst->MT[(i * r + j) * K_pad];
        int64_t dE = 0;
        if (!W) {
            for (int k = 0; k < K; ++k) {
                int dp = eps * (Cbi[k] - Cai[k] + Caj[k] - Cbj[k]) - 2 * mij[k];
                dE += 1LL * dp * (2 * P[k] + dp);
            }
        } else {
            for (int k = 0; k < K; ++k) {
                int dp = eps * (Cbi[k] - Cai[k] + Caj[k] - Cbj[k]) - 2 * mij[k];
                dE += 1LL * W[k] * dp * (2 * P[k] + dp);
            }
        }
        return dE;
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
    inline int64_t eval_move3(int i, int j, int a, int b, int flip_b, const int* W = nullptr) const {
        const int* Cai = &C[(a * r + i) * K_pad];
        const int* Caj = &C[(a * r + j) * K_pad];
        const int* Cbi = &C[(b * r + i) * K_pad];
        const int* Cbj = &C[(b * r + j) * K_pad];
        const int* mij = &inst->MT[(i * r + j) * K_pad];
        int64_t dE = 0;
        int fb = flip_b ? -1 : 1;
        if (!W) {
            for (int k = 0; k < K; ++k) {
                int dp = fb * (Cbi[k] - Cbj[k]) + (Caj[k] - Cai[k] - 2 * mij[k]);
                dE += 1LL * dp * (2 * P[k] + dp);
            }
        } else {
            for (int k = 0; k < K; ++k) {
                int dp = fb * (Cbi[k] - Cbj[k]) + (Caj[k] - Cai[k] - 2 * mij[k]);
                dE += 1LL * W[k] * dp * (2 * P[k] + dp);
            }
        }
        return dE;
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

static void save_witness(const std::string& path, const Instance& inst, const State& st) {
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
}

struct BestMove {
    int type = 0; // 1, 2, 3
    int i = 0, j = 0, a = 0, b = 0, p = 0;
    int64_t dW = (1LL << 60);
    int64_t dE = (1LL << 60);
};

static void worker_tabu(int tid, const Instance* inst, double max_secs, const std::string& out_path) {
    auto t_start = std::chrono::steady_clock::now();
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL ^ ((uint64_t)(tid + 1) * 0xbf58476d1ce4e5b9ULL));

    State st(inst);
    st.init_random(rng);
    std::vector<int> tabu(inst->r, 0);
    std::vector<int> W(inst->K, 1);

    int64_t local_best = st.E;
    int iter = 0;
    int stall = 0;
    int tenure = std::max(5, inst->r / 4);
    int kick_limit = std::max(800, inst->r * 40);

    std::vector<int> pos[4], neg[4];

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

        auto consider = [&](int type, int i, int j, int a, int b, int p, int64_t dW, int64_t dE) {
            bool is_tabu = (tabu[i] > iter || tabu[j] > iter);
            if (is_tabu && (st.E + dE >= local_best)) return;
            if (dW < best.dW || (dW == best.dW && (dE < best.dE || (dE == best.dE && (rng() % (++tie_cnt + 1) == 0))))) {
                if (dW < best.dW || dE < best.dE) tie_cnt = 0;
                best = {type, i, j, a, b, p, dW, dE};
            }
        };

        // Move 1: same-matrix sign swaps
        for (int a = 0; a < 4; ++a) {
            for (int i : pos[a]) {
                for (int j : neg[a]) {
                    int64_t dW = st.eval_move1(i, j, a, W.data());
                    int64_t dE = (stall == 0) ? dW : st.eval_move1(i, j, a, nullptr);
                    consider(1, i, j, a, a, 0, dW, dE);
                }
            }
        }
        // Move 2: cross-matrix same-sign swaps
        for (int a = 0; a < 4; ++a) {
            for (int b = a + 1; b < 4; ++b) {
                for (int i : pos[a]) {
                    for (int j : pos[b]) {
                        int64_t dW = st.eval_move2(i, j, a, b, 1, W.data());
                        int64_t dE = (stall == 0) ? dW : st.eval_move2(i, j, a, b, 1, nullptr);
                        consider(2, i, j, a, b, 1, dW, dE);
                    }
                }
                for (int i : neg[a]) {
                    for (int j : neg[b]) {
                        int64_t dW = st.eval_move2(i, j, a, b, -1, W.data());
                        int64_t dE = (stall == 0) ? dW : st.eval_move2(i, j, a, b, -1, nullptr);
                        consider(2, i, j, a, b, -1, dW, dE);
                    }
                }
            }
        }
        // Move 3: zero-sum pair transfers (only evaluate when best isn't already huge improvement or always for r<=45)
        for (int a = 0; a < 4; ++a) {
            for (int i : pos[a]) {
                for (int j : neg[a]) {
                    for (int b = 0; b < 4; ++b) {
                        if (b == a) continue;
                        for (int fb = 0; fb < 2; ++fb) {
                            int64_t dW = st.eval_move3(i, j, a, b, fb, W.data());
                            int64_t dE = (stall == 0) ? dW : st.eval_move3(i, j, a, b, fb, nullptr);
                            consider(3, i, j, a, b, fb, dW, dE);
                        }
                    }
                }
            }
        }

        if (best.type == 1) st.apply_move1(best.i, best.j, best.a);
        else if (best.type == 2) st.apply_move2(best.i, best.j, best.a, best.b, best.p);
        else if (best.type == 3) st.apply_move3(best.i, best.j, best.a, best.b, best.p);

        int t_val = tenure + (int)(rng() % 5);
        tabu[best.i] = iter + t_val;
        tabu[best.j] = iter + t_val;

        if (st.E < local_best) {
            local_best = st.E;
            stall = 0;
            int64_t prev = g_best_E.load(std::memory_order_relaxed);
            while (st.E < prev && !g_best_E.compare_exchange_weak(prev, st.E)) {}
            if (st.E < prev) {
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                std::lock_guard<std::mutex> lk(g_io_mtx);
                std::cout << "[T" << tid << " " << el << "s iter=" << iter << "] new best E=" << st.E << std::endl;
            }
            if (st.E == 0) {
                st.recompute();
                if (st.E == 0) {
                    if (!g_found.exchange(true)) {
                        std::lock_guard<std::mutex> lk(g_io_mtx);
                        save_witness(out_path, *inst, st);
                        std::cout << "*** FOUND EXACT VERIFIED T-MATRIX SOLUTION (E=0) SAVED TO " << out_path << " ***" << std::endl;
                    }
                    return;
                }
            }
        } else {
            ++stall;
            // Guided Local Search weight bump on stalled worst shift
            if (stall % 150 == 0) {
                int worst_k = 0;
                double worst_util = -1.0;
                for (int k = 0; k < inst->K; ++k) {
                    double u = (double)(1LL * st.P[k] * st.P[k]) / (double)W[k];
                    if (u > worst_util) { worst_util = u; worst_k = k; }
                }
                W[worst_k] += 1;
            }
            if (stall >= kick_limit) {
                // Reset GLS weights and apply random restart or heavy kick
                std::fill(W.begin(), W.end(), 1);
                if (rng() % 3 == 0) {
                    st.init_random(rng);
                    local_best = st.E;
                } else {
                    // Apply 8 random moves
                    for (int kk = 0; kk < 8; ++kk) {
                        for (int a = 0; a < 4; ++a) { pos[a].clear(); neg[a].clear(); }
                        for (int i = 1; i < inst->r; ++i) {
                            if (st.sgn[i] == 1) pos[st.mat[i]].push_back(i);
                            else neg[st.mat[i]].push_back(i);
                        }
                        int a = (int)(rng() & 3ULL);
                        int b = (a + 1 + (int)(rng() % 3)) & 3;
                        if (!pos[a].empty() && !pos[b].empty()) {
                            st.apply_move2(pos[a][rng() % pos[a].size()], pos[b][rng() % pos[b].size()], a, b, 1);
                        } else if (!pos[a].empty() && !neg[a].empty()) {
                            st.apply_move1(pos[a][rng() % pos[a].size()], neg[a][rng() % neg[a].size()], a);
                        }
                    }
                    local_best = st.E;
                }
                std::fill(tabu.begin(), tabu.end(), 0);
                stall = 0;
            }
        }
    }
}

static void worker_sa(int tid, const Instance* inst, double max_secs, const std::string& out_path) {
    auto t_start = std::chrono::steady_clock::now();
    std::mt19937_64 rng(0x517cc1b727220a95ULL ^ ((uint64_t)(tid + 1) * 0x94d049bb133111ebULL));
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    State st(inst);
    st.init_random(rng);

    // Maintain live lists pos[4], neg[4] and index-in-list for O(1) random move sampling!
    int pos_list[4][128], neg_list[4][128];
    int pos_sz[4], neg_sz[4];
    int idx_in_list[128];

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
    double T_hi = std::max(8.0, 0.8 * inst->K);
    double T_lo = 0.7;
    const uint64_t cycle_len = 800000ULL;
    double alpha = std::pow(T_lo / T_hi, 1.0 / (double)cycle_len);
    double T = T_hi;
    uint64_t step = 0;
    int64_t local_best = st.E;

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
            local_best = st.E;
            int64_t prev = g_best_E.load(std::memory_order_relaxed);
            while (st.E < prev && !g_best_E.compare_exchange_weak(prev, st.E)) {}
            if (st.E < prev) {
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                std::lock_guard<std::mutex> lk(g_io_mtx);
                std::cout << "[SA-T" << tid << " " << el << "s step=" << step << " T=" << T << "] new best E=" << st.E << std::endl;
            }
            if (st.E == 0) {
                st.recompute();
                if (st.E == 0) {
                    if (!g_found.exchange(true)) {
                        std::lock_guard<std::mutex> lk(g_io_mtx);
                        save_witness(out_path, *inst, st);
                        std::cout << "*** FOUND EXACT VERIFIED T-MATRIX SOLUTION (E=0) SAVED TO " << out_path << " ***" << std::endl;
                    }
                    return;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 9) {
        std::cerr << "Usage: ./tmatrix_solver <t> <g> <m1> <m2> <m3> <m4> <secs> <out_json> [threads=1] [mode=sa|tabu|hybrid]\n";
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

    Instance inst = build_instance(t, g, m1, m2, m3, m4);
    self_check_deltas(inst);
    std::cout << "tmatrix_solver: t=" << t << " g=" << g << " |U|=" << inst.h
              << " r=" << inst.r << " K=" << inst.K
              << " m=(" << m1 << "," << m2 << "," << m3 << "," << m4 << ")"
              << " threads=" << threads << " secs=" << secs << " mode=" << mode << std::endl;

    std::vector<std::thread> pool;
    for (int tid = 0; tid < threads; ++tid) {
        if (mode == "tabu" || (mode == "hybrid" && tid % 2 == 1))
            pool.emplace_back(worker_tabu, tid, &inst, secs, out_path);
        else
            pool.emplace_back(worker_sa, tid, &inst, secs, out_path);
    }
    for (auto& th : pool) th.join();

    if (!g_found.load()) {
        std::cout << "Finished " << secs << "s without E=0. Global best E=" << g_best_E.load() << std::endl;
    }
    return 0;
}

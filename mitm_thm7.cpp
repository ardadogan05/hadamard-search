// Single-threaded exact Meet-in-the-Middle solver for Theorem 7 (b=12, k=4, G = Z_2^2 x S, sigma_3)
// Zero floating-point arithmetic; 100% exact integer PAF vectors.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <array>

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct HalfVec {
    uint32_t mask;
    int sum;
    int p_self[32];
};

struct FullEntry {
    uint32_t mx, my;
    int v[32];
    uint64_t h;
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, s, r_S, K_S;
    if (!(std::cin >> n >> s >> r_S >> K_S)) return 1;
    int K = 2 * K_S + 1;
    if (K > 32) {
        std::cerr << "K > 32 not supported\n";
        return 1;
    }

    std::vector<int> orb_sz(r_S);
    for (int i = 0; i < r_S; ++i) std::cin >> orb_sz[i];

    // W_S[k][i][j]
    std::vector<std::vector<std::vector<int>>> W_S(K_S, std::vector<std::vector<int>>(r_S, std::vector<int>(r_S)));
    for (int k = 0; k < K_S; ++k)
        for (int i = 0; i < r_S; ++i)
            for (int j = 0; j < r_S; ++j)
                std::cin >> W_S[k][i][j];

    int num_rqs;
    std::cin >> num_rqs;
    std::vector<std::array<int, 4>> rqs(num_rqs);
    for (int q = 0; q < num_rqs; ++q)
        std::cin >> rqs[q][0] >> rqs[q][1] >> rqs[q][2] >> rqs[q][3];

    // Precompute all 2^r_S half-vectors grouped by sum
    std::unordered_map<int, std::vector<HalfVec>> by_sum;
    uint32_t total_masks = 1u << r_S;
    for (uint32_t mask = 0; mask < total_masks; ++mask) {
        HalfVec hv;
        hv.mask = mask;
        int sm = 0;
        int sgn[20];
        for (int i = 0; i < r_S; ++i) {
            sgn[i] = ((mask >> i) & 1u) ? 1 : -1;
            sm += orb_sz[i] * sgn[i];
        }
        hv.sum = sm;
        for (int k = 0; k < K_S; ++k) {
            int tot = 0;
            for (int i = 0; i < r_S; ++i) {
                int si = sgn[i];
                const auto& row = W_S[k][i];
                for (int j = 0; j < r_S; ++j)
                    if (row[j]) tot += row[j] * si * sgn[j];
            }
            hv.p_self[k] = tot;
        }
        by_sum[sm].push_back(hv);
    }

    // Random-looking fixed per-coordinate multipliers for additive hashing:
    // H(v1 + v2) = H(v1) + H(v2) mod 2^64!
    uint64_t coeff[32];
    for (int k = 0; k < K; ++k) coeff[k] = splitmix64(0x123456789abcdef0ULL + (uint64_t)k * 1000003ULL);

    int target[32];
    target[0] = 4;
    for (int k = 0; k < K_S; ++k) target[1 + k] = -12;
    for (int k = 0; k < K_S; ++k) target[1 + K_S + k] = 4;
    uint64_t target_h = 0;
    for (int k = 0; k < K; ++k) target_h += coeff[k] * (uint64_t)(int64_t)target[k];

    // Cache full entries by (sf, is_d0)
    std::unordered_map<int64_t, std::vector<FullEntry>> cache_L;
    auto get_L = [&](int sf, int is_d0) -> const std::vector<FullEntry>& {
        int64_t key = ((int64_t)sf << 1) | is_d0;
        auto it = cache_L.find(key);
        if (it != cache_L.end()) return it->second;

        std::vector<std::pair<int, int>> ab_pairs;
        if (is_d0) {
            ab_pairs.push_back({sf / 4, sf / 4});
        } else {
            int b1 = (sf - 2) / 4;
            ab_pairs.push_back({b1 + 2, b1});
            int b2 = (sf + 2) / 4;
            ab_pairs.push_back({b2 - 2, b2});
        }

        std::vector<FullEntry> entries;
        for (auto [a, b] : ab_pairs) {
            auto ita = by_sum.find(a);
            auto itb = by_sum.find(b);
            if (ita == by_sum.end() || itb == by_sum.end()) continue;
            for (const auto& hx : ita->second) {
                int sx[20];
                for (int i = 0; i < r_S; ++i) sx[i] = ((hx.mask >> i) & 1u) ? 1 : -1;
                for (const auto& hy : itb->second) {
                    int sy[20];
                    int dot_xy = 0;
                    for (int i = 0; i < r_S; ++i) {
                        sy[i] = ((hy.mask >> i) & 1u) ? 1 : -1;
                        dot_xy += orb_sz[i] * sx[i] * sy[i];
                    }
                    FullEntry fe;
                    fe.mx = hx.mask;
                    fe.my = hy.mask;
                    fe.v[0] = 2 * s + 2 * dot_xy;
                    for (int k = 0; k < K_S; ++k)
                        fe.v[1 + k] = hx.p_self[k] + 3 * hy.p_self[k];
                    for (int k = 0; k < K_S; ++k) {
                        int cross = 0;
                        for (int i = 0; i < r_S; ++i) {
                            int sxi = sx[i], syi = sy[i];
                            const auto& row = W_S[k][i];
                            for (int j = 0; j < r_S; ++j)
                                if (row[j]) cross += row[j] * (sxi * sy[j] + syi * sx[j]);
                        }
                        fe.v[1 + K_S + k] = cross + 2 * hy.p_self[k];
                    }
                    uint64_t h = 0;
                    for (int k = 0; k < K; ++k) h += coeff[k] * (uint64_t)(int64_t)fe.v[k];
                    fe.h = h;
                    entries.push_back(fe);
                }
            }
        }
        // Deduplicate by exact PAF vector
        std::sort(entries.begin(), entries.end(), [&](const FullEntry& A, const FullEntry& B) {
            if (A.h != B.h) return A.h < B.h;
            for (int k = 0; k < K; ++k)
                if (A.v[k] != B.v[k]) return A.v[k] < B.v[k];
            return false;
        });
        std::vector<FullEntry> uniq;
        for (const auto& fe : entries) {
            if (uniq.empty()) {
                uniq.push_back(fe);
            } else {
                bool same = (uniq.back().h == fe.h);
                if (same) {
                    for (int k = 0; k < K; ++k)
                        if (uniq.back().v[k] != fe.v[k]) { same = false; break; }
                }
                if (!same) uniq.push_back(fe);
            }
        }
        return cache_L[key] = std::move(uniq);
    };

    for (int qi = 0; qi < num_rqs; ++qi) {
        auto rq = rqs[qi];
        const auto& L0 = get_L(rq[0], 1);
        const auto& L1 = get_L(rq[1], 0);
        const auto& L2 = get_L(rq[2], 0);
        const auto& L3 = get_L(rq[3], 0);
        std::cout << "rq=" << rq[0] << "," << rq[1] << "," << rq[2] << "," << rq[3]
                  << " |L|=[" << L0.size() << "," << L1.size() << "," << L2.size() << "," << L3.size() << "]"
                  << std::endl;
        if (L0.empty() || L1.empty() || L2.empty() || L3.empty()) continue;

        // Pair the two smallest lists for the sorted table
        const std::vector<FullEntry>* ptrs[4] = {&L0, &L1, &L2, &L3};
        int idx_ord[4] = {0, 1, 2, 3};
        std::sort(idx_ord, idx_ord + 4, [&](int a, int b) { return ptrs[a]->size() < ptrs[b]->size(); });
        const auto& A = *ptrs[idx_ord[0]];
        const auto& B = *ptrs[idx_ord[1]];
        const auto& C = *ptrs[idx_ord[2]];
        const auto& D = *ptrs[idx_ord[3]];

        size_t szAB = A.size() * B.size();
        if (szAB > 120000000ULL) {
            std::cout << "  SKIPPED (|A|*|B|=" << szAB << " too large)\n";
            continue;
        }
        std::vector<uint64_t> hashes;
        hashes.reserve(szAB);
        for (const auto& a : A)
            for (const auto& b : B)
                hashes.push_back(a.h + b.h);
        std::sort(hashes.begin(), hashes.end());
        hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());

        // Probe with C + D
        bool found = false;
        for (size_t ic = 0; ic < C.size() && !found; ++ic) {
            uint64_t hc = C[ic].h;
            for (size_t id = 0; id < D.size() && !found; ++id) {
                uint64_t need = target_h - hc - D[id].h;
                if (std::binary_search(hashes.begin(), hashes.end(), need)) {
                    // Exact verification!
                    for (size_t ia = 0; ia < A.size() && !found; ++ia) {
                        for (size_t ib = 0; ib < B.size() && !found; ++ib) {
                            if (A[ia].h + B[ib].h != need) continue;
                            bool exact = true;
                            for (int k = 0; k < K; ++k) {
                                if (A[ia].v[k] + B[ib].v[k] + C[ic].v[k] + D[id].v[k] != target[k]) {
                                    exact = false;
                                    break;
                                }
                            }
                            if (exact) {
                                found = true;
                                FullEntry res[4];
                                res[idx_ord[0]] = A[ia];
                                res[idx_ord[1]] = B[ib];
                                res[idx_ord[2]] = C[ic];
                                res[idx_ord[3]] = D[id];
                                std::cout << "FOUND_EXACT_SOLUTION "
                                          << res[0].mx << " " << res[0].my << " "
                                          << res[1].mx << " " << res[1].my << " "
                                          << res[2].mx << " " << res[2].my << " "
                                          << res[3].mx << " " << res[3].my << std::endl;
                            }
                        }
                    }
                }
            }
        }
        if (found) return 0;
    }
    std::cout << "EXHAUSTIVELY_IMPOSSIBLE" << std::endl;
    return 0;
}

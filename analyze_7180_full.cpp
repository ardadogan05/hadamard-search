// Single-threaded exact joint (mod 46, mod 78) analyzer for n=7180, m=1794, g=29.
// Strictly 100% exact integer arithmetic (zero floating point).

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

static constexpr int N_ORD = 7180;
static constexpr int M_LEN = 1794;
static constexpr int TARGET_E78 = 7088;   // 7180 - 4*(1794/78)
static constexpr int SCALED_E = 154;      // (7088 - 8*39) / 44
static constexpr int MAX_A = 15;          // 154 / 10
static constexpr int TARGET_PAF = -92;    // -4*(1794/78)

static constexpr int WEIGHTS[10] = {1, 6, 6, 3, 6, 3, 6, 3, 2, 3};

static constexpr int TARGET_Q26 = 850;    // (7088 + 2*(-92) - 8*13) / 8
static constexpr int TARGET_M6  = 745;    // (7088 + 12*(-92) - 8*3) / 8

template <int MAX_VAL>
struct Bitset {
    static constexpr int NW = (MAX_VAL + 64) / 64;
    static constexpr uint64_t TOP_MASK =
        ((MAX_VAL + 1) % 64 == 0) ? ~0ULL : ((1ULL << ((MAX_VAL + 1) % 64)) - 1ULL);
    uint64_t w[NW] = {};

    inline void set(int bit) {
        w[bit >> 6] |= (1ULL << (bit & 63));
    }
    inline bool test(int bit) const {
        return (w[bit >> 6] >> (bit & 63)) & 1ULL;
    }
    inline bool any() const {
        uint64_t acc = 0;
        for (int i = 0; i < NW; ++i) acc |= w[i];
        return acc != 0;
    }
    inline void or_with(const Bitset& o) {
        for (int i = 0; i < NW; ++i) w[i] |= o.w[i];
    }
    inline int min_bit() const {
        for (int i = 0; i < NW; ++i) {
            if (w[i]) return (i << 6) + __builtin_ctzll(w[i]);
        }
        return MAX_VAL + 1;
    }
    inline void or_shl(const Bitset& src, int s) {
        int word_shift = s >> 6;
        int bit_shift = s & 63;
        if (bit_shift == 0) {
            for (int i = word_shift; i < NW; ++i) {
                w[i] |= src.w[i - word_shift];
            }
        } else {
            int rshift = 64 - bit_shift;
            w[word_shift] |= (src.w[0] << bit_shift);
            for (int i = word_shift + 1; i < NW; ++i) {
                w[i] |= (src.w[i - word_shift] << bit_shift) | (src.w[i - word_shift - 1] >> rshift);
            }
        }
        w[NW - 1] &= TOP_MASK;
    }
};

using Bitset155 = Bitset<SCALED_E>;
using Bitset851 = Bitset<TARGET_Q26>;
using Bitset746 = Bitset<TARGET_M6>;

template <int MAX_VAL>
static inline Bitset<MAX_VAL> convolve(const Bitset<MAX_VAL>& a, const Bitset<MAX_VAL>& b) {
    Bitset<MAX_VAL> res;
    for (int word = 0; word < Bitset<MAX_VAL>::NW; ++word) {
        uint64_t x = a.w[word];
        while (x) {
            int bit = (word << 6) + __builtin_ctzll(x);
            res.or_shl(b, bit);
            x &= x - 1;
        }
    }
    return res;
}

template <int MAX_VAL>
static inline bool sum_hits_target(const Bitset<MAX_VAL>& a, const Bitset<MAX_VAL>& b) {
    for (int word = 0; word < Bitset<MAX_VAL>::NW; ++word) {
        uint64_t x = a.w[word];
        while (x) {
            int bit = (word << 6) + __builtin_ctzll(x);
            if (b.test(MAX_VAL - bit)) return true;
            x &= x - 1;
        }
    }
    return false;
}

static inline bool sum_hits_154(const Bitset155& a, const Bitset155& b) {
    return sum_hits_target<SCALED_E>(a, b);
}

struct HalfVec {
    int8_t v[10];
    uint8_t E;
    int32_t even_paf[5]; // shifts 26, 6, 12, 2, 4
};

struct ProfileKey {
    int16_t e0, e1, e2;
    bool operator<(const ProfileKey& o) const {
        if (e0 != o.e0) return e0 < o.e0;
        if (e1 != o.e1) return e1 < o.e1;
        return e2 < o.e2;
    }
    bool operator==(const ProfileKey& o) const {
        return e0 == o.e0 && e1 == o.e1 && e2 == o.e2;
    }
};

struct ProfileData {
    ProfileKey key;
    Bitset155 emask;
    Bitset851 qmask;
    Bitset746 mmask;
    vector<HalfVec> vecs; // sorted by E
    int E_offset[SCALED_E + 2];
    vector<int32_t> uniq_even[5][SCALED_E + 1];
    vector<array<int32_t, 5>> uniq_even5[SCALED_E + 1];
    int32_t min_even[5][SCALED_E + 1];
    int32_t max_even[5][SCALED_E + 1];
};

struct Row46 {
    int16_t e[3], o[3];
    int32_t p0, p1;
    int pe_id, po_id;
    Bitset155 emask;
    Bitset851 qmask;
    Bitset746 mmask;
    int min_E;
    int min_q;
    int min_m;
    bool is_mod46_witness_row = false;
};

struct BGroup {
    int32_t p0, p1;
    int64_t key;
    Bitset155 emask;
    Bitset851 qmask;
    Bitset746 mmask;
    int min_E;
    int min_q;
    int min_m;
    vector<int> row_ids;
};

struct Term { int i, j, c; };
static vector<Term> even_mats[5];
static vector<Term> odd_mats[6];
static constexpr int EVEN_SHIFTS[5] = {26, 6, 12, 2, 4};
static constexpr int ODD_SHIFTS[6]  = {39, 13, 3, 15, 1, 5};

static void build_shift_matrices() {
    vector<vector<int>> orbits39;
    vector<bool> seen(39, false);
    for (int x = 0; x < 39; ++x) {
        if (!seen[x]) {
            vector<int> orb;
            int cur = x;
            while (!seen[cur]) {
                seen[cur] = true;
                orb.push_back(cur);
                cur = (cur * 29) % 39;
            }
            sort(orb.begin(), orb.end());
            orbits39.push_back(orb);
        }
    }
    sort(orbits39.begin(), orbits39.end());
    int orb39_map[39];
    for (int j = 0; j < 10; ++j) {
        if ((int)orbits39[j].size() != WEIGHTS[j]) {
            fprintf(stderr, "Orbit weight mismatch at %d\n", j);
            exit(1);
        }
        for (int r : orbits39[j]) orb39_map[r] = j;
    }

    for (int k = 0; k < 5; ++k) {
        int s = EVEN_SHIFTS[k];
        int mat[10][10] = {};
        for (int x = 0; x < 78; x += 2) {
            mat[orb39_map[x % 39]][orb39_map[(x + s) % 39]]++;
        }
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 10; ++j)
                if (mat[i][j]) even_mats[k].push_back({i, j, mat[i][j]});
    }
    for (int k = 0; k < 6; ++k) {
        int s = ODD_SHIFTS[k];
        int mat[10][10] = {};
        for (int x = 0; x < 78; x += 2) {
            mat[orb39_map[x % 39]][orb39_map[(x + s) % 39]]++;
        }
        for (int x = 1; x < 78; x += 2) {
            mat[orb39_map[(x + s) % 39]][orb39_map[x % 39]]++;
        }
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 10; ++j)
                if (mat[i][j]) odd_mats[k].push_back({i, j, mat[i][j]});
    }
}

static inline int32_t eval_even(const int8_t* v, int k) {
    int32_t s = 0;
    for (const auto& t : even_mats[k]) s += t.c * (int32_t)v[t.i] * (int32_t)v[t.j];
    return s;
}

static inline int32_t eval_odd(const int8_t* e, const int8_t* o, int k) {
    int32_t s = 0;
    for (const auto& t : odd_mats[k]) s += t.c * (int32_t)e[t.i] * (int32_t)o[t.j];
    return s;
}

struct Arr6Hash {
    size_t operator()(const array<int32_t, 6>& a) const noexcept {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < 6; ++i)
            h ^= (uint64_t)(uint32_t)a[i] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

struct Arr12Hash {
    size_t operator()(const array<int32_t, 12>& a) const noexcept {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (int i = 0; i < 12; ++i)
            h ^= (uint64_t)(uint32_t)a[i] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

enum KillReason {
    KILL_ENERGY = 0,
    KILL_EVEN_26 = 1,
    KILL_EVEN_6 = 2,
    KILL_EVEN_12 = 3,
    KILL_EVEN_2 = 4,
    KILL_EVEN_4 = 5,
    KILL_ODD_39 = 6,
    KILL_ODD_13 = 7,
    KILL_ODD_3 = 8,
    KILL_ODD_15 = 9,
    KILL_ODD_1 = 10,
    KILL_ODD_5 = 11,
    KILL_JOINT_EVEN = 12,
    KILL_JOINT_ALL11 = 13,
    SURVIVED_ALL11 = 14,
    INCONCLUSIVE_CAPPED = 15,
    NUM_REASONS = 16
};

static const char* REASON_NAMES[NUM_REASONS] = {
    "energy(7088)", "even_26", "even_6", "even_12", "even_2", "even_4",
    "odd_39", "odd_13", "odd_3", "odd_15", "odd_1", "odd_5",
    "joint_even", "joint_all11", "SURVIVED_ALL11", "INCONCLUSIVE_CAPPED"
};

static inline uint64_t pack_ep(int E, int32_t p) {
    return ((uint64_t)(uint32_t)E << 32) | (uint32_t)(p + 1000000);
}

// Reusable scratch buffers to avoid heap churn across millions of witnesses.
struct Scratch {
    vector<pair<int, int32_t>> p[4];
    vector<uint64_t> left_v;
    vector<const HalfVec*> hv[8];
};

static bool check_2x2_ep(
    const vector<pair<int, int32_t>>& p0,
    const vector<pair<int, int32_t>>& p1,
    const vector<pair<int, int32_t>>& p2,
    const vector<pair<int, int32_t>>& p3,
    int max_EL, int max_ER,
    vector<uint64_t>& left_v
) {
    left_v.clear();
    for (const auto& x : p0) {
        for (const auto& y : p1) {
            int e = x.first + y.first;
            if (e <= max_EL) left_v.push_back(pack_ep(e, x.second + y.second));
        }
    }
    if (left_v.empty()) return false;
    sort(left_v.begin(), left_v.end());
    left_v.erase(unique(left_v.begin(), left_v.end()), left_v.end());
    for (const auto& x : p2) {
        for (const auto& y : p3) {
            int e = x.first + y.first;
            if (e <= max_ER) {
                uint64_t need = pack_ep(SCALED_E - e, TARGET_PAF - x.second - y.second);
                if (binary_search(left_v.begin(), left_v.end(), need)) return true;
            }
        }
    }
    return false;
}

static KillReason check_witness_all11(
    const vector<ProfileData>& profiles,
    const int pid[8],
    FILE* survivor_out,
    const Row46* rows[4],
    Scratch& sc
) {
    Bitset155 pref[9], suff[9];
    pref[0].set(0);
    for (int i = 0; i < 8; ++i) pref[i + 1] = convolve(pref[i], profiles[pid[i]].emask);
    suff[8].set(0);
    for (int i = 7; i >= 0; --i) suff[i] = convolve(suff[i + 1], profiles[pid[i]].emask);

    int allowed_E[8][160];
    int allowed_cnt[8];
    int min_E[8];
    for (int i = 0; i < 8; ++i) {
        Bitset155 others = convolve(pref[i], suff[i + 1]);
        const Bitset155& em = profiles[pid[i]].emask;
        int cnt = 0;
        for (int word = 0; word < 3; ++word) {
            uint64_t x = em.w[word];
            while (x) {
                int bit = (word << 6) + __builtin_ctzll(x);
                if (others.test(SCALED_E - bit)) allowed_E[i][cnt++] = bit;
                x &= x - 1;
            }
        }
        if (cnt == 0) return KILL_ENERGY;
        allowed_cnt[i] = cnt;
        min_E[i] = allowed_E[i][0];
    }

    int min_EL = min_E[0] + min_E[1] + min_E[2] + min_E[3];
    int min_ER = min_E[4] + min_E[5] + min_E[6] + min_E[7];
    int max_EL = SCALED_E - min_ER;
    int max_ER = SCALED_E - min_EL;

    // Stage 1: Individual even shifts (26, 6, 12, 2, 4) with fast interval pre-check!
    for (int k = 0; k < 5; ++k) {
        // Fast exact interval check across the 8 halves (ignoring E correlation first: O(8*cnt))
        int32_t sum_lo = 0, sum_hi = 0;
        for (int i = 0; i < 8; ++i) {
            const auto& pd = profiles[pid[i]];
            int32_t lo = 1e9, hi = -1e9;
            for (int j = 0; j < allowed_cnt[i]; ++j) {
                int e = allowed_E[i][j];
                if (pd.min_even[k][e] < lo) lo = pd.min_even[k][e];
                if (pd.max_even[k][e] > hi) hi = pd.max_even[k][e];
            }
            sum_lo += lo;
            sum_hi += hi;
        }
        if (TARGET_PAF < sum_lo || TARGET_PAF > sum_hi)
            return (KillReason)(KILL_EVEN_26 + k);

        // Exact 4+4 MITM on (E, even_paf[k])
        int max_p[4] = {
            max_EL - min_E[2] - min_E[3],
            max_EL - min_E[0] - min_E[1],
            max_ER - min_E[6] - min_E[7],
            max_ER - min_E[4] - min_E[5]
        };
        for (int pair_i = 0; pair_i < 4; ++pair_i) {
            int a = 2 * pair_i, b = 2 * pair_i + 1;
            auto& out = sc.p[pair_i];
            out.clear();
            const auto& pa = profiles[pid[a]].uniq_even[k];
            const auto& pb = profiles[pid[b]].uniq_even[k];
            int lim = max_p[pair_i];
            for (int ia = 0; ia < allowed_cnt[a]; ++ia) {
                int ea = allowed_E[a][ia];
                for (int ib = 0; ib < allowed_cnt[b]; ++ib) {
                    int eb = allowed_E[b][ib];
                    if (ea + eb > lim) break;
                    for (int32_t va : pa[ea])
                        for (int32_t vb : pb[eb])
                            out.push_back({ea + eb, va + vb});
                }
            }
            sort(out.begin(), out.end());
            out.erase(unique(out.begin(), out.end()), out.end());
            if (out.empty()) return (KillReason)(KILL_EVEN_26 + k);
        }
        if (!check_2x2_ep(sc.p[0], sc.p[1], sc.p[2], sc.p[3], max_EL, max_ER, sc.left_v))
            return (KillReason)(KILL_EVEN_26 + k);
    }

    // Collect allowed HalfVec pointers for each of the 8 halves
    for (int i = 0; i < 8; ++i) {
        sc.hv[i].clear();
        const auto& pd = profiles[pid[i]];
        for (int j = 0; j < allowed_cnt[i]; ++j) {
            int e = allowed_E[i][j];
            for (int idx = pd.E_offset[e]; idx < pd.E_offset[e + 1]; ++idx) {
                sc.hv[i].push_back(&pd.vecs[idx]);
            }
        }
    }

    int max_row_E[4] = {
        SCALED_E - (min_E[2] + min_E[3] + min_E[4] + min_E[5] + min_E[6] + min_E[7]),
        SCALED_E - (min_E[0] + min_E[1] + min_E[4] + min_E[5] + min_E[6] + min_E[7]),
        SCALED_E - (min_E[0] + min_E[1] + min_E[2] + min_E[3] + min_E[6] + min_E[7]),
        SCALED_E - (min_E[0] + min_E[1] + min_E[2] + min_E[3] + min_E[4] + min_E[5])
    };

    // Stage 2: Individual odd shifts (39, 13, 3, 15, 1, 5) BEFORE joint_even!
    for (int k = 0; k < 6; ++k) {
        for (int r = 0; r < 4; ++r) {
            auto& out = sc.p[r];
            out.clear();
            int lim = max_row_E[r];
            for (const HalfVec* e : sc.hv[2 * r]) {
                for (const HalfVec* o : sc.hv[2 * r + 1]) {
                    int er = (int)e->E + (int)o->E;
                    if (er > lim) break;
                    out.push_back({er, eval_odd(e->v, o->v, k)});
                }
            }
            sort(out.begin(), out.end());
            out.erase(unique(out.begin(), out.end()), out.end());
            if (out.empty()) return (KillReason)(KILL_ODD_39 + k);
        }
        if (!check_2x2_ep(sc.p[0], sc.p[1], sc.p[2], sc.p[3], max_EL, max_ER, sc.left_v))
            return (KillReason)(KILL_ODD_39 + k);
    }

    // Stage 3a: Joint (E, 26, 6, 12) pre-filter via 4+4 MITM (much smaller than 6-tuple!)
    {
        auto build_pair3 = [&](int a, int b, int max_e) {
            vector<array<int32_t, 4>> out;
            const auto& pa = profiles[pid[a]].uniq_even5;
            const auto& pb = profiles[pid[b]].uniq_even5;
            for (int ia = 0; ia < allowed_cnt[a]; ++ia) {
                int ea = allowed_E[a][ia];
                for (int ib = 0; ib < allowed_cnt[b]; ++ib) {
                    int eb = allowed_E[b][ib];
                    if (ea + eb > max_e) break;
                    for (const auto& va : pa[ea])
                        for (const auto& vb : pb[eb])
                            out.push_back({ea + eb, va[0]+vb[0], va[1]+vb[1], va[2]+vb[2]});
                }
            }
            sort(out.begin(), out.end());
            out.erase(unique(out.begin(), out.end()), out.end());
            return out;
        };
        auto p01 = build_pair3(0, 1, max_EL - min_E[2] - min_E[3]);
        auto p23 = build_pair3(2, 3, max_EL - min_E[0] - min_E[1]);
        auto p45 = build_pair3(4, 5, max_ER - min_E[6] - min_E[7]);
        auto p67 = build_pair3(6, 7, max_ER - min_E[4] - min_E[5]);

        if (p01.size() * p23.size() <= 2000000ULL || p45.size() * p67.size() <= 2000000ULL) {
            bool swap_lr = (p01.size() * p23.size() > p45.size() * p67.size());
            const auto& L0 = swap_lr ? p45 : p01;
            const auto& L1 = swap_lr ? p67 : p23;
            const auto& R0 = swap_lr ? p01 : p45;
            const auto& R1 = swap_lr ? p23 : p67;
            int mL = swap_lr ? max_ER : max_EL;
            int mR = swap_lr ? max_EL : max_ER;

            struct Arr4Hash {
                size_t operator()(const array<int32_t, 4>& a) const noexcept {
                    uint64_t h = 0x9e3779b97f4a7c15ULL;
                    for (int i = 0; i < 4; ++i)
                        h ^= (uint64_t)(uint32_t)a[i] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                    return (size_t)h;
                }
            };
            unordered_set<array<int32_t, 4>, Arr4Hash> left_set;
            left_set.reserve(min<size_t>(L0.size() * L1.size() + 16, 1000000));
            for (const auto& x : L0)
                for (const auto& y : L1)
                    if (x[0] + y[0] <= mL)
                        left_set.insert({x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3]});
            bool hit = false;
            for (const auto& x : R0) {
                for (const auto& y : R1) {
                    if (x[0] + y[0] <= mR) {
                        array<int32_t, 4> need = {
                            SCALED_E - x[0] - y[0],
                            TARGET_PAF - x[1] - y[1],
                            TARGET_PAF - x[2] - y[2],
                            TARGET_PAF - x[3] - y[3]
                        };
                        if (left_set.count(need)) { hit = true; break; }
                    }
                }
                if (hit) break;
            }
            if (!hit) return KILL_JOINT_EVEN;
        }
    }

    // Stage 3b: Joint ALL 5 even shifts + energy via 4+4 MITM
    {
        auto build_pair5 = [&](int a, int b, int max_e) {
            vector<array<int32_t, 6>> out;
            const auto& pa = profiles[pid[a]].uniq_even5;
            const auto& pb = profiles[pid[b]].uniq_even5;
            for (int ia = 0; ia < allowed_cnt[a]; ++ia) {
                int ea = allowed_E[a][ia];
                for (int ib = 0; ib < allowed_cnt[b]; ++ib) {
                    int eb = allowed_E[b][ib];
                    if (ea + eb > max_e) break;
                    for (const auto& va : pa[ea])
                        for (const auto& vb : pb[eb])
                            out.push_back({ea + eb, va[0]+vb[0], va[1]+vb[1], va[2]+vb[2], va[3]+vb[3], va[4]+vb[4]});
                }
            }
            sort(out.begin(), out.end());
            out.erase(unique(out.begin(), out.end()), out.end());
            return out;
        };
        auto p01 = build_pair5(0, 1, max_EL - min_E[2] - min_E[3]);
        auto p23 = build_pair5(2, 3, max_EL - min_E[0] - min_E[1]);
        auto p45 = build_pair5(4, 5, max_ER - min_E[6] - min_E[7]);
        auto p67 = build_pair5(6, 7, max_ER - min_E[4] - min_E[5]);

        if (p01.size() * p23.size() > 2000000ULL && p45.size() * p67.size() > 2000000ULL)
            return INCONCLUSIVE_CAPPED;

        bool swap_lr = (p01.size() * p23.size() > p45.size() * p67.size());
        const auto& L0 = swap_lr ? p45 : p01;
        const auto& L1 = swap_lr ? p67 : p23;
        const auto& R0 = swap_lr ? p01 : p45;
        const auto& R1 = swap_lr ? p23 : p67;
        int mL = swap_lr ? max_ER : max_EL;
        int mR = swap_lr ? max_EL : max_ER;

        unordered_set<array<int32_t, 6>, Arr6Hash> left_set;
        left_set.reserve(min<size_t>(L0.size() * L1.size() + 16, 2000000));
        for (const auto& x : L0)
            for (const auto& y : L1)
                if (x[0] + y[0] <= mL)
                    left_set.insert({x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3], x[4]+y[4], x[5]+y[5]});
        bool hit = false;
        for (const auto& x : R0) {
            for (const auto& y : R1) {
                if (x[0] + y[0] <= mR) {
                    array<int32_t, 6> need = {
                        SCALED_E - x[0] - y[0],
                        TARGET_PAF - x[1] - y[1],
                        TARGET_PAF - x[2] - y[2],
                        TARGET_PAF - x[3] - y[3],
                        TARGET_PAF - x[4] - y[4],
                        TARGET_PAF - x[5] - y[5]
                    };
                    if (left_set.count(need)) { hit = true; break; }
                }
            }
            if (hit) break;
        }
        if (!hit) return KILL_JOINT_EVEN;
    }

    // Stage 4: FULL JOINT ALL 11 SHIFTS + ENERGY (12-tuple) via 2+2 MITM
    {
        struct RowFull {
            array<int32_t, 12> paf;
            const HalfVec* e;
            const HalfVec* o;
        };
        vector<RowFull> rf[4];
        for (int r = 0; r < 4; ++r) {
            unordered_set<array<int32_t, 12>, Arr12Hash> seen;
            for (const HalfVec* e : sc.hv[2 * r]) {
                for (const HalfVec* o : sc.hv[2 * r + 1]) {
                    int er = (int)e->E + (int)o->E;
                    if (er > max_row_E[r]) break;
                    array<int32_t, 12> p;
                    p[0] = er;
                    for (int k = 0; k < 5; ++k) p[1 + k] = e->even_paf[k] + o->even_paf[k];
                    for (int k = 0; k < 6; ++k) p[6 + k] = eval_odd(e->v, o->v, k);
                    if (seen.insert(p).second) {
                        rf[r].push_back({p, e, o});
                    }
                }
            }
        }
        size_t szL = rf[0].size() * rf[1].size();
        size_t szR = rf[2].size() * rf[3].size();
        if (szL > 10000000ULL && szR > 10000000ULL) return INCONCLUSIVE_CAPPED;

        bool swap_lr = (szL > szR);
        const auto& L0 = swap_lr ? rf[2] : rf[0];
        const auto& L1 = swap_lr ? rf[3] : rf[1];
        const auto& R0 = swap_lr ? rf[0] : rf[2];
        const auto& R1 = swap_lr ? rf[1] : rf[3];
        int mL = swap_lr ? max_ER : max_EL;
        int mR = swap_lr ? max_EL : max_ER;

        unordered_map<array<int32_t, 12>, pair<int, int>, Arr12Hash> left_map;
        left_map.reserve(min<size_t>(L0.size() * L1.size() + 16, 2000000));
        for (int i = 0; i < (int)L0.size(); ++i) {
            for (int j = 0; j < (int)L1.size(); ++j) {
                if (L0[i].paf[0] + L1[j].paf[0] <= mL) {
                    array<int32_t, 12> sum;
                    for (int k = 0; k < 12; ++k) sum[k] = L0[i].paf[k] + L1[j].paf[k];
                    left_map.emplace(sum, make_pair(i, j));
                }
            }
        }
        for (int i = 0; i < (int)R0.size(); ++i) {
            for (int j = 0; j < (int)R1.size(); ++j) {
                if (R0[i].paf[0] + R1[j].paf[0] <= mR) {
                    array<int32_t, 12> need;
                    need[0] = SCALED_E - R0[i].paf[0] - R1[j].paf[0];
                    for (int k = 1; k < 12; ++k) need[k] = TARGET_PAF - R0[i].paf[k] - R1[j].paf[k];
                    auto it = left_map.find(need);
                    if (it != left_map.end()) {
                        if (survivor_out) {
                            const RowFull* sol[4];
                            if (!swap_lr) {
                                sol[0] = &L0[it->second.first];
                                sol[1] = &L1[it->second.second];
                                sol[2] = &R0[i];
                                sol[3] = &R1[j];
                            } else {
                                sol[0] = &R0[i];
                                sol[1] = &R1[j];
                                sol[2] = &L0[it->second.first];
                                sol[3] = &L1[it->second.second];
                            }
                            fprintf(survivor_out, "WITNESS\n");
                            for (int r = 0; r < 4; ++r) {
                                fprintf(survivor_out, "MOD46 %d %d %d %d %d %d\n",
                                        rows[r]->e[0], rows[r]->e[1], rows[r]->e[2],
                                        rows[r]->o[0], rows[r]->o[1], rows[r]->o[2]);
                                fprintf(survivor_out, "MOD78_E");
                                for (int c = 0; c < 10; ++c) fprintf(survivor_out, " %d", (int)sol[r]->e->v[c]);
                                fprintf(survivor_out, "\nMOD78_O");
                                for (int c = 0; c < 10; ++c) fprintf(survivor_out, " %d", (int)sol[r]->o->v[c]);
                                fprintf(survivor_out, "\n");
                            }
                            fflush(survivor_out);
                        }
                        return SURVIVED_ALL11;
                    }
                }
            }
        }
        return KILL_JOINT_ALL11;
    }
}

static bool check_witness_even26_exact(
    const vector<ProfileData>& profiles,
    const int pid[8],
    Scratch& sc
) {
    Bitset155 pref[9], suff[9];
    pref[0].set(0);
    for (int i = 0; i < 8; ++i) pref[i + 1] = convolve(pref[i], profiles[pid[i]].emask);
    suff[8].set(0);
    for (int i = 7; i >= 0; --i) suff[i] = convolve(suff[i + 1], profiles[pid[i]].emask);

    int allowed_E[8][160];
    int allowed_cnt[8];
    int min_E[8];
    for (int i = 0; i < 8; ++i) {
        Bitset155 others = convolve(pref[i], suff[i + 1]);
        const Bitset155& em = profiles[pid[i]].emask;
        int cnt = 0;
        for (int word = 0; word < 3; ++word) {
            uint64_t x = em.w[word];
            while (x) {
                int bit = (word << 6) + __builtin_ctzll(x);
                if (others.test(SCALED_E - bit)) allowed_E[i][cnt++] = bit;
                x &= x - 1;
            }
        }
        if (cnt == 0) return false;
        allowed_cnt[i] = cnt;
        min_E[i] = allowed_E[i][0];
    }

    int32_t sum_lo = 0, sum_hi = 0;
    for (int i = 0; i < 8; ++i) {
        const auto& pd = profiles[pid[i]];
        int32_t lo = 1e9, hi = -1e9;
        for (int j = 0; j < allowed_cnt[i]; ++j) {
            int e = allowed_E[i][j];
            if (pd.min_even[0][e] < lo) lo = pd.min_even[0][e];
            if (pd.max_even[0][e] > hi) hi = pd.max_even[0][e];
        }
        sum_lo += lo;
        sum_hi += hi;
    }
    if (TARGET_PAF < sum_lo || TARGET_PAF > sum_hi) return false;

    int min_EL = min_E[0] + min_E[1] + min_E[2] + min_E[3];
    int min_ER = min_E[4] + min_E[5] + min_E[6] + min_E[7];
    int max_EL = SCALED_E - min_ER;
    int max_ER = SCALED_E - min_EL;
    int max_p[4] = {
        max_EL - min_E[2] - min_E[3],
        max_EL - min_E[0] - min_E[1],
        max_ER - min_E[6] - min_E[7],
        max_ER - min_E[4] - min_E[5]
    };
    for (int pair_i = 0; pair_i < 4; ++pair_i) {
        int a = 2 * pair_i, b = 2 * pair_i + 1;
        auto& out = sc.p[pair_i];
        out.clear();
        const auto& pa = profiles[pid[a]].uniq_even[0];
        const auto& pb = profiles[pid[b]].uniq_even[0];
        int lim = max_p[pair_i];
        for (int ia = 0; ia < allowed_cnt[a]; ++ia) {
            int ea = allowed_E[a][ia];
            for (int ib = 0; ib < allowed_cnt[b]; ++ib) {
                int eb = allowed_E[b][ib];
                if (ea + eb > lim) break;
                for (int32_t va : pa[ea])
                    for (int32_t vb : pb[eb])
                        out.push_back({ea + eb, va + vb});
            }
        }
        sort(out.begin(), out.end());
        out.erase(unique(out.begin(), out.end()), out.end());
        if (out.empty()) return false;
    }
    return check_2x2_ep(sc.p[0], sc.p[1], sc.p[2], sc.p[3], max_EL, max_ER, sc.left_v);
}

int main(int argc, char** argv) {
    bool energy_only = false;
    bool prefilter_only = false;
    bool exact_even26 = false;
    string data_path = "/tmp/data_7180.txt";
    string out_path = "/tmp/survivors_7180.txt";
    int start_quad = 0;
    int max_quads = -1;
    uint64_t witness_cap = 0;
    uint64_t per_quad_cap = 0;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--energy-only") energy_only = true;
        else if (string(argv[i]) == "--prefilter-only") prefilter_only = true;
        else if (string(argv[i]) == "--exact-even26") exact_even26 = true;
        else if (string(argv[i]) == "--data" && i + 1 < argc) data_path = argv[++i];
        else if (string(argv[i]) == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (string(argv[i]) == "--start-quad" && i + 1 < argc) start_quad = atoi(argv[++i]);
        else if (string(argv[i]) == "--max-quads" && i + 1 < argc) max_quads = atoi(argv[++i]);
        else if (string(argv[i]) == "--witness-cap" && i + 1 < argc) witness_cap = strtoull(argv[++i], nullptr, 10);
        else if (string(argv[i]) == "--per-quad-cap" && i + 1 < argc) per_quad_cap = strtoull(argv[++i], nullptr, 10);
    }
    bool need_vecs = !energy_only;

    auto t_start = chrono::steady_clock::now();
    build_shift_matrices();

    ifstream in(data_path);
    if (!in) {
        fprintf(stderr, "Cannot open %s\n", data_path.c_str());
        return 1;
    }
    int n_a2, n_a0, n_quads;
    in >> n_a2 >> n_a0 >> n_quads;

    map<ProfileKey, int> prof_map;
    vector<ProfileData> profiles;
    auto get_prof_id = [&](int16_t e0, int16_t e1, int16_t e2) -> int {
        ProfileKey k{e0, e1, e2};
        auto it = prof_map.find(k);
        if (it != prof_map.end()) return it->second;
        int id = (int)profiles.size();
        prof_map[k] = id;
        profiles.push_back(ProfileData{});
        profiles.back().key = k;
        return id;
    };

    vector<vector<Row46>> rows2(n_a2), rows0(n_a0);
    for (int i = 0; i < n_a2; ++i) {
        int cnt; in >> cnt;
        rows2[i].resize(cnt);
        for (int j = 0; j < cnt; ++j) {
            auto& r = rows2[i][j];
            in >> r.e[0] >> r.e[1] >> r.e[2] >> r.o[0] >> r.o[1] >> r.o[2] >> r.p0 >> r.p1;
            r.pe_id = get_prof_id(r.e[0], r.e[1], r.e[2]);
            r.po_id = get_prof_id(r.o[0], r.o[1], r.o[2]);
        }
    }
    for (int i = 0; i < n_a0; ++i) {
        int cnt; in >> cnt;
        rows0[i].resize(cnt);
        for (int j = 0; j < cnt; ++j) {
            auto& r = rows0[i][j];
            in >> r.e[0] >> r.e[1] >> r.e[2] >> r.o[0] >> r.o[1] >> r.o[2] >> r.p0 >> r.p1;
            r.pe_id = get_prof_id(r.e[0], r.e[1], r.e[2]);
            r.po_id = get_prof_id(r.o[0], r.o[1], r.o[2]);
        }
    }
    vector<array<int, 4>> quads(n_quads);
    for (int i = 0; i < n_quads; ++i) {
        in >> quads[i][0] >> quads[i][1] >> quads[i][2] >> quads[i][3];
    }

    int mask_weight[1024];
    vector<int> masks_by_sum[80];
    for (int m = 0; m < 1024; ++m) {
        int w = 0;
        for (int i = 0; i < 10; ++i) if ((m >> i) & 1) w += WEIGHTS[i];
        mask_weight[m] = w;
        masks_by_sum[2 * w].push_back(m);
    }

    size_t tot_vecs = 0;
    for (auto& pd : profiles) {
        const auto& ms = masks_by_sum[pd.key.e0 + 39];
        const auto& mu = masks_by_sum[pd.key.e1 + 39];
        const auto& mv = masks_by_sum[pd.key.e2 + 39];

        struct RawVec {
            int8_t v[10];
            uint8_t E;
            bool operator<(const RawVec& o) const {
                if (E != o.E) return E < o.E;
                return memcmp(v, o.v, 10) < 0;
            }
            bool operator==(const RawVec& o) const {
                return E == o.E && memcmp(v, o.v, 10) == 0;
            }
        };
        vector<RawVec> raw;
        for (int lm : mu) {
            for (int rm : mv) {
                int agr = 1023 ^ (lm ^ rm);
                int a = mask_weight[agr];
                if (a > MAX_A) continue;
                for (int bm : ms) {
                    int b = a - 2 * mask_weight[agr & (lm ^ bm)];
                    int E = 11 * a + b;
                    if (E > SCALED_E) continue;
                    pd.emask.set(E);
                    if (need_vecs) {
                        RawVec item;
                        for (int i = 0; i < 10; ++i) {
                            int s_i = 2 * ((bm >> i) & 1) - 1;
                            int t_i = ((lm >> i) & 1) + ((rm >> i) & 1) - 1;
                            item.v[i] = (int8_t)(s_i + 22 * t_i);
                        }
                        item.E = (uint8_t)E;
                        raw.push_back(item);
                    }
                }
            }
        }
        if (need_vecs) {
            sort(raw.begin(), raw.end());
            raw.erase(unique(raw.begin(), raw.end()), raw.end());
            pd.vecs.resize(raw.size());
            for (size_t i = 0; i < raw.size(); ++i) {
                memcpy(pd.vecs[i].v, raw[i].v, 10);
                pd.vecs[i].E = raw[i].E;
                for (int k = 0; k < 5; ++k) pd.vecs[i].even_paf[k] = eval_even(pd.vecs[i].v, k);
                int E_half = 39 + 44 * (int)pd.vecs[i].E;
                int Q = E_half + 2 * pd.vecs[i].even_paf[0];
                int M = E_half + 6 * pd.vecs[i].even_paf[1] + 6 * pd.vecs[i].even_paf[2];
                if (Q < 13 || ((Q - 13) & 7) != 0 || M < 3 || ((M - 3) & 7) != 0) {
                    fprintf(stderr, "INTERNAL ERROR: invalid Q=%d or M=%d\n", Q, M);
                    exit(1);
                }
                int q = (Q - 13) >> 3;
                int m = (M - 3) >> 3;
                if (q <= TARGET_Q26) pd.qmask.set(q);
                if (m <= TARGET_M6) pd.mmask.set(m);
            }
            int idx = 0;
            for (int e = 0; e <= SCALED_E; ++e) {
                pd.E_offset[e] = idx;
                while (idx < (int)pd.vecs.size() && pd.vecs[idx].E == e) ++idx;
                if (idx > pd.E_offset[e]) {
                    for (int k = 0; k < 5; ++k) {
                        auto& u = pd.uniq_even[k][e];
                        for (int j = pd.E_offset[e]; j < idx; ++j) u.push_back(pd.vecs[j].even_paf[k]);
                        sort(u.begin(), u.end());
                        u.erase(unique(u.begin(), u.end()), u.end());
                        pd.min_even[k][e] = u.front();
                        pd.max_even[k][e] = u.back();
                    }
                    auto& u5 = pd.uniq_even5[e];
                    for (int j = pd.E_offset[e]; j < idx; ++j) {
                        const auto& ep = pd.vecs[j].even_paf;
                        u5.push_back({ep[0], ep[1], ep[2], ep[3], ep[4]});
                    }
                    sort(u5.begin(), u5.end());
                    u5.erase(unique(u5.begin(), u5.end()), u5.end());
                }
            }
            if (idx != (int)pd.vecs.size()) {
                fprintf(stderr, "INTERNAL ERROR: E_offset sweep mismatch (%d != %zu)\n", idx, pd.vecs.size());
                exit(1);
            }
            pd.E_offset[SCALED_E + 1] = idx;
            tot_vecs += pd.vecs.size();
        }
    }

    auto build_groups = [&](vector<vector<Row46>>& rows_by_a) {
        vector<vector<BGroup>> groups(rows_by_a.size());
        for (size_t a = 0; a < rows_by_a.size(); ++a) {
            map<pair<int32_t, int32_t>, int> gmap;
            for (int r_id = 0; r_id < (int)rows_by_a[a].size(); ++r_id) {
                auto& r = rows_by_a[a][r_id];
                r.emask = convolve(profiles[r.pe_id].emask, profiles[r.po_id].emask);
                r.min_E = r.emask.min_bit();
                if (need_vecs) {
                    r.qmask = convolve(profiles[r.pe_id].qmask, profiles[r.po_id].qmask);
                    r.mmask = convolve(profiles[r.pe_id].mmask, profiles[r.po_id].mmask);
                    r.min_q = r.qmask.min_bit();
                    r.min_m = r.mmask.min_bit();
                } else {
                    r.min_q = 0;
                    r.min_m = 0;
                }
                auto pk = make_pair(r.p0, r.p1);
                auto it = gmap.find(pk);
                if (it == gmap.end()) {
                    int gid = (int)groups[a].size();
                    gmap[pk] = gid;
                    BGroup g;
                    g.p0 = r.p0; g.p1 = r.p1;
                    g.key = (int64_t)r.p0 * 100000LL + r.p1;
                    g.emask = r.emask;
                    g.qmask = r.qmask;
                    g.mmask = r.mmask;
                    g.min_E = r.min_E;
                    g.min_q = r.min_q;
                    g.min_m = r.min_m;
                    g.row_ids.push_back(r_id);
                    groups[a].push_back(move(g));
                } else {
                    auto& g = groups[a][it->second];
                    g.emask.or_with(r.emask);
                    if (r.min_E < g.min_E) g.min_E = r.min_E;
                    if (need_vecs) {
                        g.qmask.or_with(r.qmask);
                        g.mmask.or_with(r.mmask);
                        if (r.min_q < g.min_q) g.min_q = r.min_q;
                        if (r.min_m < g.min_m) g.min_m = r.min_m;
                    }
                    g.row_ids.push_back(r_id);
                }
            }
            sort(groups[a].begin(), groups[a].end(), [](const BGroup& x, const BGroup& y) {
                return x.p0 < y.p0;
            });
        }
        return groups;
    };

    auto g2 = build_groups(rows2);
    auto g0 = build_groups(rows0);

    auto t_prep = chrono::steady_clock::now();
    printf("Precomputed %zu profiles (vecs=%zu) in %.3fs\n",
           profiles.size(), tot_vecs,
           chrono::duration<double>(t_prep - t_start).count());
    fflush(stdout);

    FILE* survivor_out = (energy_only || prefilter_only) ? nullptr : fopen(out_path.c_str(), "w");

    uint64_t counts[NUM_REASONS] = {};
    uint64_t total_mod46 = 0;
    uint64_t total_energy_ok = 0;
    uint64_t total_q26_ok = 0;
    uint64_t total_m6_ok = 0;
    uint64_t total_exact_even26_ok = 0;
    int end_q = (max_quads > 0) ? min(n_quads, start_quad + max_quads) : n_quads;
    Scratch sc;

    bool hit_cap = false;
    int quads_exhausted = 0;
    int quads_capped = 0;
    for (int qi = start_quad; qi < end_q && !hit_cap; ++qi) {
        uint64_t quad_survivors = 0;
        bool quad_capped = false;
        int i1 = quads[qi][0], i2 = quads[qi][1], i3 = quads[qi][2], i4 = quads[qi][3];
        const auto& G1 = g2[i1];
        const auto& G2 = g0[i2];
        const auto& G3 = g0[i3];
        const auto& G4 = g0[i4];
        const auto& R1 = rows2[i1];
        const auto& R2 = rows0[i2];
        const auto& R3 = rows0[i3];
        const auto& R4 = rows0[i4];
        if (G1.empty() || G2.empty() || G3.empty() || G4.empty()) { quads_exhausted++; continue; }

        int min_p0_1 = G1[0].p0, min_p0_2 = G2[0].p0, min_p0_3 = G3[0].p0, min_p0_4 = G4[0].p0;
        if (min_p0_1 + min_p0_2 + min_p0_3 + min_p0_4 > N_ORD) { quads_exhausted++; continue; }

        int maxL = N_ORD - min_p0_3 - min_p0_4;
        int maxR = N_ORD - min_p0_1 - min_p0_2;

        struct PairItem {
            int idx1, idx2;
            Bitset155 emask;
            Bitset851 qmask;
            Bitset746 mmask;
            int min_q, min_m;
        };
        unordered_map<int64_t, vector<PairItem>> left_map;
        left_map.reserve(G1.size() * G2.size() / 2 + 16);

        for (int a = 0; a < (int)G1.size(); ++a) {
            if (G1[a].p0 + min_p0_2 > maxL) break;
            for (int b = 0; b < (int)G2.size(); ++b) {
                if (G1[a].p0 + G2[b].p0 > maxL) break;
                int64_t k = G1[a].key + G2[b].key;
                PairItem pi;
                pi.idx1 = a;
                pi.idx2 = b;
                pi.emask = convolve(G1[a].emask, G2[b].emask);
                if (need_vecs) {
                    pi.min_q = G1[a].min_q + G2[b].min_q;
                    pi.min_m = G1[a].min_m + G2[b].min_m;
                    pi.qmask = convolve(G1[a].qmask, G2[b].qmask);
                    pi.mmask = convolve(G1[a].mmask, G2[b].mmask);
                } else {
                    pi.min_q = 0;
                    pi.min_m = 0;
                }
                left_map[k].push_back(move(pi));
            }
        }
        if (left_map.empty()) { quads_exhausted++; continue; }

        const int64_t target_key = (int64_t)N_ORD * 100000LL;
        for (int c = 0; c < (int)G3.size() && !hit_cap && !quad_capped; ++c) {
            if (G3[c].p0 + min_p0_4 > maxR) break;
            for (int d = 0; d < (int)G4.size() && !hit_cap && !quad_capped; ++d) {
                if (G3[c].p0 + G4[d].p0 > maxR) break;
                auto it = left_map.find(target_key - G3[c].key - G4[d].key);
                if (it == left_map.end()) continue;
                Bitset155 em_cd = convolve(G3[c].emask, G4[d].emask);
                Bitset851 qm_cd;
                Bitset746 mm_cd;
                int min_q_cd = 0, min_m_cd = 0;
                if (need_vecs) {
                    min_q_cd = G3[c].min_q + G4[d].min_q;
                    min_m_cd = G3[c].min_m + G4[d].min_m;
                    qm_cd = convolve(G3[c].qmask, G4[d].qmask);
                    mm_cd = convolve(G3[c].mmask, G4[d].mmask);
                }

                for (const auto& pi : it->second) {
                    const auto& list1 = G1[pi.idx1].row_ids;
                    const auto& list2 = G2[pi.idx2].row_ids;
                    const auto& list3 = G3[c].row_ids;
                    const auto& list4 = G4[d].row_ids;

                    auto count_group = [&]() -> uint64_t {
                        if (i2 != i3 && i3 != i4) {
                            return (uint64_t)list1.size() * list2.size() * list3.size() * list4.size();
                        }
                        uint64_t cnt = 0;
                        for (int r2 : list2)
                            for (int r3 : list3) {
                                if (i2 == i3 && r2 > r3) continue;
                                for (int r4 : list4) {
                                    if (i3 == i4 && r3 > r4) continue;
                                    cnt += list1.size();
                                }
                            }
                        return cnt;
                    };

                    if (!sum_hits_154(pi.emask, em_cd)) {
                        uint64_t cnt = count_group();
                        total_mod46 += cnt;
                        counts[KILL_ENERGY] += cnt;
                        continue;
                    }
                    if (need_vecs && !prefilter_only) {
                        if (pi.min_q + min_q_cd > TARGET_Q26 || !sum_hits_target<TARGET_Q26>(pi.qmask, qm_cd)) {
                            uint64_t cnt = count_group();
                            total_mod46 += cnt;
                            counts[KILL_EVEN_26] += cnt;
                            continue;
                        }
                        if (pi.min_m + min_m_cd > TARGET_M6 || !sum_hits_target<TARGET_M6>(pi.mmask, mm_cd)) {
                            uint64_t cnt = count_group();
                            total_mod46 += cnt;
                            counts[KILL_JOINT_EVEN] += cnt;
                            continue;
                        }
                    }

                    struct Pair34 {
                        int r3, r4;
                        Bitset155 em34;
                        Bitset851 qm34;
                        Bitset746 mm34;
                        int min_q34, min_m34;
                    };
                    vector<Pair34> pairs34;
                    pairs34.reserve(list3.size() * list4.size());
                    for (int r3 : list3) {
                        for (int r4 : list4) {
                            if (i3 == i4 && r3 > r4) continue;
                            Pair34 p34;
                            p34.r3 = r3;
                            p34.r4 = r4;
                            p34.em34 = convolve(R3[r3].emask, R4[r4].emask);
                            if (need_vecs) {
                                p34.min_q34 = R3[r3].min_q + R4[r4].min_q;
                                p34.min_m34 = R3[r3].min_m + R4[r4].min_m;
                                p34.qm34 = convolve(R3[r3].qmask, R4[r4].qmask);
                                p34.mm34 = convolve(R3[r3].mmask, R4[r4].mmask);
                            } else {
                                p34.min_q34 = 0;
                                p34.min_m34 = 0;
                            }
                            pairs34.push_back(move(p34));
                        }
                    }

                    for (int r1 : list1) {
                        for (int r2 : list2) {
                            Bitset155 em12 = convolve(R1[r1].emask, R2[r2].emask);
                            int min_q12 = R1[r1].min_q + R2[r2].min_q;
                            int min_m12 = R1[r1].min_m + R2[r2].min_m;
                            bool have_qm12 = false, have_mm12 = false;
                            Bitset851 qm12;
                            Bitset746 mm12;
                            for (const auto& p34 : pairs34) {
                                int r3 = p34.r3, r4 = p34.r4;
                                if (i2 == i3 && r2 > r3) continue;
                                total_mod46++;
                                if (!sum_hits_154(em12, p34.em34)) {
                                    counts[KILL_ENERGY]++;
                                    continue;
                                }
                                total_energy_ok++;
                                if (energy_only) continue;

                                // Fast additive mod-26 energy check (qmask: E + 2*paf_26 == 6904)
                                if (min_q12 + p34.min_q34 > TARGET_Q26) {
                                    counts[KILL_EVEN_26]++;
                                    continue;
                                }
                                if (!have_qm12) {
                                    qm12 = convolve(R1[r1].qmask, R2[r2].qmask);
                                    have_qm12 = true;
                                }
                                if (!sum_hits_target<TARGET_Q26>(qm12, p34.qm34)) {
                                    counts[KILL_EVEN_26]++;
                                    continue;
                                }
                                total_q26_ok++;

                                // Fast additive mod-6 energy check (mmask: E + 6*paf_6 + 6*paf_12 == 5984)
                                if (min_m12 + p34.min_m34 > TARGET_M6) {
                                    counts[KILL_JOINT_EVEN]++;
                                    continue;
                                }
                                if (!have_mm12) {
                                    mm12 = convolve(R1[r1].mmask, R2[r2].mmask);
                                    have_mm12 = true;
                                }
                                if (!sum_hits_target<TARGET_M6>(mm12, p34.mm34)) {
                                    counts[KILL_JOINT_EVEN]++;
                                    continue;
                                }
                                total_m6_ok++;

                                int pid[8] = {
                                    R1[r1].pe_id, R1[r1].po_id,
                                    R2[r2].pe_id, R2[r2].po_id,
                                    R3[r3].pe_id, R3[r3].po_id,
                                    R4[r4].pe_id, R4[r4].po_id
                                };

                                if (prefilter_only) {
                                    if (exact_even26) {
                                        if (check_witness_even26_exact(profiles, pid, sc)) {
                                            total_exact_even26_ok++;
                                        } else {
                                            counts[KILL_EVEN_26]++;
                                        }
                                    }
                                    continue;
                                }

                                quad_survivors++;
                                const Row46* rptrs[4] = {&R1[r1], &R2[r2], &R3[r3], &R4[r4]};
                                KillReason kr = check_witness_all11(profiles, pid, survivor_out, rptrs, sc);
                                counts[kr]++;
                                if (witness_cap > 0 && total_m6_ok >= witness_cap) {
                                    hit_cap = true;
                                    break;
                                }
                                if (per_quad_cap > 0 && quad_survivors >= per_quad_cap) {
                                    quad_capped = true;
                                    break;
                                }
                            }
                            if (hit_cap || quad_capped) break;
                        }
                        if (hit_cap || quad_capped) break;
                    }
                    if (hit_cap || quad_capped) break;
                }
            }
        }
        if (quad_capped || hit_cap) quads_capped++;
        else quads_exhausted++;
    }

    if (survivor_out) fclose(survivor_out);
    auto t_end = chrono::steady_clock::now();
    printf("DONE (quads [%d, %d) of %d: %d exhausted, %d capped%s) in %.3fs:\n",
           start_quad, end_q, n_quads, quads_exhausted, quads_capped,
           hit_cap ? ", GLOBAL_CAP" : "",
           chrono::duration<double>(t_end - t_start).count());
    printf("  Total canonical mod-46 witnesses : %llu\n", (unsigned long long)total_mod46);
    printf("  Survived mod-78 energy (7088)    : %llu\n", (unsigned long long)total_energy_ok);
    if (need_vecs) {
        printf("  Survived mod-26 energy (q=850)   : %llu\n", (unsigned long long)total_q26_ok);
        printf("  Survived mod-6  energy (m=745)   : %llu\n", (unsigned long long)total_m6_ok);
        if (prefilter_only && exact_even26) {
            printf("  Survived exact (E=7088,p26=-92)  : %llu\n", (unsigned long long)total_exact_even26_ok);
        }
    }
    for (int k = 0; k < NUM_REASONS; ++k) {
        if (counts[k] > 0 || k == SURVIVED_ALL11 || k == INCONCLUSIVE_CAPPED) {
            printf("  %-22s : %llu\n", REASON_NAMES[k], (unsigned long long)counts[k]);
        }
    }
    return 0;
}

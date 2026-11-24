// Exhaust the nonzero GF(2^29) norm fiber for one fixed order-4412 triple.
// Input is exported by analyze_4412.py --export-binary-scan.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace {
constexpr int length = 1103;
constexpr int orbit_count = 59;
constexpr int shift_count = 29;
constexpr int orbit_size = 19;
constexpr uint32_t high_bit = 1U << 29;
constexpr uint32_t modulus = high_bit | 5;
constexpr uint32_t order = high_bit - 1;
constexpr uint64_t orbit_bits = (uint64_t{1} << orbit_count) - 1;

// x is primitive modulo x^29+x^2+1. These are exact multiplication by x
// and x^-1. One step visits another pair with the same product.
uint32_t times_x(uint32_t a) {
    return (a << 1) ^ ((a & (high_bit >> 1)) ? modulus : 0);
}
uint32_t over_x(uint32_t a) {
    return (a ^ ((a & 1U) ? modulus : 0)) >> 1;
}

uint32_t multiply(uint32_t a, uint32_t b) {
    uint32_t result = 0;
    for (; b; b >>= 1, a = times_x(a)) {
        if (b & 1U) result ^= a;
    }
    return result;
}

uint32_t power_of_x(uint32_t exponent) {
    uint32_t value = 1;
    for (uint32_t x = 2; exponent; exponent >>= 1, x = multiply(x, x)) {
        if (exponent & 1U) value = multiply(value, x);
    }
    return value;
}

using DecodeTable = std::array<std::array<uint64_t, 256>, 4>;

DecodeTable make_decode_table(const std::array<uint64_t, orbit_count>& inverse, int first_bit) {
    DecodeTable table{};
    for (int byte = 0; byte < 4; ++byte) {
        for (unsigned value = 1; value < 256; ++value) {
            int bit = 8 * byte + std::countr_zero(value);
            uint64_t column = bit < 29 ? inverse[first_bit + bit] : 0;
            table[byte][value] = table[byte][value & (value - 1)] ^ column;
        }
    }
    return table;
}

uint64_t decode_partial(const DecodeTable& table, uint32_t x) {
    return table[0][x & 255] ^ table[1][(x >> 8) & 255] ^
           table[2][(x >> 16) & 255] ^ table[3][x >> 24];
}

struct ShiftCheck {
    int shift;
    int target;
    std::array<int, orbit_count> diagonal{};
    // For i<j store the binary digits of the combined weight C_ij+C_ji.
    // Each term in the overlap is then counted by one mask and popcount.
    std::array<std::array<uint64_t, 4>, orbit_count> weights{};

    bool matches(uint64_t mask) const {
        int overlap = 0;
        for (uint64_t bits = mask; bits; bits &= bits - 1) {
            int i = std::countr_zero(bits);
            const auto& w = weights[i];
            overlap += diagonal[i] + std::popcount(w[0] & mask) +
                       2 * std::popcount(w[1] & mask) +
                       4 * std::popcount(w[2] & mask) +
                       8 * std::popcount(w[3] & mask);
        }
        return overlap == target;
    }
};

bool build_check(ShiftCheck& check, const std::array<int, length>& ids) {
    int counts[orbit_count][orbit_count]{};
    for (int x = 0; x < length; ++x) {
        int i = ids[x], j = ids[(x + check.shift) % length];
        if (i > j) std::swap(i, j);
        ++counts[i][j];
    }
    for (int i = 0; i < orbit_count; ++i) {
        check.diagonal[i] = counts[i][i];
        for (int j = i + 1; j < orbit_count; ++j) {
            int weight = counts[i][j];
            if (weight >= 16) return false;
            for (int bit = 0; bit < 4; ++bit) {
                if (weight & (1 << bit)) check.weights[i][bit] |= uint64_t{1} << j;
            }
        }
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL [FIRST_EXPONENT [COUNT]]\n";
        return 1;
    }
    uint64_t first = argc >= 3 ? std::stoull(argv[2]) : 0;
    if (first >= order) {
        std::cerr << "First exponent must lie in [0, 2^29-1).\n";
        return 1;
    }
    uint64_t count = argc >= 4 ? std::stoull(argv[3]) : order - first;
    if (!count || count > order - first) {
        std::cerr << "Count must be positive and stay within the nonzero fiber.\n";
        return 1;
    }
    if (power_of_x(order) != 1 || power_of_x(order / 233) == 1 ||
        power_of_x(order / 1103) == 1 || power_of_x(order / 2089) == 1) {
        std::cerr << "Field element x failed its primitive-order check.\n";
        return 1;
    }

    std::ifstream input(argv[1]);
    uint32_t norm;
    int size;
    if (!(input >> norm >> size) || norm == 0 || norm > order || size < 0 || size > length ||
        size % orbit_size > 1) {
        std::cerr << "Invalid norm or row size.\n";
        return 1;
    }
    std::array<uint64_t, orbit_count> inverse;
    for (auto& column : inverse) {
        if (!(input >> column) || column > orbit_bits) {
            std::cerr << "Invalid binary inverse column.\n";
            return 1;
        }
    }
    std::array<int, length> ids;
    std::array<int, orbit_count> sizes{};
    for (int& id : ids) {
        if (!(input >> id) || id < 0 || id >= orbit_count) {
            std::cerr << "Invalid orbit identifier.\n";
            return 1;
        }
        ++sizes[id];
    }
    if (ids[0] != 0 || sizes[0] != 1 ||
        std::any_of(sizes.begin() + 1, sizes.end(), [](int n) { return n != orbit_size; })) {
        std::cerr << "Unexpected orbit sizes.\n";
        return 1;
    }
    std::array<ShiftCheck, shift_count> checks;
    std::array<bool, orbit_count> seen_orbits{};
    seen_orbits[0] = true;
    for (auto& check : checks) {
        if (!(input >> check.shift >> check.target) || check.shift <= 0 || check.shift >= length ||
            !build_check(check, ids)) {
            std::cerr << "Invalid shift or orbit coefficient.\n";
            return 1;
        }
        int positive = ids[check.shift], negative = ids[length - check.shift];
        if (positive == negative || seen_orbits[positive] || seen_orbits[negative]) {
            std::cerr << "Shifts must cover distinct orbit pairs.\n";
            return 1;
        }
        seen_orbits[positive] = seen_orbits[negative] = true;
    }
    // Check the most atypical overlap requirement first.
    auto distance = [size](int target) { return std::abs(int64_t{length} * target - int64_t{size} * size); };
    std::stable_sort(checks.begin(), checks.end(), [&](const ShiftCheck& a, const ShiftCheck& b) {
        return distance(a.target) > distance(b.target);
    });

    const auto forward = make_decode_table(inverse, 1);
    const auto backward = make_decode_table(inverse, 30);
    uint64_t parity = (size % 2) ? inverse[0] : 0;
    uint64_t zero_bit = size % orbit_size;
    int minus_orbits = size / orbit_size + zero_bit;
    uint32_t a = power_of_x(first);
    uint32_t b = multiply(norm, power_of_x(order - first));

    auto start = std::chrono::steady_clock::now();
    uint64_t row_size_hits = 0;
    std::array<uint64_t, shift_count> shift_hits{};
    uint64_t scanned = 0;
    for (; scanned < count; ++scanned) {
        uint64_t mask = parity ^ decode_partial(forward, a) ^ decode_partial(backward, b);
        a = times_x(a);
        b = over_x(b);
        if ((mask & 1) != zero_bit || std::popcount(mask) != minus_orbits) continue;
        ++row_size_hits;
        int shift = 0;
        for (; shift < shift_count && checks[shift].matches(mask); ++shift) {
            ++shift_hits[shift];
        }
        if (shift == shift_count) {
            ++scanned;
            std::cout << "FOUND exponent=" << (first + scanned - 1) << " minus_orbit_mask=" << mask << '\n';
            break;
        }
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "first=" << first << " scanned=" << scanned << " requested=" << count
              << " row_size_hits=" << row_size_hits << " seconds=" << seconds << "\n";
    std::cout << "Shift survivors:";
    for (int j = 0; j < shift_count; ++j) std::cout << ' ' << checks[j].shift << ':' << shift_hits[j];
    std::cout << '\n';
    return shift_hits.back() ? 0 : 2;
}

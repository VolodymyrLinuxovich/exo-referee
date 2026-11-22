// fastci: the Referee's statistics hot path in C++.
//
//   auc            ROC AUC by the rank-sum identity, ties get their average rank
//   paired_ci      bootstrap 95% CI of AUC(cand) - AUC(inc) on the same resampled objects
//   paired_ci_star the same, but resampling host stars (clusters): planets around one star are
//                  not independent, so resampling objects makes the interval too narrow
//
// Exposed with a C ABI for ctypes (exo/fastci.py). Build: make -C exo/native
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace {

// AUC of scores p against 0/1 labels y, over n items. Returns 0.5 if one class is missing.
double auc_impl(const double* p, const int8_t* y, int64_t n, std::vector<int64_t>& idx) {
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return p[a] < p[b]; });
    double pos_rank_sum = 0.0;
    int64_t n1 = 0;
    int64_t i = 0;
    while (i < n) {
        int64_t j = i;
        while (j + 1 < n && p[idx[j + 1]] == p[idx[i]]) {
            ++j;
        }
        const double avg_rank = 0.5 * double(i + j) + 1.0;   // ranks are 1-based
        for (int64_t k = i; k <= j; ++k) {
            if (y[idx[k]]) {
                pos_rank_sum += avg_rank;
                ++n1;
            }
        }
        i = j + 1;
    }
    const int64_t n0 = n - n1;
    if (n1 == 0 || n0 == 0) {
        return 0.5;
    }
    return (pos_rank_sum - double(n1) * double(n1 + 1) / 2.0) / (double(n1) * double(n0));
}

void percentiles(std::vector<double>& d, double* lo, double* hi) {
    std::sort(d.begin(), d.end());
    auto at = [&](double q) {
        // linear interpolation, same as numpy's default percentile
        const double pos = q * double(d.size() - 1);
        const auto k = static_cast<size_t>(pos);
        const double f = pos - double(k);
        return k + 1 < d.size() ? d[k] * (1.0 - f) + d[k + 1] * f : d[k];
    };
    *lo = at(0.025);
    *hi = at(0.975);
}

}  // namespace

extern "C" {

double fastci_auc(const double* p, const int8_t* y, int64_t n) {
    std::vector<int64_t> idx;
    return auc_impl(p, y, n, idx);
}

// Bootstrap over objects. Resamples that contain a single class are skipped, as in referee.py.
void fastci_paired_ci(const double* c, const double* b, const int8_t* y, int64_t n,
                      int64_t reps, uint64_t seed, double* lo, double* hi) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> pick(0, n - 1);
    std::vector<double> cs(n), bs(n), diffs;
    std::vector<int8_t> ys(n);
    std::vector<int64_t> idx;
    diffs.reserve(reps);
    for (int64_t r = 0; r < reps; ++r) {
        int64_t pos = 0;
        for (int64_t k = 0; k < n; ++k) {
            const int64_t i = pick(rng);
            cs[k] = c[i];
            bs[k] = b[i];
            ys[k] = y[i];
            pos += y[i];
        }
        if (pos == 0 || pos == n) {
            continue;
        }
        diffs.push_back(auc_impl(cs.data(), ys.data(), n, idx) - auc_impl(bs.data(), ys.data(), n, idx));
    }
    percentiles(diffs, lo, hi);
}

// Bootstrap over clusters (host stars). group[i] is the star index of object i, in [0, n_groups).
void fastci_paired_ci_star(const double* c, const double* b, const int8_t* y, const int64_t* group,
                           int64_t n, int64_t n_groups, int64_t reps, uint64_t seed, double* lo, double* hi) {
    std::vector<std::vector<int64_t>> members(n_groups);
    for (int64_t i = 0; i < n; ++i) {
        members[group[i]].push_back(i);
    }
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> pick(0, n_groups - 1);
    std::vector<double> cs, bs, diffs;
    std::vector<int8_t> ys;
    std::vector<int64_t> idx;
    diffs.reserve(reps);
    for (int64_t r = 0; r < reps; ++r) {
        cs.clear();
        bs.clear();
        ys.clear();
        int64_t pos = 0;
        for (int64_t g = 0; g < n_groups; ++g) {
            for (const int64_t i : members[pick(rng)]) {
                cs.push_back(c[i]);
                bs.push_back(b[i]);
                ys.push_back(y[i]);
                pos += y[i];
            }
        }
        const auto m = static_cast<int64_t>(ys.size());
        if (pos == 0 || pos == m) {
            continue;
        }
        diffs.push_back(auc_impl(cs.data(), ys.data(), m, idx) - auc_impl(bs.data(), ys.data(), m, idx));
    }
    percentiles(diffs, lo, hi);
}

}  // extern "C"

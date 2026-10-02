// wavelet_noise.hpp: a hierarchical Gaussian white noise field for cosmological
// initial conditions, computable for any block of cells at any level without
// generating the rest of the field.
//
// SPDX-License-Identifier: MIT
//
// Field definition (base B, seeds s_l):
//   level l has n_l = B * 2^l cells per side, periodic, unit variance per cell.
//   Cell index i sits at x = i / n_l (box units), the corner convention.
//   level 0:  w_0[i,j,k] = G(s_0, 0, 0, i, j, k)
//   level l:  w_l = IDWT_db4(approx = roll(w_{l-1}, +1), details_b = G(s_l, l, b, m)),
//             b = 1..7 over coarse cells m. Band b has bit 2 = x, bit 1 = y, bit 0 = z
//             set where that axis carries the high-pass (pywt keys 'aad'=1 ... 'ddd'=7).
//   1D synthesis (periodization, matches pywt.idwt):
//             fine[p] = sum_{t: p+3-t even} a[(p+3-t)/2] rec_lo[t] + d[(p+3-t)/2] rec_hi[t]
//   roll(w, +1)[m] = w[m-1] aligns the db4 low-pass with the coarse cell, so no fractional
//   shifts are needed between levels.
//   G = Box-Muller on Threefry2x64-20, key {seed, (level << 8) | band},
//       counter {i | j << 21 | k << 42, 0}, indices taken modulo the grid size of that level.
//   s_l is the seed of the numbers level l adds. Giving levels 0..L new seeds draws a new
//   realization of the scales down to level L with the same finer detail numbers; giving
//   levels above L new seeds keeps the scales of level L and draws new smaller ones.
//
// Text form "wavelet-B:s[,n:s...]": the first entry is level 0, so B is the base, and s is the
// seed of every level. Each further n:s gives the level with n cells per side, and every finer
// level, the seed s. "wavelet-8:777" is one seed for all levels of base 8; "wavelet-8:777,128:5"
// keeps the scales of the 64^3 level and redraws everything finer.
//
// The orthonormal transform keeps every level white with unit variance. Low-k modes of
// level l are those of level l-1 up to the db4 low-pass response. Values agree to rounding
// (about 1e-15) between builds; FMA contraction and the math library can change the last bits.

#ifndef WAVELET_NOISE_HPP
#define WAVELET_NOISE_HPP

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace wavelet_noise {

  // Threefry2x64 with 20 rounds, as in Random123 (Salmon et al. 2011).
  inline void tf2x64_20(const uint64_t key[2], const uint64_t ctr[2], uint64_t out[2])
  {
    static const int rot[8] = {16, 42, 12, 31, 16, 32, 24, 21};
    const uint64_t ks[3] = {key[0], key[1], 0x1BD11BDAA9FC1A22ULL ^ key[0] ^ key[1]};
    uint64_t x0 = ctr[0] + ks[0];
    uint64_t x1 = ctr[1] + ks[1];
    for (int r = 0; r < 20; ++r) {
      x0 += x1;
      x1 = (x1 << rot[r % 8]) | (x1 >> (64 - rot[r % 8]));
      x1 ^= x0;
      if (r % 4 == 3) {
        const int s = r / 4 + 1;
        x0 += ks[s % 3];
        x1 += ks[(s + 1) % 3] + uint64_t(s);
      }
    }
    out[0] = x0;
    out[1] = x1;
  }

  inline double gaussian(uint64_t seed, int level, int band, uint64_t i, uint64_t j, uint64_t k)
  {
    const uint64_t key[2] = {seed, (uint64_t(level) << 8) | uint64_t(band)};
    const uint64_t ctr[2] = {i | (j << 21) | (k << 42), 0};
    uint64_t r[2];
    tf2x64_20(key, ctr, r);
    const double ulp = 1.0 / 9007199254740992.0;      // 2^-53
    const double u1 = 1.0 - double(r[0] >> 11) * ulp; // (0, 1]
    const double u2 = double(r[1] >> 11) * ulp;       // [0, 1)
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586477 * u2);
  }

  // The counter packs 21 bits per index, so a level has at most 2^21 cells per side; with
  // base >= 4 that allows levels 0 to 19.
  constexpr long max_cells = 1L << 21;
  constexpr int max_level = 19;

  // The base and the seed of the numbers each level adds.
  struct Field {
    int base;
    uint64_t seed[max_level + 1];
    Field(int base, uint64_t s) : base(base)
    {
      for (uint64_t &x : seed)
        x = s;
    }
  };

  // Level whose grid has `res` cells per side, or -1 if res is not base * 2^l with l <= max_level.
  inline int level_for(int base, long res)
  {
    for (int l = 0; base > 0 && l <= max_level && (long(base) << l) <= res; ++l)
      if ((long(base) << l) == res)
        return l;
    return -1;
  }

  // Field from its text form (see the top of the file).
  inline Field parse(const std::string &text)
  {
    const std::string tag = "wavelet-";
    const auto bad = [&](const std::string &why) {
      return std::invalid_argument("wavelet_noise: '" + text + "': " + why);
    };
    const auto number = [&](const std::string &digits) {
      if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos)
        throw bad("'" + digits + "' is not a non-negative integer");
      try {
        return std::stoull(digits);
      } catch (const std::out_of_range &) {
        throw bad("'" + digits + "' does not fit in 64 bits");
      }
    };
    if (text.compare(0, tag.size(), tag) != 0)
      throw bad("does not start with " + tag);
    Field field(0, 0);
    int last = -1;
    for (size_t pos = tag.size(), end; pos <= text.size(); pos = end + 1) {
      end = std::min(text.find(',', pos), text.size());
      const std::string item = text.substr(pos, end - pos);
      const size_t colon = item.find(':');
      if (colon == std::string::npos)
        throw bad("'" + item + "' is not n:seed");
      const uint64_t n = number(item.substr(0, colon)), s = number(item.substr(colon + 1));
      if (last < 0) {
        if (n < 4 || n > uint64_t(max_cells))
          throw bad("the base must be between 4 and 2^21");
        field = Field(int(n), s);
        last = 0;
        continue;
      }
      const int l = n <= uint64_t(max_cells) ? level_for(field.base, long(n)) : -1;
      if (l <= last)
        throw bad("n = " + std::to_string(n) + " must be base * 2^l, larger than the previous entry and at most 2^21");
      std::fill(field.seed + l, field.seed + max_level + 1, s);
      last = l;
    }
    return field;
  }

  namespace detail {

    inline long floordiv2(long v) { return (v >= 0) ? v / 2 : -((-v + 1) / 2); }

    inline uint64_t wrap(long v, long n) { return uint64_t(((v % n) + n) % n); }

    // A block of values on an unwrapped index range [lo[d], lo[d] + n[d]).
    struct Block {
      long lo[3] = {0, 0, 0};
      long n[3] = {0, 0, 0};
      std::vector<double> v;
      double &at(long i, long j, long k) { return v[(size_t(i - lo[0]) * n[1] + (j - lo[1])) * n[2] + (k - lo[2])]; }
    };

    // Coarse index range [mlo, mhi] that fine indices [p0, p0 + np) depend on.
    inline void coarse_range(long p0, long np, long &mlo, long &mhi)
    {
      mlo = floordiv2(p0 - 4);
      mhi = floordiv2(p0 + np + 2);
    }

    // Synthesise along one axis: low and high share their index ranges; the output
    // covers fine range [f0, f0 + fn) on `axis` and the input ranges on the other axes.
    inline Block synth_axis(const Block &low, const Block &high, int axis, long f0, long fn)
    {
      static const double rec_lo[8] = {0.2303778133088965,    0.7148465705529157,   0.6308807679298589,
                                       -0.027983769416859854, -0.18703481171909309, 0.030841381835560764,
                                       0.0328830116668852,    -0.010597401785069032};
      static const double rec_hi[8] = {-0.010597401785069032, -0.0328830116668852,   0.030841381835560764,
                                       0.18703481171909309,   -0.027983769416859854, -0.6308807679298589,
                                       0.7148465705529157,    -0.2303778133088965};
      Block out;
      for (int d = 0; d < 3; ++d) {
        out.lo[d] = low.lo[d];
        out.n[d] = low.n[d];
      }
      out.lo[axis] = f0;
      out.n[axis] = fn;
      out.v.assign(size_t(out.n[0]) * out.n[1] * out.n[2], 0.0);

      const long stride_in[3] = {low.n[1] * low.n[2], low.n[2], 1};
      const long stride_out[3] = {out.n[1] * out.n[2], out.n[2], 1};
      const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
      for (long u = 0; u < out.n[a1]; ++u)
        for (long w = 0; w < out.n[a2]; ++w) {
          const long base_in = u * stride_in[a1] + w * stride_in[a2];
          const long base_out = u * stride_out[a1] + w * stride_out[a2];
          for (long q = 0; q < fn; ++q) {
            const long p = f0 + q;
            double s = 0.0;
            for (int t = (p + 3) & 1; t < 8; t += 2) {
              const long m = (p + 3 - t) / 2 - low.lo[axis];
              const long idx = base_in + m * stride_in[axis];
              s += low.v[idx] * rec_lo[t] + high.v[idx] * rec_hi[t];
            }
            out.v[base_out + q * stride_out[axis]] = s;
          }
        }
      return out;
    }

    inline Block fill_block(const Field &field, int level, const long lo[3], const long n[3])
    {
      const long nl = long(field.base) << level;
      if (level == 0) {
        Block out;
        for (int d = 0; d < 3; ++d) {
          out.lo[d] = lo[d];
          out.n[d] = n[d];
        }
        out.v.resize(size_t(n[0]) * n[1] * n[2]);
        for (long i = lo[0]; i < lo[0] + n[0]; ++i)
          for (long j = lo[1]; j < lo[1] + n[1]; ++j)
            for (long k = lo[2]; k < lo[2] + n[2]; ++k)
              out.at(i, j, k) = gaussian(field.seed[0], 0, 0, wrap(i, nl), wrap(j, nl), wrap(k, nl));
        return out;
      }

      // Coarse cells this block depends on; the approximation is the parent level rolled by +1.
      long clo[3], cn[3];
      for (int d = 0; d < 3; ++d) {
        long mlo, mhi;
        coarse_range(lo[d], n[d], mlo, mhi);
        clo[d] = mlo;
        cn[d] = mhi - mlo + 1;
      }
      const long plo[3] = {clo[0] - 1, clo[1] - 1, clo[2] - 1};
      Block parent = fill_block(field, level - 1, plo, cn);
      for (int d = 0; d < 3; ++d)
        parent.lo[d] = clo[d]; // roll(w, +1)[m] = w[m - 1]

      const long nc = nl / 2;
      Block bands[8];
      bands[0] = std::move(parent);
      for (int b = 1; b < 8; ++b) {
        Block &B = bands[b];
        for (int d = 0; d < 3; ++d) {
          B.lo[d] = clo[d];
          B.n[d] = cn[d];
        }
        B.v.resize(size_t(cn[0]) * cn[1] * cn[2]);
        for (long i = clo[0]; i < clo[0] + cn[0]; ++i)
          for (long j = clo[1]; j < clo[1] + cn[1]; ++j)
            for (long k = clo[2]; k < clo[2] + cn[2]; ++k)
              B.at(i, j, k) = gaussian(field.seed[level], level, b, wrap(i, nc), wrap(j, nc), wrap(k, nc));
      }

      // z, then y, then x. Band bit 0 = z, bit 1 = y, bit 2 = x.
      Block bz[4];
      for (int b = 0; b < 4; ++b)
        bz[b] = synth_axis(bands[2 * b], bands[2 * b + 1], 2, lo[2], n[2]);
      Block by[2];
      for (int b = 0; b < 2; ++b)
        by[b] = synth_axis(bz[2 * b], bz[2 * b + 1], 1, lo[1], n[1]);
      return synth_axis(by[0], by[1], 0, lo[0], n[0]);
    }

  } // namespace detail

  // Field values at `level` on global indices [x0[d], x0[d] + n[d]), periodic, written
  // row-major with the last index fastest. x0 may be negative or exceed the grid.
  inline void fill(const Field &field, int level, const long x0[3], const long n[3], double *out)
  {
    if (field.base < 4 || level < 0 || level > max_level || (long(field.base) << level) > max_cells || n[0] < 0 ||
        n[1] < 0 || n[2] < 0)
      throw std::invalid_argument("wavelet_noise::fill: need base >= 4, at most 2^21 cells per side, extents >= 0");
    if (n[0] == 0 || n[1] == 0 || n[2] == 0)
      return;
    const long nl = long(field.base) << level;
    const long lo[3] = {(x0[0] % nl + nl) % nl, (x0[1] % nl + nl) % nl, (x0[2] % nl + nl) % nl};
    detail::Block b = detail::fill_block(field, level, lo, n);
    for (size_t q = 0; q < b.v.size(); ++q)
      out[q] = b.v[q];
  }

} // namespace wavelet_noise

#endif

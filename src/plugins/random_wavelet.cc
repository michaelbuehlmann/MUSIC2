// WAVELET: hierarchical white noise from wavelet_noise.hpp (db4 wavelet levels with
// base * 2^l cells per side). Any block of any level can be evaluated without the
// rest of the field, so a refinement patch costs only its own volume.
//
// MUSIC level L (2^L cells) takes the smallest noise level with n' = base * 2^l >= 2^L
// cells, keeps the Fourier modes |k| < 2^(L-1) and moves the values from the noise
// cell corners (x = j / n') to the MUSIC cell centres (x = (i + 1/2) / 2^L), both in
// the unshifted box. On levelmin the transform covers the periodic box and is exact.
// A patch is transformed on itself plus fft_margin cells per side, then cut back; the
// error this leaves is confined to modes close to the Nyquist frequency.
//
//   [random]
//   generator  = WAVELET
//   seed       = wavelet-8:777  (the field's text form, the same string as HACC's I_SEED: base 8,
//                                seed 777; wavelet-8:777,128:42 gives levels from 128^3 on seed 42)
//   fft_margin = 32    (default 32; >= the box size gives the exact periodic answer)
//
// WAVELET_DUMP_DIR=dir writes the delivered noise of each level to dir/level_LL.bin
// (float64, row-major) with its geometry in dir/level_LL.txt.

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numeric>

#include "random.hh"
#include "wavelet_noise.hpp"

class RNG_wavelet : public RNG_plugin
{
protected:
  std::string seed_;
  wavelet_noise::Field field_;
  int fft_margin_, levelmin_poisson_;
  long shift_[3];

  // smallest m >= n that is a multiple of unit and has no prime factor above 7
  static long fft_size(long n, long unit)
  {
    for (long m = (n + unit - 1) / unit * unit;; m += unit)
    {
      long r = m;
      for (long p : {2, 3, 5, 7})
        while (r % p == 0)
          r /= p;
      if (r == 1)
        return m;
    }
  }

  static long floordiv(long a, long b) { return (a >= 0) ? a / b : -((-a + b - 1) / b); }

public:
  explicit RNG_wavelet(config_file &cf)
      : RNG_plugin(cf), seed_(cf.get_value<std::string>("random", "seed")), field_(wavelet_noise::parse(seed_))
  {
    fft_margin_ = pcf_->get_value_safe<int>("random", "fft_margin", 32);
    if (fft_margin_ < 0)
    {
      music::elog.Print("WAVELET: need fft_margin >= 0");
      throw std::runtime_error("WAVELET: need fft_margin >= 0");
    }
    music::ilog.Print("WAVELET: seed %s, fft_margin %d", seed_.c_str(), fft_margin_);
  }

  bool is_multiscale() const { return true; }

  void initialize_for_grid_structure(const refinement_hierarchy &refh)
  {
    prefh_ = &refh;
    levelmin_poisson_ = pcf_->get_value<int>("setup", "levelmin");
    shift_[0] = pcf_->get_value<int>("setup", "shift_x");
    shift_[1] = pcf_->get_value<int>("setup", "shift_y");
    shift_[2] = pcf_->get_value<int>("setup", "shift_z");
  }

  void fill_grid(int level, DensityGrid<real_t> &R);
};

void RNG_wavelet::fill_grid(int level, DensityGrid<real_t> &R)
{
  double tstart = get_wtime();

  const long N = 1L << level;
  int lw = 0;
  while ((long(field_.base) << lw) < N)
    ++lw;
  const long np = long(field_.base) << lw, g = std::gcd(np, N), unit = N / g;
  const long lfac = 1L << (level - levelmin_poisson_);

  // a:  unshifted index of R(0,0,0) on the 2^L grid (patch offset minus its padding)
  // M:  FFT region on the 2^L grid, R plus el cells on the left; capped at the periodic box
  // Mp: the same region on the n' grid, starting at noise index j0
  long n[3], a[3], M[3], el[3], Mp[3], j0[3];
  double phase[3];
  for (int d = 0; d < 3; ++d)
  {
    n[d] = R.size(d);
    a[d] = prefh_->offset_abs(level, d) - (n[d] - (long)prefh_->size(level, d)) / 2 - lfac * shift_[d];
    M[d] = std::min(N, fft_size(n[d] + 2 * fft_margin_, unit));
    el[d] = (M[d] - n[d]) / 2;
    Mp[d] = M[d] / unit * (np / g);
    j0[d] = floordiv((a[d] - el[d]) * np, N);
    // half a cell to the centre, plus where the region starts between noise cells (in 2^L cells)
    const long r = (a[d] - el[d]) * np - j0[d] * N;
    phase[d] = 2.0 * M_PI * (0.5 + double(r) / double(np)) / double(M[d]);
  }

  music::ilog.Print("WAVELET: level %d from noise level %d (%ld^3), region %ldx%ldx%ld -> %ldx%ldx%ld", level, lw, np,
                    Mp[0], Mp[1], Mp[2], M[0], M[1], M[2]);

  // noise on the region in 32^3 tiles: fill() is exact on any block, so the tiles are independent
  const size_t nzp = 2 * (Mp[2] / 2 + 1);
  real_t *src = new real_t[size_t(Mp[0]) * Mp[1] * nzp];
  const long tile = 32, nt[3] = {(Mp[0] + tile - 1) / tile, (Mp[1] + tile - 1) / tile, (Mp[2] + tile - 1) / tile};

#pragma omp parallel for schedule(dynamic)
  for (long t = 0; t < nt[0] * nt[1] * nt[2]; ++t)
  {
    const long t0[3] = {t / (nt[1] * nt[2]) * tile, (t / nt[2]) % nt[1] * tile, t % nt[2] * tile};
    const long x0[3] = {j0[0] + t0[0], j0[1] + t0[1], j0[2] + t0[2]};
    const long nb[3] = {std::min(tile, Mp[0] - t0[0]), std::min(tile, Mp[1] - t0[1]), std::min(tile, Mp[2] - t0[2])};
    std::vector<double> blk((size_t)nb[0] * nb[1] * nb[2]);
    wavelet_noise::fill(field_, lw, x0, nb, blk.data());
    for (long i = 0; i < nb[0]; ++i)
      for (long j = 0; j < nb[1]; ++j)
        for (long k = 0; k < nb[2]; ++k)
          src[(size_t(t0[0] + i) * Mp[1] + (t0[1] + j)) * nzp + t0[2] + k] = blk[(size_t(i) * nb[1] + j) * nb[2] + k];
  }

  complex_t *csrc = reinterpret_cast<complex_t *>(src);
  fftw_plan_t pf = FFTW_API(plan_dft_r2c_3d)(Mp[0], Mp[1], Mp[2], src, csrc, FFTW_ESTIMATE);
  FFTW_API(execute)(pf);
  FFTW_API(destroy_plan)(pf);

  // keep |k| < M/2, shift, and normalise so that each kept mode has unit-variance white noise power
  const size_t nzc = M[2] / 2 + 1, nzcp = Mp[2] / 2 + 1;
  real_t *dst = new real_t[size_t(M[0]) * M[1] * 2 * nzc];
  complex_t *cdst = reinterpret_cast<complex_t *>(dst);
  const double norm = 1.0 / std::sqrt(double(M[0]) * M[1] * M[2] * double(Mp[0]) * Mp[1] * Mp[2]);

#pragma omp parallel for
  for (long i = 0; i < M[0]; ++i)
    for (long j = 0; j < M[1]; ++j)
      for (long k = 0; k < (long)nzc; ++k)
      {
        const long kx = (i <= M[0] / 2) ? i : i - M[0], ky = (j <= M[1] / 2) ? j : j - M[1];
        const size_t qd = (size_t(i) * M[1] + j) * nzc + k;
        if (2 * std::abs(kx) == M[0] || 2 * std::abs(ky) == M[1] || 2 * k == M[2])
        {
          RE(cdst[qd]) = 0.0;
          IM(cdst[qd]) = 0.0;
          continue;
        }
        const size_t qs = (size_t((kx + Mp[0]) % Mp[0]) * Mp[1] + (ky + Mp[1]) % Mp[1]) * nzcp + k;
        const std::complex<double> v = std::complex<double>(RE(csrc[qs]), IM(csrc[qs])) *
                                       std::polar(norm, phase[0] * kx + phase[1] * ky + phase[2] * k);
        RE(cdst[qd]) = v.real();
        IM(cdst[qd]) = v.imag();
      }
  delete[] src;

  fftw_plan_t pb = FFTW_API(plan_dft_c2r_3d)(M[0], M[1], M[2], cdst, dst, FFTW_ESTIMATE);
  FFTW_API(execute)(pb);
  FFTW_API(destroy_plan)(pb);

  double sum = 0.0, sum2 = 0.0;
#pragma omp parallel for reduction(+ : sum, sum2)
  for (long i = 0; i < n[0]; ++i)
    for (long j = 0; j < n[1]; ++j)
      for (long k = 0; k < n[2]; ++k)
      {
        const real_t v = dst[(size_t(i + el[0]) * M[1] + (j + el[1])) * 2 * nzc + (k + el[2])];
        R(i, j, k) = v;
        sum += v;
        sum2 += v * v;
      }
  delete[] dst;

  const double count = double(n[0]) * n[1] * n[2];
  music::ilog.Print("WAVELET: level %d done in %.2fs, mean %g, variance %g", level, get_wtime() - tstart, sum / count,
                    sum2 / count - (sum / count) * (sum / count));

  if (const char *dir = std::getenv("WAVELET_DUMP_DIR"))
  {
    char fname[512];
    snprintf(fname, 512, "%s/level_%02d.txt", dir, level);
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "level %d\nseed %s\nnoise_level %d\nnoise_n %ld\nfft_margin %d\n", level, seed_.c_str(), lw, np,
            fft_margin_);
    fprintf(fp, "size %ld %ld %ld\n", n[0], n[1], n[2]);
    fprintf(fp, "patch_size %zu %zu %zu\n", prefh_->size(level, 0), prefh_->size(level, 1), prefh_->size(level, 2));
    fprintf(fp, "offset_abs %ld %ld %ld\n", (long)prefh_->offset_abs(level, 0), (long)prefh_->offset_abs(level, 1),
            (long)prefh_->offset_abs(level, 2));
    fprintf(fp, "shift %ld %ld %ld\nlfac %ld\n", shift_[0], shift_[1], shift_[2], lfac);
    fprintf(fp, "first_cell %ld %ld %ld\n", a[0], a[1], a[2]);
    fprintf(fp, "fft_size %ld %ld %ld\nfft_left %ld %ld %ld\n", M[0], M[1], M[2], el[0], el[1], el[2]);
    fprintf(fp, "noise_size %ld %ld %ld\nnoise_start %ld %ld %ld\n", Mp[0], Mp[1], Mp[2], j0[0], j0[1], j0[2]);
    fclose(fp);

    snprintf(fname, 512, "%s/level_%02d.bin", dir, level);
    fp = fopen(fname, "wb");
    std::vector<double> row(n[2]);
    for (long i = 0; i < n[0]; ++i)
      for (long j = 0; j < n[1]; ++j)
      {
        for (long k = 0; k < n[2]; ++k)
          row[k] = R(i, j, k);
        fwrite(row.data(), sizeof(double), n[2], fp);
      }
    fclose(fp);
  }
}

namespace
{
  RNG_plugin_creator_concrete<RNG_wavelet> creator("WAVELET");
}

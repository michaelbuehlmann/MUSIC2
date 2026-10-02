// GenericIO output for HACC, modelled on monofonIC's genericio plugin. Writes one file with the
// columns HACC reads as INPUT_TYPE GENERIC_IO: x, y, z, vx, vy, vz, id, phi and mask, and with
// GenericIO_HACCHydro = yes also mass, hh, uu, rho, mu, zmet and yhe for hacc_hydro.
//
//   [output]
//   format              = genericio
//   filename            = ics.gio
//   GenericIO_HACCHydro = no    (default; yes for hacc_hydro: per-particle masses, zooms, gas)
//
// Units are HACC's: comoving positions in Mpc/h in [0, boxlength), velocities dx/dt in km/s and
// masses in Msun/h with HACC's critical density. Positions are those of the parent box: MUSIC's
// recentring shift (setup/shift_x, _y, _z) is undone, so a zoom shares the coordinates of the box
// it refines.
//
// Dark matter has mask 0, gas the baryon bit (1 << 2). IDs count the particles, finest level first.
// With GenericIO_HACCHydro they are 2k for dark matter and 2k+1 for gas, as in hacc_hydro's own
// initial conditions, so the gas and dark matter particles of one fine cell share k.
//
// Gravity-only HACC needs equal masses, so it takes one level and no baryons. With baryons, gas
// exists on the finest level only, half a fine cell off the dark matter (MUSIC's SPH staggering),
// and coarser levels hold total-matter particles, as in the SWIFT plugin.
//
// GenericIO writes through MPI. MUSIC runs without it, so the plugin starts MPI for the write and
// writes the file from one rank; HACC redistributes the particles when it reads them.

#ifdef ENABLE_GENERICIO

#include <GenericIO.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cosmology_calculator.hh"
#include "output.hh"

extern std::unique_ptr<cosmology::calculator> the_cosmo_calc;

class genericio_output_plugin : public output_plugin
{
protected:
  static constexpr double rho_crit_ = 2.77536627e11; // h^2 Msun / Mpc^3, as in HACC
  static constexpr uint16_t baryon_mask_ = 1 << 2;    // hacc_hydro MASK_SPECIES

  bool hydro_, baryons_;
  double boxlength_, astart_, omega_m_, omega_b_, shift_[3];
  float uu_, mu_;
  std::vector<size_t> nlevel_;           // leaf cells per level, finest first
  std::vector<double> dm_x_[3], gas_x_[3]; // box units, before wrapping
  std::vector<double> dm_v_[3], gas_v_[3]; // km/s, dx/dt

  // calls f(level, i, j, k) for every leaf cell, finest level first
  template <typename F> static void for_leaf_cells(const grid_hierarchy &gh, F f)
  {
    for (int l = gh.levelmax(); l >= int(gh.levelmin()); --l)
      for (unsigned i = 0; i < gh.get_grid(l)->size(0); ++i)
        for (unsigned j = 0; j < gh.get_grid(l)->size(1); ++j)
          for (unsigned k = 0; k < gh.get_grid(l)->size(2); ++k)
            if (gh.is_in_mask(l, i, j, k) && !gh.is_refined(l, i, j, k))
              f(l, i, j, k);
  }

  void collect_positions(int coord, const grid_hierarchy &gh, double offset, std::vector<double> &out)
  {
    out.clear();
    for_leaf_cells(gh, [&](int l, unsigned i, unsigned j, unsigned k) {
      double x[3];
      gh.cell_pos(l, i, j, k, x);
      out.push_back(x[coord] + offset + (*gh.get_grid(l))(i, j, k));
    });
    nlevel_.clear();
    for (int l = gh.levelmax(); l >= int(gh.levelmin()); --l)
      nlevel_.push_back(gh.count_leaf_cells(l, l));
  }

  void collect_velocities(const grid_hierarchy &gh, std::vector<double> &out)
  {
    out.clear();
    for_leaf_cells(gh, [&](int l, unsigned i, unsigned j, unsigned k) {
      out.push_back((*gh.get_grid(l))(i, j, k) * boxlength_ / astart_);
    });
  }

  // MPI for the GenericIO write, started here unless something else already runs it. GenericIO
  // computes checksums in OpenMP threads while only this thread calls MPI, hence FUNNELED.
  struct mpi_session {
    bool owned = false;
    int provided = MPI_THREAD_FUNNELED;
    mpi_session()
    {
      int running = 0;
      MPI_Initialized(&running);
      if (running)
        return;
      MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
      owned = true;
    }
    ~mpi_session()
    {
      if (owned)
        MPI_Finalize();
    }
  };

  float position(double x, int coord) const
  {
    double p = std::fmod(x * boxlength_ - shift_[coord], boxlength_);
    if (p < 0)
      p += boxlength_;
    const float pf = float(p);
    return pf < float(boxlength_) ? pf : 0.0f;
  }

public:
  explicit genericio_output_plugin(config_file &cf) : output_plugin(cf)
  {
    hydro_ = cf.get_value_safe<bool>("output", "GenericIO_HACCHydro", false);
    baryons_ = cf.get_value_safe<bool>("setup", "baryons", false);
    boxlength_ = cf.get_value<double>("setup", "boxlength");
    astart_ = 1.0 / (1.0 + cf.get_value<double>("setup", "zstart"));
    const cosmology::parameters &cp = the_cosmo_calc->get_parameters();
    omega_m_ = cp["Omega_m"];
    omega_b_ = cp["Omega_b"];

    const double cell = boxlength_ / double(1ul << levelmin_);
    shift_[0] = cf.get_value_safe<int>("setup", "shift_x", 0) * cell;
    shift_[1] = cf.get_value_safe<int>("setup", "shift_y", 0) * cell;
    shift_[2] = cf.get_value_safe<int>("setup", "shift_z", 0) * cell;

    if (!hydro_ && (levelmin_ != levelmax_ || baryons_))
      throw std::runtime_error("genericio: gravity-only HACC needs equal masses (levelmin = levelmax, no baryons); "
                               "set [output] GenericIO_HACCHydro = yes for hacc_hydro");

    if (baryons_)
      cf.insert_value("setup", "do_SPH", "yes"); // gas on the staggered grid, as for the other particle codes

    // initial gas temperature and internal energy, as in monofonIC's genericio plugin
    const double h = cp["h"], YHe = cp["YHe"], gamma = cf.get_value_safe<double>("cosmology", "gamma", 5.0 / 3.0);
    const double npol = std::fabs(1.0 - gamma) > 1e-7 ? 1.0 / (gamma - 1.0) : 1.0;
    const double adec = 1.0 / (160.0 * std::pow(omega_b_ * h * h / 0.022, 2.0 / 5.0));
    const double Tini = astart_ < adec ? cp["Tcmb"] / astart_ : cp["Tcmb"] / astart_ / astart_ * adec;
    const double mu = Tini > 1e4 ? 4.0 / (8.0 - 5.0 * YHe) : 4.0 / (1.0 + 3.0 * (1.0 - YHe));
    const double ceint = 1.3806e-16 / 1.6726e-24 * Tini * npol / mu / 1e10; // (km/s)^2
    uu_ = float(ceint / astart_ / astart_);
    mu_ = float(mu);
    if (hydro_ && baryons_)
      music::ilog.Print("GenericIO: initial gas temperature %.2e K, internal energy %.2e (km/s)^2", Tini, ceint);
  }

  void write_dm_mass(const grid_hierarchy &) {}
  void write_dm_density(const grid_hierarchy &) {}
  void write_dm_potential(const grid_hierarchy &) {}
  void write_gas_density(const grid_hierarchy &) {}
  void write_gas_potential(const grid_hierarchy &) {}

  void write_dm_position(int coord, const grid_hierarchy &gh) { collect_positions(coord, gh, 0.0, dm_x_[coord]); }
  void write_dm_velocity(int coord, const grid_hierarchy &gh) { collect_velocities(gh, dm_v_[coord]); }

  void write_gas_position(int coord, const grid_hierarchy &gh)
  {
    collect_positions(coord, gh, 0.5 / double(1ul << gh.levelmax()), gas_x_[coord]);
  }
  void write_gas_velocity(int coord, const grid_hierarchy &gh) { collect_velocities(gh, gas_v_[coord]); }

  void finalize()
  {
    const size_t ndm = dm_x_[0].size(), nfine = nlevel_.front(), ngas = baryons_ ? nfine : 0;
    for (int d = 0; d < 3; ++d)
      if (dm_v_[d].size() != ndm || (baryons_ && (gas_x_[d].size() != ndm || gas_v_[d].size() != ndm)))
        throw std::runtime_error("genericio: inconsistent particle counts between fields");

    // coarse particles stand for all matter: weight the gas fields in
    if (baryons_) {
      const double fb = omega_b_ / omega_m_;
      for (int d = 0; d < 3; ++d)
        for (size_t i = nfine; i < ndm; ++i) {
          dm_x_[d][i] = (1 - fb) * dm_x_[d][i] + fb * gas_x_[d][i];
          dm_v_[d][i] = (1 - fb) * dm_v_[d][i] + fb * gas_v_[d][i];
        }
    }

    const size_t n = ndm + ngas;
    std::vector<float> x(n), y(n), z(n), vx(n), vy(n), vz(n), phi(n, 0.0f);
    std::vector<int64_t> id(n);
    std::vector<uint16_t> mask(n, 0);
    std::vector<float> mass, hh, uu, rho, mu, zmet, yhe;
    if (hydro_) {
      mass.resize(n);
      hh.resize(n);
      uu.resize(n, 0.0f);
      rho.resize(n, float(omega_b_ * rho_crit_));
      mu.resize(n, 0.0f);
      zmet.resize(n, 0.0f);
      yhe.resize(n, 0.0f);
    }

    size_t q = 0;
    auto add = [&](const std::vector<double> *pos, const std::vector<double> *vel, size_t i, int64_t pid, int level,
                   double omega, bool gas) {
      x[q] = position(pos[0][i], 0);
      y[q] = position(pos[1][i], 1);
      z[q] = position(pos[2][i], 2);
      vx[q] = float(vel[0][i]);
      vy[q] = float(vel[1][i]);
      vz[q] = float(vel[2][i]);
      id[q] = pid;
      if (gas)
        mask[q] = baryon_mask_;
      if (hydro_) {
        const double spacing = boxlength_ / double(1ul << level);
        mass[q] = float(omega * rho_crit_ * spacing * spacing * spacing);
        hh[q] = float(spacing);
        if (gas) {
          uu[q] = uu_;
          mu[q] = mu_;
        }
      }
      ++q;
    };

    size_t i = 0;
    for (size_t l = 0; l < nlevel_.size(); ++l) {
      const int level = int(levelmax_) - int(l);
      const double omega = (baryons_ && l == 0) ? omega_m_ - omega_b_ : omega_m_;
      for (size_t c = 0; c < nlevel_[l]; ++c, ++i)
        add(dm_x_, dm_v_, i, hydro_ ? 2 * int64_t(i) : int64_t(i), level, omega, false);
    }
    for (size_t k = 0; k < ngas; ++k)
      add(gas_x_, gas_v_, k, 2 * int64_t(k) + 1, int(levelmax_), omega_b_, true);

    {
      mpi_session mpi; // declared before the writer, so the writer is gone before MPI stops
      if (mpi.provided < MPI_THREAD_FUNNELED)
        throw std::runtime_error("genericio: MPI does not provide MPI_THREAD_FUNNELED, which GenericIO's threaded "
                                 "checksums need");
      gio::GenericIO writer(MPI_COMM_SELF, fname_, gio::GenericIO::FileIOPOSIX);
      writer.setPhysOrigin(0.0, -1);
      writer.setPhysScale(boxlength_, -1);
      writer.setNumElems(n);
      writer.addVariable("x", x);
      writer.addVariable("y", y);
      writer.addVariable("z", z);
      writer.addVariable("vx", vx);
      writer.addVariable("vy", vy);
      writer.addVariable("vz", vz);
      writer.addVariable("id", id);
      writer.addVariable("phi", phi);
      writer.addVariable("mask", mask);
      if (hydro_) {
        writer.addVariable("mass", mass);
        writer.addVariable("hh", hh);
        writer.addVariable("uu", uu);
        writer.addVariable("rho", rho);
        writer.addVariable("mu", mu);
        writer.addVariable("zmet", zmet);
        writer.addVariable("yhe", yhe);
      }
      writer.write();
    }
    music::ilog.Print("GenericIO: wrote %zu dark matter and %zu gas particles to %s", ndm, ngas, fname_.c_str());
  }
};

namespace
{
  output_plugin_creator_concrete<genericio_output_plugin> creator("genericio");
}

#endif // ENABLE_GENERICIO

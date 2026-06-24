# TB_Ag_excitons — real-time tight-binding chain dynamics

A Julia code for real-time (time-dependent) simulations of electrons (and
optionally nuclei) on one-dimensional periodic tight-binding chains — written
with dimerized Ag atomic chains / excitons in mind, but generic to any
two-site-per-cell 1D ring.

Given an input file, the code:

1. builds a tight-binding Fock matrix for the chain (distance-dependent
   hopping + nearest-neighbor repulsion),
2. diagonalizes it to get the ground state,
3. propagates the single-particle density matrix in time under an external
   light pulse (vector potential / Peierls phase), optionally with
   electron-electron interaction (a Hartree term and/or a long-range "fxc"
   kernel) and optionally with the nuclei moving self-consistently (Ehrenfest
   dynamics),
4. optionally repeats this for a second chain that is electrostatically
   (Coulomb) coupled to the first.

## Physical model

- **Lattice.** `nchain` dimer unit cells (2 atoms each) arranged on a closed
  ring of `2*nchain` sites, total length `boxl = nchain1 * lattice`. The
  two intra-cell/inter-cell bond lengths are set by the dimerization
  parameter `dimer` (`dimer = 0` → uniform chain).
- **Electronic Hamiltonian.** Tight-binding, one orbital per site. The
  hopping integral is a *linear* function of the instantaneous bond length,
  and a power-law repulsive potential acts between bonded neighbors. Both are
  pre-tabulated over a grid of distances (`construct_potential`,
  [TLS_module.jl:682](td_code/TLS_module.jl#L682)) and looked up by index
  during the dynamics instead of being recomputed every step.
- **Ground state.** The static (rigid-lattice) Hamiltonian is diagonalized
  and the lower half of the bands is filled (closed shell, half filling). If
  the chain is undimerized the two states at the (then degenerate) Fermi
  level are each given occupation 1 instead of 2/0
  (`construct_gs_dens`, [TLS_module.jl:50](td_code/TLS_module.jl#L50)).
- **Time propagation.** The AO-basis density matrix ρ(t) is propagated under
  the von Neumann equation `iħ ρ̇ = [F(t), ρ(t)]`. Three propagator schemes
  are available (`propflag`, see below).
- **Light pulse.** Modeled as a vector potential via Peierls substitution: a
  phase `exp(i·A(t)·d_ij)` is applied to each hopping matrix element, where
  `d_ij` is the (instantaneous) bond length. Several pulse shapes are
  available (`vpot`, see below).
- **Electron-electron interaction (optional).** A mean-field correction is
  added to the diagonal of the Fock matrix, proportional to the deviation of
  the site population from its ground-state value, contracted with a kernel
  matrix (`buildxc`/`Hxc`) made of two independent, additive terms, each
  evaluated for *every* pair of sites (including the same site with itself —
  there's no separate on-site/off-site case):
  - **Hartree**: `1/sqrt(d² + 1/U²)`, strength fixed at 1 and softened by
    `hartreeu` (`U`) so that its on-site (`d=0`) value is exactly `U` — the
    classic chemical-hardness term, here with an explicit `1/d`-decaying tail
    at long range instead of being purely on-site.
  - **fxc**: `fxcalpha/sqrt(d² + fxcgamma²)` — a long-range-corrected
    exchange-correlation kernel, also nonzero on-site (`fxcalpha/fxcgamma` at
    `d=0`).

  Because both terms are evaluated on-site too, they can partially cancel
  there — e.g. with the input-file sign convention (`fxcalpha` is the
  *negative* of the kernel strength), setting `hartreeu = -fxcalpha/fxcgamma`
  makes the on-site/self-charge contribution vanish exactly, leaving only the
  two terms' differently-shaped long-range tails. Turned on whenever
  `hartreeu` and/or `fxcalpha` is non-zero.
  - **Hubbard (optional, `hubbardu`)**: a third, *independent* term,
    `Honsite`, adding `hubbardu*(ρ_jj - q0_jj)` to site `jj`'s diagonal with
    **no coupling to any other site at all** — unlike `hartreeu`, which still
    has a `1/U` spatial softening radius and so couples to neighbors even at
    large `U`. This matters because the mean-field model is prone to a real
    (non-numerical) collapse instability once `fxcalpha`/`hartreeu`/the
    interchain coupling exceed a critical strength relative to distance (see
    **Self-consistent ground state** below): the spatially-extended
    `hartreeu` kernel preferentially damps long-wavelength collapse modes but
    can leave other, more local modes under-damped, whereas a purely on-site
    term damps every mode equally and was confirmed (empirically, by
    checking real-time stability from machine-precision noise with no
    applied field) to produce a genuinely stable moderate-coupling ground
    state where `hartreeu` alone could not. Default `0` (off).
- **Nuclear dynamics (optional, Ehrenfest).** Ions move under velocity
  Verlet using forces derived from the bond-order matrix (off-diagonal ρ),
  the repulsive potential, and (if active) the electron-electron and
  inter-chain force corrections, with an optional velocity-damping term.
- **Two-chain mode (optional).** A second, independent chain (its own
  electrons *and* nuclei) sharing the same box length, displaced by a fixed
  perpendicular distance `rchain`. The chains have **no direct hopping**
  between them — they interact only electrostatically, through a
  `1/sqrt(d² + rchain²)` Coulomb kernel coupling each chain's charge
  fluctuations to the other's Fock matrix and ionic forces
  (`buildintchain`/`Hint`/`build_dint`).

## Repository layout

```
td_code/
  units.jl        - atomic-unit conversion constants (Å, eV, fs → a.u.)
  IOmodule.jl      - Input struct, input-file parser, all file output/restart I/O
  TLS_module.jl    - the physics: Hamiltonian, propagators, pulses, forces, MD
  run_td.jl        - main driver script (reads input, runs the simulation)
example/
  inp.in           - example input (64-pair single chain)
  output.out       - reference output for that run
  traj.lammpstrj   - reference trajectory
  restart.dat      - reference restart checkpoint
example_double_chain/
  inp.in           - example input (two-chain mode, 64- and 62-atom chains)
  output.out, traj.lammpstrj, restart.dat - reference output for that run
```

`run_td.jl` is the entry point; `TLS_module.jl` includes `IOmodule.jl`, which
includes `units.jl`, so all definitions are available after
`include("TLS_module.jl")` alone (the extra `include`s at the top of
`run_td.jl` are redundant but harmless).

## Workflow (what `run_td.jl` actually does)

1. **Read input.** `read_input_file("inp.in")` parses `key = value` lines into
   an `Input` struct, converting eV/Å/fs quantities to atomic units as it
   goes (see Units below). `write_header` writes the run summary to
   `output.out` and truncates `traj.lammpstrj`.
2. **Build the system(s).**
   - `construct_potential` tabulates hop(r) and repulsion(r) (+ derivatives).
   - If `mingeom = 1` (and `restart = 0`), `minimize_dimer` first relaxes
     `dimer1` (and, in two-chain mode, `dimer2`) by steepest descent to the
     dimerization that minimizes the total energy at the given `lattice`,
     before anything below is built from it — see **Geometry optimization**
     below.
   - `construct_aob_hamiltonian` builds the static (equilibrium-geometry)
     tight-binding Hamiltonian in the atomic-orbital (site) basis.
   - It is diagonalized (`eigen`) to get eigenvalues/eigenvectors, which fix
     the ground-state density matrix (`construct_gs_dens`) and initial ion
     positions (`construct_rion`, zero velocity) unless `restart = 1`, in
     which case ρ, ion positions and velocities are read from
     `restart.dat`.
   - In two-chain mode this is done twice (chain 1 and chain 2), with chain
     2's geometry rescaled by `nchain1/nchain2` so both rings share `boxl`.
   - If `scfgs = 1` (and `restart = 0`), `scf_ground_state` then relaxes this
     density self-consistently under the chain(s)' own Hxc kernel (and, in
     two-chain mode, the interchain `Hint` term) before propagation starts —
     see **Self-consistent ground state** below.
3. **Propagate.** For each of `steps = floor(time/tstep)` steps,
   `propagate(...)` ([TLS_module.jl:487](td_code/TLS_module.jl#L487) for one
   chain, [:562](td_code/TLS_module.jl#L562) for two):
   - rebuilds the off-diagonal Fock matrix from the current ion positions
     (`construct_rdep_hamiltonian`),
   - applies the Peierls phase for the configured pulse type (`vpot`),
   - adds the Hartree/fxc mean-field correction if active (`Hxc`), and, in
     two-chain mode, the inter-chain Coulomb correction (`Hint`),
   - if `ehrenfest = 1`, advances the nuclei one velocity-Verlet step
     (`velverlet`) using forces from `build_fion`/`build_dint`; with
     `ehrenfest = 0` (default) the ions stay frozen at their initial
     positions and only the electronic dynamics evolves. In that case the
     Hartree/fxc interaction matrix (`buildxc`/`buildintchain`) is also
     position-dependent only through the (now-fixed) ion positions, so
     `run_td.jl` computes it once before the time loop and passes it into
     every `propagate` call via the `xc_cache` keyword instead of rebuilding
     it — by far the dominant per-step cost otherwise — from scratch every
     step,
   - advances ρ by one step using the scheme selected by `propflag`:
     - `0` ("RC"): a single explicit midpoint (2nd-order Runge-Kutta-like)
       step with the Fock matrix frozen over the step,
     - `1` ("RCsc"): predictor-corrector — ρ is pushed a half step, the Fock
       matrix (and pulse/interaction terms) is rebuilt from that
       intermediate ρ, then the full step is taken with the updated field,
     - `2` ("UnitProp"): exact unitary propagation, `ρ(t+dt) = e^{-iFdt} ρ(t) e^{iFdt}`,
       with F evaluated as in the `RCsc` path.
   - Every `savefreq` steps: electronic energy, nuclear energy
     (`ionenergy`), and bond current are appended to in-memory buffers, and a
     trajectory frame is appended to `traj.lammpstrj`.
   - Every `savefreq*bufflen` steps: the buffers are flushed to `output.out`
     and a restart checkpoint is written to `restart.dat`.
4. Any remaining buffered data is flushed at the end of the run.

## Units

Internally everything is atomic units (`ħ = mₑ = e = 1`; see the constants
at the top of [TLS_module.jl](td_code/TLS_module.jl)). The **input file** is
written in "lab" units (eV, Å, fs) and converted on read using the factors in
[units.jl](td_code/units.jl).

All **user-facing output** is converted back to lab units for readability:
the `output.out` header, eigenvalues, and time-series body (time in fs,
energies in eV, current in e/fs — see column units in the file itself), and
the ion positions/box bounds in `traj.lammpstrj` (Å).

`restart.dat` is the one exception: it is a purely internal checkpoint (only
this program reads it back) and is kept in raw atomic units, exactly mirroring
the in-memory ρ/`rion`/`vion` arrays.

## Requirements

Julia 1.10 (tested). Packages used: `LinearAlgebra`, `SparseArrays`,
`DelimitedFiles`, `Printf` (all standard library — no
`Project.toml`/`Manifest.toml` is checked in, but nothing beyond the
standard library needs installing).

## Running

```bash
cd example/                 # or example_double_chain/, or any directory containing an inp.in
julia /path/to/td_code/run_td.jl
```

The script always reads `inp.in` and writes `output.out`, `traj.lammpstrj`,
and `restart.dat` **in the current working directory**, so run it from inside
the directory holding your input file (copy `run_td.jl` there, or invoke it
by full path as above).

To restart from a checkpoint, set `restart = 1` in `inp.in` — `rion`/`vion`/ρ
are then read from `restart.dat` instead of being (re)initialized.

## Input file reference (`inp.in`)

One `key = value` line per parameter (anything after the value on the line is
ignored, so trailing `# comment` works). Unrecognized keys are silently
ignored; omitted keys fall back to the defaults below. Columns marked
*(a.u.)* are **not** converted — they must be supplied directly in atomic
units, unlike everything else.

| Key | Meaning | Input units | Default |
|---|---|---|---|
| `chains` | number of chains: `1` or `2` | – | required |
| `nchain1` | number of dimer pairs in chain 1 (`2*nchain1` sites) | – | required |
| `nchain2` | number of dimer pairs in chain 2 (two-chain mode only) | – | required |
| `hop` | reference hopping integral at `req` | eV | -0.0246 |
| `hopslope` | slope `d(hop)/dr` of the hopping-vs-distance relation | eV/Å | 0.371 |
| `req` | equilibrium nearest-neighbor distance | Å | 4.922 |
| `lattice` | unit-cell length (`boxl = nchain1*lattice`) | Å | 11.338 |
| `dimer1` | dimerization of chain 1 (0 = uniform chain) | fraction | 0 |
| `dimer2` | dimerization of chain 2 | fraction | 0 |
| `rchain` | perpendicular distance between the two chains | Å | 10 |
| `hartreeu` | Hartree term strength (`U`); also its on-site/chemical-hardness value | eV | 0.115 |
| `hubbardu` | strength of a purely on-site (no spatial extent) Hubbard-like repulsion, independent of `hartreeu` | eV | 0 |
| `fxcalpha` | fxc kernel strength (input convention: negative of the kernel's physical strength) | eV | -0.00735 |
| `fxcgamma` | softening radius of the fxc kernel | Å | 0.492 |
| `p` | exponent of the repulsive potential | – | 15 |
| `pref` | prefactor of the repulsive potential (relative to `hop`) | – | 0.231 |
| `atmass` | mass of each atom | **a.u.** (electron masses) | 198046 |
| `damp` | `0` = no damping, `1` = velocity damping of the ions | – | 0 |
| `dampforce` | damping (friction) coefficient, used if `damp=1` | **a.u.** | 0.1 |
| `ehrenfest` | `0` = frozen ions (electronic dynamics only), `1` = Ehrenfest dynamics (ions move via velocity Verlet) | – | 0 |
| `mingeom` | `0` = use `dimer1`/`dimer2` as given, `1` = relax them by steepest descent before propagation (ignored if `restart=1`) — see **Geometry optimization** | – | 0 |
| `mingeomtol` | convergence threshold on the residual dimerizing force (per cell) | eV/Å | 1e-5 |
| `mingeomstep` | initial steepest-descent step size, refined every iteration by backtracking line search | **a.u.** | 50.0 |
| `mingeomiter` | maximum number of steepest-descent iterations | – | 2000 |
| `scfgs` | `0` = use the bare/decoupled chain ground state as the dynamics' starting point, `1` = relax it self-consistently under Hxc/Hint before propagation (ignored if `restart=1`) — see **Self-consistent ground state** | – | 0 |
| `scfmix` | linear mixing fraction for the SCF density update (0-1; lower if it fails to converge at strong coupling) | – | 0.3 |
| `scftol` | convergence threshold on the max per-site population change between SCF iterations | – | 1e-7 |
| `scfiter` | maximum number of SCF iterations | – | 500 |
| `scfnoise` | amplitude of the random on-site symmetry-breaking potential applied for the first `scfnoiseiter` iterations | eV | 1e-3 |
| `scfnoiseiter` | number of initial SCF iterations over which the symmetry-breaking noise is applied | – | 5 |
| `time` | total propagation time | fs | 0.005 |
| `tstep` | integration time step | fs | 0.005 |
| `savefreq` | save a data point/trajectory frame every N steps | steps | 100 |
| `bufflen` | flush buffers to disk every `savefreq*bufflen` steps | steps | 100 |
| `restart` | `0` = fresh start, `1` = read `restart.dat` | – | 0 |
| `propflag` | propagator: `0`=RC, `1`=RCsc, `2`=UnitProp (see Workflow) | – | 0 |
| `vpot` | pulse shape: `0`=none, `1`=step, `2`=Gaussian, `3`=laser, `4`=ramp | – | 0 |
| `vamp` | amplitude of the driving electric field | eV/Å | 0 |
| `vtime` | duration of the pulse / turn-on time | fs | 0.005 |
| `vfreq` | photon energy (ħω) of the laser, only used if `vpot=3` | eV | 1.0 |

`steps` is derived automatically as `floor(time/tstep)`.

## Geometry optimization (`mingeom`)

For a fixed `lattice`, the dimerization that minimizes the total energy
(electronic + repulsive) isn't generally known in advance — e.g. before
starting an Ehrenfest run from the relaxed geometry. Setting `mingeom = 1`
finds it automatically: `minimize_dimer`
([TLS_module.jl](td_code/TLS_module.jl)) runs steepest descent (with a
backtracking line search) on the single dimerization coordinate, reusing
`construct_aob_hamiltonian`/`construct_gs_dens` to get the exact ground state
at each trial geometry, before the rest of the setup (Hamiltonian, ground
state, initial `rion`) is built from the result. `dimer1` (and `dimer2`, in
two-chain mode) is overwritten with the optimized value, so everything
downstream — including the run summary and `output.out`'s eigenvalues — is
fully consistent with the relaxed geometry. A short convergence report
(iteration count, optimized dimerization, residual force) is printed to the
console and appended to `output.out`.

Two implementation notes:
- The optimizer evaluates hop(r)/repulsion(r) from their exact continuous
  formulas rather than `pottable`'s 10000-point grid. Close to the minimum,
  the dimerizing force is a small difference between two much larger,
  nearly-equal bond terms, and `pottable`'s discretization noise (negligible
  for MD forces) is large enough relative to that small difference to stall
  convergence — confirmed numerically when developing this feature.
- A perfectly uniform starting chain (`dimer1 = 0`) sits exactly at an
  unstable, symmetry-protected zero-force point (true for any half-filled
  bipartite ring, a basic feature of the Peierls instability), so
  `minimize_dimer` nudges it off-center before the first step. Because the
  ring has no preferred handedness, the optimized `dimer1`'s **sign** is then
  arbitrary (both signs are equally valid, degenerate ground states); only
  its magnitude is physically meaningful.

## Self-consistent ground state (`scfgs`)

By default the dynamics starts from the ground state of each chain's *bare*,
isolated, non-interacting Hamiltonian — the Hxc/Hint mean-field terms are
only switched on once propagation begins. This is fine when the interaction
is weak (the bare state is a good approximation to the true interacting
ground state, and propagating from it just produces small RPA-like
oscillations — the expected excitonic response). But above a critical
coupling strength (set by `fxcalpha`/`hartreeu` and, in two-chain mode, by
`1/rchain` through the always-on `Hint` term), the bare state stops being
even a local minimum of the true self-consistent mean-field energy — it
becomes an unstable saddle point, and propagating from it diverges to NaN in
finite time. This is a real instability of the underlying equations, not a
numerical-integration artifact: it happens with the exact unitary propagator
just as with the explicit ones, persists with no applied field at all (pure
floating-point round-off is enough to seed an exponential, textbook-clean
e^(Γt) growth), and the growth rate is essentially independent of `tstep`.

Setting `scfgs = 1` relaxes the actual self-consistent (interacting) ground
state before propagation: `scf_ground_state` ([TLS_module.jl](td_code/TLS_module.jl))
iterates "build the interacting Fock matrix from the current trial
densities → diagonalize → refill the lowest half-band → linearly mix with
the previous trial" to convergence, overwriting the chains' starting density
with the result (`gspop`, the *reference* density used inside Hxc/Hint's
charge-fluctuation kernel, is left untouched — only the propagated starting
state changes). A small random on-site potential is injected for the first
`scfnoiseiter` iterations to break the exact symmetry that would otherwise
pin the iteration at the unstable saddle (the bare state itself), the same
role as `mingeom`'s off-center nudge.

Three things worth knowing if you hit this:
- Above threshold, the self-consistent solution the SCF actually finds can
  be a **full charge-domain collapse** (site populations saturating at the
  model's hard `[0,2]` bound, splitting each chain into two macroscopic
  domains) rather than a moderate, exciton-like state — this genuinely is a
  stable fixed point (propagates cleanly), just not a useful one. Whether a
  moderate minimum exists at all depends on the stabilizing terms below.
- `hartreeu` alone is not a reliable fix for the collapse, even though it is
  nominally a repulsive/stabilizing term: because its kernel has a `1/U`
  spatial extent (so it still couples to neighboring sites even at large
  `U`), increasing it can simply move the SCF to a *different* unstable
  stationary point rather than a genuinely stable one — confirmed by
  checking real-time stability with no applied field, not just static SCF
  self-consistency (which only guarantees a stationary point, not a stable
  one).
- A purely on-site `hubbardu` term (see **Electron-electron interaction**
  above) was confirmed to genuinely stabilize a moderate ground state once
  strong enough (order a few eV in the cases tested), including at the
  shortest interchain distances tested. There is a real critical threshold
  below which it still fails, consistent with genuine stabilization rather
  than masking the issue.
- The instability threshold (in terms of physical `fxcalpha`/`rchain`) was
  checked to be essentially chain-length-independent over an 8x range in
  `nchain` (confirmed both empirically and via the leading eigenvalue of the
  linearized self-consistency map) — it is governed by the kernel's
  near-field behavior, not by how the long-range tail is summed over the
  ring.

## Output files

- **`output.out`** — run summary header (parameters in eV/Å/fs, then the
  static eigenvalue spectrum in eV), followed by a time-series table:
  - single chain: `time(fs)  electronic_energy(eV)  nuclear_energy(eV)  current(e/fs)`
  - two chains: `time(fs)  electronic_energy1(eV)  electronic_energy2(eV)  nuclear_energy1(eV)  nuclear_energy2(eV)  current1(e/fs)  current2(e/fs)`

  `current` is the bond current evaluated on the first bond of the chain
  (`2·Im(ρ₁₂F₂₁)`), used as a representative value.

- **`traj.lammpstrj`** — ion trajectory in LAMMPS dump format (viewable in
  OVITO/VMD), positions and box bounds in Å. One atom per site, `x` =
  position along the chain (`y=z=0`, chain 2 in two-chain mode is offset in
  `y` by `rchain` purely for visual separation). The `vx` velocity field is
  **repurposed** to carry the site-population deviation from its
  ground-state value, offset by `+2.0` (so values plot centered around 2
  rather than 0); `vy`,`vz` are unused.

- **`restart.dat`** — binary dump (via `Serialization.serialize`) of the
  complex AO-basis density matrix followed by the ion position and velocity
  arrays; for two chains, chain 1's block is followed by chain 2's block in
  the same layout. Read back with `read_restart`. Binary rather than a
  comma-delimited text dump because the text format scaled badly: at ~4000
  sites a single checkpoint took seconds to write and tens of seconds to
  read back, and produced a ~700MB file for what is ~270MB of actual data.

## Known issues

Found while reading/exercising the code — useful to know before relying on
results or extending it further:

- ~~Two-chain mode (`chains = 2`) does not run~~ — **fixed**, see
  [example_double_chain/](example_double_chain/). Three bugs, all in
  `run_td.jl`: (1) `construct_gs_dens` calls for both chains
  ([line 52](td_code/run_td.jl#L52), [line 65](td_code/run_td.jl#L65)) were
  missing the required `dimer` argument — `MethodError` on the first run; (2)
  the call to `propagate` ([line 140](td_code/run_td.jl#L140)) passed
  `fockaob1` for *both* chains' static-Hamiltonian argument instead of
  `fockaob1, fockaob2` — harmless when `nchain1 == nchain2` but a confirmed
  `BoundsError` otherwise (verified by reproducing it with `nchain2 >
  nchain1`, then confirming the fix resolves it); (3) the leftover-buffer
  flush at the end of a run ([line 170](td_code/run_td.jl#L170)) called the
  3-column `write_buffer` instead of the 7-column one used everywhere else,
  silently dropping nuclear-energy/current data (and corrupting the column
  count) for any run whose last partial buffer wasn't empty — the now-unused
  3-column overload was removed from `IOmodule.jl`. The new example was
  verified end-to-end: atom/box counts, chain-2's `rchain` offset and
  rescaled bond spacing, and the nuclear energy's scaling with system size
  all check out numerically.
- ~~The long-range fxc kernel had no effect, on electrons or on ions~~ —
  **fixed**, and the e-e interaction model rewritten to match the project's
  reference write-up (`Excitons in closed chains`, eqs. 5-7). `buildxc`
  ([TLS_module.jl:312](td_code/TLS_module.jl#L312)) originally wrapped *both*
  the on-site Hubbard assignment and the off-diagonal long-range-kernel/force
  formulas inside a single `if ii==jj`, where the bond vector `dij` is
  identically `0` — so the off-diagonal term never executed, `fxcalpha` never
  affected anything, and the force-derivative matrix was always zero. The
  `hubbard` parameter has since been retired (it overlapped with the other
  two terms); `buildxc` now implements exactly the two terms from the report
  for every site pair, including the self (`d=0`) term:
  `1/sqrt(d²+1/U²) + fxcalpha/sqrt(d²+fxcgamma²)`, with `U = hartreeu`. At
  `d=0` this gives `U + fxcalpha/fxcgamma` per site — reproducing the report's
  "self-charge cancellation when `U = α/γ`" result (verified numerically:
  setting `hartreeu = -fxcalpha/fxcgamma` makes the on-site contribution
  exactly `0`). Also verified: pure-Hartree (`fxcalpha=0`) gives exactly `U`
  on-site with a smooth `1/d`-decaying tail; pure-fxc (`hartreeu=0`) matches
  the previously-verified fxc-only kernel and stays NaN/Inf-free.
- ~~`propflag` was silently ignored~~ — **fixed.** `read_input_file`
  ([IOmodule.jl](td_code/IOmodule.jl)) had no parsing branch for the
  `propflag` key, so `input.propflag` always stayed at its struct default
  (`0`, "RC") no matter what the input file said — confirmed by parsing the
  example's `inp.in` (which sets `propflag = 1`) and getting `0` back. On top
  of that, `propflag = 1`'s code path referenced a bare `tstep` instead of
  `input.tstep` ([TLS_module.jl:557](td_code/TLS_module.jl#L557) and the
  two-chain equivalent at lines 670-671) — unreachable before, so this never
  surfaced, but would have crashed with `UndefVarError` the moment the
  parsing bug was fixed. Both are fixed now; verified `propflag` 0/1/2 all
  run without error and that 1 ("RCsc") and 2 ("UnitProp") agree closely with
  each other while differing slightly from 0 ("RC"), as expected for
  genuinely different integrators.
- ~~`VPgaus` (`vpot=2`) had its pulse width scaling backwards~~ — **fixed.**
  `exp(-1 * (t - stop/2)^2 / 2 * (stop/10)^2)`
  ([TLS_module.jl:110](td_code/TLS_module.jl#L110)) — because `/` and `*`
  have equal precedence in Julia — evaluated as
  `exp( (-(t-stop/2)^2/2) * (stop/10)^2 )`, giving a Gaussian with `σ =
  10/stop` instead of the intended `σ = stop/10` (matching `VPstep`'s
  convention of a width proportional to its duration parameter). Fixed by
  parenthesizing the denominator: `exp(-1 * (t-stop/2)^2 / (2*(stop/10)^2))`.
  Verified by fitting the actual curve's width before/after the fix (matched
  `10/stop` before, `stop/10` after, at several `stop` values) and by running
  a full simulation with `vpot=2`: the resulting current now peaks
  symmetrically at exactly `t = vtime/2`, as expected for a correctly-shaped
  pulse.
- ~~`vpot = 4` was inconsistently wired up~~ — **fixed.** `VPconst` (a
  constant field with a smooth turn-on) was reachable only from the
  `propflag ∈ {1,2}` code path — under `propflag=0` it was silently a no-op
  (confirmed: before the fix, a `vpot=4` run under `propflag=0` produced
  exactly zero current; after, it produces the same growing current as
  `propflag=1`/`2`). Also missing: the "after the pulse window" continuation
  branch in 3 of the 4 vector-potential blocks (only the
  `propflag ∈ {1,2}`/single-chain one had it), a mention in the input-file
  comments, and a case in `write_header`'s run summary. All four are fixed
  now and verified consistent across `propflag` 0/1/2.
- ~~The hopping-vs-distance slope was hard-coded~~ — **fixed.** The constant
  `0.007215487659` (a.u./bohr) inside `construct_potential` is now the input
  parameter `hopslope` (eV/Å, default unchanged — verified the existing
  examples produce bit-for-bit identical output when `hopslope` is omitted).
  The other hard-coded value mentioned previously was inside `Hcharge2`,
  which has since been deleted (see below).
- ~~Several functions looked unfinished/experimental and unreachable from
  the real workflow~~ — **removed.** Confirmed-unused (never called, by
  grepping every call site): `get_gs`, `Hcharge`, `Hcharge2`,
  `kickit`/`kickit_aob`/`kickit_str`, `Erandom`, `Eblackbody`,
  `BlackBodyPlanck`/`BlackBodyRash`, `Pop2Temp`, the two-argument `Propagate`
  wrappers, plus three superseded overloads sharing a name with a live
  function (the 5-arg dipole-based `buildxc`, the 1-arg `buildintchain`, the
  4-arg `build_fion`) and `write_pbuffer` in `IOmodule.jl`. Also dropped
  `using Plots` and `using Statistics` from `run_td.jl` (`Statistics` turned
  out to be just as unused as `Plots` — no `mean`/`std`/etc. call anywhere).
  Verified both examples produce bit-for-bit identical output after all of
  this was removed.

All issues found during this review are now fixed; this section is kept as
a record of what changed and how each fix was verified.

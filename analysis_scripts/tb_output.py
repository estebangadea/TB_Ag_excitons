"""Shared parsing helpers for td_code's output.out files.

output.out mixes a free-form text header (system parameters, optional
geometry-optimization log, one 'Eigenvalues:' block per chain) with a
fixed-width time-series table. All readers below locate sections by
content (the 'Eigenvalues:' marker, the 'time(fs)' column header) rather
than by a fixed line count, so they keep working regardless of header
length (e.g. mingeom logging, double-chain runs).
"""
import re

import numpy as np

# Same atomic-unit conversion factors as td_code/units.jl.
ANGSTROM = 1.0 / 0.529177210903
ELECTRONVOLT = 1.0 / 27.211386245988
FEMTOSECOND = 1.0 / 0.024188843265857


def _read_lines(filename):
    with open(filename) as f:
        return f.readlines()


def find_table_start(lines):
    """Line index of the time-series column-header row ('time(fs) ...')."""
    for i, line in enumerate(lines):
        if line.strip().startswith('time(fs)'):
            return i
    raise ValueError("could not find a 'time(fs)' header row in the output file")


def load_timeseries(filename):
    """Load the time-dependent observable table as a dict of name -> array."""
    lines = _read_lines(filename)
    header_idx = find_table_start(lines)
    names = lines[header_idx].split()
    data = np.loadtxt(filename, skiprows=header_idx + 1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {name: data[:, i] for i, name in enumerate(names)}


def load_eigenvalue_blocks(filename):
    """List of eigenvalue arrays, one per 'Eigenvalues:' block, in file order
    (chain 1 then chain 2 for a double-chain run)."""
    lines = _read_lines(filename)
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == 'Eigenvalues:':
            i += 1
            vals = []
            while i < len(lines) and lines[i].strip() != '':
                vals.append(float(lines[i]))
                i += 1
            blocks.append(np.array(vals))
        else:
            i += 1
    return blocks


def _grep_float(lines, pattern):
    rx = re.compile(pattern)
    for line in lines:
        m = rx.search(line)
        if m:
            return float(m.group(1))
    return None


def load_vpot_params(filename):
    """Parse the vector-potential block written by IOmodule.jl's write_header.

    Returns a dict: kind in {'none','step','pulse','laser','ramp'}, amp
    (eV/AA), duration (fs), freq (eV, laser only)."""
    lines = _read_lines(filename)
    head = lines[:find_table_start(lines)]

    kind = 'none'
    for line in head:
        if 'Vector potential step' in line:
            kind = 'step'
        elif 'Vector potential pulse' in line:
            kind = 'pulse'
        elif 'Vector potential laser pulse' in line:
            kind = 'laser'
        elif 'Vector potential ramp' in line:
            kind = 'ramp'

    return {
        'kind': kind,
        'amp': _grep_float(head, r'amplitud\s*=\s*([-\d.eE+]+)') or 0.0,
        'duration': _grep_float(head, r'duration\s*=\s*([-\d.eE+]+)') or 0.0,
        'freq': _grep_float(head, r'frequency\s*=\s*([-\d.eE+]+)') or 0.0,
    }


def vector_potential(t_fs, params):
    """Evaluate A(t) (in eV/AA, same scale as the 'amplitud' input) at times
    t_fs (fs), replicating VPstep/VPgaus/VPlaser/VPconst from TLS_module.jl.

    t_fs and the header's 'duration'/'amplitud'/'frequency' are converted to
    atomic units (matching read_input_file's unit handling) before evaluating
    the formulas, then the result is converted back to eV/AA.
    """
    t_fs = np.asarray(t_fs, dtype=float)
    kind = params['kind']
    if kind == 'none':
        return np.zeros_like(t_fs)

    ti = t_fs * FEMTOSECOND
    stop = params['duration'] * FEMTOSECOND
    height = params['amp'] * ELECTRONVOLT / ANGSTROM

    if kind == 'step':
        A = np.where(
            ti < stop,
            height / stop * (stop / (2 * np.pi) * np.sin(ti * 2 * np.pi / stop + np.pi) + ti),
            height,
        )
    elif kind == 'pulse':
        # propagate()'s "ii < vtime/tstep" branch is the only place vpot==2 is
        # handled; once ti >= stop no branch re-applies the Peierls phase, so
        # the field actually seen by the Hamiltonian drops to zero (not the
        # raw VPgaus()'s ti>=stop value, which is never used in that case).
        A = np.where(
            ti < stop,
            height * np.exp(-(ti - stop / 2.0) ** 2 / (2 * (stop / 10.0) ** 2)),
            0.0,
        )
    elif kind == 'ramp':
        ramp_on = height / stop * (stop ** 2 / (4 * np.pi ** 2) * (np.cos(2 * np.pi * ti / stop) - 1) + ti ** 2 / 2)
        A = np.where(ti < stop, ramp_on, height * ti - height * stop / 2)
    elif kind == 'laser':
        freq = params['freq'] * ELECTRONVOLT
        k1 = -2 / freq * np.cos(stop * freq / -2 + freq * ti)
        k2 = stop / (-2 * np.pi + stop * freq) * np.cos(stop * freq / -2 - 2 * np.pi * ti / stop + freq * ti)
        k3 = stop / (2 * np.pi + stop * freq) * np.cos(stop * freq / -2 + 2 * np.pi * ti / stop + freq * ti)
        k4 = 4 * np.pi ** 2 * np.cos(stop * freq / 2) / (stop ** 2 * freq ** 3 - 4 * np.pi ** 2 * freq)
        A_on = height * (0.5 * (k1 + k2 + k3) - k4) + ti * 0.0005
        A_off = ti * 0.0005
        A = np.where(ti < stop, A_on, A_off)
    else:
        raise ValueError(f"unknown vector potential kind: {kind}")

    return A * ANGSTROM / ELECTRONVOLT

#!/usr/bin/env python3
"""Absorption spectrum from the bond current(s) logged in a td_code
output.out file: FFT of the mean-subtracted current (report eq. 3; for a
double-chain run, current1+current2, matching the report's sec. 2 "the
polarization of the system is obtained from the sum of the bond currents of
both chains"). The single-particle band gap (from the same file's
'Eigenvalues:' block) is shaded for reference, as in report Figs. 2-4.

Usage:
    python3 plot_spectrum.py [output.out] [--fmax 5.0] [--save fig.png]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tb_output import load_timeseries, load_eigenvalue_blocks, ELECTRONVOLT, FEMTOSECOND

# hbar in eV*fs = (eV/Hartree) * (fs/a.u. time), so that
# E(eV) = hbar_eVfs * omega(rad/fs) for a frequency measured off a time axis in fs.
HBAR_EVFS = (1.0 / ELECTRONVOLT) * (1.0 / FEMTOSECOND)


def band_gap_eV(filename):
    """Smallest |eigenvalue| across all chains in the file (half the
    HOMO-LUMO gap), or None if no 'Eigenvalues:' block is present."""
    blocks = load_eigenvalue_blocks(filename)
    if not blocks:
        return None
    return min(np.min(np.abs(levels)) for levels in blocks)


def spectrum(current, dt):
    """FFT magnitude spectrum (energy in eV, intensity) of a current trace,
    after subtracting its mean (report eq. 3)."""
    current = current - np.mean(current)
    y_fft = np.fft.fft(current)
    freq = np.fft.fftfreq(len(current), dt)  # cycles / fs
    positive = freq >= 0
    energy = 2 * np.pi * HBAR_EVFS * freq[positive]
    intensity = np.abs(y_fft[positive])
    return energy, intensity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', default='output.out', help='path to output.out')
    parser.add_argument('--fmax', type=float, default=2.2, help='max frequency to plot, in eV (default 5.0)')
    parser.add_argument('--no-gap', action='store_true', help="don't shade the single-particle band gap")
    parser.add_argument('--no-per-chain', action='store_true',
                         help="for a double-chain run, don't also plot each chain's own spectrum")
    parser.add_argument('--offset', type=float, default=0.15,
                         help='vertical offset between stacked per-chain curves, as a fraction '
                              "of the total spectrum's peak intensity (default 0.15)")
    parser.add_argument('--save', help='save the figure to this path instead of showing it')
    args = parser.parse_args()

    cols = load_timeseries(args.output)
    t = cols['time(fs)']
    dt = t[1] - t[0]

    current_cols = [name for name in cols if name.startswith('current')]
    total_current = sum(cols[name] for name in current_cols)
    energy, intensity = spectrum(total_current, dt)

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=180)

    per_chain = len(current_cols) > 1 and not args.no_per_chain
    ax.plot(energy, intensity, color='C0', lw=1.5, label='total' if per_chain else None)

    if per_chain:
        step = args.offset * intensity.max()
        for i, name in enumerate(current_cols, start=1):
            e_i, inten_i = spectrum(cols[name], dt)
            ax.plot(e_i, inten_i - i * step, color=f'C{i}', lw=1, alpha=0.8, label=f'chain {i}')
        ax.legend(fontsize=7, frameon=False)

    ax.set_xlim(0.0, args.fmax)
    ax.set_xlabel('frequency (eV)')
    ax.set_ylabel('intensity')

    if not args.no_gap:
        gap = band_gap_eV(args.output)
        if gap is not None:
            ax.axvspan(0.0, 2 * gap, facecolor='k', alpha=0.2)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save)
    else:
        plt.show()


if __name__ == '__main__':
    main()

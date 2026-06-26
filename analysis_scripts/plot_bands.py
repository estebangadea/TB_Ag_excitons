#!/usr/bin/env python3
"""Reconstruct the band structure from the 'Eigenvalues:' block(s) of a
td_code output.out file (one block per chain).

The ring Hamiltonian is solved in real space (report sec. 1), so the
eigenvalues come out sorted by energy with no k-label attached. But a
closed, dimerized nc-cell chain is exactly the folded two-band SSH model:
every interior k is doubly degenerate (+-k), E(k) is extremal at k=0 and at
the reduced-zone edge k=+-pi/a, and nothing else. That fixes the k assigned
to each eigenvalue: walking the sorted valence levels away from the band
minimum walks k away from 0 out to the zone edge (one at a time, then in
+-k pairs); the conduction branch is the same walk in reverse since it
peaks at k=0 instead. See report sec. 1 ("We can recover the band structure
by assigning the eigenstate to its corresponding k-point").

Usage:
    python3 plot_bands.py [output.out] [--save fig.png]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tb_output import load_eigenvalue_blocks


def half_band(levels_from_zone_center):
    """levels_from_zone_center: nc eigenvalues of one branch (valence or
    conduction), ordered so index 0 sits at k=0 and the energy moves
    monotonically out to the zone edge. Returns (k, E) for k in [0, 1]
    (units of pi/a), one point per distinct |k|, averaging each (nearly
    exact) +-k degenerate pair down to a single value."""
    nc = len(levels_from_zone_center)
    dk = 2.0 / nc
    k = [0.0]
    e = [levels_from_zone_center[0]]
    i = 1
    m = 1
    while i < nc:
        if i + 1 < nc:
            k.append(m * dk)
            e.append(0.5 * (levels_from_zone_center[i] + levels_from_zone_center[i + 1]))
            i += 2
        else:
            k.append(1.0)  # nc even: lone Brillouin-zone edge point, k = pi/a
            e.append(levels_from_zone_center[i])
            i += 1
        m += 1
    return np.array(k), np.array(e)


def full_band(k_half, e_half):
    """Mirror a [0, 1] half-band out to the full [-1, 1] zone."""
    k_full = np.concatenate((-k_half[::-1], k_half[1:]))
    e_full = np.concatenate((e_half[::-1], e_half[1:]))
    return k_full, e_full


def reconstruct_bands(levels):
    """levels: all 2*nc eigenvalues of one chain. Returns (kv, ev), (kc, ec)
    for the valence and conduction bands over the full reduced zone."""
    levels = np.asarray(levels)
    nc = len(levels) // 2
    valence = np.sort(levels[:nc])           # ascending energy = increasing |k|
    conduction = np.sort(levels[nc:])[::-1]  # descending energy = increasing |k|

    kv, ev = full_band(*half_band(valence))
    kc, ec = full_band(*half_band(conduction))
    return (kv, ev), (kc, ec)


def plot_one_chain(ax, levels, title=None):
    (kv, ev), (kc, ec) = reconstruct_bands(levels)
    ax.plot(kv, ev, '.-', color='C0')
    ax.plot(kc, ec, '.-', color='C1')
    ax.set_xlim(-1, 1)
    ax.set_xlabel('k (π/a)')
    ax.set_ylabel('energy (eV)')
    if title:
        ax.set_title(title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', default='output.out', help='path to output.out')
    parser.add_argument('--save', help='save the figure to this path instead of showing it')
    args = parser.parse_args()

    blocks = load_eigenvalue_blocks(args.output)
    if not blocks:
        raise SystemExit(f"no 'Eigenvalues:' block found in {args.output}")

    fig, axes = plt.subplots(1, len(blocks), figsize=(4.5 * len(blocks), 4), squeeze=False)
    titles = ['chain 1', 'chain 2'] if len(blocks) > 1 else [None]
    for ax, levels, title in zip(axes[0], blocks, titles):
        plot_one_chain(ax, levels, title=title)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=180)
    else:
        plt.show()


if __name__ == '__main__':
    main()

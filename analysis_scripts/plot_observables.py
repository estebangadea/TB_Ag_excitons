#!/usr/bin/env python3
"""Plot the time-dependent observables logged in a td_code output.out file:
vector potential, electronic energy, nuclear energy and bond current.
Works for both single-chain and double-chain runs.

Usage:
    python3 plot_observables.py [output.out] [--save fig.png]
"""
import argparse

import matplotlib.pyplot as plt

from tb_output import load_timeseries, load_vpot_params, vector_potential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', default='output.out', help='path to output.out')
    parser.add_argument('--save', help='save the figure to this path instead of showing it')
    args = parser.parse_args()

    cols = load_timeseries(args.output)
    vpot = load_vpot_params(args.output)
    t = cols['time(fs)']
    double_chain = 'current2(e/fs)' in cols

    fig, axes = plt.subplots(4, 1, figsize=(6, 9), sharex=True)

    axes[0].plot(t, vector_potential(t, vpot), color='k')
    axes[0].set_ylabel('A(t) (eV/Å)')

    if double_chain:
        for suffix, label in (('1', 'chain 1'), ('2', 'chain 2')):
            axes[1].plot(t, cols[f'electronic_energy{suffix}(eV)'], label=label)
            axes[2].plot(t, cols[f'nuclear_energy{suffix}(eV)'], label=label)
            axes[3].plot(t, cols[f'current{suffix}(e/fs)'], label=label)
        for ax in axes[1:]:
            ax.legend()
    else:
        axes[1].plot(t, cols['electronic_energy(eV)'])
        axes[2].plot(t, cols['nuclear_energy(eV)'])
        axes[3].plot(t, cols['current(e/fs)'])

    axes[1].set_ylabel('electronic energy (eV)')
    axes[2].set_ylabel('nuclear energy (eV)')
    axes[3].set_ylabel('current (e/fs)')
    axes[3].set_xlabel('time (fs)')
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=180)
    else:
        plt.show()


if __name__ == '__main__':
    main()

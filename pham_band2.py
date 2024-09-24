#!/usr/bin/env python

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# Constants
KB = 8.617330337217213e-05  # Boltzmann constant in eV/K
THZ_TO_EV = 0.00413566733   # Conversion factor from THz to eV

def phonon_angular_momentum(freq: np.ndarray, polar_vec: np.ndarray, temp: float = 300) -> np.ndarray:
    """
    Calculate the phonon angular momentum.

    Args:
        freq (np.ndarray): Phonon frequencies in THz.
        polar_vec (np.ndarray): Phonon eigenvectors.
        temp (float): Temperature in Kelvin.

    Returns:
        np.ndarray: Phonon angular momentum in units of hbar.
    """
    if np.isclose(temp, 0.0):
        nbose = 0.5
    else:
        nbose = 0.5 + 1. / (np.exp(freq * THZ_TO_EV / (KB * temp)) - 1.0)

    ixyz = [[1, 2], [2, 0], [0, 1]]
    nqpts, nbnds, _, _ = polar_vec.shape

    J0 = np.zeros((3, nqpts, nbnds))
    for ii in range(3):
        e = polar_vec[..., ixyz[ii]]
        J0[ii] = 2.0 * np.sum(e[:, :, :, 0].conj() * e[:, :, :, 1], axis=2).imag

    return J0 * nbose

def read_ph_yaml(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[str], np.ndarray]:
    """
    Read phonon data from a YAML file.

    Args:
        filename (str): Path to the YAML file.

    Returns:
        Tuple containing:
        - Bcell (np.ndarray): Reciprocal lattice.
        - dists (np.ndarray): Distances along the path.
        - freqs (np.ndarray): Phonon frequencies.
        - qpoints (np.ndarray): Q-points.
        - segment_nqpoint (List[int]): Number of q-points in each segment.
        - labels (List[str]): Labels for high-symmetry points.
        - eigvec (np.ndarray): Phonon eigenvectors.
    """
    def open_file(file_path):
        _, ext = os.path.splitext(file_path)
        if ext == '.xz' or ext == '.lzma':
            import lzma
            return lzma.open(file_path)
        elif ext == '.gz':
            import gzip
            return gzip.open(file_path)
        else:
            return open(file_path, 'r')

    with open_file(filename) as f:
        data = yaml.load(f, Loader=Loader)

    freqs, dists, qpoints, labels, eigvec = [], [], [], [], []
    Acell = np.array(data['lattice'])
    Bcell = np.array(data['reciprocal_lattice'])

    for v in data['phonon']:
        labels.append(v.get('label', None))
        freqs.append([f['frequency'] for f in v['band']])
        if 'eigenvector' in v['band'][0]:
            eigvec.append([np.array(f['eigenvector']) for f in v['band']])
        qpoints.append(v['q-position'])
        dists.append(v['distance'])

    if all(x is None for x in labels):
        if 'labels' in data:
            ss = np.array(data['labels'])
            labels = list(ss[0])
            for ii, f in enumerate(ss[:-1, 1] == ss[1:, 0]):
                if not f:
                    labels[-1] += r'|' + ss[ii+1, 0]
                labels.append(ss[ii+1, 1])
        else:
            labels = []

    return (Bcell, np.array(dists), np.array(freqs), np.array(qpoints),
            data['segment_nqpoint'], labels, np.array(eigvec))

def get_data(args):
    """
    Calculate and save phonon angular momentum data, organized by band.

    Args:
        args: Parsed command-line arguments.
    """
    Bcell, D1, F1, Q1, B1, L1, E1 = read_ph_yaml(args.yaml)

    if len(E1) == 0:
        raise ValueError("PHONON EIGENVECTORS MUST NOT BE EMPTY!")

    E1 = E1[..., 0] + 1j * E1[..., 1]
    nqpts, nbnds, _, _ = E1.shape

    Jxyz = phonon_angular_momentum(F1, E1, args.temperature)

    try:
        with open(args.output_file, 'w') as f:
            f.write("# Data organized by band\n")
            f.write("# Each band is separated by a blank line\n")
            f.write("# Columns: Distance\tFrequency\tJx\tJy\tJz\n\n")

            for band in range(nbnds):
                f.write(f"# Band {band + 1}\n")
                np.savetxt(f, 
                           np.column_stack((D1,
                                            F1[:, band],
                                            Jxyz[0, :, band],
                                            Jxyz[1, :, band],
                                            Jxyz[2, :, band])),
                           fmt='%.4f', delimiter='\t')
                f.write("\n")  # Add a blank line between bands

        print(f"Data successfully written to {args.output_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")

def get_data_prev(args):
    """
    Calculate and save phonon angular momentum data.
    It doesn't distinguish between bands
    Args:
        args: Parsed command-line arguments.
    """
    Bcell, D1, F1, Q1, B1, L1, E1 = read_ph_yaml(args.yaml)

    if len(E1) == 0:
        raise ValueError("PHONON EIGENVECTORS MUST NOT BE EMPTY!")

    E1 = E1[..., 0] + 1j * E1[..., 1]
    nqpts, nbnds, _, _ = E1.shape

    Jxyz = phonon_angular_momentum(F1, E1, args.temperature)

    try:
        np.savetxt(args.output_file, 
                   np.column_stack((np.tile(D1[:, np.newaxis], (1, nbnds)).flatten(),
                                    F1.flatten(),
                                    Jxyz[0].flatten(),
                                    Jxyz[1].flatten(),
                                    Jxyz[2].flatten())),
                   fmt='%.4f', delimiter='\t',
                   header='Distance\tFrequency\tJx\tJy\tJz')
    except IOError as e:
        print(f"Error writing to file: {e}")

def create_plot(args, D1, F1, Jxyz, B1, L1):
    """
    Create the plot for phonon angular momentum.

    Args:
        args: Parsed command-line arguments.
        D1 (np.ndarray): Distances along the path.
        F1 (np.ndarray): Phonon frequencies.
        Jxyz (np.ndarray): Phonon angular momentum.
        B1 (List[int]): Number of q-points in each segment.
        L1 (List[str]): Labels for high-symmetry points.

    Returns:
        Tuple[plt.Figure, np.ndarray]: Figure and axes objects.
    """
    nsubs = 3 if args.direction == 'a' else 1

    if args.figsize is None:
        args.figsize = (6.4, 9.0) if args.layout == 'v' else (12.0, 4.8)

    fig = plt.figure(figsize=args.figsize, dpi=300)

    layout = np.arange(nsubs, dtype=int).reshape((-1, 1) if args.layout == 'v' else (1, -1))
    axes = fig.subplot_mosaic(layout, empty_sentinel=-1)
    axes = np.array([ax for ax in axes.values()])

    caxs = []
    for ax in axes:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right' if args.layout == 'v' else 'top', 
                                  size='2%' if args.layout == 'v' else '4%', 
                                  pad=0.04 if args.layout == 'v' else 0.03)
        caxs.append(cax)

    return fig, axes, caxs

def plot_pam(args, ax, D1, F1, Jxyz, B1, L1, which_j, cax):
    """
    Plot phonon angular momentum for a specific direction.

    Args:
        args: Parsed command-line arguments.
        ax (plt.Axes): Matplotlib axes object.
        D1 (np.ndarray): Distances along the path.
        F1 (np.ndarray): Phonon frequencies.
        Jxyz (np.ndarray): Phonon angular momentum.
        B1 (List[int]): Number of q-points in each segment.
        L1 (List[str]): Labels for high-symmetry points.
        which_j (int): Index of the direction to plot.
        cax (plt.Axes): Colorbar axes object.
    """
    ax.axhline(y=0, ls='-', lw=0.5, color='k', alpha=0.6)

    norm = mpl.colors.Normalize(
        vmin=Jxyz[which_j].min() if args.normalization == 'per_direction' else Jxyz.min(),
        vmax=Jxyz[which_j].max() if args.normalization == 'per_direction' else Jxyz.max()
    )
    s_m = mpl.cm.ScalarMappable(cmap=["PiYG", 'PuOr', "seismic"][which_j], norm=norm)

    if args.plt_type == 'scatter':
        ax.scatter(
            np.tile(D1, (F1.shape[1], 1)).T,
            F1,
            s=np.abs(Jxyz[which_j]) * 20,
            c=Jxyz[which_j],
            cmap=s_m.cmap,
            norm=norm
        )
    else:  # 'colormap'
        for jj in range(F1.shape[1]):
            ik = 0
            for nseg in B1:
                x, y = D1[ik:ik+nseg], F1[ik:ik+nseg, jj]
                ax.plot(x, y, lw=2.0, color='gray')
                ik += nseg

    for jj in np.cumsum(B1)[:-1]:
        ax.axvline(x=D1[jj], ls='--', color='gray', alpha=0.8, lw=0.5)

    ax.grid(True, ls='--', lw=0.5, color='gray', alpha=0.5)
    ax.set_xlim(D1.min(), D1.max())
    ax.set_xticks(D1[np.r_[[0], np.cumsum(B1)-1]])
    if L1:
        ax.set_xticklabels(L1)

    if (args.layout == 'h' and which_j == 0) or args.layout == 'v':
        ax.set_ylabel('Frequency (THz)', labelpad=5)

    cbar = plt.colorbar(s_m, cax=cax, extend='both', shrink=0.5,
                        orientation='vertical' if args.layout == 'v' else 'horizontal')
    cbar.ax.tick_params(which='both', labelsize='small')

    if args.layout == 'h':
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.set_xlabel(r'$J_{}\,/\,\hbar$'.format('xyz'[which_j]))
    else:
        cbar.ax.text(1.60, 1.02, r'$J_{}\,/\,\hbar$'.format('xyz'[which_j]),
                     ha="left", va="bottom", fontweight='bold',
                     transform=cbar.ax.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, lw=0.5))

def pam_plot(args):
    """
    Plot phonon angular momentum for all bands.

    Args:
        args: Parsed command-line arguments.
    """
    Bcell, D1, F1, Q1, B1, L1, E1 = read_ph_yaml(args.yaml)

    if len(E1) == 0:
        raise ValueError("PHONON EIGENVECTORS MUST NOT BE EMPTY!")

    E1 = E1[..., 0] + 1j * E1[..., 1]
    Jxyz = phonon_angular_momentum(F1, E1, args.temperature)

    fig, axes, caxs = create_plot(args, D1, F1, Jxyz, B1, L1)

    for ii, (ax, cax) in enumerate(zip(axes, caxs)):
        which_j = ii if args.direction == 'a' else 'xyz'.index(args.direction)
        plot_pam(args, ax, D1, F1, Jxyz, B1, L1, which_j, cax)

    plt.tight_layout(pad=1.0)
    plt.savefig(args.figname)

def pam_per_band(args):
    """
    Plot phonon angular momentum for a single band.

    Args:
        args: Parsed command-line arguments.
    """
    Bcell, D1, F1, Q1, B1, L1, E1 = read_ph_yaml(args.yaml)

    if len(E1) == 0:
        raise ValueError("PHONON EIGENVECTORS MUST NOT BE EMPTY!")

    E1 = E1[..., 0] + 1j * E1[..., 1]
    Jxyz = phonon_angular_momentum(F1, E1, args.temperature)

    fig, axes, caxs = create_plot(args, D1, F1, Jxyz, B1, L1)

    for ii, (ax, cax) in enumerate(zip(axes, caxs)):
        which_j = ii if args.direction == 'a' else 'xyz'.index(args.direction)
        plot_pam(args, ax, D1, F1[:, [args.idx]], Jxyz[:, :, [args.idx]], B1, L1, which_j, cax)

    plt.tight_layout(pad=1.0)
    plt.savefig(args.figname)

def plot_angular_momentum_vs_distance(args):
    """
    Plot phonon angular momentum for a single band.

    Args:
        args: Parsed command-line arguments.
    """
    Bcell, D1, F1, Q1, B1, L1, E1 = read_ph_yaml(args.yaml)

    if len(E1) == 0:
        raise ValueError("PHONON EIGENVECTORS MUST NOT BE EMPTY!")

    E1 = E1[..., 0] + 1j * E1[..., 1]
    Jxyz = phonon_angular_momentum(F1, E1, args.temperature)

    nsubs = 3 if args.direction == 'a' else 1

    if args.figsize is None:
        args.figsize = (6.4, 9.0) if args.layout == 'v' else (12.0, 4.8)

    fig = plt.figure(figsize=args.figsize, dpi=300)

    layout = np.arange(nsubs, dtype=int).reshape((-1, 1) if args.layout == 'v' else (1, -1))
    axes = fig.subplot_mosaic(layout, empty_sentinel=-1)
    axes = np.array([ax for ax in axes.values()])


    for ii, ax in enumerate(axes):
        which_j = ii if args.direction == 'a' else 'xyz'.index(args.direction)
        J_band = Jxyz[which_j, :, args.idx]
        ax.axhline(y=0, ls='-', lw=0.5, color='k', alpha=0.6)
        norm = mpl.colors.Normalize(
        vmin=J_band if args.normalization == 'per_direction' else Jxyz[:, :, args.idx].min(),
        vmax=J_band if args.normalization == 'per_direction' else Jxyz[:, :, args.idx].max()
        )
        pam_cmaps = ["PiYG", 'PuOr', "seismic"]
        s_m = mpl.cm.ScalarMappable(cmap=pam_cmaps[which_j], norm=norm)
        s_m.set_array([J_band])

        if args.plt_type == 'scatter':
            ax.scatter(
                D1,
                J_band,
                s=np.abs(J_band) * 20,
                c=J_band,
                cmap=s_m.cmap,
                norm=norm
            )
        else:  # 'colormap'
            for jj in range(F1.shape[1]):
                ik = 0
                for nseg in B1:
                    x, y = D1[ik:ik+nseg], F1[ik:ik+nseg, jj]
                    z = J_band
                    ax.plot(x, z, lw=2.0, color='gray')
                    ik += nseg

        for jj in np.cumsum(B1)[:-1]:
            ax.axvline(x=D1[jj], ls='--', color='gray', alpha=0.8, lw=0.5)

        ax.grid(True, ls='--', lw=0.5, color='gray', alpha=0.5)
        ax.set_xlim(D1.min(), D1.max())
        ax.set_xticks(D1[np.r_[[0], np.cumsum(B1)-1]])
        if L1:
            ax.set_xticklabels(L1)

        if (args.layout == 'h' and which_j == 0) or args.layout == 'v':
            ax.set_ylabel(r'$J_{xyz}\,/\,\hbar$')

        ax.set_title(r'$J_{}\,/\,\hbar$'.format('xyz'[which_j]))

    plt.tight_layout(pad=1.0)
    plt.savefig("PAMvsPATH.png")

def parse_cml_args(cml):
    """
    Parse command-line arguments.

    Args:
        cml (List[str]): Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Phonon Angular Momentum Calculation and Plotting")
    parser.add_argument('-i', dest='yaml', default='band.yaml', help='Input YAML file with phonon data')
    parser.add_argument('-t', dest='temperature', type=float, default=0, help='Temperature in Kelvin')
    parser.add_argument('-a', '--action', choices=['data', 'bands', 'band'], default='bands', help='Action to perform')
    parser.add_argument('-d', dest='direction', choices=['x', 'y', 'z', 'a'], default='a', help='PAM component(s) to plot')
    parser.add_argument('-s', '--figsize', nargs=2, type=float, help='Figure size (width height)')
    parser.add_argument('-o', dest='figname', default='pam.png', help='Output figure filename')
    parser.add_argument('--plt-type', choices=['scatter', 'colormap'], default='scatter', help='Plot type')
    parser.add_argument('--layout', choices=['h', 'v'], default='v', help='Layout of subfigures')
    parser.add_argument('-od', '--output_data', dest='output_file', default='', help='Output data file')
    parser.add_argument('-n', '--normalization', choices=['per_direction', 'all'], default='per_direction', help='Normalization method')
    parser.add_argument('-idx', '--index', dest='idx',  type=int, default=0, help='Band number to plot')
    
    return parser.parse_args(cml)

def main(cml):
    """
    Main function to execute the script.

    Args:
        cml (List[str]): Command-line arguments.
    """
    args = parse_cml_args(cml)
    
    try:
        if args.action == "bands":
            pam_plot(args)
        elif args.action == "band":
            pam_per_band(args)
            plot_angular_momentum_vs_distance(args)
        
        if args.output_file:
            get_data(args)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
# plotting/plotter.py

import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    """Generates and saves plots from simulation analysis data."""
    def __init__(self):
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_ramachandran(self, phi_degrees, psi_degrees, output_file, title='Ramachandran Plot'):
        """Generates and saves a Ramachandran plot."""
        print(f"  > Generating Ramachandran plot: {title}")
        fig, ax = plt.subplots(figsize=(8, 8))

        hb = ax.hexbin(phi_degrees, psi_degrees, gridsize=50, cmap='viridis', mincnt=1)
        fig.colorbar(hb, ax=ax, label='Number of Frames in Bin')

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(r'$\phi$ Angle (degrees)', fontsize=12)
        ax.set_ylabel(r'$\psi$ Angle (degrees)', fontsize=12)
        
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-180, 181, 60))
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"  > Ramachandran plot saved to {output_file}")
        plt.close(fig)

    def plot_vdos(self, freqs_cm, vdos, output_file, max_freq_cm=200):
        """Generates and saves a VDOS plot."""
        print(f"  > Generating VDOS plot...")
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(freqs_cm, vdos, color='navy')
        ax.set_xlabel('Frequency (cm⁻¹)', fontsize=14)
        ax.set_ylabel('Intensity (arbitrary units)', fontsize=14)
        ax.set_title('Vibrational Density of States (VDOS)', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(0, max_freq_cm)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"  > VDOS plot saved to {output_file}")
        plt.close(fig)

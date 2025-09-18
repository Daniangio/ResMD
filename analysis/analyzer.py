# analysis/analyzer.py

import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals, align
import numpy as np

class TrajectoryAnalyzer:
    """Handles analysis of simulation trajectories using MDAnalysis."""
    def __init__(self, topology_file, trajectory_file):
        """
        Initializes the analyzer.

        Args:
            topology_file (str): Path to the topology file (e.g., a PDB file).
            trajectory_file (str): Path to the trajectory file (e.g., DCD, XTC).
        """
        print(f"  > Loading trajectory '{trajectory_file}'...")
        self.universe = mda.Universe(topology_file, trajectory_file)

    def calculate_ramachandran(self):
        """
        Calculates phi and psi dihedral angles for a protein trajectory.
        """
        print("  > Calculating Ramachandran angles...")
        # Select all protein atoms for the analysis
        protein_selection = self.universe.select_atoms('protein')
        
        # Aligning the trajectory to a reference structure is good practice
        # to remove global rotation and translation.
        aligner = align.AlignTraj(self.universe, self.universe, select='protein', in_memory=True).run()
        
        # Run the Ramachandran analysis
        rama = dihedrals.Ramachandran(protein_selection).run()
        
        # The result is in r.angles, with shape (n_frames, n_residues, 2)
        # We flatten the array to get a list of all phi/psi pairs and convert to degrees.
        # MDAnalysis returns radians, so we convert by multiplying by 180/pi.
        phi_degrees = rama.results.angles[:, :, 0].flatten() # * np.pi / 180
        psi_degrees = rama.results.angles[:, :, 1].flatten() # * np.pi / 180
        
        print(f"  > Calculated angles for {len(phi_degrees)} data points.")
        return phi_degrees, psi_degrees

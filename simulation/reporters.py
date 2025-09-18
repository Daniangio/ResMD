# simulation/reporters.py

import numpy as np
from openmm import unit

class VelocityReporter:
    """
    An OpenMM Reporter that collects and saves particle velocities.

    This reporter buffers velocities in memory during the simulation and saves
    them to a single NumPy '.npy' file upon completion.
    """
    def __init__(self, file, reportInterval, atom_indices=None, dtype=np.float32):
        """
        Create a VelocityReporter.

        Args:
            file (str): The file path to save the velocities to (e.g., 'velocities.npy').
            reportInterval (int): The interval (in number of time steps) at which to report velocities.
            atom_indices (list, optional): A list of atom indices to save. If None, saves all atoms.
            dtype (data-type, optional): The NumPy data type to save the velocities as.
        """
        self._file = file
        self._reportInterval = reportInterval
        self._atom_indices = atom_indices
        self._dtype = dtype
        self._velocities = []

    def describeNextReport(self, simulation):
        """
        Describes the data required for the next report.

        Returns:
            tuple: A tuple containing the number of steps, and booleans for whether
                   positions, velocities, forces, and energies are needed.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, True, False, False)

    def report(self, simulation, state):
        """
        Generate a report. This is called by the Simulation object.

        Args:
            simulation (Simulation): The Simulation object.
            state (State): The State object containing the simulation data.
        """
        # Get all velocities from the state object
        velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometers / unit.picoseconds)
        
        # Select the specified subset of atoms
        if self._atom_indices is not None:
            velocities = velocities[self._atom_indices]
        
        # Append the velocities for the current frame to our buffer
        self._velocities.append(velocities)

    def finalize(self):
        """
        Finalizes the report by saving the collected velocities to a file.
        This should be called by the runner after the simulation is complete.
        """
        print(f"  > Finalizing velocity report...")
        if self._velocities:
            velocity_array = np.array(self._velocities, dtype=self._dtype)
            np.save(self._file, velocity_array)
            print(f"  > Velocities for {len(self._velocities)} frames saved to {self._file}")
        else:
            print(f"  > No velocities were collected to save.")
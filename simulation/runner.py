# simulation/runner.py

import openmm as mm
from openmm import app, unit
from mdareporter import MDAReporter
from .reporters import VelocityReporter

class SimulationRunner:
    """
    Sets up and runs a highly configurable OpenMM simulation.
    This class handles system setup, equilibration, and production runs.
    It can be configured to optionally collect velocities and apply custom forces.
    """
    def __init__(self, config):
        """
        Initializes the SimulationRunner with a configuration dictionary.

        Args:
            config (dict): A dictionary containing all simulation parameters.
                Required keys: 'pdb_input_file', 'trajectory_output'.
                Optional keys: 'velocities_output', 'force_generator', 'simulation_time_ps',
                               'temperature_k', 'time_step_fs', 'platform_name', 'solvated_pdb_output', etc.
        """
        print(f"--- Initializing Simulation for: {config.get('name', 'Unnamed')} ---")
        self.config = config
        self.pdb_input_file = config['pdb_input_file']
        
        # Extract parameters with defaults
        self.temperature = config.get('temperature_k', 300) * unit.kelvin
        self.friction_coeff = config.get('friction_coeff_ps', 1.0) / unit.picosecond
        self.time_step = config.get('time_step_fs', 2.0) * unit.femtoseconds
        self.platform_name = config.get('platform_name', 'CUDA')
        self.platform_properties = config.get('platform_properties', {'CudaPrecision': 'mixed'})

        # I/O and Reporting
        self.trajectory_output = config['trajectory_output']
        self.log_output = config.get('log_output', 'log')

        # Optional custom force generator
        self.force_generator = config.get('force_generator', None)

        # Internal state
        self.modeller = None
        self.system = None
        self.simulation = None

    def _setup_system(self):
        """Loads PDB, adds solvent, creates the System, and applies custom forces."""
        print("  > Setting up system...")
        pdb = app.PDBFile(self.pdb_input_file)
        self.modeller = app.Modeller(pdb.topology, pdb.positions)
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        solvated_pdb_output = self.config.get('solvated_pdb_output', None)
        if isinstance(solvated_pdb_output, str):
            print("  > Adding solvent...")
            self.modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * unit.nanometer)
            print(f"  > Saving solvated system to {solvated_pdb_output}...")
            with open(solvated_pdb_output, 'w') as f:
                app.PDBFile.writeFile(self.modeller.topology, self.modeller.positions, f)

        self.system = forcefield.createSystem(self.modeller.topology, nonbondedMethod=app.PME,
                                              nonbondedCutoff=1.0 * unit.nanometer,
                                              constraints=app.HBonds)
        if self.force_generator:
            print("  > Applying custom external force...")
            self.system = self.force_generator(self.system, self.modeller.topology)

    def _create_simulation(self):
        """Creates the OpenMM Simulation object."""
        print("  > Creating simulation object...")
        integrator = mm.LangevinIntegrator(self.temperature, self.friction_coeff, self.time_step)
        
        try:
            platform = mm.Platform.getPlatformByName(self.platform_name)
        except Exception:
            print(f"Warning: Platform '{self.platform_name}' not found. Falling back to 'CPU'.")
            platform = mm.Platform.getPlatformByName('CPU')
            self.platform_properties = None

        self.simulation = app.Simulation(self.modeller.topology, self.system, integrator, platform, self.platform_properties)
        self.simulation.context.setPositions(self.modeller.positions)

    def _add_reporters(self, total_steps):
        """Creates and adds all configured reporters to the simulation."""
        print("  > Adding reporters...")
        
        # Standard trajectory reporter
        self.simulation.reporters.append(MDAReporter(
            self.trajectory_output,
            self.config.get('traj_reporting_interval_steps', 100),
            selection='protein',
        ))

        # Log file reporter
        log_interval = self.config.get('log_reporting_interval_steps', 100)
        log_file = self.config.get('log_output', 'output/sim_log.txt')
        self.simulation.reporters.append(app.StateDataReporter(
            log_file, log_interval, step=True, potentialEnergy=True, temperature=True,
            progress=True, remainingTime=True, speed=True, totalSteps=total_steps
        ))

        velocities_output = self.config.get('velocities_output', None)
        if isinstance(velocities_output, str):
            vel_interval = self.config.get('vel_reporting_interval_steps', 1)
            # Get indices for the peptide to avoid saving solvent velocities
            peptide_indices = [a.index for a in self.modeller.topology.atoms() if a.residue.name != 'HOH']
            self.simulation.reporters.append(VelocityReporter(
                velocities_output, vel_interval, atom_indices=peptide_indices
            ))

    def run(self):
        """Executes the full simulation workflow."""
        self._setup_system()
        self._create_simulation()

        print("  > Minimizing energy...")
        self.simulation.minimizeEnergy()
        
        print("  > Equilibrating system...")
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        equilibration_steps = self.config.get('equilibration_steps', 10000)
        self.simulation.step(equilibration_steps)

        # --- Production Run ---
        simulation_time_ps = self.config.get('simulation_time_ps', 1000)
        total_steps = int((simulation_time_ps * unit.picoseconds) / self.time_step)
        
        self._add_reporters(total_steps)

        print(f"  > Starting production run for {simulation_time_ps} ps...")
        update_interval = self.config.get('time_param_update_interval')

        if update_interval is None:
            self.simulation.step(total_steps)
        else:
            # Run with manual parameter updates if configured
            num_chunks = total_steps // update_interval
            for i in range(num_chunks):
                current_time_ps = self.simulation.context.getState().getTime().value_in_unit(unit.picosecond)
                self.simulation.context.setParameter("time", current_time_ps)
                if i < num_chunks - 1: self.simulation.step(update_interval)
                else:
                    steps_left = total_steps % update_interval
                    if steps_left > 0: self.simulation.step(steps_left)

        # --- Finalize ---
        print("  > Finalizing reporters...")
        for reporter in self.simulation.reporters:
            if hasattr(reporter, 'finalize'):
                reporter.finalize()

        print(f"--- Simulation Finished: {self.config.get('name', 'Unnamed')} ---")
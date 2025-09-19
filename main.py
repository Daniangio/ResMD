import os
import argparse
from functools import partial

# --- Import All Necessary Tools ---
from simulation.runner import SimulationRunner
from simulation.forces import create_oscillating_efield_force
from analysis.analyzer import TrajectoryAnalyzer
from plotting.plotter import Plotter

# python scheduler.py 33.0cm.t1=33.0 33.0cm.t2=33.0 33.0cm.t3=33.0 baseline=-1

# --- Configuration ---
PDB_INPUT_FILE = 'ala.pdb'
OUTPUT_DIR = 'output'
SIMULATION_TIME_PS = 1000
FIELD_AMPLITUDE = 100.0

# --- Analysis Function (now part of this script) ---
def analyze_trajectory(topology_file, trajectory_file, plot_output_path):
    """
    Runs the full analysis and plotting workflow for a single trajectory file.
    """
    print(f"\n--- Starting Analysis for {os.path.basename(trajectory_file)} ---")
    
    # 1. Initialize the analyzer
    analyzer = TrajectoryAnalyzer(
        topology_file=topology_file,
        trajectory_file=trajectory_file
    )

    # 2. Perform the calculation
    phi, psi = analyzer.calculate_ramachandran()

    # 3. Generate and save the plot
    plotter = Plotter()
    basename = os.path.basename(trajectory_file).replace('.dcd', '')
    plot_title = f"Ramachandran Plot for {basename}"

    plotter.plot_ramachandran(
        phi_degrees=phi,
        psi_degrees=psi,
        output_file=plot_output_path,
        title=plot_title
    )
    print("--- Analysis Complete ---")

# --- Simulation Config Generator ---
def generate_config(job_name, frequency=None, temperature=300.0):
    """Generates a single, complete simulation configuration dictionary."""
    config = {
        'name': job_name,
        'pdb_input_file': PDB_INPUT_FILE,
        'temperature_k': temperature,
        "solvate": True,
        "simulation_time_ps": SIMULATION_TIME_PS,
        "log_output": os.path.join(OUTPUT_DIR, f'{job_name}.log'),
        "trajectory_output": os.path.join(OUTPUT_DIR, f'{job_name}_traj.dcd'),
        # "callbacks_step_interval": 100,
        # "callbacks": [
        #     lambda sim, step: print(f"Reached step {step}. Simulation time: {sim.context.getParameter('time')}"),
        #     lambda sim, step: print(f"Atom coordinates at step {step}: {sim.context.getState(getPositions=True).getPositions(asNumpy=True)}"),
        # ]
    }

    if frequency is not None and frequency > 0:
        print(f"Setting up forced simulation with frequency = {frequency} cm-1")
        config['force_generator'] = partial(
            create_oscillating_efield_force,
            frequency_cm=frequency,
            amplitude_kj_mol_nm_e=FIELD_AMPLITUDE
        )
        # Use the special integrator for driven simulations
        config['integrator'] = "nemd_langevin"
    else:
        # Use the standard integrator for baseline simulations
        config['integrator'] = "langevin"
    
    return config

# --- Main Job Execution Function ---
def run_job(config):
    """
    Runs a full job: simulation followed by analysis.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # --- STAGE 1: SIMULATION ---
    try:
        print(f"--- Starting simulation for job: {config['name']} ---")
        runner = SimulationRunner(config)
        runner.run()
        print(f"--- Successfully completed simulation: {config['name']} ---")
    except Exception as e:
        print(f"--- ERROR in simulation for {config['name']}: {e} ---")
        return # Exit if simulation fails

    # --- STAGE 2: ANALYSIS ---
    trajectory_file = config.get('trajectory_output')
    if trajectory_file and os.path.exists(trajectory_file):
        plot_output_path = os.path.join(OUTPUT_DIR, f"ramachandran_{config['name']}.png")
        analyze_trajectory(
            topology_file=PDB_INPUT_FILE,
            trajectory_file=trajectory_file,
            plot_output_path=plot_output_path
        )
    else:
        print(f"--- Skipping analysis for '{config['name']}' (trajectory file not found) ---")

def main():
    """Parses command-line arguments to run a specific job."""
    parser = argparse.ArgumentParser(description="Run a single OpenMM simulation and analysis job.")
    parser.add_argument('--job', type=str, required=True, help="The name of the simulation job.")
    parser.add_argument('--freq', type=float, default=None, help="Frequency in cm-1 for forced simulations.")
    args = parser.parse_args()

    config = generate_config(args.job, args.freq)
    run_job(config)

if __name__ == "__main__":
    main()
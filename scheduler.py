# scheduler.py

import os
import subprocess
import time
import argparse
from itertools import cycle

# --- Scheduler Configuration ---
GPUS_TO_USE = [0, 1, 2, 3]
MAX_CONCURRENT_JOBS = len(GPUS_TO_USE)

def run_scheduler(job_definitions):
    """
    Parses job definitions and manages simulation jobs on a set of GPUs.
    """
    jobs_to_run = []

    # --- New Job Generation Logic ---
    if not job_definitions:
        # If no jobs are provided, default to a single baseline run.
        print("-> No jobs specified. Defaulting to a single baseline simulation.")
        jobs_to_run.append({'name': 'baseline_sim', 'freq': -1.0})
    else:
        print("-> Parsing job definitions...")
        for job_def in job_definitions:
            try:
                # Ensure the definition is in the correct NAME=FREQ format
                if '=' not in job_def:
                    raise ValueError("Job definition must be in NAME=FREQ format")
                
                name, freq_str = job_def.split('=', 1)
                freq = float(freq_str)
                
                # A frequency of 0 or less signifies a baseline (no-force) simulation.
                jobs_to_run.append({'name': name, 'freq': freq})
                print(f"  - Queued job: '{name}' with frequency: {freq if freq > 0 else 'Baseline'}")

            except ValueError as e:
                print(f"  - WARNING: Could not parse job '{job_def}'. Skipping. Reason: {e}")
                continue

    # --- Command Building ---
    commands_to_run = []
    for job in jobs_to_run:
        command = ['python', 'main.py', '--job', job['name']]
        # Pass the frequency to the worker script.
        # The worker knows that freq <= 0 means no external force.
        command.extend(['--freq', str(job['freq'])])
        commands_to_run.append(command)

    # --- Process Execution Logic (Unchanged) ---
    gpu_cycler = cycle(GPUS_TO_USE)
    running_processes = []
    
    print(f"\n--- Starting scheduler with {len(commands_to_run)} jobs on {len(GPUS_TO_USE)} GPUs ---")

    while commands_to_run:
        running_processes = [p for p in running_processes if p.poll() is None]
        
        while len(running_processes) < MAX_CONCURRENT_JOBS and commands_to_run:
            command = commands_to_run.pop(0)
            gpu_id = next(gpu_cycler)
            job_env = os.environ.copy()
            job_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            job_name = command[3]

            print(f"-> Launching job '{job_name}' on GPU {gpu_id}...")
            process = subprocess.Popen(command, env=job_env)
            running_processes.append(process)
        
        time.sleep(5)

    print("\n--- All jobs launched. Waiting for the final jobs to complete... ---")
    for process in running_processes:
        process.wait()

    print("\n--- All jobs have finished. Scheduler complete. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scheduler for OpenMM simulations. Use NAME=FREQ format for jobs.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        'jobs', 
        nargs='*', 
        help="""List of jobs to run in NAME=FREQ format.
Examples:
  - Run only a baseline: my_baseline=-1
  - Run baseline and two forced simulations: baseline=0 trial_A=50.5 trial_B=100.0
  - Run two trials at the same frequency: run1_50cm=50.0 run2_50cm=50.0"""
    )
    args = parser.parse_args()
    
    run_scheduler(args.jobs)
# simulation/forces.py

import openmm as mm
from openmm import app

def create_oscillating_efield_force(system, topology, frequency_cm, amplitude_kj_mol_nm_e):
    """
    Creates and adds a CustomExternalForce representing an oscillating electric field
    that interacts with particle charges.

    Args:
        system (mm.System): The OpenMM System object to which the force will be added.
        topology (app.Topology): The OpenMM Topology, used to get particle charges.
        frequency_cm (float): The target frequency of the field in cm⁻¹.
        amplitude_kj_mol_nm_e (float): The amplitude of the electric field in kJ/(mol*nm*e).

    Returns:
        mm.System: The modified system with the new force added.
    """
    # Convert frequency from cm⁻¹ to picoseconds⁻¹ for OpenMM's time units
    # 1 cm⁻¹ = 0.0299792458 THz = 0.0299792458 ps⁻¹
    frequency_ps = frequency_cm * 0.0299792458
    angular_frequency = 2 * 3.1415926535 * frequency_ps

    # Energy expression: E = -q * F * z, where F is the oscillating field F(t)
    # The force is the negative gradient of the potential energy: Force = -∇E
    # Here, we define the energy, and OpenMM computes the force automatically.
    energy_expression = f"-charge * {amplitude_kj_mol_nm_e} * cos({angular_frequency} * time) * z"

    print(f"  > Applying force with energy expression: {energy_expression}")
    force = mm.CustomExternalForce(energy_expression)

    # Define the per-particle parameter 'charge'
    force.addPerParticleParameter("charge")
    # Define global parameter 'time'
    force.addGlobalParameter("time", 0.0)

    # We need to get the charges from a force field-generated system
    # This is a standard way to retrieve charges for a given topology
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    temp_system = forcefield.createSystem(topology)
    nonbonded_force = [f for f in temp_system.getForces() if isinstance(f, mm.NonbondedForce)][0]

    print("  > Applying force to all charged particles (water)...")
    num_forced_atoms = 0
    for atom in topology.atoms():
        # Only apply the force to water molecules as requested in the original code
        if atom.residue.name == 'HOH':
            charge, _, _ = nonbonded_force.getParticleParameters(atom.index)
            # For CustomExternalForce, addParticle only takes the particle index and its parameters
            force.addParticle(atom.index, [charge])
            num_forced_atoms += 1

    print(f"  > Force will be applied to {num_forced_atoms} water atoms.")
    system.addForce(force)
    return system
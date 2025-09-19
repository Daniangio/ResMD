# simulation/integrators.py

import math
import openmm as mm
from openmm import unit


class TimeLangevinIntegrator(mm.CustomIntegrator):
    def __init__(self, temperature, friction, dt):

        temperature = temperature.value_in_unit(unit.kelvin)
        gamma = friction.value_in_unit(unit.picosecond**-1)
        dt_ps = dt.value_in_unit(unit.picosecond)

        # compute thermostat constants numerically (dimensionless)
        a = math.exp(-gamma * dt_ps)
        b = math.sqrt(max(0.0, 1.0 - a * a))

        super().__init__(dt_ps)

        # add constants (a and b are dimensionless floats)
        self.addGlobalVariable("a", a)
        self.addGlobalVariable("b", b)

        # kT as a Quantity (leave it as a unit Quantity so expressions handle units)
        kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.addGlobalVariable("kT", kT)

        # per-DOF helper variable used by constraint code
        self.addPerDofVariable("x1", 0)

        # integration steps (OpenMM doc pattern for Langevin-middle)
        self.addComputePerDof("v", "v + dt*f/m")
        self.addConstrainVelocities()                # enforce velocity constraints if any
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()                 # enforce position constraints
        self.addComputePerDof("v", "v + (x - x1)/dt")

        # Increment integrator's "time" and sync to context
        self.addComputeGlobal("time", "time + dt")
        self.addUpdateContextState()

class TimeVerletIntegrator(mm.CustomIntegrator):
    """
    Velocity Verlet scheme with 'time' context parameter.
    """

    def __init__(self, dt):
        dt_ps = dt.value_in_unit(unit.picosecond)
        super().__init__(dt_ps)

        # per-DOF helper variable to handle constraints
        self.addPerDofVariable("x0", 0)

        # allow context-dependent forces to update state
        self.addUpdateContextState()

        # Verlet integration steps (OpenMM doc pattern)
        self.addComputePerDof("x0", "x")
        self.addComputePerDof("v", "v + 0.5*dt*f/m")
        self.addComputePerDof("x", "x + dt*v")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + 0.5*dt*f/m + (x - x0)/dt")
        self.addConstrainVelocities()

        # Increment integrator's "time" and sync to context
        self.addComputeGlobal("time", "time + dt")
        self.addUpdateContextState()


# --- INTEGRATOR FOR NON-EQUILIBRIUM SIMULATIONS ---
class CustomNEMDIntegrator(mm.CustomIntegrator):
    """
    A custom Langevin-middle integrator that applies the thermostat
    only to a specified group of atoms (the solute).
    """
    def __init__(self, temperature, friction, dt):
        temperature = temperature.value_in_unit(unit.kelvin)
        gamma = friction.value_in_unit(unit.picosecond**-1)
        dt_ps = dt.value_in_unit(unit.picosecond)

        a = math.exp(-gamma * dt_ps)
        b = math.sqrt(max(0.0, 1.0 - a * a))

        super().__init__(dt_ps)

        # --- Global variables ---
        self.addGlobalVariable("a", a)
        self.addGlobalVariable("b", b)
        kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.addGlobalVariable("kT", kT)
        
        # --- Per-DOF variables ---
        self.addPerDofVariable("x1", 0)
        # This is our thermostat switch: 1 for solute, 0 for solvent
        self.addPerDofVariable("is_thermostatted", 0)

        # --- Integration steps ---
        self.addComputePerDof("v", "v + dt*f/m")
        self.addConstrainVelocities()
        self.addComputePerDof("x", "x + 0.5*dt*v")
        
        # --- CONDITIONAL THERMOSTAT STEP ---
        # If is_thermostatted=1, apply Langevin update.
        # If is_thermostatted=0, this becomes v=v (no change).
        self.addComputePerDof("v", "(a*v + b*sqrt(kT/m)*gaussian) * is_thermostatted + v * (1-is_thermostatted)")
        
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x - x1)/dt")

        # --- Time tracking ---
        self.addComputeGlobal("time", "time + dt")
        self.addUpdateContextState()


def create_integrator(kind, temperature, friction, dt, **kwargs):
    """
    Factory function for creating integrators with time tracking.
    """
    if kind.lower() == "langevin":
        return TimeLangevinIntegrator(temperature, friction, dt)
    elif kind.lower() == "verlet":
        return TimeVerletIntegrator(dt)
    elif kind.lower() == "nemd_langevin":
        return CustomNEMDIntegrator(temperature, friction, dt)
    else:
        raise ValueError(f"Unknown integrator kind: {kind}")

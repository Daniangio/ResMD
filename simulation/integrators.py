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


def create_integrator(kind, temperature, friction, dt):
    """
    Factory function for creating integrators with time tracking.
    Args:
        kind (str): "langevin" or "verlet" (extendable).
        temperature (unit.Quantity): temperature for Langevin.
        friction (unit.Quantity): friction coefficient (1/time).
        dt (unit.Quantity): timestep.
    Returns:
        mm.Integrator
    """
    if kind.lower() == "langevin":
        return TimeLangevinIntegrator(temperature, friction, dt)
    elif kind.lower() == "verlet":
        return TimeVerletIntegrator(dt)
    else:
        raise ValueError(f"Unknown integrator kind: {kind}")

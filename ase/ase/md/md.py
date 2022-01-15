"""Molecular Dynamics."""

import warnings
import numpy as np

from ase.optimize.optimize import Dynamics
from ase.md.logger import MDLogger
from ase.io.trajectory import Trajectory
from ase import units


def process_temperature(temperature, temperature_K, orig_unit):
    """Handle that temperature can be specified in multiple units.

    For at least a transition period, molecular dynamics in ASE can
    have the temperature specified in either Kelvin or Electron
    Volt.  The different MD algorithms had different defaults, by
    forcing the user to explicitly choose a unit we can resolve
    this.  Using the original method then will issue a
    FutureWarning.

    Four parameters:

    temperature: None or float
        The original temperature specification in whatever unit was
        historically used.  A warning is issued if this is not None and
        the historical unit was eV.

    temperature_K: None or float
        Temperature in Kelvin.

    orig_unit: str
        Unit used for the `temperature`` parameter.  Must be 'K' or 'eV'.

    Exactly one of the two temperature parameters must be different from 
    None, otherwise an error is issued.

    Return value: Temperature in Kelvin.
    """
    if (temperature is not None) + (temperature_K is not None) != 1:
        raise TypeError("Exactly one of the parameters 'temperature',"
                        + " and 'temperature_K', must be given")
    if temperature is not None:
        w = "Specify the temperature in K using the 'temperature_K' argument"
        if orig_unit == 'K':
            return temperature
        elif orig_unit == 'eV':
            warnings.warn(FutureWarning(w))
            return temperature / units.kB
        else:
            raise ValueError("Unknown temperature unit " + orig_unit)

    assert temperature_K is not None
    return temperature_K


class MolecularDynamics(Dynamics):
    """Base-class for all MD classes."""

    def __init__(self, atoms, timestep, trajectory, logfile=None,
                 loginterval=1, append_trajectory=False):
        """Molecular Dynamics object.

        Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        timestep: float
            The time step in ASE time units.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        loginterval: int (optional)
            Only write a log line for every *loginterval* time steps.
            Default: 1

        append_trajectory: boolean (optional)
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.
        """
        # dt as to be attached _before_ parent class is initialized
        self.dt = timestep

        Dynamics.__init__(self, atoms, logfile=None, trajectory=None)

        self.masses = self.atoms.get_masses()
        self.max_steps = None

        if 0 in self.masses:
            warnings.warn('Zero mass encountered in atoms; this will '
                          'likely lead to errors if the massless atoms '
                          'are unconstrained.')

        self.masses.shape = (-1, 1)

        if not self.atoms.has('momenta'):
            self.atoms.set_momenta(np.zeros([len(self.atoms), 3]))

        # Trajectory is attached here instead of in Dynamics.__init__
        # to respect the loginterval argument.
        if trajectory is not None:
            if isinstance(trajectory, str):
                mode = "a" if append_trajectory else "w"
                trajectory = self.closelater(
                    Trajectory(trajectory, mode=mode, atoms=atoms)
                )
            self.attach(trajectory, interval=loginterval)

        if logfile:
            logger = self.closelater(
                MDLogger(dyn=self, atoms=atoms, logfile=logfile))
            self.attach(logger, loginterval)

    def todict(self):
        return {'type': 'molecular-dynamics',
                'md-type': self.__class__.__name__,
                'timestep': self.dt}

    def irun(self, steps=50):
        """ Call Dynamics.irun and adjust max_steps """
        self.max_steps = steps + self.nsteps
        return Dynamics.irun(self)

    def run(self, steps=50):
        """ Call Dynamics.run and adjust max_steps """
        self.max_steps = steps + self.nsteps
        return Dynamics.run(self)

    def get_time(self):
        return self.nsteps * self.dt

    def converged(self):
        """ MD is 'converged' when number of maximum steps is reached. """
        return self.nsteps >= self.max_steps

    def _get_com_velocity(self, velocity):
        """Return the center of mass velocity.
        Internal use only. This function can be reimplemented by Asap.
        """
        return np.dot(self.masses.ravel(), velocity) / self.masses.sum()

    # Make the process_temperature function available to subclasses
    # as a static method.  This makes it easy for MD objects to use
    # it, while functions in md.velocitydistribution have access to it
    # as a function.
    _process_temperature = staticmethod(process_temperature)

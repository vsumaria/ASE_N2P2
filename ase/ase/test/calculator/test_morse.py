import numpy as np
from scipy.optimize import check_grad

from ase import Atoms
from ase.vibrations import Vibrations
from ase.calculators.morse import MorsePotential, fcut, fcut_d
from ase.build import bulk

De = 5.
Re = 3.
rho0 = 2.


def test_gs_minimum_energy():
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, Re]])
    atoms.calc = MorsePotential(epsilon=De, r0=Re)
    assert atoms.get_potential_energy() == -De


def test_gs_vibrations(testdir):
    # check ground state vibrations
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, Re]])
    atoms.calc = MorsePotential(epsilon=De, r0=Re, rho0=rho0)
    vib = Vibrations(atoms)
    vib.run()


def test_cutoff():
    # check that fcut_d is the derivative of fcut
    r1 = 2.0
    r2 = 3.0
    r = np.linspace(r1 - 0.5, r2 + 0.5, 100)
    for R in r:
        assert check_grad(fcut, fcut_d, np.array([R]), r1, r2) < 1e-5


def test_forces():
    atoms = bulk('Cu', cubic=True)
    atoms.calc = MorsePotential(A=4.0, epsilon=1.0, r0=2.55)
    atoms.rattle(0.1)
    forces = atoms.get_forces()
    numerical_forces = atoms.calc.calculate_numerical_forces(atoms, d=1e-5)
    assert np.abs(forces - numerical_forces).max() < 1e-5

from math import pi, sqrt, cos

import pytest
import numpy as np

from ase import Atoms
from ase import data
from ase.lattice.cubic import FaceCenteredCubic


symb = 'Cu'
Z = data.atomic_numbers[symb]
a0 = data.reference_states[Z]['a']  # type: ignore


def checkang(a, b, phi):
    "Check the angle between two vectors."
    cosphi = np.dot(a, b) / sqrt(np.dot(a, a) * np.dot(b, b))
    assert np.abs(cosphi - cos(phi)) < 1e-10


@pytest.fixture
def atoms():
    # (100) oriented block
    atoms = FaceCenteredCubic(size=(5, 5, 5), symbol=symb, pbc=(1, 1, 0))
    assert len(atoms) == 5 * 5 * 5 * 4
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 2)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(5 * a0 - c[2, 2]) < 1e-10
    return atoms


def test_vacuum_one_direction(atoms):
    vac = 10.0
    atoms.center(axis=2, vacuum=vac)
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 2)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(4.5 * a0 + 2 * vac - c[2, 2]) < 1e-10


def test_vacuum_all_directions(atoms):
    vac = 4.0
    atoms.center(vacuum=vac)
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 2)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(4.5 * a0 + 2 * vac - c[0, 0]) < 1e-10
    assert np.abs(4.5 * a0 + 2 * vac - c[1, 1]) < 1e-10
    assert np.abs(4.5 * a0 + 2 * vac - c[2, 2]) < 1e-10


@pytest.fixture
def atoms_guc():
    return FaceCenteredCubic(size=(5, 5, 5),
                             directions=[[1, 0, 0], [0, 1, 0], [1, 0, 1]],
                             symbol=symb, pbc=(1, 1, 0))


def test_general_unit_cell(atoms_guc):
    atoms = atoms_guc
    assert len(atoms) == 5 * 5 * 5 * 2
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 4)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(2.5 * a0 - c[2, 2]) < 1e-10


def test_vacuum_one_direction_guc(atoms_guc):
    atoms = atoms_guc
    vac = 10.0
    atoms.center(axis=2, vacuum=vac)
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 4)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(2 * a0 + 2 * vac - c[2, 2]) < 1e-10

    # Recenter without specifying vacuum
    atoms.center()
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 4)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(2 * a0 + 2 * vac - c[2, 2]) < 1e-10

    a2 = atoms.copy()

    # Add vacuum in all directions
    vac = 4.0
    atoms.center(vacuum=vac)
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 4)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(4.5 * a0 + 2 * vac - c[1, 1]) < 1e-10
    assert np.abs(2 * a0 + 2 * vac - c[2, 2]) < 1e-10

    # One axis at the time:
    for i in range(3):
        a2.center(vacuum=vac, axis=i)

    assert abs(atoms.positions - a2.positions).max() < 1e-12
    assert abs(atoms.cell - a2.cell).max() < 1e-12


def test_center_empty():
    atoms = Atoms()
    atoms.center()
    assert atoms == Atoms()


def test_center_nocell():
    atoms = Atoms('H', positions=[[1., 2., 3.]])
    atoms.center()
    assert atoms.positions == pytest.approx(0)

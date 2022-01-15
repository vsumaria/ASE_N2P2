# type: ignore
import pytest
import numpy as np
from ase.calculators.turbomole import TurbomoleOptimizer
from ase.calculators.turbomole import Turbomole
from ase.build import molecule


@pytest.fixture
def atoms():
    return molecule('H2O')


@pytest.fixture
def calc():
    params = {
        'title': 'water',
        'basis set name': 'sto-3g hondo',
        'total charge': 0,
        'multiplicity': 1,
        'use dft': True,
        'density functional': 'b-p',
        'use resolution of identity': True,
    }
    return Turbomole(**params)


def test_turbomole_optimizer_class(atoms, calc):
    optimizer = TurbomoleOptimizer(atoms, calc)
    optimizer.run(steps=1)
    assert isinstance(optimizer.todict(), dict)


def test_turbomole_optimizer(atoms, calc):
    optimizer = calc.get_optimizer(atoms)
    optimizer.run(fmax=0.01, steps=5)
    assert np.linalg.norm(calc.get_forces(atoms)) < 0.01

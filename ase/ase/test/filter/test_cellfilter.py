import numpy as np
import pytest

from ase.units import GPa
from ase.build import bulk
from ase.calculators.test import gradient_test
from ase.constraints import UnitCellFilter, ExpCellFilter
from ase.optimize import LBFGS, MDMin
from ase.io import Trajectory


@pytest.fixture
def atoms(asap3):
    rng = np.random.RandomState(0)
    atoms = bulk('Cu', cubic=True)
    atoms.positions[:, 0] *= 0.995
    atoms.cell += rng.uniform(-1e-2, 1e-2, size=9).reshape((3, 3))
    atoms.calc = asap3.EMT()
    return atoms


@pytest.mark.parametrize('cellfilter', [UnitCellFilter, ExpCellFilter])
def test_pressure(atoms, cellfilter):
    xcellfilter = cellfilter(atoms, scalar_pressure=10.0 * GPa)

    # test all derivatives
    f, fn = gradient_test(xcellfilter)
    assert abs(f - fn).max() < 5e-6

    opt = LBFGS(xcellfilter)
    opt.run(1e-3)

    # check pressure is within 0.1 GPa of target
    sigma = atoms.get_stress() / GPa
    pressure = -(sigma[0] + sigma[1] + sigma[2]) / 3.0
    assert abs(pressure - 10.0) < 0.1


@pytest.mark.parametrize('cellfilter', [UnitCellFilter, ExpCellFilter])
def test_cellfilter(atoms, cellfilter):
    xcellfilter = cellfilter(atoms)
    f, fn = gradient_test(xcellfilter)
    assert abs(f - fn).max() < 3e-6


# XXX This test should have some assertions!  --askhl
def test_unitcellfilter(asap3, testdir):
    cu = bulk('Cu') * (6, 6, 6)
    cu.calc = asap3.EMT()
    f = UnitCellFilter(cu, [1, 1, 1, 0, 0, 0])
    opt = LBFGS(f)

    with Trajectory('Cu-fcc.traj', 'w', cu) as t:
        opt.attach(t)
        opt.run(5.0)
    # No assertions??


def test_unitcellfilter_hcp(asap3, testdir):
    cu = bulk('Cu', 'hcp', a=3.6 / 2.0**0.5)
    cu.cell[1, 0] -= 0.05
    cu *= (6, 6, 3)
    cu.calc = asap3.EMT()
    print(cu.get_forces())
    print(cu.get_stress())
    f = UnitCellFilter(cu)
    opt = MDMin(f, dt=0.01)
    with Trajectory('Cu-hcp.traj', 'w', cu) as t:
        opt.attach(t)
        opt.run(0.2)
    # No assertions??

import pytest
import numpy as np
from ase.build import bulk, molecule
from ase.units import Hartree


calc = pytest.mark.calculator

required_quantities = {'eigenvalues',
                       'fermilevel',
                       'version',
                       'forces',
                       'energy',
                       'free_energy',
                       'stress',
                       'ibz_kpoints',
                       'kpoint_weights'}


def run(atoms):
    atoms.get_forces()
    print(sorted(atoms.calc.results))
    for key, value in atoms.calc.results.items():
        if isinstance(value, np.ndarray):
            print(key, value.shape, value.dtype)
        else:
            print(key, value)

    for name in required_quantities:
        assert name in atoms.calc.results

    return atoms.calc.results


@pytest.mark.calculator_lite
@calc('abinit')
def test_si(factory):
    atoms = bulk('Si')
    atoms.calc = factory.calc(nbands=4 * len(atoms), kpts=[4, 4, 4])
    run(atoms)


@pytest.mark.calculator_lite
@pytest.mark.parametrize('pps', ['fhi', 'paw'])
@calc('abinit')
def test_au(factory, pps):
    atoms = bulk('Au')
    atoms.calc = factory.calc(
        pps=pps,
        nbands=10 * len(atoms),
        tsmear=0.1,
        occopt=3,
        kpts=[2, 2, 2],
        pawecutdg=6.0 * Hartree,
    )
    # Somewhat awkward to set pawecutdg also when we are not doing paw,
    # but it's an error to pass None as pawecutdg.
    run(atoms)


@pytest.fixture
def fe_atoms():
    return bulk('Fe')


def getkwargs(**kw):
    return dict(nbands=8, kpts=[2, 2, 2])


@pytest.mark.calculator_lite
@calc('abinit', occopt=7, **getkwargs())
@calc('abinit', spinmagntarget=2.3, **getkwargs())
def test_fe_magmom(factory, fe_atoms):
    fe_atoms.calc = factory.calc()
    run(fe_atoms)


@calc('abinit', nbands=8)
def test_h2o(factory):
    atoms = molecule('H2O', vacuum=2.5)
    atoms.calc = factory.calc()
    run(atoms)


@calc('abinit', nbands=8, occopt=7)
def test_o2(factory):
    atoms = molecule('O2', vacuum=2.5)
    atoms.calc = factory.calc()
    run(atoms)
    magmom = atoms.get_magnetic_moment()
    assert magmom == pytest.approx(2, 1e-2)
    print('magmom', magmom)


@pytest.mark.skip('expensive')
@calc('abinit')
def test_manykpts(factory):
    atoms = bulk('Au') * (2, 2, 2)
    atoms.rattle(stdev=0.01)
    atoms.symbols[:2] = 'Cu'
    atoms.calc = factory.calc(nbands=len(atoms) * 7, kpts=[8, 8, 8])
    run(atoms, 'manykpts')


@pytest.mark.skip('expensive')
@calc('abinit')
def test_manyatoms(factory):
    atoms = bulk('Ne', cubic=True) * (4, 2, 2)
    atoms.rattle(stdev=0.01)
    atoms.calc = factory.calc(nbands=len(atoms) * 5)
    run(atoms, 'manyatoms')

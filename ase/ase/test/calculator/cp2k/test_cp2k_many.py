"""Tests for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from ase.build import molecule
from ase.optimize import BFGS
import pytest
from ase.calculators.calculator import CalculatorSetupError
from ase import units
from ase.atoms import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet


@pytest.fixture
def atoms():
    return molecule('H2', vacuum=2.0)


def test_geoopt(cp2k_factory, atoms):
    calc = cp2k_factory.calc(label='test_H2_GOPT', print_level='LOW')
    atoms.calc = calc

    with BFGS(atoms, logfile=None) as gopt:
        gopt.run(fmax=1e-6)

    dist = atoms.get_distance(0, 1)
    dist_ref = 0.7245595
    assert (dist - dist_ref) / dist_ref < 1e-7

    energy_ref = -30.7025616943
    energy = atoms.get_potential_energy()
    assert (energy - energy_ref) / energy_ref < 1e-10


def test_h2_lda(cp2k_factory, atoms):
    calc = cp2k_factory.calc(label='test_H2_LDA')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -30.6989595886
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10


def test_h2_libxc(cp2k_factory, atoms):
    calc = cp2k_factory.calc(
        xc='XC_GGA_X_PBE XC_GGA_C_PBE',
        pseudo_potential="GTH-PBE",
        label='test_H2_libxc')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -31.591716529642
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10


def test_h2_ls(cp2k_factory, atoms):
    inp = """&FORCE_EVAL
               &DFT
                 &QS
                   LS_SCF ON
                 &END QS
               &END DFT
             &END FORCE_EVAL"""
    calc = cp2k_factory.calc(label='test_H2_LS', inp=inp)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -30.6989581747
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 5e-7


def test_h2_pbe(cp2k_factory, atoms):
    calc = cp2k_factory.calc(xc='PBE', label='test_H2_PBE')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    energy_ref = -31.5917284949
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10


def test_md(cp2k_factory):
    calc = cp2k_factory.calc(label='test_H2_MD')
    positions = [(0, 0, 0), (0, 0, 0.7245595)]
    atoms = Atoms('HH', positions=positions, calculator=calc)
    atoms.center(vacuum=2.0)

    MaxwellBoltzmannDistribution(atoms, temperature_K=0.5 * 300,
                                 force_temp=True)
    energy_start = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    with VelocityVerlet(atoms, 0.5 * units.fs) as dyn:
        dyn.run(20)

    energy_end = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    assert abs(energy_start - energy_end) < 1e-4


def test_o2(cp2k_factory):
    calc = cp2k_factory.calc(
        label='test_O2', uks=True, cutoff=150 * units.Rydberg,
        basis_set="SZV-MOLOPT-SR-GTH")
    o2 = molecule('O2', calculator=calc)
    o2.center(vacuum=2.0)
    energy = o2.get_potential_energy()
    energy_ref = -861.057011375
    diff = abs((energy - energy_ref) / energy_ref)
    assert diff < 1e-10


def test_restart(cp2k_factory, atoms):
    calc = cp2k_factory.calc()
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('test_restart')  # write a restart
    calc2 = cp2k_factory.calc(restart='test_restart')  # load a restart
    assert not calc2.calculation_required(atoms, ['energy'])


def test_unknown_keywords(cp2k_factory):
    with pytest.raises(CalculatorSetupError):
        cp2k_factory.calc(dummy_nonexistent_keyword='hello')

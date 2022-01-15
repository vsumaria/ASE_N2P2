"""Test module for explicitly unittesting parts of the VASP calculator"""

import os
import sys
import pytest

from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError, get_calculator_class
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell, check_atoms_type


@pytest.fixture
def atoms():
    return molecule('H2', vacuum=5, pbc=True)


@pytest.fixture(autouse=True)
def always_mock_calculate(mock_vasp_calculate):
    """No tests in this module may execute VASP"""
    yield


def test_verify_no_run():
    """Verify that we get an error if we try and execute the calculator,
    due to the fixture.
    """
    calc = Vasp()
    with pytest.raises(AssertionError):
        calc._run()


def test_check_atoms(atoms):
    """Test checking atoms passes for a good atoms object"""
    check_atoms(atoms)
    check_pbc(atoms)
    check_cell(atoms)


@pytest.mark.parametrize(
    'bad_atoms',
    [
        None,
        'a_string',
        # We cannot handle lists of atoms either
        [molecule('H2', vacuum=5)],
    ])
def test_not_atoms(bad_atoms):
    """Check that passing in objects which are not
    actually Atoms objects raises a setup error """

    with pytest.raises(CalculatorSetupError):
        check_atoms_type(bad_atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(bad_atoms)

    # Test that error is also raised properly when launching
    # from calculator
    calc = Vasp()
    with pytest.raises(CalculatorSetupError):
        calc.calculate(atoms=bad_atoms)


@pytest.mark.parametrize('pbc', [
    3 * [False],
    [True, False, True],
    [False, True, False],
])
def test_bad_pbc(atoms, pbc):
    """Test handling of PBC"""
    atoms.pbc = pbc

    check_cell(atoms)  # We have a cell, so this should not raise

    # Check that our helper functions raises the expected error
    with pytest.raises(CalculatorSetupError):
        check_pbc(atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(atoms)

    # Check we also raise in the calculator when launching
    # a calculation, but before VASP is actually executed
    calc = Vasp()
    atoms.calc = calc
    with pytest.raises(CalculatorSetupError):
        atoms.get_potential_energy()


def test_vasp_no_cell(testdir):
    """Check missing cell handling."""
    # Molecules come with no unit cell
    atoms = molecule('CH4')
    # We should not have a cell
    assert atoms.cell.rank == 0

    with pytest.raises(CalculatorSetupError):
        check_cell(atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(atoms)

    with pytest.raises(RuntimeError):
        atoms.write('POSCAR')

    calc = Vasp()
    atoms.calc = calc
    with pytest.raises(CalculatorSetupError):
        atoms.get_total_energy()


def test_vasp_name():
    """Test the calculator class has the expected name"""
    expected = 'vasp'
    assert Vasp.name == expected  # Test class attribute
    assert Vasp().name == expected  # Ensure instance attribute hasn't changed


def test_vasp_get_calculator():
    cls_ = get_calculator_class('vasp')

    assert cls_ == Vasp

    # Test we get the correct calculator when loading from name
    assert get_calculator_class(Vasp.name) == cls_


@pytest.mark.parametrize('envvar', Vasp.env_commands)
def test_make_command_envvar(envvar, monkeypatch, clear_vasp_envvar):
    """Test making a command based on the environment variables"""
    # Environment should be cleared by the "clear_vasp_envvar" fixture
    assert envvar not in os.environ
    cmd_str = 'my command'
    monkeypatch.setenv(envvar, cmd_str)
    calc = Vasp()

    cmd = calc.make_command()
    if envvar == 'VASP_SCRIPT':
        # This envvar uses the `sys.exe` to create the command
        exe = sys.executable
        assert cmd == f'{exe} {cmd_str}'
    else:
        # We just use the exact string
        assert cmd == cmd_str


def test_make_command_no_envvar(monkeypatch, clear_vasp_envvar):
    """Test we raise when making a command with not enough information"""
    # Environment should be cleared by the "clear_vasp_envvar" fixture
    calc = Vasp()
    with pytest.raises(CalculatorSetupError):
        calc.make_command()


def test_make_command_explicit(monkeypatch):
    """Test explicitly passing a command to the calculator"""
    # The command should be whatever we put in, if we explicitly set something
    # envvars should not matter
    for envvar in Vasp.env_commands:
        # Populate the envvars with some strings, they should not matter
        monkeypatch.setenv(envvar, 'something')
    calc = Vasp()
    my_cmd = 'my command'
    cmd = calc.make_command(my_cmd)
    assert cmd == my_cmd

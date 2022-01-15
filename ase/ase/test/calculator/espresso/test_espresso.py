import pytest
from ase.build import bulk
from ase.calculators.espresso import EspressoProfile, Espresso


def test_version():
    txt = """
     Program PWSCF v.6.4.1 starts on  5Aug2021 at 11: 2:26

     This program is part of the open-source Quantum ESPRESSO suite
    """
    assert EspressoProfile.parse_version(txt) == '6.4.1'


def verify(calc):
    assert calc.get_fermi_level() is not None
    assert calc.get_ibz_k_points() is not None
    assert calc.get_eigenvalues(spin=0, kpt=0) is not None
    assert calc.get_number_of_spins() is not None
    assert calc.get_k_point_weights() is not None


@pytest.mark.calculator_lite
def test_main(espresso_factory):
    atoms = bulk('Si')
    atoms.calc = espresso_factory.calc()
    atoms.get_potential_energy()
    verify(atoms.calc)


@pytest.mark.calculator_lite
def test_smearing(espresso_factory):
    atoms = bulk('Cu')
    input_data = {'system': {'occupations': 'smearing',
                             'smearing': 'fermi-dirac',
                             'degauss': 0.02}}
    atoms.calc = espresso_factory.calc(input_data=input_data)
    atoms.get_potential_energy()
    verify(atoms.calc)


def test_warn_label():
    with pytest.warns(FutureWarning):
        Espresso(label='hello')


def test_error_command():
    with pytest.raises(RuntimeError):
        Espresso(command='hello')

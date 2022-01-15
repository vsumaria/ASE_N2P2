import pytest
from ase.build import bulk


@pytest.mark.calculator_lite
@pytest.mark.calculator('exciting')
def test_exciting_bulk(factory):
    atoms = bulk('Si')
    atoms.calc = factory.calc()
    energy = atoms.get_potential_energy()
    print(energy)

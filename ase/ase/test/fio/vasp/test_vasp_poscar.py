import pytest
from ase.io import read, write
from ase.build import bulk
from ase.calculators.calculator import compare_atoms


@pytest.fixture
def atoms():
    _atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1, cubic=True)
    _atoms.wrap()
    return _atoms


@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('vasp5', [True, False])
def test_read_write_roundtrip(atoms, vasp5, filename):
    write(filename, atoms, vasp5=vasp5)
    atoms_loaded = read(filename)

    assert len(compare_atoms(atoms, atoms_loaded)) == 0


@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('kwargs', [{}, {'vasp5': True}])
def test_write_vasp5(atoms, filename, kwargs):
    """Test that we write the symbols to the POSCAR/CONTCAR
    with vasp5=True (which should also be the default)"""
    write(filename, atoms, format='vasp', **kwargs)
    with open(filename) as file:
        lines = file.readlines()
    # Test the 5th line, which should be the symbols
    assert lines[5].strip().split() == list(atoms.symbols)

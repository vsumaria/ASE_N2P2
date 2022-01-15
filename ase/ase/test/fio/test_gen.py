import pytest
import numpy as np

from ase import Atoms
from ase.io import read, write


@pytest.mark.parametrize('pbc', [False, [True, True, False], True])
@pytest.mark.parametrize('cell', [None, [[2.5, 0, 0], [2, 4, 0], [1, 2, 3]]])
@pytest.mark.parametrize('write_format', ['gen', 'dftb'])
def test_gen(pbc, cell, write_format):
    atoms = Atoms(symbols='OCO', pbc=pbc, cell=cell,
                  positions=[[-0.1, 1.2, 0.3],
                             [-0.1, 0.0, 0.2],
                             [0.4, -0.9, 0.0]])
    write('test.gen', atoms, format=write_format)

    atoms_new = read('test.gen')
    assert np.all(atoms_new.numbers == atoms.numbers)
    assert np.allclose(atoms_new.positions, atoms.positions)

    if atoms.pbc.any():
        assert np.all(atoms_new.pbc)
        if atoms.cell is not None:
            assert np.allclose(atoms_new.cell, atoms.cell)
    else:
        assert np.all(~atoms_new.pbc)
        assert np.allclose(atoms_new.cell, 0.)


def test_gen_multiple():
    # Try with multiple images. This is not supported by the
    # format and should fail
    atoms = Atoms('H2')

    with pytest.raises(ValueError):
        write('test.gen', [atoms, atoms])

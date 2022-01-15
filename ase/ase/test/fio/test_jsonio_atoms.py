from ase.io.jsonio import encode, decode
from ase.build import bulk, molecule
import numpy as np


def test_jsonio_atoms():

    def assert_equal(atoms1, atoms2):
        assert atoms1 == atoms2
        assert set(atoms1.arrays) == set(atoms2.arrays)
        for name in atoms1.arrays:
            assert np.array_equal(atoms1.arrays[name], atoms2.arrays[name]), name

    atoms = bulk('Ti')
    print('atoms', atoms)
    txt = encode(atoms)
    print('encoded', txt)

    atoms1 = decode(txt)
    print('decoded', atoms1)
    txt1 = encode(atoms1)
    assert txt == txt1
    assert_equal(atoms, atoms1)

    BeH = molecule('BeH')
    assert BeH.has('initial_magmoms')
    new_BeH = decode(encode(BeH))
    assert_equal(BeH, new_BeH)
    assert new_BeH.has('initial_magmoms')

    from ase.constraints import FixAtoms
    atoms = bulk('Ti')
    atoms.constraints = FixAtoms(indices=[0])
    newatoms = decode(encode(atoms))
    c1 = atoms.constraints
    c2 = newatoms.constraints
    assert len(c1) == len(c2) == 1
    # Can we check constraint equality somehow?
    # Would make sense for FixAtoms
    assert np.array_equal(c1[0].index, c2[0].index)


def test_jsonio_cartesian():

    from ase.constraints import FixCartesian

    atoms = bulk('Ag', cubic=True)
    a = [0, 1]
    mask = [[False, False, True], [False, False, True]]

    atoms.constraints = FixCartesian(a, mask=mask)
    new_atoms = decode(encode(atoms))
    c1 = atoms.constraints
    c2 = new_atoms.constraints
    assert len(c1) == len(c2) == 1
    assert np.array_equal(c1[0].index, c2[0].index)
    assert np.array_equal(c1[0].mask, c2[0].mask)

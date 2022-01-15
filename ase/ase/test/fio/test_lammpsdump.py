import numpy as np
import pytest

from ase.io.formats import ioformats, match_magic

# some of the possible bound parameters
bounds_parameters = [
    ("pp pp pp", (True, True, True)),

    ("ss mm ff", (False, False, False)),
    ("fs sm mf", (False, False, False)),
    ("sf ms ff", (False, False, False)),

    ("pp ms ff", (True, False, False)),
    ("ff pp ff", (False, True, False)),
    ("ff mm pp", (False, False, True)),

    ("pp ff pp", (True, False, True)),
]

ref_positions = np.array([[0.5, 0.6, 0.7],
                          [0.6, 0.1, 1.9],
                          [0.45, 0.32, 0.67]])


@pytest.fixture
def fmt():
    return ioformats['lammps-dump-text']


@pytest.fixture
def lammpsdump():
    def factory(bounds="pp pp pp",
                position_cols="x y z",
                have_element=True,
                have_id=False,
                have_type=True):

        _element = "element" if have_element else "unk0"
        _id = "id" if have_id else "unk1"
        _type = "type" if have_type else "unk2"

        buf = f"""\
        ITEM: TIMESTEP
        0
        ITEM: NUMBER OF ATOMS
        3
        ITEM: BOX BOUNDS {bounds}
        0.0e+00 4e+00
        0.0e+00 5.0e+00
        0.0e+00 2.0e+01
        ITEM: ATOMS {_element} {_id} {_type} {position_cols}
        C  1 1 0.5 0.6 0.7
        C  3 1 0.6 0.1 1.9
        Si 2 2 0.45 0.32 0.67
        """

        return buf

    return factory


def lammpsdump_headers():
    actual_magic = 'ITEM: TIMESTEP'
    yield actual_magic
    yield f'anything\n{actual_magic}\nanything'


@pytest.mark.parametrize('header', lammpsdump_headers())
def test_recognize_lammpsdump(header):
    fmt_name = 'lammps-dump-text'
    fmt = match_magic(header.encode('ascii'))
    assert fmt.name == fmt_name


def test_lammpsdump_order(fmt, lammpsdump):
    # reordering of atoms by the `id` column
    ref_order = np.array([1, 3, 2])
    atoms = fmt.parse_atoms(lammpsdump(have_id=True))
    assert atoms.cell.orthorhombic
    assert pytest.approx(atoms.cell.lengths()) == [4., 5., 20.]
    assert pytest.approx(atoms.positions) == ref_positions[ref_order - 1]


def test_lammpsdump_element(fmt, lammpsdump):
    # Test lammpsdump with elements column given
    atoms = fmt.parse_atoms(lammpsdump())
    assert np.all(atoms.get_atomic_numbers() == np.array([6, 6, 14]))


def test_lammpsdump_errors(fmt, lammpsdump):
    # elements not given
    with pytest.raises(ValueError,
                       match="Cannot determine atom types.*"):
        _ = fmt.parse_atoms(lammpsdump(have_element=False, have_type=False))

    # positions not given
    with pytest.raises(ValueError,
                       match="No atomic positions found in LAMMPS output"):
        _ = fmt.parse_atoms(lammpsdump(position_cols="unk_x unk_y unk_z"))


@pytest.mark.parametrize("cols,scaled", [
    ("xs ys zs", True),
    ("xsu ysu zsu", True),
    ("x y z", False),
    ("xu yu zu", False),
])
def test_lammpsdump_position_reading(fmt, lammpsdump, cols, scaled):
    # test all 4 kinds of definitions of positions

    atoms = fmt.parse_atoms(lammpsdump(position_cols=cols))
    assert atoms.cell.orthorhombic
    assert pytest.approx(atoms.cell.lengths()) == [4., 5., 20.]

    if scaled:
        assert pytest.approx(atoms.positions) == ref_positions * np.array(
            [4, 5, 20]).T
    else:
        assert pytest.approx(atoms.positions) == ref_positions


@pytest.mark.parametrize("bounds,expected", bounds_parameters)
def test_lammpsdump_bounds(fmt, lammpsdump, bounds, expected):
    # Test lammpsdump with all possible boundaries
    atoms = fmt.parse_atoms(lammpsdump(bounds=bounds))
    assert pytest.approx(atoms.cell.lengths()) == [4., 5., 20.]
    assert np.all(atoms.get_pbc() == expected)

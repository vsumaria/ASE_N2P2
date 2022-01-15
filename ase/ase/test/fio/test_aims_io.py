import numpy as np
from ase.build import bulk
from ase.io.aims import read_aims as read
from ase.io.aims import parse_geometry_lines
from pytest import approx

format = "aims"

atoms = bulk("Si")
atoms.positions[0, 0] -= 0.01

file = "geometry.in"


# check cartesian
def test_cartesian(atoms=atoms):
    """write cartesian coords and check if structure was preserved"""
    atoms.write(file, format=format)

    new_atoms = read((file))

    assert np.allclose(atoms.positions, new_atoms.positions)


# check scaled
def test_scaled(atoms=atoms):
    """write fractional coords and check if structure was preserved"""
    atoms.write(file, format=format, scaled=True, wrap=False)

    new_atoms = read(file)

    assert np.allclose(atoms.positions, new_atoms.positions), (
        atoms.positions,
        new_atoms.positions,
    )


# this should fail
def test_scaled_wrapped(atoms=atoms):
    """write fractional coords and check if structure was preserved"""
    atoms.write(file, format=format, scaled=True, wrap=True)

    new_atoms = read(file)

    try:
        assert np.allclose(atoms.positions, new_atoms.positions), (
            atoms.positions,
            new_atoms.positions,
        )
    except AssertionError:
        atoms.wrap()
        assert np.allclose(atoms.positions, new_atoms.positions), (
            atoms.positions,
            new_atoms.positions,
        )


sample_geometry_1 = """\
lattice_vector 4.5521460059804628 0.0000000000000000 0.0000000000000000
lattice_vector -2.2760730029902314 3.9422740829149499 0.0000000000000000 # Dummy comment
lattice_vector 0.0000000000000000 0.0000000000000000 7.1603474299999998
atom_frac 0.0000000000000000 0.0000000000000000 0.0000000000000000 Pb # Dummy comment
atom_frac 0.6666666666666666 0.3333333333333333 0.7349025600000001 I
atom_frac 0.3333333333333333 0.6666666666666666 0.2650974399999999 I
#=======================================================
# Parametric constraints
#=======================================================
symmetry_n_params 3 2 1
symmetry_params a c d0_z
symmetry_lv a, 0, 0
symmetry_lv -0.5*a, 0.8660254037844*a, 0
symmetry_lv 0, 0, c
symmetry_frac 0, 0, 0
symmetry_frac 0.6666666666667, 0.3333333333333, 1.0-d0_z
symmetry_frac 0.3333333333333, 0.6666666666667, d0_z
"""

sample_geometry_2 = """\
atom 0.0000000000000000 0.0000000000000000 0.0000000000000000 Pb # Dummy comment
    constrain_relaxation .true.
atom 0.6666666666666666 0.3333333333333333 0.7349025600000001 I
    initial_moment 1
    initial_charge -1
atom 0.3333333333333333 0.6666666666666666 0.2650974399999999 I
    constrain_relaxation y
    initial_charge -1
    initial_moment 1
"""

expected_symbols = ['Pb', 'I', 'I']
expected_scaled_positions = np.array([
    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
    [0.6666666666666666, 0.3333333333333333, 0.7349025600000000],
    [0.3333333333333333, 0.6666666666666666, 0.2650974400000000],
])
expected_charges = np.array([0, -1, -1])
expected_moments = np.array([0, 1, 1])
expected_lattice_vectors = np.array([
    [4.5521460059804628, 0.0000000000000000, 0.0000000000000000],
    [-2.2760730029902314, 3.9422740829149499, 0.0000000000000000],
    [0.0000000000000000, 0.0000000000000000, 7.1603474299999998],
])


def test_parse_geometry_lines():
    lines = sample_geometry_1.splitlines()
    atoms = parse_geometry_lines(lines, 'sample_geometry_1.in')
    assert all(atoms.symbols == expected_symbols)
    assert atoms.get_scaled_positions() == approx(expected_scaled_positions)
    assert atoms.get_cell()[:] == approx(expected_lattice_vectors)
    assert all(atoms.pbc)

    lines = sample_geometry_2.splitlines()
    atoms = parse_geometry_lines(lines, 'sample_geometry_2.in')
    assert all(atoms.symbols == expected_symbols)
    assert atoms.get_scaled_positions() == approx(expected_scaled_positions)
    assert atoms.get_initial_charges() == approx(expected_charges)
    assert atoms.get_initial_magnetic_moments() == approx(expected_moments)
    assert all(atoms.pbc == [0, 0, 0])
    assert len(atoms.constraints) == 2

from ase.build import bulk, molecule
from ase.phonons import Phonons
from ase.calculators.emt import EMT


def check_set_atoms(atoms, set_atoms, expected_atoms):
    """ Perform a test that .set_atoms() only displaces the expected atoms. """
    atoms.calc = EMT()
    phonons = Phonons(atoms, EMT())
    phonons.set_atoms(set_atoms)

    # TODO: For now, because there is no public API to iterate over/inspect
    #       displacements, we run and check the number of files in the cache.
    #       Later when the requisite API exists, we should use it both to
    #       check the actual atom indices and to avoid computation.
    phonons.run()
    assert len(phonons.cache) == 6 * len(expected_atoms) + 1


def test_set_atoms_indices(testdir):
    check_set_atoms(molecule('CO2'), set_atoms=[0, 1], expected_atoms=[0, 1])


def test_set_atoms_symbol(testdir):
    check_set_atoms(molecule('CO2'), set_atoms=['O'], expected_atoms=[1, 2])


def test_check_eq_forces(testdir):
    atoms = bulk('C')
    atoms.calc = EMT()

    phonons = Phonons(atoms, EMT(), supercell=(1, 2, 1))
    phonons.run()
    fmin, fmax, _i_min, _i_max = phonons.check_eq_forces()
    assert fmin < fmax


# Regression test for #953;  data stored for eq should resemble data for displacements
def test_check_consistent_format(testdir):
    atoms = molecule('H2')
    atoms.calc = EMT()

    phonons = Phonons(atoms, EMT())
    phonons.run()

    # Check that the data stored for `eq` is shaped like the data stored for displacements.
    eq_data = phonons.cache['eq']
    disp_data = phonons.cache['0x-']
    assert isinstance(eq_data, dict) and isinstance(disp_data, dict)
    assert set(eq_data) == set(disp_data), "dict keys mismatch"
    for array_key in eq_data:
        assert eq_data[array_key].shape == disp_data[array_key].shape, array_key

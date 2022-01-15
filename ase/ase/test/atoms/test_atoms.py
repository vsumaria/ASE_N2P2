import pytest
import numpy as np
from ase import Atoms


def test_atoms():
    from ase import Atoms
    print(Atoms())
    print(Atoms('H2O'))
    #...


def test_numbers_input():
    numbers = np.array([[0, 1], [2, 3]])
    with pytest.raises(Exception, match='"numbers" must be 1-dimensional.'):
        Atoms(positions=np.zeros((2, 3)), numbers=numbers, cell=np.eye(3))

    Atoms(positions=np.zeros((2, 3)), numbers=[0, 1], cell=np.eye(3))


def test_bad_array_shape():
    with pytest.raises(ValueError, match='wrong length'):
        Atoms().set_masses([1, 2])

    with pytest.raises(ValueError, match='wrong length'):
        Atoms('H').set_masses([])

    with pytest.raises(ValueError, match='wrong shape'):
        Atoms('H').set_masses(np.ones((1, 3)))


def test_set_masses():
    atoms = Atoms('AgAu')
    m0 = atoms.get_masses()
    atoms.set_masses([1, None])
    assert atoms.get_masses() == pytest.approx([1, m0[1]])


@pytest.mark.parametrize('zlength', [0, 10])
def test_com(zlength):
    """Test that atoms.get_center_of_mass(scaled=True) works"""

    d = 1.142
    a = Atoms('CO', positions=[(2, 0, 0), (2, -d, 0)], pbc=True)
    a.set_cell(np.array(((4, -4, 0), (0, 5.657, 0), (0, 0, zlength))))

    def array_almost_equal(a1, a2, tol=np.finfo(type(1.0)).eps):
        return (np.abs(a1 - a2) < tol).all()

    scaledref = np.array((0.5, 0.23823622, 0.))
    assert array_almost_equal(a.get_center_of_mass(scaled=True),
                              scaledref, tol=1e-8)

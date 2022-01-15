import pytest
from ase.build import molecule
from numpy.testing import assert_allclose


@pytest.fixture()
def atoms():
    return molecule('H2O')


@pytest.mark.calculator('nwchem')
@pytest.mark.parametrize(
    'theory,eref,forces,pbc,kwargs',
    [
        ['dft', -2051.9802410863354, True, False, dict(basis='3-21G')],
        ['scf', -2056.7877421222634, True, False, dict(basis='3-21G')],
        ['mp2', -2060.1413846247333, True, False, dict(basis='3-21G')],
        ['ccsd', -2060.3418911515882, False, False, dict(basis='3-21G')],
        ['tce', -2060.319141863451, False, False, dict(
            basis='3-21G',
            tce={'ccd': None}
        )],
        ['tddft', -2044.3908422254976, True, False, dict(
            basis='3-21G',
            tddft=dict(
                nroots=2,
                algorithm=1,
                notriplet=None,
                target=1,
                civecs=None,
                grad={'root': 1},
            )
        )],
        ['pspw', -465.1290581383751, False, True, dict()],
        ['band', -465.1290611316276, False, True, dict()],
        ['paw', -2065.6600649367365, False, True, dict()]
    ]
)
def test_nwchem(factory, atoms, theory, eref, forces, pbc, kwargs):
    calc = factory.calc(label=theory, theory=theory, **kwargs)
    if pbc:
        atoms.center(vacuum=2)
        atoms.pbc = True
    atoms.calc = calc
    assert_allclose(atoms.get_potential_energy(), eref, atol=1e-4, rtol=1e-4)
    if forces:
        assert_allclose(atoms.get_forces(),
                        calc.calculate_numerical_forces(atoms),
                        atol=1e-4, rtol=1e-4)

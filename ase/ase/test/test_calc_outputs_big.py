import numpy as np
import pytest
from ase.outputs import Properties, all_outputs


@pytest.fixture
def rng():
    return np.random.RandomState(17)


@pytest.fixture
def props(rng):
    nspins, nkpts, nbands = 2, 3, 5
    natoms = 4

    results = dict(
        natoms=natoms,
        energy=rng.random(),
        free_energy=rng.random(),
        energies=rng.random(natoms),
        forces=rng.random((natoms, 3)),
        stress=rng.random(6),
        stresses=rng.random((natoms, 6)),
        nspins=nspins,
        nkpts=nkpts,
        nbands=nbands,
        eigenvalues=rng.random((nspins, nkpts, nbands)),
        occupations=rng.random((nspins, nkpts, nbands)),
        fermi_level=rng.random(),
        ibz_kpoints=rng.random((nkpts, 3)),
        kpoint_weights=rng.random(nkpts),
        dipole=rng.random(3),
        charges=rng.random(natoms),
        magmom=rng.random(),
        magmoms=rng.random(natoms),
        polarization=rng.random(3),
        born_charges=rng.random((natoms, 3, 3)),
        dielectric_tensor=rng.random((3, 3)),

    )
    return Properties(results)


def test_properties_big(props):
    for name in all_outputs:
        assert name in props, name
        obj = props[name]
        print(name, obj)


def test_singlepoint_roundtrip(props):
    from ase.build import bulk
    from ase.calculators.singlepoint import (SinglePointDFTCalculator,
                                             arrays_to_kpoints)

    atoms = bulk('Au') * (1, 1, props['natoms'])

    kpts = arrays_to_kpoints(props['eigenvalues'], props['occupations'],
                             props['kpoint_weights'])
    calc = SinglePointDFTCalculator(atoms=atoms, kpts=kpts,
                                    efermi=props['fermi_level'],
                                    forces=props['forces'])

    props1 = calc.properties()
    print(props1)

    assert set(props1) >= {
        'eigenvalues', 'occupations', 'kpoint_weights', 'fermi_level'}

    for prop in props1:
        assert props[prop] == pytest.approx(props1[prop])

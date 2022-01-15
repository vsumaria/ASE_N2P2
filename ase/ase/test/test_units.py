"""This test cross-checks our implementation of CODATA against the
implementation that SciPy brings with it.
"""

import numpy as np
import pytest

import ase.units
from ase.units import CODATA, create_units
import scipy.constants.codata


# Scipy lacks data for some of the codata versions:
codata_scipy_versions = set(CODATA) - {'1998', '1986'}


@pytest.mark.parametrize('version', sorted(codata_scipy_versions))
def test_units(version):
    name_map = {'_c': 'speed of light in vacuum',
                '_mu0': 'mag. const.',
                '_Grav': 'Newtonian constant of gravitation',
                '_hplanck': 'Planck constant',
                '_e': 'elementary charge',
                '_me': 'electron mass',
                '_mp': 'proton mass',
                '_Nav': 'Avogadro constant',
                '_k': 'Boltzmann constant',
                '_amu': 'atomic mass unit-kilogram relationship'}

    scipy_CODATA = getattr(scipy.constants.codata,
                           f'_physical_constants_{version}', None)
    if version == '2018' and scipy_CODATA is None:
        pytest.skip('No CODATA for 2018 with this scipy')

    assert scipy_CODATA is not None

    for asename, scipyname in name_map.items():
        aseval = CODATA[version][asename]
        try:
            scipyval = scipy_CODATA[scipyname][0]
        except KeyError:
            # XXX Can we be more specific?
            continue  # 2002 in scipy contains too little data

        assert np.isclose(aseval, scipyval), scipyname


def test_create_units():
    """Check that units are created and allow attribute access."""

    # just use current CODATA version
    new_units = ase.units.create_units(ase.units.__codata_version__)
    assert new_units.eV == new_units['eV'] == ase.units.eV
    for unit_name in new_units:
        assert getattr(new_units, unit_name) == getattr(ase.units, unit_name)
        assert new_units[unit_name] == getattr(ase.units, unit_name)


def test_bad_codata():
    name = 'my_bad_codata_version'
    with pytest.raises(NotImplementedError, match=name):
        create_units(name)

"""
Check the many ways of specifying KPOINTS
"""
import os
import pytest

from ase.build import bulk

from .filecmp_ignore_whitespace import filecmp_ignore_whitespace

calc = pytest.mark.calculator


@pytest.fixture
def atoms():
    return bulk('Al', 'fcc', a=4.5, cubic=True)


def check_kpoints_line(n, contents):
    """Assert the contents of a line"""
    with open('KPOINTS', 'r') as fd:
        lines = fd.readlines()
    assert lines[n].strip() == contents


@pytest.fixture
def write_kpoints(atoms):
    """Helper fixture to write the input kpoints file"""
    def _write_kpoints(factory, **kwargs):
        calc = factory.calc(**kwargs)
        calc.initialize(atoms)
        calc.write_kpoints(atoms=atoms)
        return atoms, calc

    return _write_kpoints


@calc('vasp')
def test_vasp_kpoints_111(factory, write_kpoints):
    # Default to (1 1 1)
    write_kpoints(factory, gamma=True)
    check_kpoints_line(2, 'Gamma')
    check_kpoints_line(3, '1 1 1')


@calc('vasp')
def test_vasp_kpoints_3_tuple(factory, write_kpoints):

    # 3-tuple prints mesh
    write_kpoints(factory, gamma=False, kpts=(4, 4, 4))
    check_kpoints_line(2, 'Monkhorst-Pack')
    check_kpoints_line(3, '4 4 4')


@calc('vasp')
def test_vasp_kpoints_auto(factory, write_kpoints):
    # Auto mode
    write_kpoints(factory, kpts=20)
    check_kpoints_line(1, '0')
    check_kpoints_line(2, 'Auto')
    check_kpoints_line(3, '20')


@calc('vasp')
def test_vasp_kpoints_1_element_list_gamma(factory, write_kpoints):
    # 1-element list ok, Gamma ok
    write_kpoints(factory, kpts=[20], gamma=True)
    check_kpoints_line(1, '0')
    check_kpoints_line(2, 'Auto')
    check_kpoints_line(3, '20')


@calc('vasp')
def test_kspacing_supress_kpoints_file(factory, write_kpoints):
    # KSPACING suppresses KPOINTS file
    Al, calc = write_kpoints(factory, kspacing=0.23)
    calc.write_incar(Al)
    assert not os.path.isfile('KPOINTS')
    with open('INCAR', 'r') as fd:
        assert ' KSPACING = 0.230000\n' in fd.readlines()


@calc('vasp')
def test_negative_kspacing_error(factory, write_kpoints):
    # Negative KSPACING raises an error
    with pytest.raises(ValueError):
        write_kpoints(factory, kspacing=-0.5)


@calc('vasp')
def test_weighted(factory, write_kpoints):
    # Explicit weighted points with nested lists, Cartesian if not specified
    write_kpoints(factory,
                  kpts=[[0.1, 0.2, 0.3, 2], [0.0, 0.0, 0.0, 1],
                        [0.0, 0.5, 0.5, 2]])

    with open('KPOINTS.ref', 'w') as fd:
        fd.write("""KPOINTS created by Atomic Simulation Environment
    3 
    Cartesian
    0.100000 0.200000 0.300000 2.000000 
    0.000000 0.000000 0.000000 1.000000 
    0.000000 0.500000 0.500000 2.000000 
    """)

    assert filecmp_ignore_whitespace('KPOINTS', 'KPOINTS.ref')


@calc('vasp')
def test_explicit_auto_weight(factory, write_kpoints):
    # Explicit points as list of tuples, automatic weighting = 1.
    write_kpoints(factory,
                  kpts=[(0.1, 0.2, 0.3), (0.0, 0.0, 0.0), (0.0, 0.5, 0.5)],
                  reciprocal=True)

    with open('KPOINTS.ref', 'w') as fd:
        fd.write("""KPOINTS created by Atomic Simulation Environment
    3 
    Reciprocal
    0.100000 0.200000 0.300000 1.0 
    0.000000 0.000000 0.000000 1.0 
    0.000000 0.500000 0.500000 1.0 
    """)

    assert filecmp_ignore_whitespace('KPOINTS', 'KPOINTS.ref')

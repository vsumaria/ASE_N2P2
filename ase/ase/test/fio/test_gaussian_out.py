from io import StringIO

import numpy as np
import pytest

from ase.io import read
from ase.io.formats import match_magic
import ase.units as units


buf = r"""
 Entering Gaussian System, Link 0=g16

...

 ******************************************
 Gaussian 16:  ES64L-G16RevA.03 25-Dec-2016
                 6-Apr-2021
 ******************************************

...

                          Input orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        1.1            2.2        3.3
      2          1           0        4.4            5.5        6.6
      3          1           0        7.7            8.8        9.9
 ---------------------------------------------------------------------

...

 SCF Done:  E(RB3LYP) =  -12.3456789     A.U. after    9 cycles

...

 -------------------------------------------------------------------
 Center     Atomic                   Forces (Hartrees/Bohr)
 Number     Number              X              Y              Z
 -------------------------------------------------------------------
      1        8              0.1              0.2            0.3
      2        1              0.4              0.5            0.6
      3        1              0.7              0.8            0.9
 -------------------------------------------------------------------
"""


def test_match_magic():
    bytebuf = buf.encode('ascii')
    assert match_magic(bytebuf).name == 'gaussian-out'


def test_gaussian_out():
    fd = StringIO(buf)
    atoms = read(fd, format='gaussian-out')
    assert str(atoms.symbols) == 'OH2'
    assert atoms.positions == pytest.approx(np.array([
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        [7.7, 8.8, 9.9],
    ]))
    assert not any(atoms.pbc)
    assert atoms.cell.rank == 0

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert energy / units.Ha == pytest.approx(-12.3456789)
    assert forces / (units.Ha / units.Bohr) == pytest.approx(np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]))

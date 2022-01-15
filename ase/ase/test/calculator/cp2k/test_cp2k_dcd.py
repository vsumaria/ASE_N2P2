"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

import subprocess

import numpy as np
import pytest

from ase.build import molecule
from ase import io
from ase.io.cp2k import iread_cp2k_dcd
from ase.calculators.calculator import compare_atoms


inp = """\
&MOTION
  &PRINT
    &TRAJECTORY SILENT
      FORMAT DCD_ALIGNED_CELL
    &END TRAJECTORY
  &END PRINT
  &MD
    STEPS 5
  &END MD
&END MOTION
&GLOBAL
  RUN_TYPE MD
&END GLOBAL
"""


@pytest.mark.calculator_lite
@pytest.mark.calculator('cp2k')
def test_dcd(factory, factories):
    # (Should the cp2k_main executable live on the cp2k factory?)
    exes = factories.executables

    cp2k_main = exes.get('cp2k_main')
    if cp2k_main is None:
        pytest.skip('Please define "cp2k_main" in testing executables.  '
                    'It should point to the main cp2k executable '
                    '(not the shell)')

    calc = factory.calc(label='test_dcd', max_scf=1, inp=inp)
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    h2.set_pbc(True)
    energy = h2.get_potential_energy()
    assert energy is not None
    subprocess.check_call([cp2k_main, '-i', 'test_dcd.inp', '-o',
                           'test_dcd.out'])
    h2_end = io.read('test_dcd-pos-1.dcd')
    assert (h2_end.symbols == 'X').all()
    traj = io.read('test_dcd-pos-1.dcd', ref_atoms=h2,
                   index=slice(0, None), aligned=True)
    ioITraj = io.iread('test_dcd-pos-1.dcd', ref_atoms=h2,
                       index=slice(0, None), aligned=True)

    with open('test_dcd-pos-1.dcd', 'rb') as fd:
        itraj = iread_cp2k_dcd(fd, indices=slice(0, None),
                               ref_atoms=h2, aligned=True)
        for i, iMol in enumerate(itraj):
            ioIMol = next(ioITraj)
            assert compare_atoms(iMol, traj[i]) == []
            assert compare_atoms(iMol, ioIMol) == []
            assert iMol.get_pbc().all()

    traj = io.read('test_dcd-pos-1.dcd', ref_atoms=h2, index=slice(0, None))
    pbc = [mol.get_pbc() for mol in traj]
    assert not np.any(pbc)

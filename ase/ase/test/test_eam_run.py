import numpy as np

from ase.calculators.eam import EAM

from ase.build import fcc111
from io import StringIO


def test_eam_run(pt_eam_potential_file):
    eam = EAM(potential=StringIO(pt_eam_potential_file.read_text()),
              form='eam', elements=['Pt'])
    slab = fcc111('Pt', size=(4, 4, 2), vacuum=10.0)
    slab.calc = eam

    assert(abs(-164.277599313 - slab.get_potential_energy()) < 1E-8)
    assert(abs(6.36379627645 - np.linalg.norm(slab.get_forces())) < 1E-8)

import numpy as np
from numpy.testing import assert_allclose
import pytest
from ase.build import bulk
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS


@pytest.mark.calculator_lite
@pytest.mark.calculator('lammpsrun')
def test_Pt_stress_cellopt(factory, pt_eam_potential_file):
    params = {}
    params['pair_style'] = 'eam'
    params['pair_coeff'] = ['1 1 {}'.format(pt_eam_potential_file)]
    # XXX Should it accept Path objects?  Yes definitely for files.
    with factory.calc(specorder=['Pt'], files=[str(pt_eam_potential_file)],
                      **params) as calc:
        rng = np.random.RandomState(17)

        atoms = bulk('Pt') * (2, 2, 2)
        atoms.rattle(stdev=0.1)
        atoms.cell += 2 * rng.random((3, 3))
        atoms.calc = calc

        assert_allclose(atoms.get_stress(), calc.calculate_numerical_stress(atoms),
                        atol=1e-4, rtol=1e-4)

        with BFGS(ExpCellFilter(atoms)) as opt:
            for i, _ in enumerate(opt.irun(fmax=0.001)):
                pass

        cell1_ref = np.array(
            [[0.16524, 3.8999, 3.92855],
             [4.211015, 0.634928, 5.047811],
             [4.429529, 3.293805, 0.447377]]
        )

        assert_allclose(np.asarray(atoms.cell), cell1_ref,
                        atol=3e-4, rtol=3e-4)
        assert_allclose(atoms.get_stress(),
                        calc.calculate_numerical_stress(atoms),
                        atol=1e-4, rtol=1e-4)

        assert i < 80, 'Expected 59 iterations, got many more: {}'.format(i)

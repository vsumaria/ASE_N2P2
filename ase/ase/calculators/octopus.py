"""ASE-interface to Octopus.

Ask Hjorth Larsen <asklarsen@gmail.com>
Carlos de Armas

http://tddft.org/programs/octopus/
"""

import numpy as np
from ase.io.octopus.input import process_special_kwargs, generate_input
from ase.io.octopus.output import read_eigenvalues_file, read_static_info
from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator)


class OctopusIOError(IOError):
    pass


class OctopusProfile:
    def __init__(self, argv):
        self.argv = argv

    def version(self):
        from subprocess import check_output
        import re
        txt = check_output(self.argv + ['--version']).decode('ascii')
        match = re.match(r'octopus\s*(.+)', txt)
        # With MPI it prints the line for each rank, but we just match
        # the first line.
        return match.group(1)

    def run(self, directory, outputfile):
        from subprocess import check_call
        with open(directory / outputfile, 'w') as fd:
            check_call(self.argv, stdout=fd, cwd=directory)


class OctopusTemplate(CalculatorTemplate):
    def __init__(self):
        super().__init__(
            name='octopus',
            implemented_properties=['energy', 'forces', 'dipole', 'stress'],
        )

    def read_results(self, directory):
        """Read octopus output files and extract data."""
        results = {}
        with open(directory / 'static/info') as fd:
            results.update(read_static_info(fd))

        # If the eigenvalues file exists, we get the eigs/occs from that one.
        # This probably means someone ran Octopus in 'unocc' mode to
        # get eigenvalues (e.g. for band structures), and the values in
        # static/info will be the old (selfconsistent) ones.
        eigpath = directory / 'static/eigenvalues'
        if eigpath.is_file():
            with open(eigpath) as fd:
                kpts, eigs, occs = read_eigenvalues_file(fd)
                kpt_weights = np.ones(len(kpts))  # XXX ?  Or 1 / len(kpts) ?
                # XXX New Octopus probably has symmetry reduction !!
            results.update(eigenvalues=eigs, occupations=occs,
                           ibz_k_points=kpts,
                           k_point_weights=kpt_weights)
        return results

    def execute(self, directory, profile):
        profile.run(directory, 'octopus.out')

    def write_input(self, directory, atoms, parameters, properties):
        txt = generate_input(atoms, process_special_kwargs(atoms, parameters))
        inp = directory / 'inp'
        inp.write_text(txt)


class Octopus(GenericFileIOCalculator):
    """Octopus calculator.

    The label is always assumed to be a directory."""

    def __init__(self,
                 profile=None,
                 directory='.',
                 **kwargs):
        """Create Octopus calculator.

        Label is always taken as a subdirectory.
        Restart is taken to be a label."""

        if profile is None:
            profile = OctopusProfile(['octopus'])

        super().__init__(profile=profile, template=OctopusTemplate(),
                         directory=directory,
                         parameters=kwargs)

    @classmethod
    def recipe(cls, **kwargs):
        from ase import Atoms
        system = Atoms()
        calc = Octopus(CalculationMode='recipe', **kwargs)
        system.calc = calc
        try:
            system.get_potential_energy()
        except OctopusIOError:
            pass
        else:
            raise OctopusIOError('Expected recipe, but found '
                                 'useful physical output!')

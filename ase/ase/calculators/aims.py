"""This module defines an ASE interface to FHI-aims.

Felix Hanke hanke@liverpool.ac.uk
Jonas Bjork j.bjork@liverpool.ac.uk
Simon P. Rittmeyer simon.rittmeyer@tum.de
"""

import os
import time
import re
from pathlib import Path

import numpy as np

from ase.units import Hartree
from ase.io.aims import write_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import kpts2mp
from ase.calculators.genericfileio import (GenericFileIOCalculator,
                                           CalculatorTemplate)


def get_aims_version(string):
    match = re.search(r'\s*FHI-aims version\s*:\s*(\S+)', string, re.M)
    return match.group(1)


def write_control(fd, atoms, parameters, debug=False):
    parameters = dict(parameters)
    lim = '#' + '=' * 79

    if parameters['xc'] == 'LDA':
        parameters['xc'] = 'pw-lda'

    cubes = parameters.pop('cubes', None)

    fd.write(lim + '\n')
    for line in ['FHI-aims file',
                 'Created using the Atomic Simulation Environment (ASE)',
                 time.asctime(),
                 ]:
        fd.write('# ' + line + '\n')
    if debug:
        fd.write(
            '# \n# List of parameters used to initialize the calculator:')
        for p, v in parameters.items():
            s = '#     {} : {}\n'.format(p, v)
            fd.write(s)
    fd.write(lim + '\n')

    assert not ('kpts' in parameters and 'k_grid' in parameters)
    assert not ('smearing' in parameters and
                'occupation_type' in parameters)

    for key, value in parameters.items():
        if key == 'kpts':
            mp = kpts2mp(atoms, parameters['kpts'])
            fd.write('%-35s%d %d %d\n' % (('k_grid',) + tuple(mp)))
            dk = 0.5 - 0.5 / np.array(mp)
            fd.write('%-35s%f %f %f\n' % (('k_offset',) + tuple(dk)))
        elif key == 'species_dir':
            continue
        elif key == 'plus_u':
            continue
        elif key == 'smearing':
            name = parameters.smearing[0].lower()
            if name == 'fermi-dirac':
                name = 'fermi'
            width = parameters['smearing'][1]
            fd.write('%-35s%s %f' % ('occupation_type', name, width))
            if name == 'methfessel-paxton':
                order = parameters['smearing'][2]
                fd.write(' %d' % order)
            fd.write('\n' % order)
        elif key == 'output':
            for output_type in value:
                fd.write('%-35s%s\n' % (key, output_type))
        elif key == 'vdw_correction_hirshfeld' and value:
            fd.write('%-35s\n' % key)
        elif isinstance(value, bool):
            fd.write('%-35s.%s.\n' % (key, str(value).lower()))
        elif isinstance(value, (tuple, list)):
            fd.write('%-35s%s\n' %
                         (key, ' '.join(str(x) for x in value)))
        elif isinstance(value, str):
            fd.write('%-35s%s\n' % (key, value))
        else:
            fd.write('%-35s%r\n' % (key, value))

    if cubes:
        cubes.write(fd)

    fd.write(lim + '\n\n')


def translate_tier(tier):
    if tier.lower() == 'first':
        return 1
    elif tier.lower() == 'second':
        return 2
    elif tier.lower() == 'third':
        return 3
    elif tier.lower() == 'fourth':
        return 4
    else:
        return -1


def write_species(fd, atoms, parameters):
    parameters = dict(parameters)
    species_path = parameters.get('species_dir')
    if species_path is None:
        species_path = os.environ.get('AIMS_SPECIES_DIR')
    if species_path is None:
        raise RuntimeError(
            'Missing species directory!  Use species_dir ' +
            'parameter or set $AIMS_SPECIES_DIR environment variable.')

    species_path = Path(species_path)

    species = set(atoms.symbols)

    tier = parameters.pop('tier', None)

    if tier is not None:
        if isinstance(tier, int):
            tierlist = np.ones(len(species), 'int') * tier
        elif isinstance(tier, list):
            assert len(tier) == len(species)
            tierlist = tier

    for i, symbol in enumerate(species):
        path = species_path / ('%02i_%s_default' % (
            atomic_numbers[symbol], symbol))
        reached_tiers = False
        with open(path) as species_fd:
            for line in species_fd:
                if tier is not None:
                    if 'First tier' in line:
                        reached_tiers = True
                        targettier = tierlist[i]
                        foundtarget = False
                        do_uncomment = True
                    if reached_tiers:
                        line, foundtarget, do_uncomment = format_tiers(
                            line, targettier, foundtarget, do_uncomment)
                fd.write(line)

        if tier is not None and not foundtarget:
            raise RuntimeError(
                "Basis tier %i not found for element %s" %
                (targettier, symbol))

        if parameters.get('plus_u') is not None:
            if symbol in parameters.plus_u:
                fd.write('plus_u %s \n' %
                         parameters.plus_u[symbol])


def format_tiers(line, targettier, foundtarget, do_uncomment):
    if 'meV' in line:
        assert line[0] == '#'
        if 'tier' in line and 'Further' not in line:
            tier = line.split(" tier")[0]
            tier = tier.split('"')[-1]
            current_tier = translate_tier(tier)
            if current_tier == targettier:
                foundtarget = True
            elif current_tier > targettier:
                do_uncomment = False
        else:
            do_uncomment = False
        outputline = line
    elif do_uncomment and line[0] == '#':
        outputline = line[1:]
    elif not do_uncomment and line[0] != '#':
        outputline = '#' + line
    else:
        outputline = line
    return outputline, foundtarget, do_uncomment


class AimsProfile:
    def __init__(self, argv):
        self.argv = argv

    def run(self, directory, outputname):
        from subprocess import check_call
        with open(directory / outputname, 'w') as fd:
            check_call(self.argv, stdout=fd, cwd=directory)


class AimsTemplate(CalculatorTemplate):
    def __init__(self):
        super().__init__(
            'aims',
            ['energy', 'free_energy',
             'forces', 'stress', 'stresses',
             'dipole', 'magmom'])

        self.outputname = 'aims.out'

    def write_input(self, directory, atoms, parameters, properties):
        parameters = dict(parameters)
        ghosts = parameters.pop('ghosts', None)
        geo_constrain = parameters.pop('geo_constrain', None)
        scaled = parameters.pop('scaled', None)
        velocities = parameters.pop('velocities', None)

        if geo_constrain is None:
            geo_constrain = 'relax_geometry' in parameters

        if scaled is None:
            scaled = np.all(atoms.pbc)
        if velocities is None:
            velocities = atoms.has('momenta')

        have_lattice_vectors = atoms.pbc.any()
        have_k_grid = ('k_grid' in parameters or
                       'kpts' in parameters)
        if have_lattice_vectors and not have_k_grid:
            raise RuntimeError('Found lattice vectors but no k-grid!')
        if not have_lattice_vectors and have_k_grid:
            raise RuntimeError('Found k-grid but no lattice vectors!')

        geometry_in = directory / 'geometry.in'

        write_aims(
            geometry_in,
            atoms,
            scaled,
            geo_constrain,
            velocities=velocities,
            ghosts=ghosts
        )

        control = directory / 'control.in'
        with open(control, 'w') as fd:
            write_control(fd, atoms, parameters)
            write_species(fd, atoms, parameters)

    def execute(self, directory, profile):
        profile.run(directory, self.outputname)

    def read_results(self, directory):
        from ase.io.aims import read_aims_output

        dst = directory / self.outputname
        atoms = read_aims_output(dst, index=-1)
        return atoms.calc.properties()
        #converged = self.read_convergence()
        #if not converged:
        #    raise RuntimeError('FHI-aims did not converge!')
        #self.read_energy()
        # if ('compute_forces' in self.parameters or
        #    'sc_accuracy_forces' in self.parameters):
        #    self.read_forces()

        # if ('sc_accuracy_stress' in self.parameters or
        #        ('compute_numerical_stress' in self.parameters
        #         and self.parameters['compute_numerical_stress']) or
        #        ('compute_analytical_stress' in self.parameters
        #         and self.parameters['compute_analytical_stress']) or
        #        ('compute_heat_flux' in self.parameters
        #         and self.parameters['compute_heat_flux'])):
        #    self.read_stress()

        # if ('compute_heat_flux' in self.parameters
        #    and self.parameters['compute_heat_flux']):
        #    self.read_stresses()

        # if ('dipole' in self.parameters.get('output', []) and
        #    not self.atoms.pbc.any()):
        #    self.read_dipole()


class Aims(GenericFileIOCalculator):
    def __init__(self, profile=None, **kwargs):
        """Construct the FHI-aims calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' and 'smearing' or any of FHI-aims'
        native keywords.


        Arguments:

        cubes: AimsCube object
            Cube file specification.

        tier: int or array of ints
            Set basis set tier for all atomic species.

        plus_u : dict
            For DFT+U. Adds a +U term to one specific shell of the species.

        kwargs : dict
            Any of the base class arguments.

        """

        if profile is None:
            profile = AimsProfile(['aims'])

        super().__init__(template=AimsTemplate(),
                         profile=profile, parameters=kwargs)

    # def get_dipole_moment(self, atoms):
    #    if ('dipole' not in self.parameters.get('output', []) or
    #        atoms.pbc.any()):
    #        raise PropertyNotImplementedError
    #    return FileIOCalculator.get_dipole_moment(self, atoms)

    # def _get_stress(self, atoms):
    #    if ('compute_numerical_stress' not in self.parameters and
    #        'compute_analytical_stress' not in self.parameters):
    #        raise PropertyNotImplementedError
    #    return FileIOCalculator.get_stress(self, atoms)

    # def get_forces(self, atoms):
    #    if ('compute_forces' not in self.parameters and
    #        'sc_accuracy_forces' not in self.parameters):
    #        raise PropertyNotImplementedError
    #    return FileIOCalculator.get_forces(self, atoms)

    # def read_dipole(self):
    #    "Method that reads the electric dipole moment from the output file."
    #    for line in open(self.out, 'r'):
    #        if line.rfind('Total dipole moment [eAng]') > -1:
    #            dipolemoment = np.array([float(f)
    #                                     for f in line.split()[6:9]])
    #    self.results['dipole'] = dipolemoment

    # def read_energy(self):
    #    for line in open(self.out, 'r'):
    #        if line.rfind('Total energy corrected') > -1:
    #            E0 = float(line.split()[5])
    #        elif line.rfind('Total energy uncorrected') > -1:
    #            F = float(line.split()[5])
    #    self.results['free_energy'] = F
    #    self.results['energy'] = E0

    # def read_stress(self):
    #    lines = open(self.out, 'r').readlines()
    #    stress = None
    #    for n, line in enumerate(lines):
    #        if (line.rfind('|              Analytical stress tensor') > -1 or
    #            line.rfind('Numerical stress tensor') > -1):
    #            stress = []
    #            for i in [n + 5, n + 6, n + 7]:
    #                data = lines[i].split()
    #                stress += [float(data[2]), float(data[3]), float(data[4])]
    #    # rearrange in 6-component form and return
    #    self.results['stress'] = np.array([stress[0], stress[4], stress[8],
    #                                       stress[5], stress[2], stress[1]])

    def read_stresses(self):
        """ Read stress per atom """
        with open(self.out) as fd:
            next(l for l in fd if
                 'Per atom stress (eV) used for heat flux calculation' in l)
            # scroll to boundary
            next(l for l in fd if '-------------' in l)

            stresses = []
            for l in [next(fd) for _ in range(len(self.atoms))]:
                # Read stresses and rearrange from
                # (xx, yy, zz, xy, xz, yz) to (xx, yy, zz, yz, xz, xy)
                xx, yy, zz, xy, xz, yz = [float(d) for d in l.split()[2:8]]
                stresses.append([xx, yy, zz, yz, xz, xy])

            self.results['stresses'] = np.array(stresses)

    # def get_stresses(self, voigt=False):
    #    """ Return stress per atom

    #    Returns an array of the six independent components of the
    #    symmetric stress tensor per atom, in the traditional Voigt order
    #    (xx, yy, zz, yz, xz, xy) or as a 3x3 matrix.  Default is 3x3 matrix.
    #    """

    #    voigt_stresses = self.results['stresses']

    #    if voigt:
    #        return voigt_stresses
    #    else:
    #        stresses = np.zeros((len(self.atoms), 3, 3))
    #        for ii, stress in enumerate(voigt_stresses):
    #            xx, yy, zz, yz, xz, xy = stress
    #            stresses[ii] = np.array([(xx, xy, xz),
    #                                     (xy, yy, yz),
    #                                     (xz, yz, zz)])
    #        return stresses

    def read_convergence(self):
        converged = False
        lines = open(self.out, 'r').readlines()
        for n, line in enumerate(lines):
            if line.rfind('Have a nice day') > -1:
                converged = True
        return converged

    # def get_number_of_iterations(self):
    #    return self.read_number_of_iterations()

    def read_number_of_iterations(self):
        niter = None
        lines = open(self.out, 'r').readlines()
        for n, line in enumerate(lines):
            if line.rfind('| Number of self-consistency cycles') > -1:
                niter = int(line.split(':')[-1].strip())
        return niter

    # def get_electronic_temperature(self):
    #      return self.read_electronic_temperature()

    def read_electronic_temperature(self):
        width = None
        lines = open(self.out, 'r').readlines()
        for n, line in enumerate(lines):
            if line.rfind('Occupation type:') > -1:
                width = float(line.split('=')[-1].strip().split()[0])
        return width

    # def get_number_of_electrons(self):
    #    return self.read_number_of_electrons()

    def read_number_of_electrons(self):
        nelect = None
        lines = open(self.out, 'r').readlines()
        for n, line in enumerate(lines):
            if line.rfind('The structure contains') > -1:
                nelect = float(line.split()[-2].strip())
        return nelect

    # def get_number_of_bands(self):
    #    return self.read_number_of_bands()

    def read_number_of_bands(self):
        nband = None
        lines = open(self.out, 'r').readlines()
        for n, line in enumerate(lines):
            if line.rfind('Number of Kohn-Sham states') > -1:
                nband = int(line.split(':')[-1].strip())
        return nband

    # def get_k_point_weights(self):
    #    return self.read_kpts(mode='k_point_weights')

    # def get_bz_k_points(self):
    #    raise NotImplementedError

    # def get_ibz_k_points(self):
    #    return self.read_kpts(mode='ibz_k_points')

    # def get_spin_polarized(self):
    #    return self.read_number_of_spins()

    # def get_number_of_spins(self):
    #    return 1 + self.get_spin_polarized()

    # def get_magnetic_moment(self, atoms=None):
    #    return self.read_magnetic_moment()

    def read_number_of_spins(self):
        spinpol = None
        lines = open(self.out, 'r').readlines()
        for n, line in enumerate(lines):
            if line.rfind('| Number of spin channels') > -1:
                spinpol = int(line.split(':')[-1].strip()) - 1
        return spinpol

    def read_magnetic_moment(self):
        magmom = None
        if not self.get_spin_polarized():
            magmom = 0.0
        else:  # only for spinpolarized system Magnetisation is printed
            for line in open(self.out, 'r').readlines():
                if line.find('N_up - N_down') != -1:  # last one
                    magmom = float(line.split(':')[-1].strip())
        return magmom

    # def get_fermi_level(self):
    #     return self.read_fermi()

    # def get_eigenvalues(self, kpt=0, spin=0):
    #    return self.read_eigenvalues(kpt, spin, 'eigenvalues')

    # def get_occupations(self, kpt=0, spin=0):
    #    return self.read_eigenvalues(kpt, spin, 'occupations')

    def read_fermi(self):
        E_f = None
        lines = open(self.out, 'r').readlines()
        for n, line in enumerate(lines):
            if line.rfind('| Chemical potential (Fermi level) in eV') > -1:
                E_f = float(line.split(':')[-1].strip())
        return E_f

    def read_kpts(self, mode='ibz_k_points'):
        """ Returns list of kpts weights or kpts coordinates.  """
        values = []
        assert mode in ['ibz_k_points', 'k_point_weights']
        lines = open(self.out, 'r').readlines()
        kpts = None
        kptsstart = None
        for n, line in enumerate(lines):
            if line.rfind('| Number of k-points') > -1:
                kpts = int(line.split(':')[-1].strip())
        for n, line in enumerate(lines):
            if line.rfind('K-points in task') > -1:
                kptsstart = n  # last occurrence of (
        assert kpts is not None
        assert kptsstart is not None
        text = lines[kptsstart + 1:]
        values = []
        for line in text[:kpts]:
            if mode == 'ibz_k_points':
                b = [float(c.strip()) for c in line.split()[4:7]]
            else:
                b = float(line.split()[-1])
            values.append(b)
        if len(values) == 0:
            values = None
        return np.array(values)

    def read_eigenvalues(self, kpt=0, spin=0, mode='eigenvalues'):
        """ Returns list of last eigenvalues, occupations
        for given kpt and spin.  """
        values = []
        assert mode in ['eigenvalues', 'occupations']
        lines = open(self.out, 'r').readlines()
        # number of kpts
        kpts = None
        for n, line in enumerate(lines):
            if line.rfind('| Number of k-points') > -1:
                kpts = int(line.split(':')[-1].strip())
                break
        assert kpts is not None
        assert kpt + 1 <= kpts
        # find last (eigenvalues)
        eigvalstart = None
        for n, line in enumerate(lines):
            # eigenvalues come after Preliminary charge convergence reached
            if line.rfind('Preliminary charge convergence reached') > -1:
                eigvalstart = n
                break
        assert eigvalstart is not None
        lines = lines[eigvalstart:]
        for n, line in enumerate(lines):
            if line.rfind('Writing Kohn-Sham eigenvalues') > -1:
                eigvalstart = n
                break
        assert eigvalstart is not None
        text = lines[eigvalstart + 1:]  # remove first 1 line
        # find the requested k-point
        nbands = self.read_number_of_bands()
        sppol = self.get_spin_polarized()
        beg = ((nbands + 4 + int(sppol) * 1) * kpt * (sppol + 1) +
               3 + sppol * 2 + kpt * sppol)
        if self.get_spin_polarized():
            if spin == 0:
                beg = beg
                end = beg + nbands
            else:
                beg = beg + nbands + 5
                end = beg + nbands
        else:
            end = beg + nbands
        values = []
        for line in text[beg:end]:
            # aims prints stars for large values ...
            line = line.replace('**************', '         10000')
            line = line.replace('***************', '          10000')
            line = line.replace('****************', '           10000')
            b = [float(c.strip()) for c in line.split()[1:]]
            values.append(b)
        if mode == 'eigenvalues':
            values = [Hartree * v[1] for v in values]
        else:
            values = [v[0] for v in values]
        if len(values) == 0:
            values = None
        return np.array(values)


class AimsCube:
    "Object to ensure the output of cube files, can be attached to Aims object"
    def __init__(self, origin=(0, 0, 0),
                 edges=[(0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)],
                 points=(50, 50, 50), plots=tuple()):
        """parameters:

        origin, edges, points:
            Same as in the FHI-aims output
        plots:
            what to print, same names as in FHI-aims """

        self.name = 'AimsCube'
        self.origin = origin
        self.edges = edges
        self.points = points
        self.plots = plots

    def ncubes(self):
        """returns the number of cube files to output """
        return len(self.plots)

    def move_to_base_name(self, basename):
        """ when output tracking is on or the base namem is not standard,
        this routine will rename add the base to the cube file output for
        easier tracking """
        for plot in self.plots:
            found = False
            cube = plot.split()
            if (cube[0] == 'total_density' or
                cube[0] == 'spin_density' or
                cube[0] == 'delta_density'):
                found = True
                old_name = cube[0] + '.cube'
                new_name = basename + '.' + old_name
            if cube[0] == 'eigenstate' or cube[0] == 'eigenstate_density':
                found = True
                state = int(cube[1])
                s_state = cube[1]
                for i in [10, 100, 1000, 10000]:
                    if state < i:
                        s_state = '0' + s_state
                old_name = cube[0] + '_' + s_state + '_spin_1.cube'
                new_name = basename + '.' + old_name
            if found:
                os.system('mv ' + old_name + ' ' + new_name)

    def add_plot(self, name):
        """ in case you forgot one ... """
        self.plots += [name]

    def write(self, file):
        """ write the necessary output to the already opened control.in """
        file.write('output cube ' + self.plots[0] + '\n')
        file.write('   cube origin ')
        for ival in self.origin:
            file.write(str(ival) + ' ')
        file.write('\n')
        for i in range(3):
            file.write('   cube edge ' + str(self.points[i]) + ' ')
            for ival in self.edges[i]:
                file.write(str(ival) + ' ')
            file.write('\n')
        if self.ncubes() > 1:
            for i in range(self.ncubes() - 1):
                file.write('output cube ' + self.plots[i + 1] + '\n')

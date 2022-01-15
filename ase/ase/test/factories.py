import os
import re
from pathlib import Path
from typing import Mapping
import configparser

import pytest

from ase.calculators.calculator import (names as calculator_names,
                                        get_calculator_class)
from ase.calculators.genericfileio import read_stdout


class NotInstalled(Exception):
    pass


def get_testing_executables():
    # TODO: better cross-platform support (namely Windows),
    # and a cross-platform global config file like /etc/ase/ase.conf
    paths = [Path.home() / '.config' / 'ase' / 'ase.conf']
    try:
        paths += [Path(x) for x in os.environ['ASE_CONFIG'].split(':')]
    except KeyError:
        pass
    conf = configparser.ConfigParser()
    conf['executables'] = {}
    effective_paths = conf.read(paths)
    return effective_paths, conf['executables']


factory_classes = {}


def factory(name):
    def decorator(cls):
        cls.name = name
        factory_classes[name] = cls
        return cls

    return decorator


def make_factory_fixture(name):
    @pytest.fixture(scope='session')
    def _factory(factories):
        factories.require(name)
        return factories[name]

    _factory.__name__ = '{}_factory'.format(name)
    return _factory


@factory('abinit')
class AbinitFactory:
    def __init__(self, executable, pp_paths):
        self.executable = executable
        self.pp_paths = pp_paths
        self._version = None

    def version(self):
        from ase.calculators.abinit import get_abinit_version
        # XXX Ugly
        if self._version is None:
            self._version = get_abinit_version(self.executable)
        return self._version

    def is_legacy_version(self):
        version = self.version()
        major_ver = int(version.split('.')[0])
        return major_ver < 9

    def _base_kw(self):
        #command = f'{self.executable} PREFIX.in > PREFIX.log'
        return dict(pp_paths=self.pp_paths,
                    ecut=150,
                    chksymbreak=0,
                    toldfe=1e-3)

    def calc(self, **kwargs):
        from ase.calculators.abinit import Abinit, AbinitProfile

        profile = AbinitProfile([self.executable])

        if self.is_legacy_version():
            raise RuntimeError('Sorry, Abinit 9+ is required.')

        kw = self._base_kw()
        assert kw['pp_paths'] is not None
        kw.update(kwargs)
        assert kw['pp_paths'] is not None
        return Abinit(profile=profile, **kw)

    @classmethod
    def fromconfig(cls, config):
        factory = AbinitFactory(config.executables['abinit'],
                                config.datafiles['abinit'])
        # XXX Hack
        factory._version = factory.version()
        return factory


@factory('aims')
class AimsFactory:
    def __init__(self, executable):
        self.executable = executable
        # XXX pseudo_dir

    def calc(self, **kwargs):
        from ase.calculators.aims import Aims, AimsProfile
        kwargs1 = dict(xc='LDA')
        kwargs1.update(kwargs)
        profile = AimsProfile([self.executable])
        return Aims(profile=profile, **kwargs1)

    def version(self):
        from ase.calculators.aims import get_aims_version
        txt = read_stdout([self.executable])
        return get_aims_version(txt)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['aims'])


@factory('asap')
class AsapFactory:
    importname = 'asap3'

    def calc(self, **kwargs):
        from asap3 import EMT
        return EMT(**kwargs)

    def version(self):
        import asap3
        return asap3.__version__

    @classmethod
    def fromconfig(cls, config):
        # XXXX TODO Clean this up.  Copy of GPAW.
        # How do we design these things?
        import importlib
        spec = importlib.util.find_spec('asap3')
        if spec is None:
            raise NotInstalled('asap3')
        return cls()


@factory('cp2k')
class CP2KFactory:
    def __init__(self, executable):
        self.executable = executable

    def version(self):
        from ase.calculators.cp2k import Cp2kShell
        shell = Cp2kShell(self.executable, debug=False)
        return shell.version

    def calc(self, **kwargs):
        from ase.calculators.cp2k import CP2K
        return CP2K(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return CP2KFactory(config.executables['cp2k'])


@factory('castep')
class CastepFactory:
    def __init__(self, executable):
        self.executable = executable

    def version(self):
        from ase.calculators.castep import get_castep_version
        return get_castep_version(self.executable)

    def calc(self, **kwargs):
        from ase.calculators.castep import Castep
        return Castep(castep_command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['castep'])


@factory('dftb')
class DFTBFactory:
    def __init__(self, executable, skt_paths):
        self.executable = executable
        assert len(skt_paths) == 1
        self.skt_path = skt_paths[0]

    def version(self):
        stdout = read_stdout([self.executable])
        match = re.search(r'DFTB\+ release\s*(\S+)', stdout, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.dftb import Dftb
        command = f'{self.executable} > PREFIX.out'
        return Dftb(
            command=command,
            slako_dir=str(self.skt_path) + '/',  # XXX not obvious
            **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['dftb'], config.datafiles['dftb'])


@factory('dftd3')
class DFTD3Factory:
    def __init__(self, executable):
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.dftd3 import DFTD3
        return DFTD3(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['dftd3'])


@factory('elk')
class ElkFactory:
    def __init__(self, executable, species_dir):
        self.executable = executable
        self.species_dir = species_dir

    def version(self):
        output = read_stdout([self.executable])
        match = re.search(r'Elk code version (\S+)', output, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.elk import ELK
        command = f'{self.executable} > elk.out'
        return ELK(command=command, species_dir=self.species_dir, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['elk'], config.datafiles['elk'][0])


@factory('espresso')
class EspressoFactory:
    def __init__(self, executable, pseudo_dir):
        self.executable = executable
        self.pseudo_dir = pseudo_dir

    def _base_kw(self):
        from ase.units import Ry
        return dict(ecutwfc=300 / Ry)

    def _profile(self):
        from ase.calculators.espresso import EspressoProfile
        return EspressoProfile([self.executable])

    def version(self):
        self._profile().version()

    def calc(self, **kwargs):
        from ase.calculators.espresso import Espresso

        pseudopotentials = {}
        for path in self.pseudo_dir.glob('*.UPF'):
            fname = path.name
            # Names are e.g. si_lda_v1.uspp.F.UPF
            symbol = fname.split('_', 1)[0].capitalize()
            pseudopotentials[symbol] = fname

        kw = self._base_kw()
        kw.update(kwargs)
        return Espresso(profile=self._profile(),
                        pseudo_dir=str(self.pseudo_dir),
                        pseudopotentials=pseudopotentials,
                        **kw)

    @classmethod
    def fromconfig(cls, config):
        paths = config.datafiles['espresso']
        assert len(paths) == 1
        return cls(config.executables['espresso'], paths[0])


@factory('exciting')
class ExcitingFactory:
    def __init__(self, executable):
        # XXX species path
        self.executable = executable

    def calc(self, **kwargs):
        from ase.calculators.exciting import Exciting
        return Exciting(bin=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['exciting'])


@factory('vasp')
class VaspFactory:
    def __init__(self, executable):
        self.executable = executable

    def version(self):
        from ase.calculators.vasp import get_vasp_version
        header = read_stdout([self.executable], createfile='INCAR')
        return get_vasp_version(header)

    def calc(self, **kwargs):
        from ase.calculators.vasp import Vasp
        # XXX We assume the user has set VASP_PP_PATH
        if Vasp.VASP_PP_PATH not in os.environ:
            # For now, we skip with a message that we cannot run the test
            pytest.skip(
                'No VASP pseudopotential path set. Set the ${} environment variable to enable.'
                .format(Vasp.VASP_PP_PATH))
        return Vasp(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['vasp'])


@factory('gpaw')
class GPAWFactory:
    importname = 'gpaw'

    def calc(self, **kwargs):
        from gpaw import GPAW
        return GPAW(**kwargs)

    def version(self):
        import gpaw
        return gpaw.__version__

    @classmethod
    def fromconfig(cls, config):
        import importlib
        spec = importlib.util.find_spec('gpaw')
        # XXX should be made non-pytest dependent
        if spec is None:
            raise NotInstalled('gpaw')
        return cls()


@factory('psi4')
class Psi4Factory:
    importname = 'psi4'

    def calc(self, **kwargs):
        from ase.calculators.psi4 import Psi4
        return Psi4(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        try:
            import psi4  # noqa
        except ModuleNotFoundError:
            raise NotInstalled('psi4')
        return cls()


@factory('gromacs')
class GromacsFactory:
    def __init__(self, executable):
        self.executable = executable

    def version(self):
        from ase.calculators.gromacs import get_gromacs_version
        return get_gromacs_version(self.executable)

    def calc(self, **kwargs):
        from ase.calculators.gromacs import Gromacs
        return Gromacs(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['gromacs'])


class BuiltinCalculatorFactory:
    def calc(self, **kwargs):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self.name)
        return cls(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls()


@factory('emt')
class EMTFactory(BuiltinCalculatorFactory):
    pass


@factory('lammpsrun')
class LammpsRunFactory:
    def __init__(self, executable):
        self.executable = executable

    def version(self):
        stdout = read_stdout([self.executable])
        match = re.match(r'LAMMPS\s*\((.+?)\)', stdout, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.lammpsrun import LAMMPS
        return LAMMPS(command=self.executable, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['lammpsrun'])


@factory('lammpslib')
class LammpsLibFactory:
    def __init__(self, potentials_path):
        # Set the path where LAMMPS will look for potential parameter files
        os.environ["LAMMPS_POTENTIALS"] = str(potentials_path)
        self.potentials_path = potentials_path

    def version(self):
        import lammps
        cmd_args = [
            "-echo", "log", "-log", "none", "-screen", "none", "-nocite"
        ]
        lmp = lammps.lammps(name="", cmdargs=cmd_args, comm=None)
        try:
            return lmp.version()
        finally:
            lmp.close()

    def calc(self, **kwargs):
        from ase.calculators.lammpslib import LAMMPSlib
        return LAMMPSlib(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.datafiles['lammps'][0])


@factory('openmx')
class OpenMXFactory:
    def __init__(self, executable, data_path):
        self.executable = executable
        self.data_path = data_path

    def version(self):
        from ase.calculators.openmx.openmx import parse_omx_version
        dummyfile = 'omx_dummy_input'
        stdout = read_stdout([self.executable, dummyfile],
                             createfile=dummyfile)
        return parse_omx_version(stdout)

    def calc(self, **kwargs):
        from ase.calculators.openmx import OpenMX
        return OpenMX(command=self.executable,
                      data_path=str(self.data_path),
                      **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['openmx'],
                   data_path=config.datafiles['openmx'][0])


@factory('octopus')
class OctopusFactory:
    def __init__(self, executable):
        self.executable = executable

    def _profile(self):
        from ase.calculators.octopus import OctopusProfile
        return OctopusProfile([self.executable])

    def version(self):
        return self._profile().version()

    def calc(self, **kwargs):
        from ase.calculators.octopus import Octopus
        return Octopus(profile=self._profile(), **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['octopus'])


@factory('siesta')
class SiestaFactory:
    def __init__(self, executable, pseudo_path):
        self.executable = executable
        self.pseudo_path = pseudo_path

    def version(self):
        from ase.calculators.siesta.siesta import get_siesta_version
        full_ver = get_siesta_version(self.executable)
        m = re.match(r'siesta-(\S+)', full_ver, flags=re.I)
        if m:
            return m.group(1)
        return full_ver

    def calc(self, **kwargs):
        from ase.calculators.siesta import Siesta
        command = '{} < PREFIX.fdf > PREFIX.out'.format(self.executable)
        return Siesta(command=command,
                      pseudo_path=str(self.pseudo_path),
                      **kwargs)

    @classmethod
    def fromconfig(cls, config):
        paths = config.datafiles['siesta']
        assert len(paths) == 1
        path = paths[0]
        return cls(config.executables['siesta'], str(path))


@factory('nwchem')
class NWChemFactory:
    def __init__(self, executable):
        self.executable = executable

    def version(self):
        stdout = read_stdout([self.executable], createfile='nwchem.nw')
        match = re.search(
            r'Northwest Computational Chemistry Package \(NWChem\) (\S+)',
            stdout, re.M)
        return match.group(1)

    def calc(self, **kwargs):
        from ase.calculators.nwchem import NWChem
        command = f'{self.executable} PREFIX.nwi > PREFIX.nwo'
        return NWChem(command=command, **kwargs)

    @classmethod
    def fromconfig(cls, config):
        return cls(config.executables['nwchem'])


@factory('plumed')
class PlumedFactory:
    def __init__(self):
        import plumed
        self.path = plumed.__spec__.origin
        
    def calc(self, **kwargs):
        from ase.calculators.plumed import Plumed
        return Plumed(**kwargs)

    @classmethod
    def fromconfig(cls, config):
        import importlib
        spec = importlib.util.find_spec('plumed')
        # XXX should be made non-pytest dependent
        if spec is None:
            raise NotInstalled('plumed')
        return cls()


class NoSuchCalculator(Exception):
    pass


class Factories:
    all_calculators = set(calculator_names)
    builtin_calculators = {'eam', 'emt', 'ff', 'lj', 'morse', 'tip3p', 'tip4p'}
    autoenabled_calculators = {'asap'} | builtin_calculators

    # TODO: Port calculators to use factories.  As we do so, remove names
    # from list of calculators that we monkeypatch:
    monkeypatch_calculator_constructors = {
        'ace',
        'aims',
        'amber',
        'crystal',
        'demon',
        'demonnano',
        'dftd3',
        'dmol',
        'exciting',
        'fleur',
        'gamess_us',
        'gaussian',
        'gulp',
        'hotbit',
        'lammpslib',
        'mopac',
        'onetep',
        'orca',
        'qchem',
        'turbomole',
    }

    def __init__(self, requested_calculators):
        executable_config_paths, executables = get_testing_executables()
        assert isinstance(executables, Mapping), executables
        self.executables = executables
        self.executable_config_paths = executable_config_paths

        datafiles_module = None
        datafiles = {}

        try:
            import asetest as datafiles_module
        except ImportError:
            pass
        else:
            datafiles.update(datafiles_module.datafiles.paths)
            datafiles_module = datafiles_module

        self.datafiles_module = datafiles_module
        self.datafiles = datafiles

        factories = {}

        for name, cls in factory_classes.items():
            try:
                factory = cls.fromconfig(self)
            except (NotInstalled, KeyError):
                pass
            else:
                factories[name] = factory

        self.factories = factories

        requested_calculators = set(requested_calculators)
        if 'auto' in requested_calculators:
            requested_calculators.remove('auto')
            requested_calculators |= set(self.factories)
        self.requested_calculators = requested_calculators

        for name in self.requested_calculators:
            if name not in self.all_calculators:
                raise NoSuchCalculator(name)

    def installed(self, name):
        return name in self.builtin_calculators | set(self.factories)

    def is_adhoc(self, name):
        return name not in factory_classes

    def optional(self, name):
        return name not in self.builtin_calculators

    def enabled(self, name):
        auto = name in self.autoenabled_calculators and self.installed(name)
        return auto or (name in self.requested_calculators)

    def require(self, name):
        # XXX This is for old-style calculator tests.
        # Newer calculator tests would depend on a fixture which would
        # make them skip.
        # Older tests call require(name) explicitly.
        assert name in calculator_names
        if not self.installed(name) and not self.is_adhoc(name):
            pytest.skip(f'Not installed: {name}')
        if name not in self.requested_calculators:
            pytest.skip(f'Use --calculators={name} to enable')

    def __getitem__(self, name):
        return self.factories[name]

    def monkeypatch_disabled_calculators(self):
        test_calculator_names = (self.autoenabled_calculators
                                 | self.builtin_calculators
                                 | self.requested_calculators)
        disable_names = self.monkeypatch_calculator_constructors - test_calculator_names
        #disable_names = self.all_calculators - test_calculator_names

        for name in disable_names:
            try:
                cls = get_calculator_class(name)
            except ImportError:
                pass
            else:

                def get_mock_init(name):
                    def mock_init(obj, *args, **kwargs):
                        pytest.skip(f'use --calculators={name} to enable')

                    return mock_init

                def mock_del(obj):
                    pass

                cls.__init__ = get_mock_init(name)
                cls.__del__ = mock_del


def get_factories(pytestconfig):
    opt = pytestconfig.getoption('--calculators')
    requested_calculators = opt.split(',') if opt else []
    return Factories(requested_calculators)


def parametrize_calculator_tests(metafunc):
    """Parametrize tests using our custom markers.

    We want tests marked with @pytest.mark.calculator(names) to be
    parametrized over the named calculator or calculators."""
    calculator_inputs = []

    for marker in metafunc.definition.iter_markers(name='calculator'):
        calculator_names = marker.args
        kwargs = dict(marker.kwargs)
        marks = kwargs.pop('marks', [])
        for name in calculator_names:
            param = pytest.param((name, kwargs), marks=marks)
            calculator_inputs.append(param)

    if calculator_inputs:
        metafunc.parametrize('factory',
                             calculator_inputs,
                             indirect=True,
                             ids=lambda input: input[0])


class CalculatorInputs:
    def __init__(self, factory, parameters=None):
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        self.factory = factory

    def require_version(self, version):
        from ase.utils import tokenize_version
        installed_version = self.factory.version()
        old = tokenize_version(installed_version) < tokenize_version(version)
        if old:
            pytest.skip('Version too old: Requires {}; got {}'
                        .format(version, installed_version))

    @property
    def name(self):
        return self.factory.name

    def __repr__(self):
        cls = type(self)
        return '{}({}, {})'.format(cls.__name__, self.name, self.parameters)

    def new(self, **kwargs):
        kw = dict(self.parameters)
        kw.update(kwargs)
        return CalculatorInputs(self.factory, kw)

    def calc(self, **kwargs):
        param = dict(self.parameters)
        param.update(kwargs)
        return self.factory.calc(**param)


class ObsoleteFactoryWrapper:
    # We use this for transitioning older tests to the new framework.
    def __init__(self, name):
        self.name = name

    def calc(self, **kwargs):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self.name)
        return cls(**kwargs)

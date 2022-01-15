from os import PathLike
from pathlib import Path
from typing import Iterable, Mapping, Any
from abc import ABC, abstractmethod

from ase.calculators.abc import GetOutputsMixin
from ase.calculators.calculator import BaseCalculator


def read_stdout(args, createfile=None):
    """Run command in tempdir and return standard output.

    Helper function for getting version numbers of DFT codes.
    Most DFT codes don't implement a --version flag, so in order to
    determine the code version, we just run the code until it prints
    a version number."""
    import tempfile
    from subprocess import Popen, PIPE
    with tempfile.TemporaryDirectory() as directory:
        if createfile is not None:
            path = Path(directory) / createfile
            path.touch()
        proc = Popen(args,
                     stdout=PIPE,
                     stderr=PIPE,
                     stdin=PIPE,
                     cwd=directory,
                     encoding='ascii')
        stdout, _ = proc.communicate()
        # Exit code will be != 0 because there isn't an input file
    return stdout


class CalculatorTemplate(ABC):
    def __init__(self, name: str, implemented_properties: Iterable[str]):
        self.name = name
        self.implemented_properties = frozenset(implemented_properties)

    @abstractmethod
    def write_input(self, directory, atoms, parameters, properties):
        ...

    @abstractmethod
    def execute(self, directory, profile):
        ...

    @abstractmethod
    def read_results(self, directory: PathLike) -> Mapping[str, Any]:
        ...


class GenericFileIOCalculator(BaseCalculator, GetOutputsMixin):
    def __init__(self, *, template, profile, directory='.', parameters=None):
        self.template = template
        self.profile = profile

        # Maybe we should allow directory to be a factory, so
        # calculators e.g. produce new directories on demand.
        self.directory = Path(directory)

        super().__init__(parameters)

    def set(self, *args, **kwargs):
        raise RuntimeError('No setting parameters for now, please.  '
                           'Just create new calculators.')

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, self.template.name)

    @property
    def implemented_properties(self):
        return self.template.implemented_properties

    @property
    def name(self):
        return self.template.name

    def calculate(self, atoms, properties, system_changes):
        self.atoms = atoms.copy()

        directory = self.directory

        self.template.write_input(directory, atoms, self.parameters,
                                  properties)
        self.template.execute(directory, self.profile)
        self.results = self.template.read_results(directory)
        # XXX Return something useful?

    def _outputmixin_get_results(self):
        return self.results

import os
import pytest
from ase import Atoms
from ase.calculators.vasp import Vasp


@pytest.fixture
def atoms_co():
    """Simple atoms object for testing with a single CO molecule"""
    d = 1.14
    atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)], pbc=True)
    atoms.center(vacuum=5)
    return atoms


@pytest.fixture
def atoms_2co():
    """Simple atoms object for testing with 2x CO molecules"""
    d = 1.14
    atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)], pbc=True)
    atoms.extend(Atoms('CO', positions=[(0, 2, 0), (0, 2, d)]))

    atoms.center(vacuum=5.)
    return atoms


@pytest.fixture
def mock_vasp_calculate(mocker):
    """Fixture which mocks the VASP run method, so a calculation cannot run.
    Acts as a safeguard for tests which want to test VASP,
    but avoid accidentally launching a calculation"""
    def _mock_run(self, command=None, out=None, directory=None):
        assert False, 'Test attempted to launch a calculation'

    # Patch the calculate and run methods, so we're certain
    # calculations aren't accidentally launched
    mocker.patch('ase.calculators.vasp.Vasp._run', _mock_run)
    yield


@pytest.fixture
def clear_vasp_envvar(monkeypatch):
    """Clear the environment variables which can be used to launch
    a VASP calculation."""
    for envvar in Vasp.env_commands:
        monkeypatch.delenv(envvar, raising=False)
        assert envvar not in os.environ
    yield

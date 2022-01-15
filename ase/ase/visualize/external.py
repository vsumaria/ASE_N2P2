from io import BytesIO
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager

from ase.io.formats import ioformats
from ase.io import write


def _pipe_to_ase_gui(atoms, repeat):
    buf = BytesIO()
    write(buf, atoms, format='traj')

    args = [sys.executable, '-m', 'ase', 'gui', '-']
    if repeat:
        args.append('--repeat={},{},{}'.format(*repeat))

    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    proc.stdin.write(buf.getvalue())
    proc.stdin.close()
    return proc


class CLIViewer:
    def __init__(self, name, fmt, argv):
        self.name = name
        self.fmt = fmt
        self.argv = argv

    @property
    def ioformat(self):
        return ioformats[self.fmt]

    @contextmanager
    def mktemp(self, atoms, data=None):
        ioformat = self.ioformat
        suffix = '.' + ioformat.extensions[0]

        if ioformat.isbinary:
            mode = 'wb'
        else:
            mode = 'w'

        with tempfile.TemporaryDirectory(prefix='ase-view-') as dirname:
            # We use a tempdir rather than a tempfile because it's
            # less hassle to handle the cleanup on Windows (files
            # cannot be open on multiple processes).
            path = Path(dirname) / f'atoms{suffix}'
            with path.open(mode) as fd:
                if data is None:
                    write(fd, atoms, format=self.fmt)
                else:
                    write(fd, atoms, format=self.fmt, data=data)
            yield path

    def view_blocking(self, atoms, data=None):
        with self.mktemp(atoms, data) as path:
            subprocess.check_call(self.argv + [str(path)])

    def view(self, atoms, data=None, repeat=None):
        """Spawn a new process in which to open the viewer."""
        if repeat is not None:
            atoms = atoms.repeat(repeat)

        proc = subprocess.Popen(
            [sys.executable, '-m', 'ase.visualize.external'],
            stdin=subprocess.PIPE)

        pickle.dump((self, atoms, data), proc.stdin)
        proc.stdin.close()
        return proc

    @classmethod
    def viewers(cls):
        # paraview_script = Path(__file__).parent / 'paraview_script.py'
        # Can we make paraview/vtkxml work on some test system?
        return [
            cls('ase_gui_cli', 'traj', [sys.executable, '-m', 'ase.gui']),
            cls('avogadro', 'cube', ['avogadro']),
            cls('gopenmol', 'extxyz', ['runGOpenMol']),
            cls('rasmol', 'proteindatabank', ['rasmol', '-pdb']),
            cls('vmd', 'cube', ['vmd']),
            cls('xmakemol', 'extxyz', ['xmakemol', '-f']),
            # cls('paraview', 'vtu',
            #     ['paraview', f'--script={paraview_script}'])
        ]


class PyViewer:
    def __init__(self, name, supports_repeat=False):
        self.name = name
        self.supports_repeat = supports_repeat

    def view(self, atoms, data=None, repeat=None):
        # Delegate to any of the below methods
        func = getattr(self, self.name)
        if self.supports_repeat:
            return func(atoms, repeat)
        else:
            if repeat is not None:
                atoms = atoms.repeat(repeat)
            return func(atoms)

    def sage(self, atoms):
        from ase.visualize.sage import view_sage_jmol
        return view_sage_jmol(atoms)

    def ngl(self, atoms):
        from ase.visualize.ngl import view_ngl
        return view_ngl(atoms)

    def x3d(self, atoms):
        from ase.visualize.x3d import view_x3d
        return view_x3d(atoms)

    def ase(self, atoms, repeat):
        return _pipe_to_ase_gui(atoms, repeat)

    @classmethod
    def viewers(cls):
        return [
            cls('ase', supports_repeat=True),
            cls('ngl'),
            cls('sage'),
            cls('x3d'),
        ]


viewers = {viewer.name: viewer
           for viewer in CLIViewer.viewers() + PyViewer.viewers()}
viewers['nglview'] = viewers['ngl']


def main():
    cli_viewer, atoms, data = pickle.load(sys.stdin.buffer)
    cli_viewer.view_blocking(atoms, data)


if __name__ == '__main__':
    main()

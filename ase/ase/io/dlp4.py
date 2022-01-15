""" Read/Write DL_POLY_4 CONFIG files """
import re

from numpy import zeros, isscalar

from ase.atoms import Atoms
from ase.units import _auf, _amu, _auv
from ase.data import chemical_symbols
from ase.calculators.singlepoint import SinglePointCalculator

__all__ = ['read_dlp4', 'write_dlp4']

# dlp4 labels will be registered in atoms.arrays[DLP4_LABELS_KEY]
DLP4_LABELS_KEY = 'dlp4_labels'
DLP4_DISP_KEY = 'dlp4_displacements'
dlp_f_si = 1.0e-10 * _amu / (1.0e-12 * 1.0e-12)  # Å*Da/ps^2
dlp_f_ase = dlp_f_si / _auf
dlp_v_si = 1.0e-10 / 1.0e-12  # Å/ps
dlp_v_ase = dlp_v_si / _auv


def _get_frame_positions(f):
    """Get positions of frames in HISTORY file."""
    # header line contains name of system
    init_pos = f.tell()
    f.seek(0)
    rl = len(f.readline())  # system name, and record size
    items = f.readline().strip().split()
    if len(items) == 5:
        classic = False
    elif len(items) == 3:
        classic = True
    else:
        raise RuntimeError("Cannot determine version of HISTORY file format.")

    levcfg, imcon, natoms = [int(x) for x in items[0:3]]
    if classic:
        # we have to iterate over the entire file
        startpos = f.tell()
        pos = []
        line = True
        while line:
            line = f.readline()
            if 'timestep' in line:
                pos.append(f.tell())
        f.seek(startpos)
    else:
        nframes = int(items[3])
        pos = [((natoms * (levcfg + 2) + 4) * i + 3)
               * rl for i in range(nframes)]

    f.seek(init_pos)
    return levcfg, imcon, natoms, pos


def read_dlp_history(f, index=-1, symbols=None):
    """Read a HISTORY file.

    Compatible with DLP4 and DLP_CLASSIC.

    *Index* can be integer or slice.

    Provide a list of element strings to symbols to ignore naming
    from the HISTORY file.
    """
    levcfg, imcon, natoms, pos = _get_frame_positions(f)
    if isscalar(index):
        selected = [pos[index]]
    else:
        selected = pos[index]

    images = []
    for fpos in selected:
        f.seek(fpos + 1)
        images.append(read_single_image(f, levcfg, imcon, natoms,
                                        is_trajectory=True, symbols=symbols))

    return images


def iread_dlp_history(f, symbols=None):
    """Generator version of read_history"""
    levcfg, imcon, natoms, pos = _get_frame_positions(f)
    for p in pos:
        f.seek(p + 1)
        yield read_single_image(f, levcfg, imcon, natoms, is_trajectory=True,
                                symbols=symbols)


def read_dlp4(f, symbols=None):
    """Read a DL_POLY_4 config/revcon file.

    Typically used indirectly through read('filename', atoms, format='dlp4').

    Can be unforgiven with custom chemical element names.
    Please complain to alin@elena.space for bugs."""

    line = f.readline()
    line = f.readline().split()
    levcfg = int(line[0])
    imcon = int(line[1])

    position = f.tell()
    line = f.readline()
    tokens = line.split()
    is_trajectory = tokens[0] == 'timestep'
    if not is_trajectory:
        # Difficult to distinguish between trajectory and non-trajectory
        # formats without reading one line too much.
        f.seek(position)

    while line:
        if is_trajectory:
            tokens = line.split()
            natoms = int(tokens[2])
        else:
            natoms = None
        yield read_single_image(f, levcfg, imcon, natoms, is_trajectory,
                                symbols)
        line = f.readline()


def read_single_image(f, levcfg, imcon, natoms, is_trajectory, symbols=None):
    cell = zeros((3, 3))
    ispbc = imcon > 0
    if ispbc or is_trajectory:
        for j in range(3):
            line = f.readline()
            line = line.split()
            for i in range(3):
                try:
                    cell[j, i] = float(line[i])
                except ValueError:
                    raise RuntimeError("error reading cell")
    if symbols:
        sym = symbols
    else:
        sym = []
    positions = []
    velocities = []
    forces = []
    charges = []
    masses = []
    disp = []

    if is_trajectory:
        counter = range(natoms)
    else:
        from itertools import count
        counter = count()

    labels = []
    mass = None
    ch = None
    d = None

    for a in counter:
        line = f.readline()
        if not line:
            a -= 1
            break

        m = re.match(r'\s*([A-Za-z][a-z]?)(\S*)', line)
        assert m is not None, line
        symbol, label = m.group(1, 2)
        symbol = symbol.capitalize()
        if is_trajectory:
            ll = line.split()
            if len(ll) == 5:
                mass, ch, d = [float(i) for i in line.split()[2:5]]
            else:
                mass, ch = [float(i) for i in line.split()[2:4]]

            charges.append(ch)
            masses.append(mass)
            disp.append(d)
        if not symbols:
            assert symbol in chemical_symbols
            sym.append(symbol)
        # make sure label is not empty
        if label:
            labels.append(label)
        else:
            labels.append('')

        x, y, z = f.readline().split()[:3]
        positions.append([float(x), float(y), float(z)])
        if levcfg > 0:
            vx, vy, vz = f.readline().split()[:3]
            velocities.append([float(vx) * dlp_v_ase, float(vy)
                               * dlp_v_ase, float(vz) * dlp_v_ase])
        if levcfg > 1:
            fx, fy, fz = f.readline().split()[:3]
            forces.append([float(fx) * dlp_f_ase, float(fy)
                           * dlp_f_ase, float(fz) * dlp_f_ase])

    if symbols and (a + 1 != len(symbols)):
        raise ValueError("Counter is at {} but you gave {} symbols"
                         .format(a + 1, len(symbols)))

    if imcon == 0:
        pbc = False
    elif imcon == 6:
        pbc = [True, True, False]
    else:
        assert imcon in [1, 2, 3]
        pbc = True

    atoms = Atoms(positions=positions,
                  symbols=sym,
                  cell=cell,
                  pbc=pbc,
                  # Cell is centered around (0, 0, 0) in dlp4:
                  celldisp=-cell.sum(axis=0) / 2)

    if is_trajectory:
        atoms.set_masses(masses)
        atoms.set_array(DLP4_DISP_KEY, disp, float)
        atoms.set_initial_charges(charges)

    atoms.set_array(DLP4_LABELS_KEY, labels, str)
    if levcfg > 0:
        atoms.set_velocities(velocities)
    if levcfg > 1:
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
    return atoms


def write_dlp4(fd, atoms, levcfg=0, title='CONFIG generated by ASE'):
    """Write a DL_POLY_4 config file.

    Typically used indirectly through write('filename', atoms, format='dlp4').

    Can be unforgiven with custom chemical element names.
    Please complain to alin@elena.space in case of bugs"""

    fd.write('{0:72s}\n'.format(title))
    natoms = len(atoms)
    npbc = sum(atoms.pbc)
    if npbc == 0:
        imcon = 0
    elif npbc == 3:
        imcon = 3
    elif tuple(atoms.pbc) == (True, True, False):
        imcon = 6
    else:
        raise ValueError('Unsupported pbc: {}.  '
                         'Supported pbc are 000, 110, and 000.'
                         .format(atoms.pbc))

    fd.write('{0:10d}{1:10d}{2:10d}\n'.format(levcfg, imcon, natoms))
    if imcon > 0:
        cell = atoms.get_cell()
        for j in range(3):
            fd.write('{0:20.10f}{1:20.10f}{2:20.10f}\n'.format(
                cell[j, 0], cell[j, 1], cell[j, 2]))
    vels = []
    forces = []
    if levcfg > 0:
        vels = atoms.get_velocities() / dlp_v_ase
    if levcfg > 1:
        forces = atoms.get_forces() / dlp_f_ase

    labels = atoms.arrays.get(DLP4_LABELS_KEY)

    for i, a in enumerate(atoms):
        sym = a.symbol
        if labels is not None:
            sym += labels[i]
        fd.write("{0:8s}{1:10d}\n{2:20.10f}{3:20.10f}{4:20.10f}\n".format(
            sym, a.index + 1, a.x, a.y, a.z))
        if levcfg > 0:
            if vels is None:
                fd.write("{0:20.10f}{1:20.10f}{2:20.10f}\n".format(
                    0.0, 0.0, 0.0))
            else:
                fd.write("{0:20.10f}{1:20.10f}{2:20.10f}\n".format(
                    vels[a.index, 0], vels[a.index, 1], vels[a.index, 2]))
        if levcfg > 1:
            if forces is None:
                fd.write("{0:20.10f}{1:20.10f}{2:20.10f}\n".format(
                    0.0, 0.0, 0.0))
            else:
                fd.write("{0:20.10f}{1:20.10f}{2:20.10f}\n".format(
                    forces[a.index, 0], forces[a.index, 1], forces[a.index, 2]))

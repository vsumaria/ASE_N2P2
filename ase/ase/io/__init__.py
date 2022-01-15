from ase.io.trajectory import Trajectory, PickleTrajectory
from ase.io.bundletrajectory import BundleTrajectory
from ase.io.netcdftrajectory import NetCDFTrajectory
from ase.io.formats import read, iread, write, string2index


class ParseError(Exception):
    """Parse error during reading of a file"""


__all__ = [
    'Trajectory', 'PickleTrajectory', 'BundleTrajectory', 'NetCDFTrajectory',
    'read', 'iread', 'write', 'string2index'
]

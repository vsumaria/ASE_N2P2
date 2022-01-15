import errno
import functools
import os
import io
import pickle
import sys
import time
import string
import warnings
from importlib import import_module
from math import sin, cos, radians, atan2, degrees
from contextlib import contextmanager, ExitStack
from math import gcd
from pathlib import PurePath, Path
import re

import numpy as np

from ase.formula import formula_hill, formula_metal

__all__ = ['exec_', 'basestring', 'import_module', 'seterr', 'plural',
           'devnull', 'gcd', 'convert_string_to_fd', 'Lock',
           'opencew', 'OpenLock', 'rotate', 'irotate', 'pbc2pbc', 'givens',
           'hsv2rgb', 'hsv', 'pickleload', 'FileNotFoundError',
           'formula_hill', 'formula_metal', 'PurePath', 'xwopen',
           'tokenize_version']


def tokenize_version(version_string: str):
    """Parse version string into a tuple for version comparisons.

    Usage: tokenize_version('3.8') < tokenize_version('3.8.1').
    """
    tokens = []
    for component in version_string.split('.'):
        match = re.match(r'(\d*)(.*)', component)
        assert match is not None, f'Cannot parse component {component}'
        number_str, tail = match.group(1, 2)
        try:
            number = int(number_str)
        except ValueError:
            number = -1
        tokens += [number, tail]
    return tuple(tokens)


# Python 2+3 compatibility stuff (let's try to remove these things):
basestring = str
pickleload = functools.partial(pickle.load, encoding='bytes')


def deprecated(msg, category=FutureWarning):
    """Return a decorator deprecating a function.

    Use like @deprecated('warning message and explanation')."""
    def deprecated_decorator(func):
        @functools.wraps(func)
        def deprecated_function(*args, **kwargs):
            warning = msg
            if not isinstance(warning, Warning):
                warning = category(warning)
            warnings.warn(warning)
            return func(*args, **kwargs)
        return deprecated_function
    return deprecated_decorator


@contextmanager
def seterr(**kwargs):
    """Set how floating-point errors are handled.

    See np.seterr() for more details.
    """
    old = np.seterr(**kwargs)
    try:
        yield
    finally:
        np.seterr(**old)


def plural(n, word):
    """Use plural for n!=1.

    >>> plural(0, 'egg'), plural(1, 'egg'), plural(2, 'egg')
    ('0 eggs', '1 egg', '2 eggs')
    """
    if n == 1:
        return '1 ' + word
    return '%d %ss' % (n, word)


class DevNull:
    encoding = 'UTF-8'
    closed = False

    _use_os_devnull = deprecated('use open(os.devnull) instead',
                                 DeprecationWarning)
    # Deprecated for ase-3.21.0.  Change to futurewarning later on.

    @_use_os_devnull
    def write(self, string):
        pass

    @_use_os_devnull
    def flush(self):
        pass

    @_use_os_devnull
    def seek(self, offset, whence=0):
        return 0

    @_use_os_devnull
    def tell(self):
        return 0

    @_use_os_devnull
    def close(self):
        pass

    @_use_os_devnull
    def isatty(self):
        return False

    @_use_os_devnull
    def read(self, n=-1):
        return ''


devnull = DevNull()


@deprecated('convert_string_to_fd does not facilitate proper resource '
            'management.  '
            'Please use e.g. ase.utils.IOContext class instead.')
def convert_string_to_fd(name, world=None):
    """Create a file-descriptor for text output.

    Will open a file for writing with given name.  Use None for no output and
    '-' for sys.stdout.
    """
    if world is None:
        from ase.parallel import world
    if name is None or world.rank != 0:
        return open(os.devnull, 'w')
    if name == '-':
        return sys.stdout
    if isinstance(name, (str, PurePath)):
        return open(str(name), 'w')  # str for py3.5 pathlib
    return name  # we assume name is already a file-descriptor


# Only Windows has O_BINARY:
CEW_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0)


@contextmanager
def xwopen(filename, world=None):
    """Create and open filename exclusively for writing.

    If master cpu gets exclusive write access to filename, a file
    descriptor is returned (a dummy file descriptor is returned on the
    slaves).  If the master cpu does not get write access, None is
    returned on all processors."""

    fd = opencew(filename, world)
    try:
        yield fd
    finally:
        if fd is not None:
            fd.close()


#@deprecated('use "with xwopen(...) as fd: ..." to prevent resource leak')
def opencew(filename, world=None):
    return _opencew(filename, world)


def _opencew(filename, world=None):
    if world is None:
        from ase.parallel import world

    closelater = []

    def opener(file, flags):
        return os.open(file, flags | CEW_FLAGS)

    try:
        error = 0
        if world.rank == 0:
            try:
                fd = open(filename, 'wb', opener=opener)
            except OSError as ex:
                error = ex.errno
            else:
                closelater.append(fd)
        else:
            fd = open(os.devnull, 'wb')
            closelater.append(fd)

        # Synchronize:
        error = world.sum(error)
        if error == errno.EEXIST:
            return None
        if error:
            raise OSError(error, 'Error', filename)

        return fd
    except BaseException:
        for fd in closelater:
            fd.close()
        raise


def opencew_text(*args, **kwargs):
    fd = opencew(*args, **kwargs)
    if fd is None:
        return None
    return io.TextIOWrapper(fd)


class Lock:
    def __init__(self, name='lock', world=None, timeout=float('inf')):
        self.name = str(name)
        self.timeout = timeout
        if world is None:
            from ase.parallel import world
        self.world = world

    def acquire(self):
        dt = 0.2
        t1 = time.time()
        while True:
            fd = opencew(self.name, self.world)
            if fd is not None:
                self.fd = fd
                break
            time_left = self.timeout - (time.time() - t1)
            if time_left <= 0:
                raise TimeoutError
            time.sleep(min(dt, time_left))
            dt *= 2

    def release(self):
        self.world.barrier()
        # Important to close fd before deleting file on windows
        # as a WinError would otherwise be raised.
        self.fd.close()
        if self.world.rank == 0:
            os.remove(self.name)
        self.world.barrier()

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, tb):
        self.release()


class OpenLock:
    def acquire(self):
        pass

    def release(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass


def search_current_git_hash(arg, world=None):
    """Search for .git directory and current git commit hash.

    Parameters:

    arg: str (directory path) or python module
        .git directory is searched from the parent directory of
        the given directory or module.
    """
    if world is None:
        from ase.parallel import world
    if world.rank != 0:
        return None

    # Check argument
    if isinstance(arg, str):
        # Directory path
        dpath = arg
    else:
        # Assume arg is module
        dpath = os.path.dirname(arg.__file__)
    # dpath = os.path.abspath(dpath)
    # in case this is just symlinked into $PYTHONPATH
    dpath = os.path.realpath(dpath)
    dpath = os.path.dirname(dpath)  # Go to the parent directory
    git_dpath = os.path.join(dpath, '.git')
    if not os.path.isdir(git_dpath):
        # Replace this 'if' with a loop if you want to check
        # further parent directories
        return None
    HEAD_file = os.path.join(git_dpath, 'HEAD')
    if not os.path.isfile(HEAD_file):
        return None
    with open(HEAD_file, 'r') as fd:
        line = fd.readline().strip()
    if line.startswith('ref: '):
        ref = line[5:]
        ref_file = os.path.join(git_dpath, ref)
    else:
        # Assuming detached HEAD state
        ref_file = HEAD_file
    if not os.path.isfile(ref_file):
        return None
    with open(ref_file, 'r') as fd:
        line = fd.readline().strip()
    if all(c in string.hexdigits for c in line):
        return line
    return None


def rotate(rotations, rotation=np.identity(3)):
    """Convert string of format '50x,-10y,120z' to a rotation matrix.

    Note that the order of rotation matters, i.e. '50x,40z' is different
    from '40z,50x'.
    """

    if rotations == '':
        return rotation.copy()

    for i, a in [('xyz'.index(s[-1]), radians(float(s[:-1])))
                 for s in rotations.split(',')]:
        s = sin(a)
        c = cos(a)
        if i == 0:
            rotation = np.dot(rotation, [(1, 0, 0),
                                         (0, c, s),
                                         (0, -s, c)])
        elif i == 1:
            rotation = np.dot(rotation, [(c, 0, -s),
                                         (0, 1, 0),
                                         (s, 0, c)])
        else:
            rotation = np.dot(rotation, [(c, s, 0),
                                         (-s, c, 0),
                                         (0, 0, 1)])
    return rotation


def givens(a, b):
    """Solve the equation system::

      [ c s]   [a]   [r]
      [    ] . [ ] = [ ]
      [-s c]   [b]   [0]
    """
    sgn = np.sign
    if b == 0:
        c = sgn(a)
        s = 0
        r = abs(a)
    elif abs(b) >= abs(a):
        cot = a / b
        u = sgn(b) * (1 + cot**2)**0.5
        s = 1. / u
        c = s * cot
        r = b * u
    else:
        tan = b / a
        u = sgn(a) * (1 + tan**2)**0.5
        c = 1. / u
        s = c * tan
        r = a * u
    return c, s, r


def irotate(rotation, initial=np.identity(3)):
    """Determine x, y, z rotation angles from rotation matrix."""
    a = np.dot(initial, rotation)
    cx, sx, rx = givens(a[2, 2], a[1, 2])
    cy, sy, ry = givens(rx, a[0, 2])
    cz, sz, rz = givens(cx * a[1, 1] - sx * a[2, 1],
                        cy * a[0, 1] - sy * (sx * a[1, 1] + cx * a[2, 1]))
    x = degrees(atan2(sx, cx))
    y = degrees(atan2(-sy, cy))
    z = degrees(atan2(sz, cz))
    return x, y, z


def pbc2pbc(pbc):
    newpbc = np.empty(3, bool)
    newpbc[:] = pbc
    return newpbc


def hsv2rgb(h, s, v):
    """http://en.wikipedia.org/wiki/HSL_and_HSV

    h (hue) in [0, 360[
    s (saturation) in [0, 1]
    v (value) in [0, 1]

    return rgb in range [0, 1]
    """
    if v == 0:
        return 0, 0, 0
    if s == 0:
        return v, v, v

    i, f = divmod(h / 60., 1)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    if i == 0:
        return v, t, p
    elif i == 1:
        return q, v, p
    elif i == 2:
        return p, v, t
    elif i == 3:
        return p, q, v
    elif i == 4:
        return t, p, v
    elif i == 5:
        return v, p, q
    else:
        raise RuntimeError('h must be in [0, 360]')


def hsv(array, s=.9, v=.9):
    array = (array + array.min()) * 359. / (array.max() - array.min())
    result = np.empty((len(array.flat), 3))
    for rgb, h in zip(result, array.flat):
        rgb[:] = hsv2rgb(h, s, v)
    return np.reshape(result, array.shape + (3,))


# This code does the same, but requires pylab
# def cmap(array, name='hsv'):
#     import pylab
#     a = (array + array.min()) / array.ptp()
#     rgba = getattr(pylab.cm, name)(a)
#     return rgba[:-1] # return rgb only (not alpha)


def longsum(x):
    """128-bit floating point sum."""
    return float(np.asarray(x, dtype=np.longdouble).sum())


@contextmanager
def workdir(path, mkdir=False):
    """Temporarily change, and optionally create, working directory."""
    path = Path(path)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    olddir = os.getcwd()
    os.chdir(str(path))  # py3.6 allows chdir(path) but we still need 3.5
    try:
        yield  # Yield the Path or dirname maybe?
    finally:
        os.chdir(olddir)


class iofunction:
    """Decorate func so it accepts either str or file.

    (Won't work on functions that return a generator.)"""
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, func):
        @functools.wraps(func)
        def iofunc(file, *args, **kwargs):
            openandclose = isinstance(file, (str, PurePath))
            fd = None
            try:
                if openandclose:
                    fd = open(str(file), self.mode)
                else:
                    fd = file
                obj = func(fd, *args, **kwargs)
                return obj
            finally:
                if openandclose and fd is not None:
                    # fd may be None if open() failed
                    fd.close()
        return iofunc


def writer(func):
    return iofunction('w')(func)


def reader(func):
    return iofunction('r')(func)


# The next two functions are for hotplugging into a JSONable class
# using the jsonable decorator.  We are supposed to have this kind of stuff
# in ase.io.jsonio, but we'd rather import them from a 'basic' module
# like ase/utils than one which triggers a lot of extra (cyclic) imports.

def write_json(self, fd):
    """Write to JSON file."""
    from ase.io.jsonio import write_json as _write_json
    _write_json(fd, self)


@classmethod  # type: ignore
def read_json(cls, fd):
    """Read new instance from JSON file."""
    from ase.io.jsonio import read_json as _read_json
    obj = _read_json(fd)
    assert type(obj) is cls
    return obj


def jsonable(name):
    """Decorator for facilitating JSON I/O with a class.

    Pokes JSON-based read and write functions into the class.

    In order to write an object to JSON, it needs to be a known simple type
    (such as ndarray, float, ...) or implement todict().  If the class
    defines a string called ase_objtype, the decoder will want to convert
    the object back into its original type when reading."""
    def jsonableclass(cls):
        cls.ase_objtype = name
        if not hasattr(cls, 'todict'):
            raise TypeError('Class must implement todict()')

        # We may want the write and read to be optional.
        # E.g. a calculator might want to be JSONable, but not
        # that .write() produces a JSON file.
        #
        # This is mostly for 'lightweight' object IO.
        cls.write = write_json
        cls.read = read_json
        return cls
    return jsonableclass


class ExperimentalFeatureWarning(Warning):
    pass


def experimental(func):
    """Decorator for functions not ready for production use."""
    @functools.wraps(func)
    def expfunc(*args, **kwargs):
        warnings.warn('This function may change or misbehave: {}()'
                      .format(func.__qualname__),
                      ExperimentalFeatureWarning)
        return func(*args, **kwargs)
    return expfunc


def lazymethod(meth):
    """Decorator for lazy evaluation and caching of data.

    Example::

      class MyClass:

         @lazymethod
         def thing(self):
             return expensive_calculation()

    The method body is only executed first time thing() is called, and
    its return value is stored.  Subsequent calls return the cached
    value."""
    name = meth.__name__

    @functools.wraps(meth)
    def getter(self):
        try:
            cache = self._lazy_cache
        except AttributeError:
            cache = self._lazy_cache = {}

        if name not in cache:
            cache[name] = meth(self)
        return cache[name]
    return getter


def atoms_to_spglib_cell(atoms):
    """Convert atoms into data suitable for calling spglib."""
    return (atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())


def warn_legacy(feature_name):
    warnings.warn(
        f'The {feature_name} feature is untested and ASE developers do not '
        'know whether it works or how to use it.  Please rehabilitate it '
        '(by writing unittests) or it may be removed.',
        FutureWarning)


def lazyproperty(meth):
    """Decorator like lazymethod, but making item available as a property."""
    return property(lazymethod(meth))


class IOContext:
    @lazyproperty
    def _exitstack(self):
        return ExitStack()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def closelater(self, fd):
        return self._exitstack.enter_context(fd)

    def close(self):
        self._exitstack.close()

    def openfile(self, file, comm=None, mode='w'):
        from ase.parallel import world
        if comm is None:
            comm = world

        if hasattr(file, 'close'):
            return file  # File already opened, not for us to close.

        if file is None or comm.rank != 0:
            return self.closelater(open(os.devnull, mode=mode))

        if file == '-':
            return sys.stdout

        return self.closelater(open(file, mode=mode))

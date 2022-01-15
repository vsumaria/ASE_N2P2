import warnings

import pytest

from ase.build import bulk, molecule, fcc111
from ase.io.animation import write_animation


def test_animate(plt):
    from matplotlib.animation import writers

    if 'html' not in writers.list():
        pytest.skip('matplotlib html writer not present')

    images = [molecule('H2O'), bulk('Cu'), fcc111('Au', size=(1, 1, 1))]

    # gif and mp4 writers may not be available.  Easiest solution is to only
    # test this using the html writer because it always exists whenever
    # matplotlib exists:
    with warnings.catch_warnings():
        try:
            from matplotlib import MatplotlibDeprecationWarning
        except ImportError:
            pass
        else:
            warnings.simplefilter('ignore', MatplotlibDeprecationWarning)
        write_animation('things.html', images, writer='html')

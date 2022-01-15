import ase.parallel as parallel
from ase.visualize.external import viewers


def view(atoms, data=None, viewer='ase', repeat=None, block=False):
    if parallel.world.size > 1:
        return

    vwr = viewers[viewer.lower()]
    handle = vwr.view(atoms, data=data, repeat=repeat)

    if block and hasattr(handle, 'wait'):
        status = handle.wait()
        if status != 0:
            raise RuntimeError(f'Viewer "{vwr.name}" failed with status '
                               '{status}')

    return handle

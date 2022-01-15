from math import radians, sin, cos

import pytest

from ase import Atoms
from ase.neb import NEB
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton, BFGS
from ase.visualize import view


@pytest.mark.calculator('nwchem')
def test_h3o2m(factory):
    # http://jcp.aip.org/resource/1/jcpsa6/v97/i10/p7507_s1
    doo = 2.74
    doht = 0.957
    doh = 0.977
    angle = radians(104.5)
    initial = Atoms('HOHOH',
                    positions=[(-sin(angle) * doht, 0, cos(angle) * doht),
                               (0., 0., 0.),
                               (0., 0., doh),
                               (0., 0., doo),
                               (sin(angle) * doht, 0., doo - cos(angle) * doht)])
    if 0:
        view(initial)

    final = Atoms('HOHOH',
                  positions=[(- sin(angle) * doht, 0., cos(angle) * doht),
                             (0., 0., 0.),
                             (0., 0., doo - doh),
                             (0., 0., doo),
                             (sin(angle) * doht, 0., doo - cos(angle) * doht)])
    if 0:
        view(final)

    # Make band:
    images = [initial.copy()]
    for i in range(3):
        images.append(initial.copy())
    images.append(final.copy())
    neb = NEB(images, climb=True)

    def calculator():
        return factory.calc(
            task='gradient',
            theory='scf',
            charge=-1
        )

    # Set constraints and calculator:
    constraint = FixAtoms(indices=[1, 3])  # fix OO
    for image in images:
        image.calc = calculator()
        image.set_constraint(constraint)

    # Relax initial and final states:
    with QuasiNewton(images[0]) as dyn1:
        dyn1.run(fmax=0.10)
    with QuasiNewton(images[-1]) as dyn2:
        dyn2.run(fmax=0.10)

    # Interpolate positions between initial and final states:
    neb.interpolate()

    for image in images:
        print(image.get_distance(1, 2), image.get_potential_energy())

    with BFGS(neb) as dyn:
        # use better basis (e.g. aug-cc-pvdz) for NEB to converge
        dyn.run(fmax=0.10)

    for image in images:
        print(image.get_distance(1, 2), image.get_potential_energy())

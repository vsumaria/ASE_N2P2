from ase import Atoms
from ase.calculators.emt import EMT
from ase.md import VelocityVerlet
from ase.io import Trajectory


def test_md(testdir):
    a = 3.6
    b = a / 2
    fcc = Atoms('Cu', positions=[(0, 0, 0)],
                cell=[(0, b, b), (b, 0, b), (b, b, 0)],
                pbc=1)
    fcc *= (2, 1, 1)
    fcc.calc = EMT()
    fcc.set_momenta([(0.9, 0.0, 0.0), (-0.9, 0, 0)])

    def f():
        print(fcc.get_potential_energy(), fcc.get_total_energy())

    with VelocityVerlet(fcc, timestep=0.1) as md:
        md.attach(f)
        with Trajectory('Cu2.traj', 'w', fcc) as traj:
            md.attach(traj.write, interval=3)
            md.run(steps=20)

    with Trajectory('Cu2.traj', 'r') as traj:
        traj[-1]

    # Really?? No assertion at all?

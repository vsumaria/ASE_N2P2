# ASE N2P2 Caculator
ASE calculator for N2p2 HDNNP potential 

Example of using the N2P2 calculator with ASE: 

```
from ase.calculators.n2p2 import N2P2Calculator

atoms = read('abc.traj')

calc = N2P2Calculator(directory = './tmp', files = ['./nnp/input.nn', './nnp/scaling.data', './nnp/weights.008.data', './nnp/weights.006.data', './nnp/weights.078.data'],)

atoms.set_calculator(atoms)
e = atoms.get_potential_energy()

```

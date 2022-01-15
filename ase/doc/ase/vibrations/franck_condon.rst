Evaluation of Franck-Condon factors
===================================

The Franck-Condon principle couples electronic and vibrational
properties.

-------------------------------
Franck-Condon factors in CH4
-------------------------------

Forces
------

We may get a good structure for methane from ``ase.build``.
This will not be the ground state structure according to
the ``EMT`` calcuator though. Therfore ``EMT`` predicts finite
forces::

  from ase.build import molecule
  from ase.calculators.emt import EMT

  atoms = molecule('CH4')
  atoms.calc = EMT()

  # evaluate forces in this configuration
  forces_a = atoms.get_forces()

Vibrational properties
----------------------

These forces can be used to calculate the corresponding
Franck-Condon factors for transitions from the ``EMT``
ground state. First we need the ``EMT`` ground state and
the vibrational properties::

  from ase.optimize import BFGS
  from ase.vibrations import Vibrations

  # relax and get vibrational properties
  opt = BFGS(atoms, logfile=None)
  opt.run(fmax=0.01)

  vibname = 'vib'
  vib = Vibrations(atoms, name=vibname)
  vib.run()
  vib.summary()

Huang-Rhys factors
------------------

The Huang-Rhys factors describe the displacement energy in
each vibrational coordinate relative to the vibrational energy.
We may get them by::
  
  from ase.vibrations.franck_condon import FranckCondon

  # FC factor for all frequencies
  fc = FranckCondon(atoms, vibname)

  HR_a, freq_a = fc.get_Huang_Rhys_factors(forces_a)

Franck-Condon factors
---------------------

The Franck-Condon factors depend on temperature due to occupation
of vibrational states. We may get them for 293 K by::

  FC, freq = fc.get_Franck_Condon_factors(293, forces_a)

where ``FC[0]`` contains the Franck-Condon factors and
``freq[0]`` the corresponding frequencies.

It is also possible to evaluate higher order transitions.
Two vibrational quanta might be considered by increasing the order::

  FC, freq = fc.get_Franck_Condon_factors(293, forces_a, order=2)

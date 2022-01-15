.. module:: ase.vibrations

Vibrational modes
=================

You can calculate the vibrational modes of an
:class:`~ase.Atoms` object in the harmonic approximation using
the :class:`Vibrations`.

.. autoclass:: Vibrations
   :members:

name is a string that is prefixed to the names of all the files
created. atoms is an Atoms object that is either at a
fully relaxed ground state or at a saddle point. freeatoms is a
list of atom indices for which the vibrational modes will be calculated,
the rest of the atoms are considered frozen. displacements is a
list of displacements, one for each free atom that are used in the
finite difference method to calculate the Hessian matrix. method is -1
for backward differences, 0 for centered differences, and 1 for
forward differences.

Old calculations
----------------

The output format of vibrational calculations was changed from ``pickle``
to ``json``. There is a tool to convert old ``pickle``-files::

  > python3 -m ase.vibrations.pickle2json mydirectory/vib.*.pckl

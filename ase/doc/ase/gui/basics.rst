=======================================
ase-gui basics and command line options
=======================================

General use
-----------

Visualizing a system with ASE's GUI is straight-forward using a regular
mouse. The scroll function allows to change the magnification, the
left mouse button selects atoms, the right mouse button allows to
rotate, and the middle button allows to translate the system on the
screen.

Depending on the number of selected atoms, :ref:`ase-gui` automatically measures
different quantities:

================================= ======================================
Selection                         measurement
================================= ======================================
single atom                       xyz position and atomic symbol
two atoms                         interatomic distance and symbols
three atoms                       all three internal angles and
                                  symbols
four atoms, selected sequentially Measures the dihedral angle,
                                  e.g. the angle between bonds 12 and 34
more than four atoms              chemical composition of selection.
================================= ======================================


Files
-----

The :ref:`ase-gui` program can read all the file formats the ASE's
:func:`~ase.io.read` function can understand.

::

  $ ase gui N2Fe110-path.traj


Selecting part of a trajectory
------------------------------

A Python-like syntax for selecting a subset of configurations can be
used.  Instead of the Python syntax ``list[start:stop:step]``, you use
:file:`filaname@start:stop:step`::

  $ ase gui x.traj@0:10:1  # first 10 images
  $ ase gui x.traj@0:10    # first 10 images
  $ ase gui x.traj@:10     # first 10 images
  $ ase gui x.traj@-10:    # last 10 images
  $ ase gui x.traj@0       # first image
  $ ase gui x.traj@-1      # last image
  $ ase gui x.traj@::2     # every second image

If you want to select the same range from many files, the you can use
the ``-n`` or ``--image-number`` option::

  $ ase gui -n -1 *.traj   # last image from all files
  $ ase gui -n 0 *.traj    # first image from all files

.. tip::

  Type :command:`ase gui -h` for a description of all command line options.


Writing files
-------------

::

  $ ase gui -n -1 a*.traj -o new.traj

Possible formats are: ``traj``, ``xyz``, ``cube``, ``pdb``, ``eps``,
``png``, and ``pov``.  For details, see the :mod:`~ase.io` module
documentation.


Interactive use
---------------

The :ref:`ase-gui` program can also be launched directly from a Python
script or interactive session:

>>> from ase import *
>>> atoms = ...
>>> view(atoms)

or

>>> view(atoms, repeat=(3, 3, 2))

or, to keep changes to your atoms:

>>> atoms.edit()

Use :meth:`ase.gui.gui.GUI.repeat_poll` to interact programmatically
with the GUI, for example to monitor an ongoing calculation
and update the display on the fly.

.. automethod:: ase.gui.gui.GUI.repeat_poll

NEB calculations
----------------

Use :menuselection:`Tools --> NEB` to plot energy barrier.

::

  $ ase gui --interpolate 3 initial.xyz final.xyz -o interpolated_path.traj


Plotting data from the command line
-----------------------------------
Plot the energy relative to the energy of the first image as a
function of the distance between atom 0 and 5::

  $ ase gui -g "d(0,5),e-E[0]" x.traj
  $ ase gui -t -g "d(0,5),e-E[0]" x.traj > x.dat  # No GUI, write data to stdout

The symbols are the same as used in the plotting data function.


Defaults
--------

Using a file ``~/.ase/gui.py``, certain defaults can be set. If it exists,
this file is executed after initializing the variables and colors
normally used in ASE. One can change the default graphs that are
plotted, and the default radii for displaying specific atoms. This
example will display the energy evolution and the maximal force in a
graph and also display Cu atoms (Z=29) with a radius of 1.6 Angstrom.

::

  gui_default_settings['gui_graphs_string'] = "i, e - min(E), fmax"
  gui_default_settings['covalent_radii'] = [[29,1.6]]


.. _high contrast:

High contrast settings
----------------------

In revision 2600 or later, it is possible to change the foreground and
background colors used to draw the atoms, for instance to draw white
graphics on a black background. This can be done in ``~/.ase/gui.py``.

::

  gui_default_settings['gui_foreground_color'] = '#ffffff' #white
  gui_default_settings['gui_background_color'] = '#000000' #black

To change the color scheme of graphs it is necessary to change the
default behaviour of Matplotlib in a similar way by using a file
``~/.matplotlib/matplotlibrc``.

::

  patch.edgecolor  : white
  text.color       : white
  axes.facecolor   : black
  axes.edgecolor   : white
  axes.labelcolor  : white
  axes.color_cycle : b, g, r, c, m, y, w
  xtick.color      : white
  ytick.color      : white
  grid.color       : white
  figure.facecolor : 0.1
  figure.edgecolor : black

Finally, the color scheme of the windows themselves (i.e. menus, buttons
and text etc.) can be changed by choosing a different desktop theme. In
Ubuntu it is possible to get white on a dark background by selecting the
theme HighContrastInverse under Appearances in the system settings dialog.

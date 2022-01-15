.. module:: ase.calculators.plumed

========
PLUMED
========

.. image:: ../../static/plumed.png

Introduction
============

Plumed_ is an open source library which allows to implement several 
kind of enhanced sampling methods and contains a variety of tools to 
analyze data obtained from molecular dynamics simulations. With this 
calculator can be carried out biased simulations incluing metadynamics 
or its variation well-tempered metadynamics, among others. Besides, it 
is possible to compute a large set of collective variables that plumed 
has already implemented for being calculated on-the-fly in MD simulations 
or for postprocessing tasks.

.. _Plumed: https://www.plumed.org/ 

Setup
=====

Typically, plumed simulations need an external file, commonly called plumed.dat
for setting up the plumed functions. In this ASE calculator interface, plumed information is
given to the calculator through a string list containing the lines that would be included
in the plumed.dat file. Something like this::

    setup = ["d: DISTANCE ATOMS=1,2",
             "PRINT ARG=d STRIDE=10 FILE=COLVAR"]

.. autoclass:: ase.calculators.plumed.Plumed

"""Module to read atoms in chemical json file format.

https://wiki.openchemistry.org/Chemical_JSON
"""
import json
import numpy as np

from ase import Atoms
from ase.cell import Cell


# contract and lower case string
def contract(dictionary):
    dcopy = {}
    for key in dictionary:
        dcopy[key.replace(' ', '').lower()] = dictionary[key]
    return dcopy


def read_cml(fileobj):
    data = contract(json.load(fileobj))
    atoms = Atoms()
    datoms = data['atoms']

    atoms = Atoms(datoms['elements']['number'])

    if 'unitcell' in data:
        cell = data['unitcell']
        a = cell['a']
        b = cell['b']
        c = cell['c']
        alpha = cell['alpha']
        beta = cell['beta']
        gamma = cell['gamma']
        atoms.cell = Cell.fromcellpar([a, b, c, alpha, beta, gamma])
        atoms.pbc = True
    
    coords = contract(datoms['coords'])
    if '3d' in coords:
        positions = np.array(coords['3d']).reshape(len(atoms), 3)
        atoms.set_positions(positions)
    else:
        positions = np.array(coords['3dfractional']).reshape(len(atoms), 3)
        atoms.set_scaled_positions(positions)
        
    yield atoms

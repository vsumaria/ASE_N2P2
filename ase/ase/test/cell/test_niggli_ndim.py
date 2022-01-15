import pytest
import numpy as np

from ase.cell import Cell


def test_niggli_0d():
    rcell, op = Cell.new().niggli_reduce()
    assert rcell.rank == 0
    assert (op == np.identity(3, dtype=int)).all()


def test_niggli_1d():
    cell = Cell.new()
    vector = [1, 2, 3]
    cell[1] = vector

    rcell, op = cell.niggli_reduce()
    assert rcell.lengths() == pytest.approx([np.linalg.norm(vector), 0, 0])
    assert Cell(op.T @ cell).cellpar() == pytest.approx(rcell.cellpar())


def test_niggli_2d():
    cell = Cell.new()
    cell[0] = [3, 4, 5]
    cell[2] = [5, 6, 7]
    rcell, op = cell.niggli_reduce()
    assert rcell.rank == 2
    assert rcell.lengths()[2] == 0
    assert Cell(op.T @ cell).cellpar() == pytest.approx(rcell.cellpar())

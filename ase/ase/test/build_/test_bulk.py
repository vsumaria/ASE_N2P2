from ase.build import bulk


def test_bulk():
    a1 = bulk('ZnS', 'wurtzite', a=3.0, u=0.23) * (1, 2, 1)
    a2 = bulk('ZnS', 'wurtzite', a=3.0, u=0.23, orthorhombic=True)
    a1.cell = a2.cell
    a1.wrap()
    assert abs(a1.positions - a2.positions).max() < 1e-14


def hasmom(*args, **kwargs):
    return bulk(*args, **kwargs).has('initial_magmoms')


def test_magnetic_or_not():
    assert hasmom('Fe')
    assert hasmom('Fe', orthorhombic=True)
    assert hasmom('Fe', cubic=True)
    assert hasmom('Fe', 'bcc', 4.0)
    assert not hasmom('Fe', 'fcc', 4.0)
    assert not hasmom('Ti')
    assert not hasmom('Ti', 'bcc', 4.0)

    assert hasmom('Co')
    assert hasmom('Ni')

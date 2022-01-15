import pytest

from ase import Atoms
from ase.db import connect
from ase.db.web import Session
from ase.calculators.calculator import compare_atoms


def get_atoms():
    atoms = Atoms('H2O',
                  [(0, 0, 0),
                   (2, 0, 0),
                   (1, 1, 0)])
    atoms.center(vacuum=5)
    atoms.set_pbc(True)
    return atoms


@pytest.fixture(scope='module')
def database(tmp_path_factory):
    with tmp_path_factory.mktemp('dbtest') as dbtest:
        db = connect(dbtest / 'test.db', append=False)
        x = [0, 1, 2]
        t1 = [1, 2, 0]
        t2 = [[2, 3], [1, 1], [1, 0]]

        atoms = get_atoms()
        db.write(atoms,
                 foo=42.0,
                 bar='abc',
                 data={'x': x,
                       't1': t1,
                       't2': t2})
        db.write(atoms)

        yield db


def handle_query(args) -> str:
    """Converts request args to ase.db query string."""
    return args['query']


@pytest.fixture(scope='module')
def client(database):
    pytest.importorskip('flask')
    import ase.db.app as app

    app.add_project(database)
    app.app.testing = True
    return app.app.test_client()


def test_add_columns(database):
    """Test that all keys can be added also for row withous keys."""
    pytest.importorskip('flask')

    session = Session('name')
    project = {'default_columns': ['bar'],
               'handle_query_function': handle_query}

    session.update('query', '', {'query': 'id=2'}, project)
    table = session.create_table(database, 'id', ['foo'])
    assert table.columns == ['bar']  # selected row doesn't have a foo key
    assert 'foo' in table.addcolumns  # ... but we can add it


def test_favicon(client):
    assert client.get('/favicon.ico').status_code == 308  # redirect
    assert client.get('/favicon.ico/').status_code == 204  # no content


def test_db_web(client):
    import io
    from ase.db.web import Session
    from ase.io import read
    c = client

    page = c.get('/').data.decode()
    sid = Session.next_id - 1
    assert 'foo' in page
    for url in [f'/update/{sid}/query/bla/?query=id=1',
                '/default/row/1']:
        resp = c.get(url)
        assert resp.status_code == 200

    for type in ['json', 'xyz', 'cif']:
        url = f'atoms/default/1/{type}'
        resp = c.get(url)
        assert resp.status_code == 200
        txt = resp.data.decode()

        print(type)
        print(txt)

        fmt = type
        if fmt == 'xyz':
            fmt = 'extxyz'
        atoms = read(io.StringIO(txt), format=fmt)
        assert (atoms.numbers == [1, 1, 8]).all()
        tol = 1e-5 if type == 'cif' else 1e-10
        assert not compare_atoms(atoms, get_atoms(), tol), type


def test_paging(database):
    """Test paging."""
    pytest.importorskip('flask')

    session = Session('name')
    project = {'default_columns': ['bar'],
               'handle_query_function': handle_query}

    session.update('query', '', {'query': ''}, project)
    table = session.create_table(database, 'id', ['foo'])
    assert len(table.rows) == 2

    session.update('limit', '1', {}, project)
    session.update('page', '1', {}, project)
    table = session.create_table(database, 'id', ['foo'])
    assert len(table.rows) == 1

    # We are now on page 2 and select something on page 1:
    session.update('query', '', {'query': 'id=1'}, project)
    table = session.create_table(database, 'id', ['foo'])
    assert len(table.rows) == 1

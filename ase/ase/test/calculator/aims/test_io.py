import re
from io import StringIO
from ase.calculators.aims import write_control
from ase.build import bulk


def test_control():
    atoms = bulk('Au')
    parameters = {'xc': 'LDA', 'oranges': 7, 'coffee': 2.3,
                  'greeting': 'hello'}
    fd = StringIO()
    write_control(fd, atoms, parameters)
    txt = fd.getvalue()
    print(txt)

    def contains(pattern):
        return re.search(pattern, txt, re.M)

    assert contains(r'oranges\s+7')
    assert contains(r'coffee\s+2.3')
    assert contains(r'xc\s+pw-lda')
    assert contains(r'greeting\s+hello')

import datetime
import sys

import sphinx_rtd_theme

sys.path.append('.')
assert sys.version_info >= (3, 6)

extensions = ['ext',
              'images',
              'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.extlinks',
              'sphinx.ext.mathjax',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx']
extlinks = {'doi': ('https://doi.org/%s', 'doi:'),
            'arxiv': ('https://arxiv.org/abs/%s', 'arXiv:')}
source_suffix = '.rst'
master_doc = 'index'
project = 'ASE'
copyright = f'{datetime.date.today().year}, ASE-developers'
templates_path = ['templates']
exclude_patterns = ['build']
default_role = 'math'
pygments_style = 'sphinx'
autoclass_content = 'both'
modindex_common_prefix = ['ase.']
nitpick_ignore = [('envvar', 'VASP_PP_PATH'),
                  ('envvar', 'ASE_ABC_COMMAND'),
                  ('envvar', 'FLEUR_INPGEN'),
                  ('envvar', 'FLEUR'),
                  ('envvar', 'LAMMPS_COMMAND'),
                  ('envvar', 'ASE_NWCHEM_COMMAND'),
                  ('envvar', 'SIESTA_COMMAND'),
                  ('envvar', 'SIESTA_PP_PATH'),
                  ('envvar', 'VASP_SCRIPT')]

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_style = 'ase.css'
html_favicon = 'static/ase.ico'
html_static_path = ['static']
html_last_updated_fmt = '%a, %d %b %Y %H:%M:%S'

latex_elements = {'papersize': 'a4paper'}
latex_show_urls = 'inline'
latex_show_pagerefs = True
latex_documents = [
    ('index', 'ASE.tex', 'ASE', 'ASE-developers', 'howto', not True)]

intersphinx_mapping = {'gpaw': ('https://wiki.fysik.dtu.dk/gpaw', None),
                       'python': ('https://docs.python.org/3.9', None)}

# Avoid GUI windows during doctest:
doctest_global_setup = """
import numpy as np
import ase.visualize as visualize
from ase import Atoms
visualize.view = lambda atoms: None
Atoms.edit = lambda self: None
"""

autodoc_mock_imports = ["kimpy"]

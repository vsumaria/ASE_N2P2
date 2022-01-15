.. _removed_features:

Legacy functionality
====================

Sometimes features are removed from ASE.  These features may still be
useful, and can be found in version history, only we can't support
them.

Common reasons for removing a feature is that is that the feature is
unused, buggy or broken, untested, undocumented, difficult to
maintain, or has possible security problems.

Below is a list of removed features.

====================================== ===== =============================
Removed in 3.23.0                      MR    Comment
====================================== ===== =============================
``ase.io.gaussian_reader``             !2329 No tests or documentation
====================================== ===== =============================

====================================== ===== =============================
Removed in 3.21.0                      MR    Comment
====================================== ===== =============================
``ase.calculators.ase_qmmm_manyqm``    !2092 Has docs but lacks real tests
``ase.build.voids``                    !2078
Unused code in ``ase.transport.tools`` !2077
``ase.io.iwm``                         !2064
``ase.visualize.primiplotter``         !2060 Moved to asap3
``ase.visualize.fieldplotter``         !2060 Moved to asap3
``ase.io.plt``                         !2057
====================================== ===== =============================



====================================== ===== =============================
Removed in 3.20.0                      MR    Comment
====================================== ===== =============================
dacapo-netcdf in ``ase.io.dacapo``     !1892
``ase.build.adsorb``                   !1845
Unused code in ``ase.utils.ff``        !1844
``ase.utils.extrapolate``              !1808 Moved to GPAW
``ase/data/tmgmjbp04n.py``             !1720
``ase/data/tmfp06d.py``                !1720
``ase/data/gmtkn30.py``                !1720
``ase/data/tmxr200x_tm3r2008.py``      !1720
``ase/data/tmxr200x_tm2r2007.py``      !1720
``ase/data/tmxr200x_tm1r2006.py``      !1720
``ase/data/tmxr200x.py``               !1720
``ase.calculators.jacapo``             !1604
``ase.calculators.dacapo``
``ase.spacegroup.findsym``                   Use spglib
====================================== ===== =============================

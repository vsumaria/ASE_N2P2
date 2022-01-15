import numpy as np

from ase.optimize.sciopt import SciPyOptimizer, OptimizerConvergenceError
   

def ode12r(f, X0, h=None, verbose=1, fmax=1e-6, maxtol=1e3, steps=100,
           rtol=1e-1, C1=1e-2, C2=2.0, hmin=1e-10, extrapolate=3,
           callback=None, apply_precon=None, converged=None, residual=None):
    """
    Adaptive ODE solver, which uses 1st and 2nd order approximations to
    estimate local error and choose a new step length.

    This optimizer is described in detail in:

                S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
                150, 094109 (2019)
                https://dx.doi.org/10.1063/1.5064465

    Parameters
    ----------

    f : function
        function returning driving force on system
    X0 : 1-dimensional array
        initial value of degrees of freedom
    h : float
        step size, if None an estimate is used based on ODE12
    verbose: int
        verbosity level. 0=no output, 1=log output (default), 2=debug output.
    fmax : float
        convergence tolerance for residual force
    maxtol: float
        terminate if reisdual exceeds this value
    rtol : float
        relative tolerance
    C1 : float
        sufficient contraction parameter
    C2 : float
        residual growth control (Inf means there is no control)
    hmin : float
        minimal allowed step size
    extrapolate : int
        extrapolation style (3 seems the most robust)
    callback : function
        optional callback function to call after each update step
    apply_precon: function
        apply a apply_preconditioner to the optimisation
    converged: function
        alternative function to check convergence, rather than
        using a simple norm of the forces.
    residual: function
        compute the residual from the current forces
        
    Returns
    -------

    X: array
        final value of degrees of freedom
    """
    
    X = X0
    Fn = f(X)
    
    if callback is None:
        def callback(X):
            pass
    callback(X)
    
    if residual is None:
        def residual(F, X):
            return np.linalg.norm(F, np.inf)
    Rn = residual(Fn, X)
    
    if apply_precon is None:
        def apply_precon(F, X):
            return F, residual(F, X)
    Fp, Rp = apply_precon(Fn, X)
    
    def log(*args):
        if verbose >= 1:
            print(*args)
                        
    def debug(*args):
        if verbose >= 2:
            print(*args)
        
    if converged is None:
        def converged(F, X):
            return residual(F, X) <= fmax
    
    if converged(Fn, X):
        log("ODE12r terminates successfully after 0 iterations")
        return X
    if Rn >= maxtol:
        raise OptimizerConvergenceError(f"ODE12r: Residual {Rn} is too large "
                                        "at iteration 0")

    # computation of the initial step
    r = residual(Fp, X)  # pick the biggest force
    if h is None:
        h = 0.5 * rtol ** 0.5 / r  # Chose a stepsize based on that force
        h = max(h, hmin)  # Make sure the step size is not too big

    for nit in range(1, steps):
        Xnew = X + h * Fp  # Pick a new position
        Fn_new = f(Xnew)  # Calculate the new forces at this position
        Rn_new = residual(Fn_new, Xnew)
        Fp_new, Rp_new = apply_precon(Fn_new, Xnew)

        e = 0.5 * h * (Fp_new - Fp)  # Estimate the area under the forces curve
        err = np.linalg.norm(e, np.inf)  # Error estimate

        # Accept step if residual decreases sufficiently and/or error acceptable
        accept = ((Rp_new <= Rp * (1 - C1 * h)) or
                  ((Rp_new <= Rp * C2) and err <= rtol))

        # Pick an extrapolation scheme for the system & find new increment
        y = Fp - Fp_new
        if extrapolate == 1:  # F(xn + h Fp)
            h_ls = h * (Fp @ y) / (y @ y)
        elif extrapolate == 2:  # F(Xn + h Fp)
            h_ls = h * (Fp @ Fp_new) / (Fp @ y + 1e-10)
        elif extrapolate == 3:  # min | F(Xn + h Fp) |
            h_ls = h * (Fp @ y) / (y @ y + 1e-10)
        else:
            raise ValueError(f'invalid extrapolate value: {extrapolate}. '
                             'Must be 1, 2 or 3')
        if np.isnan(h_ls) or h_ls < hmin:  # Rejects if increment is too small
            h_ls = np.inf

        h_err = h * 0.5 * np.sqrt(rtol / err)

        # Accept the step and do the update
        if accept:
            X = Xnew
            Rn = Rn_new
            Fn = Fn_new
            Fp = Fp_new
            Rp = Rp_new
            callback(X)

            # We check the residuals again
            if converged(Fn, X):
                log(f"ODE12r: terminates successfully "
                    f"after {nit} iterations.")
                return X
            if Rn >= maxtol:
                log(f"ODE12r: Residual {Rn} is too "
                    f"large at iteration number {nit}")
                return X

            # Compute a new step size.
            # Based on the extrapolation and some other heuristics
            h = max(0.25 * h,
                    min(4 * h, h_err, h_ls))  # Log steep-size analytic results

            debug(f"ODE12r:      accept: new h = {h}, |F| = {Rp}")
            debug(f"ODE12r:                hls = {h_ls}")
            debug(f"ODE12r:               herr = {h_err}")
        else:
            # Compute a new step size.
            h = max(0.1 * h, min(0.25 * h, h_err,
                                 h_ls))
            debug(f"ODE12r:      reject: new h = {h}")
            debug(f"ODE12r:               |Fnew| = {Rp_new}")
            debug(f"ODE12r:               |Fold| = {Rp}")
            debug(f"ODE12r:        |Fnew|/|Fold| = {Rp_new/Rp}")

        # abort if step size is too small
        if abs(h) <= hmin:
            raise OptimizerConvergenceError('ODE12r terminates unsuccessfully'
                                            f' Step size {h} too small')

    raise OptimizerConvergenceError(f'ODE12r terminates unsuccessfully after '
                                    f'{steps} iterations.')


class ODE12r(SciPyOptimizer):
    """
    Optimizer based on adaptive ODE solver :func:`ode12r`
    """
    def __init__(self, atoms, logfile='-', trajectory=None,
                 callback_always=False, alpha=1.0, master=None,
                 force_consistent=None, precon=None, verbose=0, rtol=1e-2):
        SciPyOptimizer.__init__(self, atoms, logfile, trajectory,
                                callback_always, alpha, master,
                                force_consistent)
        from ase.optimize.precon.precon import make_precon  # avoid circular dep
        self.precon = make_precon(precon)
        self.verbose = verbose
        self.rtol = rtol

    def apply_precon(self, Fn, X):
        self.atoms.set_positions(X.reshape(len(self.atoms), 3))
        Fn, Rn = self.precon.apply(Fn, self.atoms)
        return Fn, Rn

    def call_fmin(self, fmax, steps):
        ode12r(lambda x: -self.fprime(x),
               self.x0(),
               fmax=fmax, steps=steps,
               verbose=self.verbose,
               apply_precon=self.apply_precon,
               callback=self.callback,
               converged=lambda F, X: self.converged(F.reshape(-1, 3)),
               rtol=self.rtol)

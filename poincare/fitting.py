# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.21.1",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")

async with app.setup(hide_code=True):
    # Setup cell for marimo notebook, can be ignored
    import marimo as mo
    import sys

    # Import packages if running on marimo playground
    if sys.executable == "/home/pyodide/this.program":
        import micropip

        await micropip.install(
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "poincare>=1.3.0", "matplotlib"],
            verbose=False,
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Fitting models to data
    Poincare can fit model parameters (and initial conditions) to experimental or expected results. The tools live in `poincare.analysis`:

    - `make_target` builds the data to fit in the same format as simulation results.
    - `fit_result` fits one time course.
    - `fit_steady_state` fits steady states measured at different parameter values.
    - `Fitter` fits several experiments at once, with shared parameters.

    All of them use [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) and return the best values together with their covariance.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fitting a time course
    First we define a model to fit, a harmonic oscillator with unknown frequency `omega`:
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from poincare import (
        Derivative,
        Parameter,
        Simulator,
        System,
        Variable,
        assign,
        initial,
        solvers,
    )

    class Oscillator(System):
        x: Variable = initial(default=1)
        vx: Derivative = x.derive(initial=0)

        omega: Parameter = assign(default=1)

        spring = vx.derive() << -(omega**2) * x

    return (
        Derivative,
        Oscillator,
        Parameter,
        Simulator,
        System,
        Variable,
        assign,
        initial,
        np,
        plt,
        solvers,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The fit runs the model many times, changing the parameters slightly to find in which direction to move them. For that to work, the simulation error must be much smaller than those changes, so we create the `Simulator` with a solver with tight tolerances (see the `ToleranceWarning` section below):
    """)
    return


@app.cell
def _(Oscillator, Simulator, solvers):
    sim = Simulator(Oscillator).with_solver(solvers.LSODA(rtol=1e-10, atol=1e-12))
    return (sim,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As data we will use the known solution for the harmonic oscillator, $x(t) = \cos(\omega t)$ with $\omega = 2$, plus some noise:
    """)
    return


@app.cell
def _(np, plt):
    t = np.linspace(0, 10, 50)  # Measurement times
    omega_real = 2  # "Actual" omega
    rng = np.random.default_rng(0)
    x_data = np.cos(omega_real * t) + rng.normal(0, 0.05, t.size)

    plt.plot(t, x_data, ".", label="x")
    plt.xlabel("time")
    plt.legend()
    return omega_real, rng, t, x_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The target has to be an xarray `Dataset` in the same format as the simulation results. The `make_target` function builds it from a dictionary `{variable: values}` and the times at which the values were measured:
    """)
    return


@app.cell
def _(Oscillator, t, x_data):
    from poincare.analysis import fit_result, make_target

    target = make_target({Oscillator.x: x_data}, save_at=t)
    target
    return fit_result, make_target, target


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We pass the simulator and the target to `fit_result`, together with a dictionary `p0` telling it which parameters to fit. For each parameter we can give:

    - an initial guess,
    - `None`, to start from the model's default value,
    - a tuple `(initial guess, lower bound, upper bound)` to add bounds (the guess can be `None`).

    The model is simulated at the times of the target, so there is no need to set them on the simulator.
    """)
    return


@app.cell
def _(Oscillator, fit_result, omega_real, sim, target):
    fit = fit_result(sim, target, p0={Oscillator.omega: omega_real * 1.1})
    fit
    return (fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The result is a dictionary `{parameter: best value}`, which can be passed directly to `with_values` to simulate the fitted model:
    """)
    return


@app.cell
def _(fit, np, plt, sim, t, x_data):
    fitted = sim.with_values(fit).solve(save_at=np.linspace(0, 10, 500))
    plt.plot(t, x_data, ".", label="data")
    plt.plot(fitted["time"], fitted["x"], label="fit")
    plt.xlabel("time")
    plt.legend()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Initial conditions, transforms and scales
    Initial conditions are fitted like parameters, by adding the variable to `p0`.

    We can also fit arbitrary functions of the variables and parameters by adding a `transform` to the `Simulator` and including it in the target. For our example we add the energy $E = \frac{1}{2} v_x^2 + \frac{1}{2} \omega^2 x^2$, which is expected to stay constant at $E = \frac{1}{2} \omega^2 = 2$:
    """)
    return


@app.cell
def _(Oscillator, make_target, np, sim, t, x_data):
    sim_E = sim.with_transform(
        {"E": Oscillator.vx**2 / 2 + Oscillator.omega**2 * Oscillator.x**2 / 2},
        append=True,
    )
    target_E = make_target({Oscillator.x: x_data, "E": np.full_like(t, 2.0)}, save_at=t)
    return sim_E, target_E


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The optional `scale` dictionary changes how much each variable weighs in the fit. Here `scale={"E": 0.5}` weighs the error in the energy half as much as the error in `x`. We also fit the initial value of `x`, bounded between 0 and 10:
    """)
    return


@app.cell
def _(Oscillator, fit_result, omega_real, sim_E, target_E):
    fit_E = fit_result(
        sim_E,
        target_E,
        p0={Oscillator.omega: omega_real * 1.1, Oscillator.x: (None, 0, 10)},
        scale={"E": 0.5},
    )
    fit_E
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Uncertainties
    Besides the best values, the result has the covariance matrix of the fitted parameters in `.pcov`. It is indexed by a pair of parameters (or their names), and its diagonal gives the variance of each parameter, so the standard error is its square root:
    """)
    return


@app.cell
def _(Oscillator, fit, np):
    print(fit.pcov)
    print("omega =", fit[Oscillator.omega], "+/-", np.sqrt(fit.pcov[Oscillator.omega, Oscillator.omega]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `np.asarray(fit.pcov)` gives the plain matrix, ordered as `fit.pcov.parameters`. If the parameters have units, `fit.pcov[p, q]` is in units of `p * q`.

    In the fit above we gave no experimental errors, so all points are assumed to have the same error, which is estimated from how far the data is from the fit. If we know the errors of our measurements, we can pass them with `sigma=`, in the same form as the target. The residuals are then divided by `sigma`, and the covariance uses those errors as they are:
    """)
    return


@app.cell
def _(Oscillator, fit_result, make_target, np, omega_real, sim, t, target):
    sigma = make_target({Oscillator.x: np.full_like(t, 0.05)}, save_at=t)

    fit_sigma = fit_result(
        sim, target, p0={Oscillator.omega: omega_real * 1.1}, sigma=sigma
    )
    print(fit_sigma)
    print("omega error:", np.sqrt(fit_sigma.pcov["omega", "omega"]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fitting steady states
    Sometimes the data are not time courses but steady states measured for different values of a parameter (or initial condition). `fit_steady_state` compares the data with a `SteadyState.sweep` over that parameter (see [Asymptotic behaviour](https://dyscolab.github.io/static_notebooks/asymptotic_behaviour.html)).

    As an example, take a mass `m` hanging from a spring of stiffness `k`, with a damping `c`. Measuring the position `x` downwards from the rest length of the spring, the equation of motion is $m \ddot{x} = m g - k x - c \dot{x}$, and the mass comes to rest at $x = m g / k$:
    """)
    return


@app.cell
def _(
    Derivative,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
    solvers,
):
    class HangingSpring(System):
        x: Variable = initial(default=0)
        v: Derivative = x.derive(initial=0)

        m: Parameter = assign(default=1)
        k: Parameter = assign(default=1)
        c: Parameter = assign(default=1)
        g: Parameter = assign(default=9.81)

        eq = v.derive() << g - (k * x + c * v) / m

    sim_spring = Simulator(HangingSpring).with_solver(
        solvers.LSODA(rtol=1e-10, atol=1e-12)
    )
    return HangingSpring, sim_spring


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We hang several masses and measure where each one comes to rest. The target is a `Dataset` whose coordinate is the swept parameter, in the same format as the output of `SteadyState.sweep`:
    """)
    return


@app.cell
def _(np, rng):
    import xarray as xr

    masses = np.array([0.5, 1.0, 2.0, 4.0])
    k_real = 4.0  # "Actual" stiffness
    x_rest = masses * 9.81 / k_real + rng.normal(0, 0.02, masses.size)

    target_steady = xr.Dataset(
        {"x": xr.DataArray(x_rest, dims="m", coords={"m": masses})}
    )
    target_steady
    return k_real, masses, target_steady


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `fit_steady_state` takes the simulator, a `SteadyState` instance, the parameter that was varied and the target, plus `p0`, `scale` and `sigma` as in `fit_result`:
    """)
    return


@app.cell
def _(HangingSpring, sim_spring, target_steady):
    from poincare import SteadyState
    from poincare.analysis import fit_steady_state

    fit_steady = fit_steady_state(
        sim_spring,
        SteadyState(),
        HangingSpring.m,
        target_steady,
        p0={HangingSpring.k: (1.0, 1e-3, 100)},
    )
    fit_steady
    return (SteadyState,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fitting several experiments at once
    Often parameters are measured in several experiments, for example with different initial conditions, or some as time courses and others as steady states. The `Fitter` class fits all of them at once, with the same values for the fitted parameters. It takes:

    - the `Simulator`,
    - a list of simulation functions, one per experiment. Each takes the simulator (with the parameters being tried) and returns a result comparable to its target, so it is where each experiment's times and conditions are set,
    - a list of targets,
    - `p0`, as in `fit_result`, and optionally `scales` and `sigma` (one per target).

    The resting positions above tell us the stiffness `k`, but nothing about the damping `c`. To measure it, we release a mass of 1 from the rest length of the spring and record how it rings down:
    """)
    return


@app.cell
def _(HangingSpring, k_real, make_target, np, plt, rng, sim_spring):
    c_real = 0.8  # "Actual" damping
    t_ring = np.linspace(0, 15, 60)
    _true = sim_spring.with_values(
        {HangingSpring.k: k_real, HangingSpring.c: c_real}
    ).solve(save_at=t_ring)
    x_ring = _true["x"].to_numpy() + rng.normal(0, 0.02, t_ring.size)
    target_ring = make_target({HangingSpring.x: x_ring}, save_at=t_ring)

    plt.plot(t_ring, x_ring, ".", label="x")
    plt.xlabel("time")
    plt.legend()
    return t_ring, target_ring


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We fit `k` and `c` to both experiments: the ring-down is compared with a simulation at the measured times, and the resting positions with a steady state sweep over the masses:
    """)
    return


@app.cell
def _(
    HangingSpring,
    SteadyState,
    masses,
    sim_spring,
    t_ring,
    target_ring,
    target_steady,
):
    from poincare.analysis import Fitter

    fitter = Fitter(
        sim_spring,
        [
            lambda sim: sim.solve(save_at=t_ring),
            lambda sim: SteadyState().sweep(sim, variable=HangingSpring.m, values=masses),
        ],
        [target_ring, target_steady],
        p0={HangingSpring.k: (1.0, 1e-3, 100), HangingSpring.c: (None, 0, 10)},
    )
    fit_both = fitter.fit()
    fit_both
    return (fitter,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `Fitter.fit()` returns the same kind of result as `fit_result`, with the covariance in `.pcov`. It can be called again with a different `p0` to try other initial guesses or bounds for the same parameters, and extra keyword arguments are passed to `scipy.optimize.least_squares`:
    """)
    return


@app.cell
def _(HangingSpring, fitter):
    fitter.fit(p0={HangingSpring.k: (3.0, 1e-3, 100), HangingSpring.c: (0.5, 0, 10)})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Warnings
    Fits can warn when their results may not be reliable. Both warnings are in `poincare.analysis.fits`.

    ### ToleranceWarning
    `least_squares` estimates how the residuals change with each parameter by changing it by a very small relative step (about $10^{-8}$). If the solver error is larger than the effect of that step, the fit can stop early at wrong values. A `ToleranceWarning` means the solver tolerances are too loose for this. With the default solver tolerances we get:
    """)
    return


@app.cell
def _(Oscillator, Simulator, fit_result, target):
    import warnings

    from poincare.analysis.fits import CovarianceWarning, ToleranceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_result(Simulator(Oscillator), target, p0={Oscillator.omega: 2.2})

    print(*[w.message for w in caught if w.category is ToleranceWarning], sep="\n")
    return CovarianceWarning, warnings


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    What to try:

    - tighten the tolerances with `sim.with_solver(...)`, for example `solvers.LSODA(rtol=1e-10, atol=1e-12)`, as we did above,
    - or, if tighter tolerances make the simulations too slow, pass a larger step to the fit, e.g. `fit_result(..., diff_step=1e-4)`.

    ### CovarianceWarning
    A `CovarianceWarning` means the fitted values may be fine, but their covariance (and so their errors) is not reliable. It appears when:

    - a parameter ended at one of its bounds: widen the bounds, or check if the model or initial guess is right,
    - there are not enough data points to estimate the noise when no `sigma` is given: add data or pass `sigma`,
    - some parameters are not determined separately by the data. For example, a capacitor `C` discharging through a resistor `R` depends only on the product `R * C`, so any pair with the same product fits equally well:
    """)
    return


@app.cell
def _(
    CovarianceWarning,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    fit_result,
    initial,
    make_target,
    np,
    solvers,
    warnings,
):
    class RCCircuit(System):
        V: Variable = initial(default=5.0)
        R: Parameter = assign(default=1.0)
        C: Parameter = assign(default=0.5)

        eq = V.derive() << -V / (R * C)

    _t = np.linspace(0, 5, 20)
    _target = make_target({RCCircuit.V: 5 * np.exp(-_t / 0.5)}, save_at=_t)

    with warnings.catch_warnings(record=True) as caught_cov:
        warnings.simplefilter("always")
        fit_rc = fit_result(
            Simulator(RCCircuit).with_solver(solvers.LSODA(rtol=1e-10, atol=1e-12)),
            _target,
            p0={RCCircuit.R: 1.2, RCCircuit.C: 0.4},
        )

    print(fit_rc)
    print(*[w.message for w in caught_cov if w.category is CovarianceWarning], sep="\n")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The product `R * C` is well determined, but `R` and `C` separately are not. In this case, fix one parameter of each group (leave it out of `p0`), or add data that distinguishes them, for example another experiment fitted together with `Fitter`.
    """)
    return


if __name__ == "__main__":
    app.run()

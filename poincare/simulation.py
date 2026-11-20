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
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "poincare>=1.2.0", "matplotlib"], verbose = False
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Simulation in poincare
    Poincare and SimBio include a number of different options for simulation, including choosing the times at which the system is integrated and saved, changing solvers and backends, and interactive simulation. Searching for steady states and limit cycles is covered in [Asymptotic behaviour](https://dyscolab.github.io/static_notebooks/asymptotic_behaviour.html).

    We will use a damped oscillator as an example:
    """)
    return


@app.cell
def _():
    import numpy as np

    from poincare import (
        Constant,
        Derivative,
        Simulator,
        System,
        Variable,
        assign,
        initial,
    )

    # Define a System
    class DampedOscillator(System):
        x: Variable = initial(default=1)
        v: Derivative = x.derive(initial=0)
        k: Constant = assign(default=1, constant=True)
        eq = v.derive() << -k * x - 0.1 * v

    sim = Simulator(DampedOscillator)
    return DampedOscillator, Simulator, np, sim


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## save_at and t_span
    Two time arguments control a simulation:

    - `save_at`: the times at which the results are returned. Each row of the output `Dataset` corresponds to one of these times.
    - `t_span`: the interval `(start, end)` over which the solver integrates. The initial values of the variables are the values at `t_span[0]`. By default, `t_span` goes from the first to the last time in `save_at`.

    When a model is simulated only once, the simplest option is to pass `save_at` to `solve()`:
    """)
    return


@app.cell
def _(np, sim):
    sim.solve(save_at=np.linspace(0, 50, 200)).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When the same model is simulated several times, the times can be stored in the simulator with `with_times()`. Like `with_values()`, it returns a new `Simulator`, and `solve()` then uses the stored times:
    """)
    return


@app.cell
def _(DampedOscillator, np, sim):
    sim_50 = sim.with_times(save_at=np.linspace(0, 50, 200))

    for _x0 in [1, 2, 3]:
        _result = sim_50.with_values({DampedOscillator.x: _x0}).solve()
        _result["x"].plot(label=f"x(0) = {_x0}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Times given to `solve()` take precedence over the stored ones, so `sim_50.solve(save_at=...)` can still be used for a one-off simulation with different times.

    ### Starting the integration before the first saved time
    Since the initial values are those at the start of `t_span`, passing only `save_at=np.linspace(90, 100, 50)` would place the initial conditions at `t = 90`. To let the transient die out without saving it, we pass a `t_span` that starts earlier: the solver integrates from `t = 0`, but only the results between `t = 90` and `t = 100` are returned.
    """)
    return


@app.cell
def _(np, sim):
    late = sim.with_times(t_span=(0, 100), save_at=np.linspace(90, 100, 50)).solve()
    late.to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare it with saving at the same times without a `t_span`: the oscillation starts again from `x = 1` at `t = 90`.
    """)
    return


@app.cell
def _(np, sim):
    sim.solve(save_at=np.linspace(90, 100, 50)).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If only `t_span` is given, the results are returned at the time steps chosen by the solver:
    """)
    return


@app.cell
def _(sim):
    sim.with_times(t_span=(0, 10)).solve()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Using different solvers and backends
    We can change the method used for simulating the system with `with_solver()`, which also returns a new `Simulator`. Since we will simulate the same model with several solvers, we store the times on the simulator first. To use Runge-Kutta of order 4(5) instead of the default LSODA (SciPy's implementation of the Adams/BDF method):
    """)
    return


@app.cell
def _(np, sim):
    from poincare import solvers

    sim_times = sim.with_times(save_at=np.linspace(0, 10, 5))
    sim_times.with_solver(solvers.RK45()).solve()
    return sim_times, solvers


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Solvers also take the relative and absolute tolerances `rtol` and `atol`. We can compare the results of different solvers at the same times:
    """)
    return


@app.cell
def _(sim_times, solvers):
    import pandas as pd

    pd.DataFrame(
        {
            name: sim_times.with_solver(solver).solve()["x"].to_series()
            for name, solver in {
                "LSODA": solvers.LSODA(),
                "RK45": solvers.RK45(),
                "DOP853": solvers.DOP853(),
                "BDF (rtol=1e-8)": solvers.BDF(rtol=1e-8, atol=1e-10),
            }.items()
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Currently the implemented methods are: Adams/BDF (LSODA), Runge-Kutta of orders 3(2) (RK23), 4(5) (RK45) and 8 (DOP853), Runge-Kutta method of Radau IIA family of order 5 (Radau), and BDF (BDF). All of them are wrappers to SciPy's implementation of the method, see [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/integrate.html) for more details.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Poincare also allows the use of different backends to compile the systems to, including [NumPy](https://numpy.org/) (by default), [Numba](https://numba.pydata.org/) and [JAX](https://docs.jax.dev/en/latest/). The backend is chosen when creating the `Simulator`:
    """)
    return


@app.cell
def _(DampedOscillator, Simulator, np):
    try:
        result2 = Simulator(DampedOscillator, backend="numba").solve(save_at=np.linspace(0, 10, 5))
    except ModuleNotFoundError:
        print("Numba does't run on pyodide, we promise it'll work if you install it locally.")
        result2 = Simulator(DampedOscillator).solve(save_at=np.linspace(0, 10, 5))
    result2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Interactive simulators

    ### In marimo
    """)
    return


@app.cell
def _():
    slider = mo.ui.slider(start=0, stop=10, step=0.1, value=0.5, label="v(0)")
    slider
    return (slider,)


@app.cell
def _(DampedOscillator, np, sim, slider):
    sim.with_values({DampedOscillator.v: slider.value}).solve(
        save_at=np.linspace(0, 10, 100)
    ).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### In Jupyter

     in Jupyter we use the `interact` method to create interactive simulations to vary parameters or initial conditions:
    """)
    return


@app.cell
def _(DampedOscillator, np, sim):
    try:
        result4 = sim.interact(
            save_at=np.linspace(0, 10, 100), values={DampedOscillator.k: (0, 10, 0.1)}
        )
    except ImportError as e:
        print(f"ImportError: {e}")
    return


if __name__ == "__main__":
    app.run()

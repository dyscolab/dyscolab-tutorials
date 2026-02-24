import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simulation in poincare
    Poincare and SimBio include a number of different options for simulation, including changing solvers and backends, searching for steady states and limit cycles with parameter sweeps and interactive simulation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using different solvers and backends
    We can change the method used for simulating the system by passing it to the `solve` method. To use Runge-Kutta  of order 4 (5) instead of the default LSODA (SciPy's implementation of Adams/BDF method):
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

    return DampedOscillator, Simulator, np


@app.cell
def _(DampedOscillator, Simulator):
    from poincare import solvers

    result = Simulator(DampedOscillator).solve(solver=solvers.RK45(), save_at=range(3))
    result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Currently the implemented methods are: Asams/BDF (LSODA), Runge-Kutta of orders 3(2) (RK32), 4(5) (RK45) and 8 (DOP853), Runge-Kutta method of Radau IIA family of order 5 (Radau), and BDF (BDF). All oft them are wrappers to SciPy's implementation  of the method, see [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/integrate.html) for more details.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Poincare also allows the use of diffent backends to compile the systems to, inculuding [NumPy](https://numpy.org/) (by default) [Numba](https://numba.pydata.org/) and [JAX](https://docs.jax.dev/en/latest/):
    """)
    return


@app.cell
def _(DampedOscillator, Simulator):
    result2 = Simulator(DampedOscillator, backend="numba").solve(save_at=range(3))
    result2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### In marimo
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive simulators

    ### In marimo
    """)
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=10, step=0.1, value=0.5, label="v(0)")
    slider
    return (slider,)


@app.cell
def _(DampedOscillator, Simulator, np, slider):
    Simulator(DampedOscillator).solve(
        save_at=np.linspace(0, 10, 100), values={DampedOscillator.v: slider.value}
    ).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### In Jupyter

     in Jupyter we use the `interact` method to create interactive simulations to vary parameters or initial conditions:
    """)
    return


@app.cell
def _(DampedOscillator, Simulator, np):
    result4 = Simulator(DampedOscillator).interact(
        save_at=np.linspace(0, 10, 100), values={DampedOscillator.k: (0, 10, 0.1)}
    )
    return


@app.cell
def _():
    # Setup cell for marimo notebook, can be ignored
    import marimo as mo
    import sys

    # Import packages if running on mairmo playground
    if sys.executable == "/home/pyodide/this.program":
        import micropip

        await micropip.install(["typing_extensions>=4.15.0", "poincare>=1.0.0b2"])
    return (mo,)


if __name__ == "__main__":
    app.run()

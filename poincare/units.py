# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.10",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

async with app.setup(hide_code=True):
    # Setup cell for marimo notebook, can be ignored
    import marimo as mo
    import sys

    # Import packages if running on marimo playground
    if sys.executable == "/home/pyodide/this.program":
        import micropip

        await micropip.install(
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "poincare>=1.2.0", "matplotlib"],
            verbose=False,
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Units in poincare
    poincaré supports physical units through
    [`pint`](https://github.com/hgrecco/pint).
    Variables, parameters and simulation times can be given as `pint` quantities,
    and the results of the simulation carry the corresponding units.

    All quantities in a model must come from the same unit registry:
    """)
    return


@app.cell
def _():
    import numpy as np
    import pint

    from poincare import (
        Derivative,
        Parameter,
        Simulator,
        System,
        Variable,
        assign,
        initial,
    )

    u = pint.UnitRegistry()
    return Derivative, Parameter, Simulator, System, Variable, assign, initial, np, pint, u


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Variables, parameters and times with units

    Consider a harmonic oscillator whose position is measured in meters and its angular frequency in hertz:
    """)
    return


@app.cell
def _(Derivative, Parameter, Simulator, System, Variable, assign, initial, np, u):
    class Oscillator(System):
        x: Variable = initial(default=1 * u.m)
        v: Derivative = x.derive(initial=0 * u.m / u.s)
        w: Parameter = assign(default=1 * u.Hz)
        eq = v.derive() << -(w**2) * x

    oscillator_result = Simulator(Oscillator).solve(save_at=np.linspace(0, 10, 100) * u.s)
    oscillator_result
    return (oscillator_result,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The variables `x` and `v` have units of `m` and `m/s`, respectively.
    `pint` raises a `DimensionalityError` if we try to add them:
    """)
    return


@app.cell
def _(oscillator_result, pint):
    try:
        _ = oscillator_result["x"] + oscillator_result["v"]
    except pint.DimensionalityError as de:
        print("DimensionalityError:", de)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Simulation times

    The times passed to the simulator must have the dimension of the independent variable,
    which poincaré infers from the model.
    Here, a solution is washed out of a tank with a residence time of 2 minutes,
    so the independent variable is a time:
    """)
    return


@app.cell
def _(Simulator, System, Variable, initial, np, u):
    class Washout(System):
        c: Variable = initial(default=10 * u.mM)
        eq = c.derive() << -c / (2 * u.min)

    washout_result = Simulator(Washout).solve(save_at=np.linspace(0, 10, 50) * u.min)
    washout_result
    return Washout, washout_result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Any unit of time can be used, and the results are returned in that unit.
    If the times have another dimension, or no units at all, `solve()` raises an error:
    """)
    return


@app.cell
def _(Simulator, Washout, np, pint, u):
    for _save_at in [np.linspace(0, 10, 50) * u.m, np.linspace(0, 10, 50)]:
        try:
            Simulator(Washout).solve(save_at=_save_at)
        except pint.PintError as pe:
            print("PintError:", pe)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Removing units

    ### From the results

    The result is an [xarray](https://docs.xarray.dev/en/stable/) `Dataset` with units handled by [`pint-xarray`](https://pint-xarray.readthedocs.io/).
    Its `.pint` accessor converts the results to other units:
    """)
    return


@app.cell
def _(washout_result):
    washout_result.pint.to(c="uM")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    and removes them with `dequantify()`, which stores the units as attributes of each variable:
    """)
    return


@app.cell
def _(washout_result):
    plain_result = washout_result.pint.dequantify()
    plain_result["c"].attrs
    return (plain_result,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    which allows to plot the DataFrame with `.plot()`:
    """)
    return


@app.cell
def _(plain_result):
    plain_result.to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### From a model

    `strip_units()` returns a copy of a model where every quantity is replaced by its magnitude in base units (seconds, meters, moles, ...).
    This is useful to work with tools that expect plain numbers.
    The removed units are kept in the `_units` attribute of the new model:
    """)
    return


@app.cell
def _(Washout):
    from poincare import strip_units

    PlainWashout = strip_units(Washout)
    PlainWashout._units
    return (PlainWashout,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The new model is simulated with times in seconds, and its results are plain numbers in base units (here, the concentration in mol/m³):
    """)
    return


@app.cell
def _(PlainWashout, Simulator, np):
    Simulator(PlainWashout).solve(save_at=np.linspace(0, 600, 50)).to_dataframe().plot()
    return


if __name__ == "__main__":
    app.run()

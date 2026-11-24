# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.10",
#     "pyzmq>=27.1.0",
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
    # Units in poincare
    Poincare supports physical units through
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
    return (
        Derivative,
        Parameter,
        Simulator,
        System,
        Variable,
        assign,
        initial,
        np,
        pint,
        u,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Variables, parameters and times with units

    Consider a harmonic oscillator whose position is measured in meters and its angular frequency in hertz:
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
    np,
    u,
):
    class Oscillator(System):
        x: Variable = initial(default=1 * u.m)
        v: Derivative = x.derive(initial=0 * u.m / u.s)
        w: Parameter = assign(default=1 * u.Hz)
        eq = v.derive() << -(w**2) * x

    oscillator_result = Simulator(Oscillator).solve(save_at=np.linspace(0, 10, 100) * u.s)
    oscillator_result
    return Oscillator, oscillator_result


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
    Units are also checked when a model is defined. If units aren't defiende consistently it will raise a `PintError`.
    """)
    return


@app.cell
def _(Derivative, Parameter, System, Variable, assign, initial, pint, u):
    try:
        class WrongUnitsOscillator(System):
            x: Variable = initial(default=1 * u.m)
            v: Derivative = x.derive(initial=0 * u.m / u.s)
            w: Parameter = assign(default=1)
            eq = v.derive() << -(w**2) * x
        
    except pint.PintError as pe:
            print("PintError:", pe)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here the definition of `v`'s default value implies that the independet variable has units in `s` since it's a first derivative and it's units are `1/s` the originigal variable's. However when we remove `w`'s units we are giving `x` the `x`'s second derivative the same units as `x`, implying an adimensional independetvariable. Poincare detects this incosistency and riases an error.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Simulation times

    The times passed to the simulator must have the dimension of the independent variable,
    which Poincare infers from the model if not given explicitly.
    In the oscillator above, `x` is in `m` and its derivative `v` in `m/s`,
    so the independent variable is a time, which is the most common case.
    That is why `save_at` was given in seconds.
    Any unit of time can be used, and the results are returned in that unit:
    """)
    return


@app.cell
def _(Oscillator, Simulator, np, u):
    Simulator(Oscillator).solve(save_at=np.linspace(0, 10_000, 100) * u.ms)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If the times have another dimension, or no units at all, `solve()` raises an error:
    """)
    return


@app.cell
def _(Oscillator, Simulator, np, pint, u):
    for _save_at in [np.linspace(0, 10, 100) * u.m, np.linspace(0, 10, 100)]:
        try:
            Simulator(Oscillator).solve(save_at=_save_at)
        except pint.PintError as pe:
            print("PintError:", pe)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Other independent variables

    The independent variable does not have to be a time.
    For example, the pressure of the atmosphere decreases with the altitude $z$ as

    $$\frac{dP}{dz} = -\frac{M g}{R T} P$$

    where $M$ is the molar mass of air, $g$ the acceleration of gravity, $R$ the gas constant and $T$ the temperature.
    Here the independent variable is a length.
    We declare it explicitly with `Independent`, giving the altitude where the integration starts (sea level):
    """)
    return


@app.cell
def _(Parameter, System, Variable, assign, initial, u):
    from poincare import Independent

    class Atmosphere(System):
        z = Independent(default=0 * u.m)
        P: Variable = initial(default=1013 * u.hPa)
        M: Parameter = assign(default=0.029 * u.kg / u.mol)
        g: Parameter = assign(default=9.81 * u.m / u.s**2)
        T: Parameter = assign(default=288 * u.K)

        eq = P.derive() << -M * g * P / (u.molar_gas_constant * T)

    return (Atmosphere,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now `save_at` must be a length, here in kilometers.
    The coordinate of the results is still called `time`, but it holds the altitudes:
    """)
    return


@app.cell
def _(Atmosphere, Simulator, np, u):
    atmosphere_result = Simulator(Atmosphere).solve(save_at=np.linspace(0, 10, 50) * u.km)
    atmosphere_result
    return (atmosphere_result,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    and times in seconds are rejected:
    """)
    return


@app.cell
def _(Atmosphere, Simulator, np, pint, u):
    try:
        Simulator(Atmosphere).solve(save_at=np.linspace(0, 10, 50) * u.s)
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
def _(atmosphere_result, u):
    atmosphere_result.pint.to(u.atm)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    and removes them with `dequantify()`, which stores the units as attributes of each variable:
    """)
    return


@app.cell
def _(atmosphere_result):
    plain_result = atmosphere_result.pint.dequantify()
    plain_result["P"].attrs
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
def _(Atmosphere):
    from poincare import strip_units

    PlainAtmosphere = strip_units(Atmosphere)
    PlainAtmosphere._units
    return (PlainAtmosphere,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The new model is simulated with plain numbers in base units, so the altitudes are given in meters, and its results are also in base units (here, the pressure in pascals):
    """)
    return


@app.cell
def _(PlainAtmosphere, Simulator, np):
    Simulator(PlainAtmosphere).solve(save_at=np.linspace(0, 10_000, 50)).to_dataframe().plot()
    return


if __name__ == "__main__":
    app.run()

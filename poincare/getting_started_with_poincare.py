# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.10",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")

async with app.setup(hide_code=True):
    # Setup cell for marimo notebook, can be ignored
    import marimo as mo
    import sys

    # Import packages if running on marimo playground
    if sys.executable == "/home/pyodide/this.program":
        import micropip

        await micropip.install(
            ["typing_extensions>=4.15.0", "poincare>=1.0.0b3", "matplotlib"]
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Getting started with Poincare
    Poincare is a python library for declaring and simulating dynamical systems. Designed around the principle of modularity, composability and reproducibility, it's intended to create a layer to separate the actual declaration of models from their simulation, allowing to easily switch methods and backends. It also makes models composable, allowing the combination of smaller systems to create larger ones, and implements a series of analysis tools such as parameter sweeps to find steady states or limit cicles.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Installation

    It can be installed from PyPI:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    pip install -U poincare
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    or conda-forge:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    conda install -c conda-forge poincare
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Creating and simultaing a system
    Systems are represented as a subclass of the `System` Class. To create one with equations:
    $$ \frac{dx}{dt} = -x \quad \text{with} \quad x(0) = 1 $$
    we can write:
    """)
    return


@app.cell
def _():
    from poincare import System, Variable, initial

    class Model(System):
        # Define a variable with name `x` with an initial value (t=0) of `1``.
        x: Variable = initial(default=1)
        # The rate of change of `x` (i.e. velocity) is assigned (<<) to `-x`.
        # This relation is assigned to a Python variable (`eq`)
        eq = x.derive() << -x

    return Model, System, Variable, initial


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To simulate that system we must create a `Simulator` for it:
    """)
    return


@app.cell
def _(Model):
    import numpy as np
    from poincare import Simulator

    _sim = Simulator(Model)
    result = _sim.solve(save_at=np.linspace(0, 10, 100))
    # Create a simulator for Model
    # Solve the model and save the resulting xarray dataset to "result"
    result.to_dataframe().plot()
    return Simulator, np, result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The output is a [xarray](https://docs.xarray.dev/en/stable/) `Dataset`,
    which can be plotted with `to_dataframe().plot()`.
    """)
    return


@app.cell
def _(result):
    # Display Datest
    result
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can change the initial conditions by passing a `values` dictionary to `solve`; it is possible to create more than one solution with different initial conditions from the same `Simulator`, avoiding model recompilation.
    """)
    return


@app.cell
def _(Model, Simulator):
    _sim = Simulator(Model)
    result1 = _sim.solve(values={Model.x: 2}, save_at=range(3))
    print("Result 1:  \n", result1)
    result2 = _sim.solve(values={Model.x: 3}, save_at=range(3))
    print("Result 2: \n", result2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Higher order systems
    To define a higher-order system, we have to explicitly define and assign an initial condition to the derivative of a variable:
    """)
    return


@app.cell
def _(Simulator, System, Variable, initial, np):
    from poincare import Derivative

    class Oscillator(System):
        x: Variable = initial(default=1)
        # Define `v` as the derivative of `x` with an intital value of `0`
        v: Derivative = x.derive(initial=0)
        eq = v.derive() << -x

    result_1 = Simulator(Oscillator).solve(save_at=np.linspace(0, 10, 100))
    result_1.to_dataframe().plot()
    return Derivative, Oscillator


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Non-autonomous systems

    To use the independent variable,
    we create an instance of `Independent`:
    """)
    return


@app.cell
def _(Simulator, System, Variable, initial):
    from poincare import Independent

    class NonAutonomous(System):
        time = Independent()
        x: Variable = initial(default=0)
        eq = x.derive() << 2 * time

    Simulator(NonAutonomous).solve(save_at=range(3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Transforming output
    We can compute transformations of the output
    by passing a dictionary `transform = {"name": expression}` of expressions to calculate:
    """)
    return


@app.cell
def _(Oscillator, Simulator, np):
    # Compute kinetic and potential energy
    result_2 = Simulator(
        Oscillator,
        transform={
            "x": Oscillator.x,
            "T": 1 / 2 * Oscillator.v**2,
            "V": 1 / 2 * Oscillator.x**2,
        },
    ).solve(save_at=np.linspace(0, 10, 100))
    result_2.to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note that the output will only save whatever is passed to transform, so the original variables must be explicitly passed in order to be included.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Constants and Parameters

    Besides variables,
    we can define parameters and constants,
    and use functions from [Symbolite](https://github.com/hgrecco/symbolite).

    ### Constants

    Constants allow to define common initial conditions for Variables and Derivatives:
    """)
    return


@app.cell
def _(Simulator, System, Variable, initial):
    from poincare import Constant, assign

    class ModelXY(System):
        c: Constant = assign(default=1, constant=True)
        x: Variable = initial(default=c)
        y: Variable = initial(default=2 * c)
        eq_x = x.derive() << -x
        eq_y = y.derive() << -y

    Simulator(ModelXY).solve(save_at=range(3))
    return ModelXY, assign


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now, we can vary their initial conditions jointly:
    """)
    return


@app.cell
def _(ModelXY, Simulator):
    Simulator(ModelXY).solve(values={ModelXY.c: 2}, save_at=range(3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    But we can break that connection by passing `y`'s initial value directly:
    """)
    return


@app.cell
def _(ModelXY, Simulator):
    Simulator(ModelXY).solve(values={ModelXY.c: 2, ModelXY.y: 2}, save_at=range(3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Parameters
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Parameters are like Variables,
    but their time evolution is given directly as a function of time,
    Variables, Constants and other Parameters:
    """)
    return


@app.cell
def _(Simulator, System, Variable, assign, initial):
    from poincare import Parameter

    class ParametrizedDecay(System):
        p: Parameter = assign(default=1)
        x: Variable = initial(default=1)
        eq = x.derive() << -p * x

    # We can use a value other than the default for parameter `p` when solving the system
    Simulator(ParametrizedDecay).solve(
        save_at=range(3), values={ParametrizedDecay.p: 0.5}
    )
    Simulator(ParametrizedDecay).solve(
        save_at=range(3), values={ParametrizedDecay.p: 2}
    )
    return (Parameter,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Functions
    Symbolite functions are accessible from the `symbolite.real` module:
    """)
    return


@app.cell
def _(Parameter, Simulator, System, Variable, assign, initial):
    from symbolite import real

    class ParametrizedForce(System):
        x: Variable = initial(default=1)
        F: Parameter = assign(default=real.sin(x))

        eq = x.derive() << F

    Simulator(ParametrizedForce).solve(save_at=range(3))
    return ParametrizedForce, real


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note how `F` is a function of `x`. Parameters which depend on variables cannot be changed after model compilation:
    """)
    return


@app.cell
def _(ParametrizedForce, Simulator, real):
    try:
        Simulator(ParametrizedForce).solve(
            save_at=range(3),
            values={ParametrizedForce.F: real.cos(ParametrizedForce.x)},
        )
    except ValueError as ve:
        # Poincare Raises a ValueError when the functional dependece of a parameter is changed
        print("ValueError:", ve)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Instead we must recompile the model with the new value:
    """)
    return


@app.cell
def _(ParametrizedForce, Simulator, real):
    # crate a new simulator with the new formula for `F`
    new_sim = Simulator(ParametrizedForce(F=real.cos(ParametrizedForce.x)))
    new_sim.solve(save_at=range(3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Units

    poincaré also supports functions through
    [`pint`](https://github.com/hgrecco/pint)
    and [`pint-pandas`](https://github.com/hgrecco/pint-pandas).
    """)
    return


@app.cell
def _(Derivative, Parameter, Simulator, System, Variable, assign, initial):
    import pint

    unit = pint.UnitRegistry()

    class UnitModel(System):
        x: Variable = initial(default=1 * unit.m)
        v: Derivative = x.derive(initial=0 * unit.m / unit.s)
        w: Parameter = assign(default=1 * unit.Hz)
        eq = v.derive() << -(w**2) * x

    result_3 = Simulator(UnitModel).solve(save_at=range(3))
    return (result_3,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The columns have units of `m` and `m/s`, respectively.
    `pint` raises a `DimensionalityError` if we try to add them:
    """)
    return


@app.cell
def _(result_3):
    from pint import DimensionalityError

    try:
        result_3["x"] + result_3["v"]
    except DimensionalityError as de:
        print("DimensionalityError: ", de)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can remove the units with:
    """)
    return


@app.cell
def _(result_3):
    result_3.pint.dequantify()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    which allows to plot the DataFrame with `.plot()`.
    """)
    return


if __name__ == "__main__":
    app.run()

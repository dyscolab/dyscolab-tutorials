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
    # Getting started with Poincare
    Poincare is a python library for declaring and simulating dynamical systems. Designed around the principle of modularity, composability and reproducibility, it's intended to create a layer to separate the actual declaration of models from their simulation, allowing to easily switch methods and backends. It also makes models composable, allowing the combination of smaller systems to create larger ones, and implements a series of analysis tools such as parameter sweeps to find steady states or limit cicles.
    """)
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
    ## Creating and simulating a system
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
    We can change the initial conditions by calling `with_values()` on the simulator and passing a dictionary; it is possible to create more than one solution with different initial conditions from the same `Simulator`, avoiding model recompilation.
    """)
    return


@app.cell
def _(Model, Simulator):
    _sim = Simulator(Model)
    result1 = _sim.with_values({Model.x: 2}).solve(save_at=range(3))
    print("Result 1:  \n", result1)
    result2 = _sim.with_values({Model.x: 3}).solve(save_at=range(3))
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
    by passing calling `with_transform()` on a `Simulator` and passing a dictionary `{"name": expression}` of expressions to calculate:
    """)
    return


@app.cell
def _(Oscillator, Simulator, np):
    # Compute kinetic and potential energy
    result_2 = Simulator(Oscillator).with_transform({
            "x": Oscillator.x,
            "T": 1 / 2 * Oscillator.v**2,
            "V": 1 / 2 * Oscillator.x**2,
        }).solve(save_at=np.linspace(0, 10, 100))
    result_2.to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note that the output will only save whatever is passed to transform, so the original variables must be explicitly passed in order to be included. This can be changed by setting `append = True` in `with_transform()`.
    """)
    return


@app.cell
def _(Oscillator, Simulator, np):
    result_2_append = Simulator(Oscillator).with_transform({
            "T": 1 / 2 * Oscillator.v**2,
            "V": 1 / 2 * Oscillator.x**2,
        }, append=True).solve(save_at=np.linspace(0, 10, 100))
    result_2_append.to_dataframe().plot()
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
    Simulator(ModelXY).with_values({ModelXY.c: 2}).solve(save_at=range(3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    But we can break that connection by passing `y`'s initial value directly:
    """)
    return


@app.cell
def _(ModelXY, Simulator):
    Simulator(ModelXY).with_values({ModelXY.c: 2, ModelXY.y: 2}).solve(save_at=range(3))
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
    Simulator(ParametrizedDecay).with_values({ParametrizedDecay.p: 0.5}).solve(save_at=range(3))
    return Parameter, ParametrizedDecay


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A parameter can also be declared without a default by calling `assign()` without arguments. The `Simulator` can still be created, but a value must be given with `with_values()` before solving:
    """)
    return


@app.cell
def _(Parameter, Simulator, System, Variable, assign, initial):
    class UnknownDecay(System):
        # `k` has no default value
        k: Parameter = assign()
        x: Variable = initial(default=1)
        eq = x.derive() << -k * x

    unknown_sim = Simulator(UnknownDecay)
    try:
        unknown_sim.solve(save_at=range(3))
    except TypeError as te:
        # Poincare raises a TypeError when solving with missing values
        print("TypeError:", te)
    return UnknownDecay, unknown_sim


@app.cell
def _(UnknownDecay, unknown_sim):
    unknown_sim.with_values({UnknownDecay.k: 2}).solve(save_at=range(3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulation times

    So far we have passed the times to `solve()` every time. When the same model is simulated more than once we can store the times on the simulator with `with_times()`. Like `with_values()`, it returns a new `Simulator`, and `solve()` can then be called without times:
    """)
    return


@app.cell
def _(ParametrizedDecay, Simulator, np):
    import matplotlib.pyplot as plt

    # Store the times on the simulator
    decay_sim = Simulator(ParametrizedDecay).with_times(save_at=np.linspace(0, 5, 50))

    _fig, _ax = plt.subplots()
    for _p in [0.5, 1, 2]:
        # Only the values change, the times are taken from the simulator
        _result = decay_sim.with_values({ParametrizedDecay.p: _p}).solve()
        _ax.plot(_result["time"], _result["x"], label=f"p = {_p}")
    _ax.set(xlabel="time", ylabel="x")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Besides `save_at`, `with_times()` and `solve()` also accept `t_span`, the interval over which the system is integrated. See [Simulation in Poincare](https://dyscolab.github.io/static_notebooks/simulation.html) for the difference between the two.
    """)
    return


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
        Force: Parameter = assign(default=real.sin(x))

        eq = x.derive() << Force

    force_sim = Simulator(ParametrizedForce).with_times(save_at=range(3))
    force_sim.solve()
    return ParametrizedForce, force_sim, real


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note how `Force` is a function of `x`. Parameters which depend on variables cannot be changed after model compilation:
    """)
    return


@app.cell
def _(ParametrizedForce, force_sim, real):
    try:
        force_sim.with_values({ParametrizedForce.Force: real.cos(ParametrizedForce.x)}).solve()
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
    # create a new simulator with the new formula for Force
    new_sim = Simulator(ParametrizedForce(Force=real.cos(ParametrizedForce.x)))
    new_sim.solve(save_at=range(3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Units

    Poincare also supports units through
    [`pint`](https://github.com/hgrecco/pint)
    and [`pint-pandas`](https://github.com/hgrecco/pint-pandas).
    Variables, parameters and simulation times can be given with units:
    """)
    return


@app.cell
def _(Derivative, Parameter, Simulator, System, Variable, assign, initial, np):
    import pint

    unit = pint.UnitRegistry()

    class UnitModel(System):
        x: Variable = initial(default=1 * unit.m)
        v: Derivative = x.derive(initial=0 * unit.m / unit.s)
        w: Parameter = assign(default=1 * unit.Hz)
        eq = v.derive() << -(w**2) * x

    Simulator(UnitModel).solve(save_at=np.linspace(0, 2, 3) * unit.s)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The columns of the result have units of `m` and `m/s`, respectively. See [Units in Poincare](https://dyscolab.github.io/static_notebooks/units.html) for more on working with units.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model Reports
    Aside for its use in simulation, Poincare is meant to be a centralized source for all information concerning the system. To ensure consistency between analytical formulations and numerical implementations, we can generate a report with all relevant model data (equations, variables and parameters).

    In a notebook, `notebook_report()` displays the report directly:
    """)
    return


@app.cell
def _(ParametrizedDecay):
    from poincare.printing import notebook_report

    notebook_report(ParametrizedDecay)
    return (notebook_report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Some arguments control how the report is generated:

    - `transform`: a dictionary `{component: latex expression}` to replace the name of certain components. E.g.: `transform={ParametrizedDecay.p: "k"}` writes "k" instead of "p" in the report.
    - `descriptions`: a dictionary `{component: description}` that adds a column with a description of each variable and parameter.
    """)
    return


@app.cell
def _(ParametrizedDecay, notebook_report):
    notebook_report(
        ParametrizedDecay,
        transform={ParametrizedDecay.p: "k"},
        descriptions={
            ParametrizedDecay.x: "amount",
            ParametrizedDecay.p: "decay rate",
        },
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - `replace_algebraics`: whether to replace parameters that depend on other parameters and variables by their expressions. It is `False` by default, so "Force" appears in the equations and is defined in the parameter table:
    """)
    return


@app.cell
def _(ParametrizedForce, notebook_report):
    notebook_report(ParametrizedForce)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Passing `replace_algebraics=True` writes its expression into the equations instead:
    """)
    return


@app.cell
def _(ParametrizedForce, notebook_report):
    notebook_report(ParametrizedForce, replace_algebraics=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To include the model in an external document, `model_report()` generates the same report as LaTeX code and writes it to a file. It takes the same `transform`, `descriptions` and `replace_algebraics` arguments:
    """)
    return


@app.cell
def _(ParametrizedDecay):
    from poincare import model_report

    model_report(
        ParametrizedDecay,
        path="Parametrized_Decay_report.tex",
        transform={ParametrizedDecay.p: "k"},
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `standalone` argument decides whether the file compiles as a standalone document (by including headers such as `\documentclass` and `\begin{document}`) or is meant to be added to an existing document. It also controls whether existing contents of the file are overwritten (if `True`) or appended to (if `False`). It is `True` by default.

    If no `path` is given, `model_report()` returns the LaTeX code as a string instead.
    """)
    return


if __name__ == "__main__":
    app.run()

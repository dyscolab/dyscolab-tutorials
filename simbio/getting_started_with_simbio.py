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
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "simbio>=1.0.1", "matplotlib"], verbose = False
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Getting started with SimBio
    SimBio is a Python-based package for simulation of Chemical Reaction Networks (CRNs). It extends [poincare](https://dyscolab.github.io/poincare/), a package for modelling dynamical systems, to add functionality for CRNs.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Installation

    Using [pixi](https://pixi.sh/latest/),
    install from PyPI with:

    ```sh
    pixi add --pypi simbio
    ```

    or install the latest development version from GitHub with:

    ```sh
    pixi add --pypi simbio@https://github.com/dyscolab/simbio.git
    ```

    Otherwise,
    use `pip` or your `pip`-compatible package manager:

    ```sh
    pip install simbio  # from PyPI
    pip install git+https://github.com/dyscolab/simbio.git  # from GitHub
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Creating and simulating a Model
    To create a model we must create class which inherits from `System`, which can contain different `Variables`, in this case, A, B and C. We can define a `RateLaw` to add the reaction 2A + B $\rightarrow$ C.
    """)
    return


@app.cell
def _():
    from simbio import System, RateLaw, Variable, initial

    class Model(System):
        # Create species A, B, and C, each with intial value 1
        A: Variable = initial(default=1)
        B: Variable = initial(default=1)
        C: Variable = initial(default=1)

        reaction = RateLaw(reactants=[2 * A, B], products=[C], rate_law=1)

    return Model, RateLaw, System, Variable, initial


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Would be equivalent to the equations:
    $$
    \begin{aligned}
    \frac{dA}{dt} &= -2 \\
    \frac{dB}{dt} &= -1 \\
    \frac{dC}{dt} &= 1
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To simulate it we must create a `Simulator` for it
    """)
    return


@app.cell
def _(Model):
    import numpy as np
    from simbio import Simulator

    sim = Simulator(Model)
    result = sim.solve(save_at=np.linspace(0, 10, 100))
    result
    return Simulator, np, result, sim


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `Simulator.solve` outputs a [xarray](https://docs.xarray.dev/en/stable/) `Dataset`. To plot it we convert it to a pandas dataframe and use the `plot()` method.
    """)
    return


@app.cell
def _(result):
    result.to_dataframe()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can vary initial conditions by passing a `values` dictionary to the `solve()`
    """)
    return


@app.cell
def _(Model, sim):
    sim.solve(save_at=range(3), values={Model.A: 2, Model.B: 0.5}).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mass action reactions
    In the previous example the reaction happened at a constant rate. We can also implement it as a Mass action reaction
    """)
    return


@app.cell
def _(System, Variable, initial):
    from simbio import MassAction

    class ModelMA(System):
        # Create species A, B, and C, each with intial value 1
        A: Variable = initial(default=1)
        B: Variable = initial(default=1)
        C: Variable = initial(default=1)

        reaction = MassAction(reactants=[2 * A, B], products=[C], rate=1)

    return (ModelMA,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Corresponds to the equations:
    $$
    \begin{aligned}
    \frac{dA}{dt} &= -2A^2 \\
    \frac{dB}{dt} &= -B\\
    \frac{dC}{dt} &= A^2B
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(ModelMA, Simulator, np):
    simMA = Simulator(ModelMA)
    result_2 = simMA.solve(save_at=np.linspace(0, 10, 100))
    result_2.to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Constants and Parameters

    Besides variables,
    we can define parameters and constants,
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Constants

    Constants allow to define common initial conditions for Variables:
    """)
    return


@app.cell
def _(RateLaw, Simulator, System, Variable, initial):
    from simbio import Constant, assign

    class ModelConst(System):
        # Create a constant c to set intial conditions for A and B
        c: Constant = assign(default=1, constant=True)

        # Create species A, B, and C, each with intial value 1
        A: Variable = initial(default=c)
        B: Variable = initial(default=2 * c)
        C: Variable = initial(default=1)

        reaction = RateLaw(reactants=[2 * A, B], products=[C], rate_law=1)

    Simulator(ModelConst).solve(save_at=range(3))
    return ModelConst, assign


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now, we can vary their initial conditions jointly:
    """)
    return


@app.cell
def _(ModelConst, Simulator):
    Simulator(ModelConst).solve(values={ModelConst.c: 2}, save_at=range(3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    But we can break that connection by passing `y`'s initial value directly:
    """)
    return


@app.cell
def _(ModelConst, Simulator):
    Simulator(ModelConst).solve(
        values={ModelConst.c: 2, ModelConst.B: 3}, save_at=range(3)
    )
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
    Variables, Constants and other Parameters. Their default value can also be changed in `values`:
    """)
    return


@app.cell
def _(RateLaw, Simulator, System, Variable, assign, initial):
    from simbio import Parameter

    class ModelParametrized(System):
        # Create species A, B, and C, each with intial value 1
        A: Variable = initial(default=1)
        B: Variable = initial(default=1)
        C: Variable = initial(default=1)

        # Create a rate parameter to control the rate
        rate: Parameter = assign(default=1)

        reaction = RateLaw(reactants=[2 * A, B], products=[C], rate_law=rate)

    Simulator(ModelParametrized).solve(
        save_at=range(3), values={ModelParametrized.rate: 3}
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Other utilities
    SimBio contains a number of built in classes to represent common reactions, which can be found in `simbio.reactions.single`:
    - `Creation(A: Species, rate: Parameter)`: A substance `A` is created from nothing at a given rate, ∅ -> A.
    - `AutoCreation(A: Species, rate: Parameter)`: A substance `A` is created at a rate proportional to its abundance, A -> 2A.
    - `Destruction(A: Species, rate: Parameter)`: A substance `A` degrades into nothing, A -> ∅.
    - `Conversion(A: Species, B: Species, rate: Parameter)`: A substance `A` converts into `B`, A -> B.
    - `Synthesis(A: Species, B: Species, AB: Species, rate: Parameter)`:  Two simple substances `A` and `B` combine to form a more complex substance `AB`, A + B -> AB.
    - `Dissociation(AB: Species, A: Species, B: Species, rate: Parameter)`: A more complex substance `AB` breaks down into its more simple parts `A` and `B`, AB -> A + B.

    All rates in these reactions are set according to `MassAction` rules, so `Synthesis(A = 2*A, B=B, AB = AB, rate=k)` would represent:
    $$
    \begin{aligned}
    \frac{dA}{dt}&=- 2 A^{2} B k\\
    \frac{dB}{dt}&= -A^{2} B k\\
    \frac{dAB}{dt}&= A^{2} B k\\
    \end{aligned}
    $$

    There are also built in two-way reactions in `simbio.reactions.compound`:
    - `ReversibleSynthesis(A: Species, B: Species, AB: Species, forward_rate: Parameter, reverse_rate: Parameter)`: A `Synthesis` and `Dissociation` reactions each with a separate `rate`, A + B <-> AB.
    - `Equilibration(A: Species, B: Species, forward_rate: Parameter, reverse_rate: Parameter)`: A forward and backward `Conversion`, each with a separate `rate`, A <-> B.
    - `CatalyzeConvert(A: Species, B: Species, AB: Species, P: Species, forward_rate: Parameter, reverse_rate: Parameter, conversion_rate: Parameter)`: A `ReversibleSynthesis` between `A`, `B`, and `AB`, and a `Conversion` from `AB` to `P` with `rate = conversion_rate`, A + B <-> A:B -> P.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For more details into SimBio's capabilities, we recommend reading [poincare's documentation](https://dyscolab.github.io/poincare/).
    """)
    return


if __name__ == "__main__":
    app.run()

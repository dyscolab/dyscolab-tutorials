import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

async with app.setup(hide_code=True):
    # Setup cell for marimo notebook, can be ignored
    import marimo as mo
    import sys

    # Import packages if running on marimo playground
    if sys.executable == "/home/pyodide/this.program":
        import micropip

        await micropip.install(
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "simbio>=1.1.0", "matplotlib", "seaborn"],
            verbose=False,
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Stochastic simulations in SimBio
    Simbio contains the capability to stochastically simulate systems using the [Gillespie algorithm](http://en.wikipedia.org/wiki/Gillespie_algorithm); this is preferable over ODE based simulations for reactions where small amounts of reactants are present. Simbio uses the [rebop](https://rebop.readthedocs.io/en/latest/) library for this, which must be installed separately:
    ```bash
    pip install rebop
    ```
    To stochastically simulate a `System`, which we can define noramlly:
    """)
    return


@app.cell
def _():
    import xarray as xr
    from simbio.rebop import RebopSimulator
    from simbio import System, Variable, initial, MassAction, RateLaw

    class Infection(System):
        S: Variable = initial(default=100)
        D: Variable = initial(default=1)
        R: Variable = initial(default=0)

        r_infect = MassAction(reactants=[S, D], products=[2 * D], rate=0.015)
        r_cure = MassAction(reactants=[D], products=[R], rate=0.1)

    return Infection, RateLaw, RebopSimulator, System, Variable, initial, xr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can create a `RebopSimulator` for it:
    """)
    return


@app.cell
def _(Infection, RebopSimulator):
    rebsim = RebopSimulator(Infection)
    result = rebsim.solve(n_points=100, upto_t=10)
    result.to_dataframe().plot()
    return rebsim, result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Unlike the regular `Simulator.solve()` which takes an iterable of times to `save_at`, `RebopSimulator.solve()` takes arguments `upto_t` to control simulation end time and `n_points` which decides how many equally spaced times are sampled. Since it doesn't count 0, `n_points` = N will a `Dataset` with a N + 1 size time coordinate:
    """)
    return


@app.cell
def _(result):
    len(result.coords["time"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To manually set a seed, we must pass it to the `rng` argument:
    """)
    return


@app.cell
def _(rebsim):
    _result = rebsim.solve(n_points=1000, upto_t=10, rng=23534256).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Of course, since the simulation is stochastic the result will be different every time its run. We can run it many times to get a better idea of the system's behaviour:
    """)
    return


@app.cell
def _(rebsim, xr):
    import seaborn.objects as so

    df_100 = (
        xr.concat(
            [rebsim.solve(n_points=0, upto_t=100) for _ in range(10)],
            dim="seed",
            join="outer",
        )
        .to_dataframe()
        .melt(ignore_index=False)
        .reset_index()
    )
    df_100
    so.Plot(df_100, x="time", y="value", color="variable", group="seed").add(
        so.Lines()
    ).show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## RateLaws and arbitrary rates
    Stochastic simulation support a reduced subset of `symbolite` functions and operators for rates both in RateLaws and MassActions.
    """)
    return


@app.cell
def _(RateLaw, RebopSimulator, System, Variable, initial):
    from symbolite import real

    class Model(System):
        A: Variable = initial(default=100)
        B: Variable = initial(default=1)

        r = RateLaw(reactants=[A], products=[B], rate_law=real.sqrt(A) + 1)

    rebsim_2 = RebopSimulator(Model)
    rebsim_2.solve(n_points=50, upto_t=1).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Rates and rate_laws can include:
    - Basic operations: +, -, *, / and \*\*.
    - Functions that can be converted to powers: `real.exp`, `real.sqrt` and `real.hypot`.
    - Functions that can be converted to multiplication: `real.degrees` and `real.radians`
    - Mathematical constants: `real.e`, `real.pi` and `real.tau`.
    """)
    return


if __name__ == "__main__":
    app.run()

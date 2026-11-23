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
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "simbio>=1.2.0", "matplotlib", "seaborn"],
            verbose=False,
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Stochastic simulations in SimBio
    SimBio contains the capability to stochastically simulate systems using the [Gillespie algorithm](http://en.wikipedia.org/wiki/Gillespie_algorithm); this is preferable over ODE based simulations for reactions where small amounts of reactants are present. SimBio uses the [rebop](https://rebop.readthedocs.io/en/latest/) library for this, which must be installed separately:
    ```bash
    pip install rebop
    ```
    To stochastically simulate a `System`, which we can define normally:
    """)
    return


@app.cell
def _():
    from simbio.rebop import RebopSimulator
    from simbio import System, Variable, initial, MassAction, RateLaw

    class Infection(System):
        S: Variable = initial(default=100)
        D: Variable = initial(default=1)
        R: Variable = initial(default=0)

        r_infect = MassAction(reactants=[S, D], products=[2 * D], rate=0.015)
        r_cure = MassAction(reactants=[D], products=[R], rate=0.1)

    return Infection, MassAction, RateLaw, RebopSimulator, System, Variable, initial


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulation times

    We create a `RebopSimulator` for it. Unlike the regular `Simulator`, which saves the solution at the times given by `save_at`, a `RebopSimulator` takes `upto_t`, the simulation end time, and `n_points`, the number of equally spaced intervals at which the state is saved. We will simulate this model many times, so we store both on the simulator with `with_times()`, and every call to `solve()` uses them:
    """)
    return


@app.cell
def _(Infection, RebopSimulator):
    rebsim = RebopSimulator(Infection).with_times(upto_t=50, n_points=100)
    result = rebsim.solve()
    result.to_dataframe().plot()
    return rebsim, result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since time 0 is not counted, `n_points` = N gives a `Dataset` with N + 1 time points:
    """)
    return


@app.cell
def _(result):
    len(result.coords["time"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Times passed to `solve()` take precedence over the stored ones, and each argument can be given on its own. Here we keep the stored `upto_t` and save at more points:
    """)
    return


@app.cell
def _(rebsim):
    rebsim.solve(n_points=1000).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If `n_points` is not given at all, the state is saved after every reaction event. For a model simulated only once, `with_times()` is not needed: `RebopSimulator(Model).solve(upto_t=..., n_points=...)` works as well.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Random seeds

    To manually set a seed, we must pass it to the `rng` argument. The same seed always gives the same result:
    """)
    return


@app.cell
def _(rebsim):
    rebsim.solve(rng=23534256).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Averaging many runs

    Of course, since the simulation is stochastic the result will be different every time it's run. To get the mean behaviour of the system, `n_runs` repeats the simulation that many times with independent random streams and returns the average. It requires `n_points`, so that all runs are saved at the same times:
    """)
    return


@app.cell
def _(rebsim):
    rebsim.solve(n_runs=200, rng=1).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The average hides how different single runs can be. The epidemic peaks at different times in each run and, since it starts from a single infected individual, now and then that individual recovers before infecting anyone and the infection dies out (in about one run out of fifteen with these parameters). The average mixes all these cases into one curve that no single run follows. When such events matter, look at single runs instead of the average, for example by solving with different seeds:
    """)
    return


@app.cell
def _(rebsim):
    import matplotlib.pyplot as plt

    for seed in range(20):
        rebsim.solve(rng=seed)["D"].plot(color="C1", alpha=0.6)
    plt.gca()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Transforming the output

    As with `Simulator`, `with_transform()` selects or computes the quantities to save. Here we save the number of infected individuals and the number of individuals that have ever been infected:
    """)
    return


@app.cell
def _(Infection, rebsim):
    rebsim.with_transform(
        {
            "infected": Infection.D,
            "ever_infected": Infection.D + Infection.R,
        }
    ).solve(n_runs=200, rng=1).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Initial values with units

    Initial values can be given with units. Their magnitudes are read as copy numbers, so they should be numbers of molecules (or individuals), for example in `count` units. Here the decay rate has units of `1/s`, so the simulation times must be given in units of time:
    """)
    return


@app.cell
def _(MassAction, RebopSimulator, System, Variable, initial):
    import pint
    from simbio import Parameter, assign

    u = pint.get_application_registry()

    class Decay(System):
        A: Variable = initial(default=1000 * u.count)
        B: Variable = initial(default=0 * u.count)
        k: Parameter = assign(default=0.5 / u.s)

        decay = MassAction(reactants=[A], products=[B], rate=k)

    decay_sim = RebopSimulator(Decay).with_times(upto_t=10 * u.s, n_points=50)
    decay_sim.solve().pint.dequantify().to_dataframe().plot()
    return Decay, Parameter, assign, decay_sim, u


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Values with units can also be passed with `with_values()`. With only 20 molecules, the fluctuations are clearly visible:
    """)
    return


@app.cell
def _(Decay, decay_sim, u):
    decay_sim.with_values({Decay.A: 20 * u.count}).solve().pint.dequantify().to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Constant species

    As in deterministic simulations (see [Getting started with SimBio](https://dyscolab.github.io/static_notebooks/getting_started_with_simbio.html)), a `Parameter` can be used as a constant species, for example a buffered substrate. It enters the rate of the reaction, but its copy number never changes:
    """)
    return


@app.cell
def _(MassAction, Parameter, RebopSimulator, System, Variable, assign, initial):
    class Buffered(System):
        # S is buffered: its value stays constant
        S: Parameter = assign(default=10)
        P: Variable = initial(default=0)
        k: Parameter = assign(default=0.1)

        conversion = MassAction(reactants=[S], products=[P], rate=k)

    buffered_sim = RebopSimulator(Buffered).with_times(upto_t=10, n_points=100)
    buffered_sim.solve().to_dataframe().plot()
    return Buffered, buffered_sim


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `P` is produced at a constant mean rate $k S$, so the average of many runs follows the straight line $P = k S t$ of the deterministic model. Changing `S` changes that rate:
    """)
    return


@app.cell
def _(Buffered, buffered_sim):
    buffered_sim.with_values({Buffered.S: 20}).solve(n_runs=200, rng=1).to_dataframe().plot()
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
    rebsim_2.solve(upto_t=1, n_points=50).to_dataframe().plot()
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

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
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "jablonski>=0.4.0", "matplotlib", "rebop"],
            verbose=False,
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Stochastic photon counting
    The simulations in [Getting started with Jablonski](https://dyscolab.github.io/static_notebooks/getting_started_with_jablonski.html) solve ODEs: the population of each level is a continuous number, and the emission is the average over a very large number of molecules. When only a few molecules are excited (single molecule spectroscopy, nanoparticles, very weak excitation) each molecule absorbs and emits at random times, and a measurement fluctuates around that average.

    Jablonski can simulate this with the [Gillespie algorithm](http://en.wikipedia.org/wiki/Gillespie_algorithm), using the [rebop](https://rebop.readthedocs.io/en/latest/) library, which must be installed separately:
    ```bash
    pip install rebop
    ```
    This is the same approach as SimBio's [stochastic simulations](https://dyscolab.github.io/static_notebooks/stochastic.html), applied to spectroscopic models. We'll use a dye with 20 molecules, a fluorescent singlet and a phosphorescent triplet:
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pint

    from jablonski import Simulator, SingletState, SpectroscopicSystem, TripletState, initial
    from jablonski.simulation import pulse_excitation, time_resolved
    from jablonski.transitions import (
        Absorption,
        Fluorescence,
        InternalConversion,
        IntersystemCrossing,
        Phosphorescence,
        ReverseIntersystemCrossing,
    )

    u = pint.get_application_registry()

    class Dye(SpectroscopicSystem):
        S0: SingletState = initial(0 * u.eV, "singlet", default=20)
        T1: TripletState = initial(1.8 * u.eV, "triplet", default=0)
        S1: SingletState = initial(2.5 * u.eV, "singlet", default=0)

        abs = Absorption(ground=S0, excited=S1, rate=1e-16 * u.cm**2)
        fl = Fluorescence(excited=S1, ground=S0, rate=0.5 / (10 * u.ns))
        ic = InternalConversion(high=S1, low=S0, rate=0.2 / (10 * u.ns))
        isc = IntersystemCrossing(source=S1, target=T1, rate=0.3 / (10 * u.ns))
        ph = Phosphorescence(excited=T1, ground=S0, rate=0.2 / (100 * u.ns))
        risc = ReverseIntersystemCrossing(source=T1, target=S0, rate=0.8 / (100 * u.ns))

    def plot_emission(ds, variable="emission", **kwargs):
        """Plot a time resolved result in photons / s against time in ns."""
        da = ds[variable].pint.to({"time": u.ns}).pint.to("1/s").pint.dequantify()
        plt.plot(da.time.values, da.values, **kwargs)
        plt.xlabel("time [ns]")
        plt.ylabel("emission [photons / s]")

    return Dye, Simulator, np, plot_emission, plt, pulse_excitation, time_resolved, u


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A single run
    Instead of a `Simulator` we create a `RebopSimulator`, and use the `rebop_*` functions from `jablonski.rebop`. They take the same excitations as their ODE counterparts, but like `RebopSimulator.solve` they use the simulation end time `upto_t` and the number of points `n_points` instead of `save_at`. Since we'll run many simulations over the same times, we set them once on the simulator with `with_times(upto_t=..., n_points=...)`, as with `save_at` for the ODE `Simulator` (see [Simulation](https://dyscolab.github.io/static_notebooks/simulation.html)). They can also be passed to each function, and an argument given there overrides the one on the simulator. The time of the result is in the unit of `upto_t`.

    Here we excite the dye with a 5 ns pulse and follow the total emission for 500 ns with `rebop_time_resolved`:
    """)
    return


@app.cell
def _(Dye, pulse_excitation, u):
    from jablonski.rebop import (
        RebopSimulator,
        rebop_spectral_time_resolved,
        rebop_time_resolved,
    )

    rsim = RebopSimulator(Dye).with_times(upto_t=500 * u.ns, n_points=500)
    pulse = pulse_excitation({Dye.abs: 1e24 / (u.cm**2 * u.s)}, width=5 * u.ns)

    single_run = rebop_time_resolved(rsim, excitation=pulse)
    single_run
    return (
        RebopSimulator,
        pulse,
        rebop_spectral_time_resolved,
        rebop_time_resolved,
        rsim,
        single_run,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The emission at each time is the photon rate given the molecules that are excited in this run: the radiative rate times the number of excited molecules. Every time a molecule is excited or decays it jumps by one molecule's rate, while the ODE result from `time_resolved` changes smoothly:
    """)
    return


@app.cell
def _(Dye, Simulator, np, plot_emission, plt, pulse, single_run, time_resolved, u):
    deterministic = time_resolved(
        Simulator(Dye), excitation=pulse, save_at=np.linspace(0, 500, 501) * u.ns
    )

    plot_emission(single_run, label="stochastic, single run")
    plot_emission(deterministic, label="deterministic")
    plt.legend()
    plt.show()
    return (deterministic,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reproducible results
    Each run gives a different result. To get the same one every time, for example in a report or a test, pass a seed (or a numpy random `Generator`) as `rng`:
    """)
    return


@app.cell
def _(pulse, rebop_time_resolved, rsim):
    run_a = rebop_time_resolved(rsim, excitation=pulse, rng=42)
    run_b = rebop_time_resolved(rsim, excitation=pulse, rng=42)
    run_a.pint.dequantify().identical(run_b.pint.dequantify())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Different seeds give different runs. Plotting a few of them shows how much a measurement on 20 molecules can vary:
    """)
    return


@app.cell
def _(plot_emission, plt, pulse, rebop_time_resolved, rsim):
    for seed in range(5):
        plot_emission(rebop_time_resolved(rsim, excitation=pulse, rng=seed), alpha=0.6)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Averaging runs
    Measurements are often repeated and averaged, for example with time-correlated single photon counting. `n_runs` runs the simulation several times, each with its own random numbers, and returns the average. As the number of runs grows, the average approaches the deterministic result:
    """)
    return


@app.cell
def _(deterministic, plot_emission, plt, pulse, rebop_time_resolved, rsim):
    for n_runs in [10, 200]:
        averaged = rebop_time_resolved(rsim, excitation=pulse, rng=1, n_runs=n_runs)
        plot_emission(averaged, label=f"average of {n_runs} runs", alpha=0.8)
    plot_emission(deterministic, label="deterministic", color="black")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With a seed, the averaged result is also reproducible.

    ## Emission lines
    `rebop_spectral_time_resolved` gives each emission line separately, like `spectral_time_resolved`, with the energy of each line in `attrs`. Here we also pass `n_points=100`, which overrides the 500 points set on the simulator while keeping its `upto_t`. The fast fluorescence (2.5 eV) is only seen during and right after the pulse, while the phosphorescence (1.8 eV) lasts much longer:
    """)
    return


@app.cell
def _(plot_emission, plt, pulse, rebop_spectral_time_resolved, rsim):
    lines = rebop_spectral_time_resolved(
        rsim, excitation=pulse, n_points=100, rng=1, n_runs=50
    )
    for line in lines.data_vars:
        plot_emission(lines, variable=line, label=f"{line} ({lines.attrs[line]})")
    plt.yscale("log")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    As in the ODE functions, `join_by_energy=True` sums the lines with the same energy and `kind` selects emission (default), `"absorption"` or a list of kinds.

    ## When does it matter?
    The fluctuations come from counting molecules: their relative size goes as $1/\sqrt{N}$ for $N$ excited molecules. We can see it by repeating a single run with more molecules, setting the initial population with `with_values`, and dividing by the number of molecules:
    """)
    return


@app.cell
def _(Dye, plot_emission, plt, pulse, rebop_time_resolved, rsim):
    for molecules in [20, 2000]:
        run = rebop_time_resolved(
            rsim.with_values({Dye.S0: molecules}), excitation=pulse, rng=3
        )
        plot_emission(run / molecules, label=f"{molecules} molecules")
    plt.ylabel("emission per molecule [photons / s]")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With thousands of molecules the stochastic result is already close to the ODE one, which is much faster to compute. Stochastic simulation is worth it when few molecules (or few photons) are involved, and to estimate how noisy a measurement on such a system will be. The Gillespie algorithm simulates every single event, so its cost grows with the number of molecules and of excitations.
    """)
    return


if __name__ == "__main__":
    app.run()

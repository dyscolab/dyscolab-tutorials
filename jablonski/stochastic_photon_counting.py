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
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "jablonski>=0.5.0", "matplotlib", "rebop"],
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
    This is the same approach as SimBio's [stochastic simulations](https://dyscolab.github.io/static_notebooks/stochastic.html), applied to spectroscopic models.

    As an example we'll look at a single dye molecule under constant laser illumination, as in a single molecule fluorescence microscope. A typical dye, like a rhodamine, is excited from $S_0$ to $S_1$ and emits fluorescence within a few ns. Once every few thousand excitations it crosses to the triplet $T_1$ instead, where it stays for microseconds. While it is in the triplet the molecule can't absorb or fluoresce, so its fluorescence switches on and off: it *blinks*. An average over many molecules hides this, but a single molecule shows it clearly.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pint

    from jablonski import Simulator, SingletState, SpectroscopicSystem, TripletState, initial
    from jablonski.simulation import step_excitation, time_resolved
    from jablonski.transitions import (
        Absorption,
        Fluorescence,
        InternalConversion,
        IntersystemCrossing,
        Phosphorescence,
        ReverseIntersystemCrossing,
    )
    from jablonski.plots import jablonski_diagram

    u = pint.get_application_registry()

    class Rhodamine(SpectroscopicSystem):
        S0: SingletState = initial(0 * u.eV, "singlet", default=1)
        T1: TripletState = initial(1.8 * u.eV, "triplet", default=0)
        S1: SingletState = initial(2.3 * u.eV, "singlet", default=0)

        abs = Absorption(ground=S0, excited=S1, rate=4e-16 * u.cm**2)
        fl = Fluorescence(excited=S1, ground=S0, rate=2.4e8 / u.s)
        ic = InternalConversion(high=S1, low=S0, rate=1e7 / u.s)
        isc = IntersystemCrossing(source=S1, target=T1, rate=2e5 / u.s)
        ph = Phosphorescence(excited=T1, ground=S0, rate=1e3 / u.s)
        # T1 -> S0 is also a crossing from a triplet to a singlet, here without emission
        triplet_decay = ReverseIntersystemCrossing(source=T1, target=S0, rate=1e5 / u.s)

    jablonski_diagram(Rhodamine)
    return Rhodamine, Simulator, np, plt, step_excitation, time_resolved, u


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A single run
    Instead of a `Simulator` we create a `RebopSimulator`, and use the `rebop_*` functions from `jablonski.rebop`. They take the same excitations as their ODE counterparts, but like `RebopSimulator.solve` they use the simulation end time `upto_t` and the number of points `n_points` instead of `save_at`. Since we'll run many simulations over the same times, we set them once on the simulator with `with_times(upto_t=..., n_points=...)`, as with `save_at` for the ODE `Simulator` (see [Simulation](https://dyscolab.github.io/static_notebooks/simulation.html)). They can also be passed to each function, and an argument given there overrides the one on the simulator. The time of the result is in the unit of `upto_t`.

    Here we switch the laser on at $t = 0$ with `step_excitation`, which keeps it on until the end of the simulation, and follow the total emission for 200 μs with `rebop_time_resolved`:
    """)
    return


@app.cell
def _(Rhodamine, step_excitation, u):
    from jablonski.rebop import (
        RebopSimulator,
        rebop_spectral_time_resolved,
        rebop_time_resolved,
    )

    rsim = RebopSimulator(Rhodamine).with_times(upto_t=200 * u.us, n_points=20000)
    laser = step_excitation({Rhodamine.abs: 1e23 / (u.cm**2 * u.s)})

    single_run = rebop_time_resolved(rsim, excitation=laser)
    single_run
    return (
        laser,
        rebop_spectral_time_resolved,
        rebop_time_resolved,
        rsim,
        single_run,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The emission at each time is the photon rate given the molecules that are excited at that instant: the radiative rate times the number of excited molecules. For a single molecule it jumps between 0 and the fluorescence rate every few ns. A detector instead counts the photons that arrive in a time bin, so to compare with a measurement we average the emission over bins of 100 points (1 μs) with xarray's `coarsen`.

    The binned single molecule trace has bright periods interrupted by dark ones, when the molecule is in the triplet. The ODE result from `time_resolved` is the average over many molecules: it starts at its highest value, when all molecules are in the singlets, and drops within a few μs as a fraction of them gets stuck in the triplet.
    """)
    return


@app.cell
def _(Rhodamine, Simulator, laser, np, plt, single_run, time_resolved, u):
    deterministic = time_resolved(
        Simulator(Rhodamine), excitation=laser, save_at=np.linspace(0, 200, 401) * u.us
    )
    deterministic = deterministic["emission"].pint.to({"time": u.us}).pint.dequantify()

    binned = single_run["emission"].pint.dequantify().coarsen(time=100, boundary="trim").mean()
    plt.plot(binned.time, binned, label="single molecule, 1 μs bins")
    plt.plot(deterministic.time, deterministic, label="deterministic", color="black")
    plt.xlabel("time [μs]"); plt.ylabel("emission [photons / s]"); plt.legend()
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
def _(laser, rebop_time_resolved, rsim):
    run_a = rebop_time_resolved(rsim, excitation=laser, rng=42)
    run_b = rebop_time_resolved(rsim, excitation=laser, rng=42)
    run_a.pint.dequantify().identical(run_b.pint.dequantify())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Different seeds give different runs. Plotting a few of them, each shifted up by $5 \cdot 10^7$ photons / s so that they don't overlap, shows that the dark periods start at random times and have random lengths:
    """)
    return


@app.cell
def _(laser, plt, rebop_time_resolved, rsim):
    for seed in range(3):
        trace = rebop_time_resolved(rsim, excitation=laser, rng=seed)
        trace = trace["emission"].pint.dequantify().coarsen(time=100, boundary="trim").mean()
        plt.plot(trace.time, trace + seed * 5e7)
    plt.xlabel("time [μs]"); plt.ylabel("emission + offset [photons / s]")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Averaging runs
    Measurements are often repeated and averaged, or made on many molecules at once. `n_runs` runs the simulation several times, each with its own random numbers, and returns the average. As the number of runs grows, the average approaches the deterministic result:
    """)
    return


@app.cell
def _(deterministic, laser, plt, rebop_time_resolved, rsim):
    for n_runs in [10, 200]:
        averaged = rebop_time_resolved(rsim, excitation=laser, rng=1, n_runs=n_runs)
        averaged = averaged["emission"].pint.dequantify().coarsen(time=100, boundary="trim").mean()
        plt.plot(averaged.time, averaged, label=f"average of {n_runs} runs", alpha=0.8)
    plt.plot(deterministic.time, deterministic, label="deterministic", color="black")
    plt.xlabel("time [μs]"); plt.ylabel("emission [photons / s]"); plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With a seed, the averaged result is also reproducible.

    ## Emission lines
    `rebop_spectral_time_resolved` gives each emission line separately, like `spectral_time_resolved`, with the energy of each line in `attrs`. Here we also pass `n_points=100`, which overrides the 20000 points set on the simulator while keeping its `upto_t`.
    """)
    return


@app.cell
def _(laser, plt, rebop_spectral_time_resolved, rsim):
    lines = rebop_spectral_time_resolved(
        rsim, excitation=laser, n_points=100, rng=1, n_runs=50
    )
    for line in lines.data_vars:
        emission = lines[line].pint.dequantify()
        plt.plot(emission.time, emission, label=f"{line} ({lines.attrs[line]})")
    plt.yscale("log"); plt.xlabel("time [μs]"); plt.ylabel("emission [photons / s]"); plt.legend()
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
def _(Rhodamine, laser, plt, rebop_time_resolved, rsim):
    for molecules in [1, 1000]:
        run = rebop_time_resolved(
            rsim.with_values({Rhodamine.S0: molecules}), excitation=laser, rng=3
        )
        run = run["emission"].pint.dequantify().coarsen(time=100, boundary="trim").mean()
        plt.plot(run.time, run / molecules, label=f"N = {molecules}")
    plt.xlabel("time [μs]"); plt.ylabel("emission per molecule [photons / s]"); plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With a thousand molecules the blinking of each one is averaged out and the stochastic result is already close to the ODE one, which is much faster to compute. Stochastic simulation is worth it when few molecules (or few photons) are involved, and to estimate how noisy a measurement on such a system will be. The Gillespie algorithm simulates every single event, so its cost grows with the number of molecules and of excitations.
    """)
    return


if __name__ == "__main__":
    app.run()

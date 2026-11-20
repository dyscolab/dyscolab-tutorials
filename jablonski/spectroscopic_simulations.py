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
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "jablonski>=0.4.0", "matplotlib"],
            verbose=False,
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Emission, absorption and quantum yield
    Besides the populations of each level, jablonski can compute the observables from spectroscopy experiments: emitted and absorbed photons over time, steady-state spectra and quantum yields. This notebook goes through these tools. If you are new to jablonski start with [Getting Started with Jablonski](https://dyscolab.github.io/static_notebooks/getting_started_with_jablonski.html).

    As an example system we use a simplified model of ruthenium tris(bipyridine) (https://doi.org/10.1016/j.ccr.2020.213758), with two absorptions: one into the singlet metal-to-ligand charge transfer state (MLCT), and one into a higher ligand-centred state (LC), which quickly relaxes to the MLCT. The singlet MLCT fluoresces weakly and crosses to the triplet MLCT, which phosphoresces.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pint

    from jablonski import Simulator, SingletState, SpectroscopicSystem, TripletState, initial
    from jablonski.plots import notebook_report
    from jablonski.transitions import (
        Absorption,
        Fluorescence,
        InternalConversion,
        IntersystemCrossing,
        Phosphorescence,
        ReverseIntersystemCrossing,
    )

    u = pint.get_application_registry()

    class Ruthenium(SpectroscopicSystem):
        ground: SingletState = initial(0 * u.eV, "singlet", default=10)
        MLCT_3: TripletState = initial(2.12 * u.eV, "triplet", default=0)
        MLCT_1: SingletState = initial(2.74 * u.eV, default=0)
        LC: SingletState = initial(4.3 * u.eV, default=0)

        abs = Absorption(ground=ground, excited=MLCT_1, rate=5.6e-17 * u.cm**2)
        abs_LC = Absorption(ground=ground, excited=LC, rate=3e-16 * u.cm**2)
        ic = InternalConversion(high=LC, low=MLCT_1, rate=1 / (100 * u.fs))
        fl = Fluorescence(excited=MLCT_1, ground=ground, rate=1e8 / u.s)
        isc = IntersystemCrossing(source=MLCT_1, target=MLCT_3, rate=1 / (1 * u.ps))
        ph = Phosphorescence(excited=MLCT_3, ground=ground, rate=1 / (0.6 * u.us) * 0.04)
        risc = ReverseIntersystemCrossing(
            source=MLCT_3, target=ground, rate=1 / (0.6 * u.us) * 0.96
        )

    sim = Simulator(Ruthenium)
    notebook_report(Ruthenium)
    return (
        Absorption,
        Fluorescence,
        IntersystemCrossing,
        Phosphorescence,
        ReverseIntersystemCrossing,
        Ruthenium,
        Simulator,
        SingletState,
        SpectroscopicSystem,
        TripletState,
        initial,
        notebook_report,
        np,
        plt,
        sim,
        u,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Excitation
    Every simulation function takes an excitation: how many photons reach each absorption. The pump of an absorption is a photon flux, in units of $1/(\text{distance}^2 \cdot \text{time})$.

    Steady-state functions take a dictionary mapping each absorption to its pump, for continuous excitation:
    """)
    return


@app.cell
def _(Ruthenium, u):
    cw = {Ruthenium.abs: 1e23 / (u.cm**2 * u.s)}
    return (cw,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Time-resolved functions take events: a dictionary mapping times to the pumps that change at that time. They are built with excitation creators from `jablonski.simulation`:

    - `step_excitation(excitation, start=0 s)` turns the pumps on at `start` and leaves them on.
    - `pulse_excitation(excitation, width, start=0 s)` turns them on at `start` and off after `width`.
    - `delta_excitation(absorption, area, start=0 s)` approximates a $\delta$ pulse with a total photon fluence `area` (photons per area), as a 50 ps pulse.

    Events from different creators can be merged with `|`.
    """)
    return


@app.cell
def _(Ruthenium, u):
    from jablonski.simulation import delta_excitation, pulse_excitation, step_excitation

    step = step_excitation({Ruthenium.abs: 1e23 / (u.cm**2 * u.s)}, start=1 * u.us)
    pulse = pulse_excitation({Ruthenium.abs: 1e23 / (u.cm**2 * u.s)}, width=2 * u.us)
    delta = delta_excitation(Ruthenium.abs, area=1e17 / u.cm**2)
    pulse
    return delta, pulse


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Time-resolved simulations
    Time-resolved functions take the times to save at from the simulator, set with `with_times(save_at=...)` (simulation times are also controlled by `t_span`, for the difference between `save_at` and `t_span`, see [Simulation](https://dyscolab.github.io/static_notebooks/simulation.html)). `piecewise(sim, events)` simulates the populations of each level. It splits the simulation at every event and integrates from the first to the last time of `save_at`.
    """)
    return


@app.cell
def _(np, plt, pulse, sim, u):
    from jablonski.simulation import piecewise

    timed_sim = sim.with_times(save_at=np.linspace(0, 5, 200) * u.us)
    populations = piecewise(timed_sim, events=pulse)
    populations[["MLCT_3"]].pint.dequantify().to_dataframe().plot()
    plt.show()
    return (timed_sim,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Usually we are interested in the light instead. `spectral_time_resolved(sim, excitation)` gives the photon rate of every radiative transition, one variable per line named `emission_<transition>` (here `emission_fl` and `emission_ph`). The energy of each line is stored in the attributes of the result.

    Results are photon rates, in photons per second.
    """)
    return


@app.cell
def _(plt, pulse, timed_sim):
    from jablonski.simulation import spectral_time_resolved

    lines = spectral_time_resolved(timed_sim, pulse)
    lines.pint.dequantify().to_dataframe().plot()
    plt.yscale("log")
    plt.ylim(1, None)
    plt.ylabel("Emission [photons / s]")
    plt.show()
    lines.attrs
    return (spectral_time_resolved,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With `join_by_energy=True` lines with the same energy are summed, and named by their energy. `time_resolved(sim, excitation)` sums all lines into a single `emission` variable, for example the decay after a $\delta$ pulse:
    """)
    return


@app.cell
def _(delta, np, plt, sim, u):
    from jablonski.simulation import time_resolved

    decay_sim = sim.with_times(save_at=np.linspace(0, 3, 200) * u.us)
    decay = time_resolved(decay_sim, delta)
    decay.pint.dequantify().to_dataframe().plot()
    plt.yscale("log")
    plt.ylabel("Emission [photons / s]")
    plt.show()
    return (time_resolved,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Steady-state simulations
    Under continuous excitation the populations reach a steady state. `spectral_steady_state(sim, excitation)` gives the photon rate of each line at that point, with the same names and attributes as `spectral_time_resolved`, and `steady_state(sim, excitation)` gives the total.
    """)
    return


@app.cell
def _(cw, sim):
    from jablonski.simulation import spectral_steady_state, steady_state

    spectral_steady_state(sim, cw)
    return spectral_steady_state, steady_state


@app.cell
def _(cw, sim, steady_state):
    steady_state(sim, cw)["emission"].data
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Choosing what to compute
    All these functions take a `kind` argument which selects the transitions:

    - `"emission"` (default): every radiative transition.
    - `"fluorescence"` or `"phosphorescence"`: only that process.
    - `"absorption"`: the rate of absorbed photons of every absorption, named `absorption_<transition>`.
    - a list of them, such as `["absorption", "fluorescence"]`.

    For example, only the phosphorescence over time:
    """)
    return


@app.cell
def _(pulse, spectral_time_resolved, timed_sim):
    spectral_time_resolved(timed_sim, pulse, kind="phosphorescence")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With `kind="absorption"`, `time_resolved` and `steady_state` return the total absorbed photons as an `absorption` variable:
    """)
    return


@app.cell
def _(plt, pulse, time_resolved, timed_sim):
    absorbed = time_resolved(timed_sim, pulse, kind="absorption")
    absorbed.pint.dequantify().to_dataframe().plot()
    plt.ylabel("Absorption [photons / s]")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To get several kinds side by side use a list with the spectral functions, which keep each line separate:
    """)
    return


@app.cell
def _(cw, sim, spectral_steady_state):
    spectral_steady_state(sim, cw, kind=["absorption", "fluorescence", "phosphorescence"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Spectra
    `spectra(sim, excitation, unit=u.nm)` gives the steady-state lines as a spectrum indexed by wavelength (or any other spectroscopic `unit`, such as `u.eV` or `1 / u.cm`). Lines with the same energy are summed. Values are photon rates, as above.
    """)
    return


@app.cell
def _(cw, sim):
    from jablonski.simulation import spectra

    spectra(sim, cw)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Jablonski lines have no width, so to plot something closer to a measured spectrum `widened_spectra` broadens each line into a gaussian of standard deviation `width` (in `unit`) and evaluates it at `samples`. `kind` works as before; here we show fluorescence and phosphorescence separately.
    """)
    return


@app.cell
def _(cw, np, plt, sim, u):
    from jablonski.simulation import widened_spectra

    samples = np.linspace(400, 700, 1000) * u.nm
    for kind in ["fluorescence", "phosphorescence"]:
        widened = widened_spectra(sim, cw, kind=kind, samples=samples, width=10)
        widened.pint.dequantify().plot(label=kind)
    plt.yscale("log")
    plt.ylim(1e-1, 1e7)
    plt.ylabel("Emission [photons / s]")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `jablonski.plots.graph_spectra` takes the same arguments and plots the widened spectrum as a line coloured by wavelength.

    ### Excitation spectra
    `excitation_emission_matrix(sim, height)` pumps each absorption in turn with the photon flux `height` and gives the emission spectrum for each, one variable per absorption.
    """)
    return


@app.cell
def _(sim, u):
    from jablonski.simulation import excitation_emission_matrix, excitation_spectra

    flux = 1e23 / (u.cm**2 * u.s)
    excitation_emission_matrix(sim, flux)
    return excitation_spectra, flux


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `excitation_spectra(sim, emission, height)` takes one emission wavelength of that matrix (the nearest line to `emission`), giving the emission at that wavelength for each absorption.
    """)
    return


@app.cell
def _(excitation_spectra, flux, sim, u):
    excitation_spectra(sim, emission=585 * u.nm, height=flux)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Quantum yield
    The quantum yield is the number of emitted photons per absorbed photon. `quantum_yield(sim, excitation)` computes it at steady state, and `kind` selects which emissions count (`"emission"`, `"fluorescence"` or `"phosphorescence"`). For this model the yield is set by the phosphorescence, about 4%:
    """)
    return


@app.cell
def _(cw, sim):
    from jablonski.simulation import quantum_yield, spectral_quantum_yield

    {
        kind: quantum_yield(sim, cw, kind=kind).pint.to("percent").data.item()
        for kind in ["emission", "fluorescence", "phosphorescence"]
    }
    return (spectral_quantum_yield,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `spectral_quantum_yield(sim, excitation)` splits it by emission wavelength:
    """)
    return


@app.cell
def _(cw, sim, spectral_quantum_yield):
    spectral_quantum_yield(sim, cw)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In this model every transition is linear in the populations, so the quantum yield doesn't depend on how hard we pump. That is not true when two excited molecules interact, as in triplet-triplet annihilation upconversion. There light is absorbed by a sensitizer, which quickly crosses to its triplet $T_1$. The sensitizer can't emit at higher energy on its own; instead it hands its triplet to a second molecule, the annihilator, and returns to its ground state:

    $$\text{sens.T1} + \text{ann.S0} \rightarrow \text{sens.S0} + \text{ann.T1}.$$

    This triplet energy transfer is modelled with `EnergyTransferUpconversion`, which describes any energy transfer from an excited molecule (`sensitizer_high` → `sensitizer_low`) to another one (`activator_low` → `activator_high`), with a rate proportional to `sens.T1 * ann.S0`.

    The annihilator triplets are long-lived, so they accumulate. When two of them meet they annihilate: one goes back to the ground state and gives its energy to the other, which ends up in the excited singlet $S_1$:

    $$\text{ann.T1} + \text{ann.T1} \rightarrow \text{ann.S1} + \text{ann.S0}.$$

    This step is modelled with `TripletTripletAnnihilation`, here passing the same triplet twice, and its rate is proportional to `ann.T1 * ann.T1`. The annihilator $S_1$ (3.0 eV) lies above the energy of the absorbed photons (2.3 eV), so its fluorescence is upconverted. Since two triplets are needed for each upconverted photon, the upconversion depends nonlinearly on the excitation.
    """)
    return


@app.cell
def _(
    Absorption,
    Fluorescence,
    IntersystemCrossing,
    ReverseIntersystemCrossing,
    SingletState,
    SpectroscopicSystem,
    TripletState,
    initial,
    u,
):
    from jablonski.transitions import (
        EnergyTransferUpconversion,
        TripletTripletAnnihilation,
    )

    class Sensitizer(SpectroscopicSystem):
        S0: SingletState = initial(0 * u.eV, default=1)
        S1: SingletState = initial(2.3 * u.eV, default=0)
        T1: TripletState = initial(1.9 * u.eV, "triplet", default=0)

        abs = Absorption(ground=S0, excited=S1, rate=1e-16 * u.cm**2)
        isc = IntersystemCrossing(source=S1, target=T1, rate=1e10 / u.s)
        decay = ReverseIntersystemCrossing(source=T1, target=S0, rate=1e4 / u.s)

    class Annihilator(SpectroscopicSystem):
        S0: SingletState = initial(0 * u.eV, default=10)
        S1: SingletState = initial(3.0 * u.eV, default=0)
        T1: TripletState = initial(1.8 * u.eV, "triplet", default=0)

        fl = Fluorescence(excited=S1, ground=S0, rate=1e8 / u.s)
        decay = ReverseIntersystemCrossing(source=T1, target=S0, rate=1e3 / u.s)

    class Upconversion(SpectroscopicSystem):
        sens = Sensitizer()
        ann = Annihilator()

        # Triplet energy transfer: sens.T1 + ann.S0 -> sens.S0 + ann.T1
        tet = EnergyTransferUpconversion(
            sensitizer_high=sens.T1,
            sensitizer_low=sens.S0,
            activator_low=ann.S0,
            activator_high=ann.T1,
            rate=1e5 / u.s,
        )
        # Triplet-triplet annihilation: ann.T1 + ann.T1 -> ann.S1 + ann.S0
        tta = TripletTripletAnnihilation(
            triplet_1=ann.T1,
            triplet_2=ann.T1,
            singlet_1=ann.S1,
            singlet_2=ann.S0,
            rate=1e4 / u.s,
        )

    return (Upconversion,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `sweeps.sweep_quantum_yield(sim, excitations)` computes the quantum yield for a list of excitations. If each excitation pumps a single absorption the result is indexed by the pump, otherwise pass `keys`. At low flux most annihilator triplets decay on their own, and the upconverted fluorescence grows with the square of the excitation, so its yield grows with the flux. At high flux annihilation dominates and the yield approaches 0.5, since two absorbed photons are needed for each upconverted one:
    """)
    return


@app.cell
def _(Simulator, Upconversion, np, plt, u):
    from jablonski.sweeps import sweep_quantum_yield

    fluxes = np.logspace(15, 23, 30) / (u.cm**2 * u.s)
    yields = sweep_quantum_yield(
        Simulator(Upconversion),
        excitations=[{Upconversion.sens.abs: flux} for flux in fluxes],
        kind="fluorescence",
    )
    yields.pint.dequantify().plot()
    plt.xscale("log")
    plt.ylabel("Upconversion quantum yield")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `sweeps.sweep_spectra` and `sweeps.sweep_spectral_steady_state` sweep spectra and steady-state lines in the same way (see [Getting Started with Jablonski](https://dyscolab.github.io/static_notebooks/getting_started_with_jablonski.html) for an example).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Constant ground states
    Under weak excitation only a tiny fraction of the molecules is excited at any time, so the ground state population barely changes. In that case it can be modelled with `ConstantGroundState(default, energy=0)`, a level whose population stays constant. It is a `Parameter` with an `energy` (in eV if it is a plain number) and can replace a singlet or triplet state in any transition. It has no equation of its own, so `ground` appears in the parameters of the report below, not in the variables, and its population can be changed with `with_values` like any other parameter.

    Here is the ruthenium model from the beginning with a constant ground state (leaving out the LC absorption, which we didn't pump):
    """)
    return


@app.cell
def _(
    Absorption,
    Fluorescence,
    IntersystemCrossing,
    Phosphorescence,
    ReverseIntersystemCrossing,
    SingletState,
    SpectroscopicSystem,
    TripletState,
    initial,
    notebook_report,
    u,
):
    from jablonski import ConstantGroundState

    class DiluteRuthenium(SpectroscopicSystem):
        ground = ConstantGroundState(default=10)
        MLCT_3: TripletState = initial(2.12 * u.eV, "triplet", default=0)
        MLCT_1: SingletState = initial(2.74 * u.eV, default=0)

        abs = Absorption(ground=ground, excited=MLCT_1, rate=5.6e-17 * u.cm**2)
        fl = Fluorescence(excited=MLCT_1, ground=ground, rate=1e8 / u.s)
        isc = IntersystemCrossing(source=MLCT_1, target=MLCT_3, rate=1 / (1 * u.ps))
        ph = Phosphorescence(excited=MLCT_3, ground=ground, rate=1 / (0.6 * u.us) * 0.04)
        risc = ReverseIntersystemCrossing(
            source=MLCT_3, target=ground, rate=1 / (0.6 * u.us) * 0.96
        )

    notebook_report(DiluteRuthenium)
    return (DiluteRuthenium,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Comparing the steady-state emission of both models shows when the approximation holds. At low flux they agree, while at high flux a significant part of the molecules sits in the long-lived triplet, the real ground state is depleted and the emission saturates. The constant ground state ignores this and keeps growing linearly:
    """)
    return


@app.cell
def _(DiluteRuthenium, Ruthenium, Simulator, np, plt, steady_state, u):
    saturation_fluxes = np.logspace(19, 24, 20) / (u.cm**2 * u.s)
    for _model in [Ruthenium, DiluteRuthenium]:
        _sim = Simulator(_model)
        _emission = [
            steady_state(_sim, {_model.abs: flux})["emission"].pint.magnitude.item()
            for flux in saturation_fluxes
        ]
        plt.plot(saturation_fluxes.magnitude, _emission, label=_model.__name__)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Photon flux [1 / (cm² s)]")
    plt.ylabel("Emission [photons / s]")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next steps
    - [Fitting spectroscopic data](https://dyscolab.github.io/static_notebooks/fitting_spectroscopic_data.html): fit model parameters to measured time-resolved emission and spectra.
    - [Stochastic photon counting](https://dyscolab.github.io/static_notebooks/stochastic_photon_counting.html): simulate individual photon counts instead of average rates.
    """)
    return


if __name__ == "__main__":
    app.run()

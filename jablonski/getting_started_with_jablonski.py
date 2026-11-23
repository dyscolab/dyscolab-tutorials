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
    # Getting Started with Jablonski
    Jablonski is a python library for simulating photochemical systems. It extends [Poincare](https://dyscolab.github.io/poincare/), a package for modelling dynamical systems. To get started let's implement a simplified version of the ruthenium tris(bipyridine) model described in https://doi.org/10.1016/j.ccr.2020.213758.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pint

    from jablonski import (
        Simulator,
        SingletState,
        SpectroscopicSystem,
        TripletState,
        Parameter,
        initial,
        assign,
    )
    from jablonski.simulation import (
        piecewise,
        pulse_excitation,
        spectral_time_resolved,
        time_resolved,
    )
    from jablonski.transitions import (
        Absorption,
        InternalConversion,
        IntersystemCrossing,
        Phosphorescence,
        ReverseIntersystemCrossing,
    )

    u = pint.get_application_registry()

    # Define Ruthenium tris(bipyridine)
    class Ruthenium(SpectroscopicSystem):
        ground: SingletState = initial(
            0 * u.eV, "singlet", default=10
        )  # Define ground state
        MLCT_3: TripletState = initial(
            2.12 * u.eV, "triplet", default=0
        )  # Triplet MLCT level
        MLCT_1: SingletState = initial(2.74, default=0)  # Singlet MLCT level

        abs = Absorption(
            ground=ground, excited=MLCT_1, rate=5.6e-17 * u.cm**2
        )  # Photon absorption, rate is cross-section
        ph = Phosphorescence(
            ground=ground, excited=MLCT_3, rate=1 / (0.6 * u.us) * 0.04
        )  # Radiative decay from  MLCT_3 to ground
        risc = ReverseIntersystemCrossing(
            source=MLCT_3, target=ground, rate=1 / (0.6 * u.us) * 0.96
        )  # Non radiative decay from  MLCT_3 to ground
        isc = IntersystemCrossing(
            source=MLCT_1, target=MLCT_3, rate=1 / (1 * u.ps)
        )  # Non radiative decay from  MLCT_1 to MLCT_3

    return (
        InternalConversion,
        Parameter,
        Phosphorescence,
        ReverseIntersystemCrossing,
        Ruthenium,
        Simulator,
        SingletState,
        SpectroscopicSystem,
        TripletState,
        assign,
        initial,
        np,
        piecewise,
        plt,
        pulse_excitation,
        spectral_time_resolved,
        time_resolved,
        u,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `notebook_report` shows a summary of the model: a Jablonski diagram, followed by its equations, variables and parameters.
    """)
    return


@app.cell
def _(Ruthenium):
    from jablonski.plots import notebook_report

    notebook_report(Ruthenium)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's break this down. First we import all necessary functions from Jablonski as well as [pint](https://pint.readthedocs.io/en/stable/), which will allow us to use units from a `UnitRegistry`.
    Then we create a class which inherits from `SpectroscopicSystem` and we define three levels: `ground`, `MLCT_1`, and `MLCT_3`. Each has an `energy`, which will by default be interpreted as eV, a `multiplicity` which is either "singlet" or "triplet", and a default initial condition representing level population (which could optionally also have units). We then create four transitions: `Absorption` represents photon absorption, which we give a ground state, an excited state and a rate, which represents the photon absorption cross section and must thus be in units of distance squared. There are three emissions: two non-radiative, `ReverseIntersystemCrossing` and `IntersystemCrossing`, and one radiative, `Phosphorescence`. This time the rate is the inverse of $\text{lifetime} \cdot \text{quantum yield}$, so it must be in units of inverse time.

    In the diagram radiative transitions are drawn as straight arrows and non-radiative ones as wiggly arrows. Transitions are translated to ODEs, which you can see in the Equations section of the report. As an example the `Absorption` above translates to:

    $$
    \begin{aligned}
    \frac{d\,\text{ground}}{dt} &= -\text{rate} \cdot \text{pump} \cdot \text{ground} \\
    \frac{d\,\text{MLCT}_1}{dt} &= +\text{rate} \cdot \text{pump} \cdot \text{ground}
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To simulate it we define a pulse with `pulse_excitation`. We give it a dictionary mapping each absorption to the height of the pulse, which represents photon flux so it must be in units of $1/(\text{distance}^2 \cdot \text{time})$, and a duration (`width`) and a `start`, which is 0 by default. Then we pass it to `piecewise` along with a `Simulator` for the system, on which we set the times to save at with `with_times`.

    `piecewise` splits the simulation at each change of the excitation and integrates from the first to the last time in `save_at` (see [the difference between `save_at` and `t_span`](https://dyscolab.github.io/static_notebooks/simulation.html)). Storing the times on the simulator is the standard way when it is used more than once, since every Jablonski simulation function takes them from there. They can also be passed to each function directly, as in `piecewise(sim, events=pulse, save_at=times)`.
    """)
    return


@app.cell
def _(Ruthenium, Simulator, np, piecewise, pulse_excitation, u):
    sim_1 = Simulator(Ruthenium).with_times(save_at=np.linspace(0, 10, 100) * u.us)
    pulse_1 = pulse_excitation(
        {Ruthenium.abs: 1e23 / (u.cm**2 * u.s)}, width=5 * u.us
    )  # Pulse in photon flux
    result_1 = piecewise(sim_1, events=pulse_1)
    result_1
    return (result_1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The result is an [xarray](https://docs.xarray.dev/en/stable/) `DataSet`, which we can plot by converting it to a [pandas](https://pandas.pydata.org/) DataFrame and using the inbuilt method.
    """)
    return


@app.cell
def _(result_1):
    result_1.pint.dequantify().to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Jablonski Systems are composable, so to create the compound with Osmium defined in the paper we can create the Osmium compound separately and then combine them.
    """)
    return


@app.cell
def _(
    Parameter,
    Phosphorescence,
    ReverseIntersystemCrossing,
    SingletState,
    SpectroscopicSystem,
    TripletState,
    assign,
    initial,
    u,
):
    # Define Osmium Compound
    class Osmium(SpectroscopicSystem):
        ground: SingletState = initial(
            0 * u.eV, "singlet", default=0
        )  # Define ground state
        MLCT_3: TripletState = initial(1.7, "triplet", default=0)  # Triplet MLCT level
        ph_rate: Parameter = assign(default=1 / (50 * u.ns) * 0.004)
        ph = Phosphorescence(
            ground=ground, excited=MLCT_3, rate=ph_rate
        )  # Radiative decay from  MLCT_3 to ground
        risc = ReverseIntersystemCrossing(
            source=MLCT_3, target=ground, rate=1 / (50 * u.ns) * 0.996
        )  # Non radiative decay from  MLCT_3 to ground

    return (Osmium,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The definition is similar, but this time we can extract a rate into a separate `Parameter`. We then create instances of both in a separate System.
    """)
    return


@app.cell
def _(InternalConversion, Osmium, Ruthenium, SpectroscopicSystem, u):
    # Combine both
    class Ru_Os(SpectroscopicSystem):
        ru = Ruthenium()
        os = Osmium()

        transfer = InternalConversion(
            high=ru.MLCT_3, low=os.MLCT_3, rate=(2.2e7 + 2e8) / u.s
        )
        back_transfer = InternalConversion(
            high=os.ground, low=ru.ground, rate=(9.1e7 + 1.2e8) / u.s
        )

    return (Ru_Os,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To combine them we create instances of both, which will add both the levels and equations defined above. We can add more equations which represent the exchange of electrons.

    ## Jablonski diagrams
    The diagram of the report can also be drawn on its own with `jablonski_diagram`, which returns a matplotlib figure and axes. Each subsystem is drawn in its own columns (one for singlets and one for triplets), so the levels of the ruthenium and the osmium complexes don't overlap, and transitions between subsystems are drawn across columns.
    """)
    return


@app.cell
def _(Ru_Os, plt):
    from jablonski.plots import jablonski_diagram

    fig, ax = jablonski_diagram(Ru_Os, figsize=(9, 4), fontsize=8)
    plt.show()
    return (jablonski_diagram,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Energy transfer upconversion (ETU) involves two molecules at once: a sensitizer gives its energy away and drops to a lower level, while an activator is promoted to a higher one. Each ETU is drawn as a pair of dashed arrows, one in the sensitizer and one in the activator, and arrows that belong together share a colour. As an example take ytterbium, which absorbs near-infrared light and transfers it to erbium, which can be promoted twice and then emits green light. Erbium ions can also transfer energy between themselves.
    """)
    return


@app.cell
def _(SingletState, SpectroscopicSystem, initial, jablonski_diagram, plt, u):
    from jablonski.transitions import (
        Absorption as _Absorption,
        EnergyTransferUpconversion,
        Fluorescence,
        InternalConversion as _InternalConversion,
    )

    class Ytterbium(SpectroscopicSystem):
        F7_2: SingletState = initial(0 * u.eV, default=1)
        F5_2: SingletState = initial(1.27 * u.eV, default=0)

        abs = _Absorption(ground=F7_2, excited=F5_2, rate=1e-20 * u.cm**2)
        decay = Fluorescence(excited=F5_2, ground=F7_2, rate=1 / (1 * u.ms))

    class Erbium(SpectroscopicSystem):
        I15_2: SingletState = initial(0 * u.eV, default=1)
        I11_2: SingletState = initial(1.27 * u.eV, default=0)
        S3_2: SingletState = initial(2.26 * u.eV, default=0)
        F7_2: SingletState = initial(2.53 * u.eV, default=0)

        relax = _InternalConversion(high=F7_2, low=S3_2, rate=1e6 / u.s)
        green = Fluorescence(excited=S3_2, ground=I15_2, rate=1 / (0.5 * u.ms))
        decay = Fluorescence(excited=I11_2, ground=I15_2, rate=1 / (5 * u.ms))

    class YbEr(SpectroscopicSystem):
        yb = Ytterbium()
        er = Erbium()

        # Ytterbium -> erbium, twice
        etu_1 = EnergyTransferUpconversion(
            sensitizer_high=yb.F5_2,
            sensitizer_low=yb.F7_2,
            activator_low=er.I15_2,
            activator_high=er.I11_2,
            rate=1e4 / u.s,
        )
        etu_2 = EnergyTransferUpconversion(
            sensitizer_high=yb.F5_2,
            sensitizer_low=yb.F7_2,
            activator_low=er.I11_2,
            activator_high=er.F7_2,
            rate=1e4 / u.s,
        )
        # Erbium -> erbium
        etu_er = EnergyTransferUpconversion(
            sensitizer_high=er.I11_2,
            sensitizer_low=er.I15_2,
            activator_low=er.I11_2,
            activator_high=er.F7_2,
            rate=1e3 / u.s,
        )

    fig_etu, ax_etu = jablonski_diagram(
        YbEr, figsize=(9, 4), fontsize=8, etu_colors=["tab:red", "tab:purple"]
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `etu_colors` takes a list of colours (or a matplotlib `cycler`). Colours are reused only when the arrows can't be confused: the two ytterbium-to-erbium transfers share the same arrow in ytterbium, so they share a colour, while the erbium-to-erbium transfer gets its own.

    ## Simulating the compound
    Simulation of the compound System is similar:
    """)
    return


@app.cell
def _(Ru_Os, Simulator, np, piecewise, pulse_excitation, u):
    pulse_2 = pulse_excitation(
        {Ru_Os.ru.abs: 1e25 / (u.cm**2 * u.s)}, width=5 * u.ns
    )

    sim_2 = Simulator(Ru_Os).with_times(save_at=np.linspace(0, 30, 100) * u.ns)
    result_2 = piecewise(sim_2, events=pulse_2)
    result_2.pint.dequantify().to_dataframe().plot()
    return pulse_2, sim_2


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Apart from simulation Jablonski has a number of analysis tools. Instead of the population of each level we can get the spectral time resolved emission with the same pulse using `spectral_time_resolved`. We get one variable per radiative transition, named `emission_<transition>`, and the energy of each line in the attributes of the result. Emissions are photon rates, in photons per second.
    """)
    return


@app.cell
def _(plt, pulse_2, sim_2, spectral_time_resolved):
    spectral = spectral_time_resolved(
        sim_2, excitation=pulse_2
    )  # Time resolved emission
    spectral.pint.dequantify().to_dataframe().plot()
    plt.yscale("log")
    plt.show()
    spectral.attrs  # Print dict with energy associated with each line
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `time_resolved` gives the total emission instead, summed over all lines.
    """)
    return


@app.cell
def _(plt, pulse_2, sim_2, time_resolved):
    total = time_resolved(sim_2, excitation=pulse_2)
    total.pint.dequantify().to_dataframe().plot()
    plt.yscale("log")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To get more realistic excitation we can use Jablonski's helper functions. `pump_from_laser` allows us to calculate the resulting excitation with the molecule in the center of a gaussian beam of a certain width (as standard deviation), power, wavelength and linewidth.
    """)
    return


@app.cell
def _(Ru_Os, u):
    from jablonski.helpers import pump_from_laser, h, c

    wavelength = (h * c) / Ru_Os.ru.abs.energy_difference
    excitation = pump_from_laser(
        Ru_Os, wavelength=wavelength, linewidth=5 * u.nm, power=1 * u.W, width=1 * u.mm
    )
    excitation
    return excitation, pump_from_laser, wavelength


@app.cell
def _(Ru_Os, excitation, u):
    excitation[Ru_Os.ru.abs].to(1 / (u.cm**2 * u.s))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since Jablonski doesn't support line profiles yet the linewidth is taken as a simple tolerance: anything within it is pumped at full power and anything outside is ignored.

    Under continuous excitation `steady_state` gives the total emission once the populations stop changing, and `spectral_steady_state` gives it per line:
    """)
    return


@app.cell
def _(excitation, sim_2):
    from jablonski.simulation import spectral_steady_state, steady_state

    steady_state(sim_2, excitation=excitation)["emission"].data
    return (spectral_steady_state,)


@app.cell
def _(excitation, sim_2, spectral_steady_state):
    spectral_steady_state(sim_2, excitation=excitation)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `spectra` gives the same steady-state lines as a spectrum, indexed by wavelength, and `widened_spectra` broadens each line into a gaussian, which is handy for plotting.
    """)
    return


@app.cell
def _(excitation, np, plt, sim_2, u):
    from jablonski.simulation import spectra, widened_spectra

    lines = spectra(sim_2, excitation=excitation)
    widened = widened_spectra(
        sim_2, excitation=excitation, samples=np.linspace(550, 800, 1000) * u.nm
    )
    widened.pint.dequantify().plot()
    plt.ylabel("Emission [photons / s]")
    plt.show()
    lines
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This can be combined with Jablonski's sweep functions. We can get the emission spectra for different intensities by using `sweeps.sweep_spectra` with pumps for different laser powers.
    """)
    return


@app.cell
def _(Ru_Os, np, plt, pump_from_laser, sim_2, u, wavelength):
    from jablonski.sweeps import sweep_spectra

    powers = np.logspace(-1, 2, 20) * u.W  # Powers to sweep
    excitations = [
        pump_from_laser(
            Ru_Os, wavelength=wavelength, linewidth=5 * u.nm, power=power, width=1 * u.mm
        )
        for power in powers
    ]
    steady = sweep_spectra(
        sim_2,
        excitations=excitations,
        keys=powers,
    )  # Sweep steady state emission spectra
    df = steady.pint.dequantify().to_dataframe().T
    df.index = df.index.map(lambda x: x.magnitude)
    df.plot()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Laser power [W]")
    plt.ylabel("Emission [photons / s]")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `keys` argument gives us the keys for our Dataset. If the excitation affects only one transition it can be omitted and defaults to the pump intensity, but in this case it makes more sense to use the laser power.

    All simulation tools (time-resolved and steady-state simulations, spectra and quantum yields) are explained in [Emission, absorption and quantum yield](https://dyscolab.github.io/static_notebooks/spectroscopic_simulations.html).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checking rates
    Jablonski contains a list of expected ranges for the rates of most transitions. To check them use `transitions.check_all(system: SpectroscopicSystem)`, which will raise a warning for every transition with a rate out of the expected range. Rates given as an expression of other parameters are not checked. For example, in the ruthenium model above:
    """)
    return


@app.cell
def _(Ruthenium):
    import warnings

    from jablonski.transitions import check_all

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_all(Ruthenium)

    mo.md("\n".join(f"- {warning.message}" for warning in caught))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The ranges are typical values, so a warning is a hint to double check the rate and its units, not necessarily an error.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Other utilities
    Jablonski contains a number of other built-in classes to represent common photochemical transitions, found in `jablonski.transitions`, simulation tools, found in `jablonski.simulation` and `jablonski.sweeps` (see [Emission, absorption and quantum yield](https://dyscolab.github.io/static_notebooks/spectroscopic_simulations.html)), and graphical output tools, found in `jablonski.plots`. Transitions and graphical tools are listed below. Since Jablonski is an extension of Poincare most Poincare utilities are also valid with Jablonski and systems can interoperate (i.e. a system can be defined partly by transitions and partly by explicit ODEs). For more information see [Poincare's documentation](https://dyscolab.github.io/poincare/#documentation).

    ### Transitions
    - `Absorption(ground: SingletState, excited: SingletState, rate: Parameter, pump: Parameter)`: A molecule absorbs a photon and is promoted from its ground (or any lower) singlet state to an excited singlet state, $S_0 \rightarrow S_1$. The effective rate is `rate * pump`.

    - `TripletTripletAbsorption(ground: TripletState, excited: TripletState, rate: Parameter, pump: Parameter)`: A molecule in a lower triplet state absorbs a photon and is promoted to a higher triplet state, $T_1 \rightarrow T_n$. The effective rate is `rate * pump`.

    - `VibrationalRelaxation(high: SpinState, low: SpinState, rate: Parameter)`: A non-radiative transition to a lower vibrational level within the same electronic state, $S_n(v') \rightarrow S_n(v)$.

    - `InternalConversion(high: SpinState, low: SpinState, rate: Parameter)`: A non-radiative transition between two electronic states of the same spin multiplicity, $S_2 \rightarrow S_1$.

    - `Fluorescence(excited: SpinState, ground: SpinState, rate: Parameter)`: A radiative transition between two electronic states of the same spin multiplicity, emitting a photon, $S_1 \rightarrow S_0 + h\nu$.

    - `IntersystemCrossing(source: SingletState, target: TripletState, rate: Parameter)`: A non-radiative transition between isoenergetic vibrational levels of electronic states with different spin multiplicity, crossing from singlet to triplet, $S_1 \rightarrow T_1$.

    - `ReverseIntersystemCrossing(source: TripletState, target: SingletState, rate: Parameter)`: A non-radiative transition between isoenergetic vibrational levels of electronic states with different spin multiplicity, crossing from triplet back to singlet, $T_1 \rightarrow S_1$.

    - `Phosphorescence(excited: SpinState, ground: SpinState, rate: Parameter)`: A radiative transition between two electronic states of different spin multiplicity, emitting a photon, $T_1 \rightarrow S_0 + h\nu$.

    - `EnergyTransferUpconversion(sensitizer_high: SpinState, sensitizer_low: SpinState, activator_low: SpinState, activator_high: SpinState, rate: Parameter)`: A sensitizer in an excited state transfers its energy to an activator: the sensitizer relaxes to a lower level while the activator is promoted to a higher one, $\text{sensitizer\_high} + \text{activator\_low} \rightarrow \text{sensitizer\_low} + \text{activator\_high}$. The four levels can belong to different molecules or to the same one, and the rate is `rate * sensitizer_high * activator_low`.

    - `TripletTripletAnnihilation(triplet_1: TripletState, triplet_2: TripletState, singlet_1: SingletState, singlet_2: SingletState, rate: Parameter)`: Two molecules in triplet states annihilate, leaving two singlet states, $T + T \rightarrow S + S$, with rate `rate * triplet_1 * triplet_2`. The states can be different levels, for example $T_1 + T_1 \rightarrow S_1 + S_0$ in triplet-triplet annihilation upconversion, and passing the same state twice gives $2T \rightarrow 2S$. The total energy of the singlets can't be higher than that of the triplets.


    All transitions use `MassAction` kinetics. For example, `Fluorescence(excited=S1, ground=S0, rate=k)` gives:

    $$
    \begin{aligned}
    \frac{dS_1}{dt} &= -kS_1 \\
    \frac{dS_0}{dt} &= +kS_1
    \end{aligned}
    $$

    ### Constant ground states
    A ground state whose population stays constant can be modelled with `ConstantGroundState`, see [Emission, absorption and quantum yield](https://dyscolab.github.io/static_notebooks/spectroscopic_simulations.html#constant-ground-states).

    ### Graphical output

    - `notebook_report(model: type[SpectroscopicSystem], transform: dict | None = None, descriptions: dict | None = None, replace_algebraics: bool = False, show: bool = True)`: Shows a report of `model` in a marimo or Jupyter notebook. Its first section is the Jablonski diagram, followed by the model's equations, variables and parameters. With `show=False` the Markdown is returned instead of displayed.

    - `model_report(model: type[SpectroscopicSystem], path: str | None = None, transform: dict | None = None, descriptions: dict | None = None, standalone: bool = True, replace_algebraics: bool = False)`: Generates a LaTeX report for `model`. Its first section is the Jablonski diagram, rendered as a PGF figure, followed by the model's equations, variables and parameters. If `path` is provided the report is written to disk; otherwise the source is returned as a string. `replace_algebraics` controls whether Parameters with algebraic dependence (i.e. that depend on other parameters or variables) are included as defined or replaced by their dependence.

    - `jablonski_diagram(system: SpectroscopicSystem, figsize: tuple[Number, Number] = (6.4, 4.8), fontsize: Number = 10, show_energy_axis: bool = True, unit: str | Unit = u.eV, etu_colors: Iterable[str] | Cycler = ETU_COLORS)`: Renders a Jablonski diagram for `system`, with energy on the y-axis in the given `unit`. Each subsystem is drawn in its own singlet and triplet columns. Radiative and non-radiative transitions are drawn by a straight and wiggly line respectively, and energy transfer upconversions by dashed lines colour-coded with `etu_colors`. Returns a `(fig, ax)` tuple.

    - `graph_spectra(sim: Simulator, excitation: dict, unit: str | Unit = u.nm, kind: SpectraKind = "emission", samples: Iterable[float] = np.linspace(380, 700, 1000), width: float = 5, figsize: tuple[Number, Number] = (6.4, 4.8))`: Plots a widened emission spectrum as a wavelength-colored line, with wavelength on the x-axis and emission intensity in photons per second on the y-axis. Returns a `(fig, ax)` tuple.
    """)
    return


if __name__ == "__main__":
    app.run()

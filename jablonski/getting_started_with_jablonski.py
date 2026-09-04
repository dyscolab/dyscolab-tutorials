import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

async with app.setup(hide_code=True):
    # Setup cell for marimo notebook, can be ignored
    import marimo as mo
    import sys

    # Import packages if running on marimo playground
    if sys.executable == "/home/pyodide/this.program":
        import micropip

        await micropip.install(
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "jablonski>=0.3.1", "matplotlib"],
            verbose=False,
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Getting Started with Jablonski
    Jablonski is a python library for simulating photochemical systems. It extends [poincare](https://dyscolab.github.io/poincare/), a package for modelling dynamical systems. To get started let's implement a simplified version of the ruthenium tris(bipyridine) model described in https://doi.org/10.1016/j.ccr.2020.213758.
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
        spectral_time_resolved_emission,
    )
    from jablonski.transitions import (
        Absorption,
        InternalConversion,
        IntersystemCrossing,
        Phosphorescence,
        ReverseIntersystemCrossing,
    )

    u = pint.get_application_registry()

    # Defiene Ruthenium tris(bipyridine)
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
        spectral_time_resolved_emission,
        u,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's break this down. First we import all necessary functions from Jablonski as well as [pint](https://pint.readthedocs.io/en/stable/), which will allow us to use units from a `UnitResgitry`.
    Then we create a class which inherits from `SpectroscopicSystem` and we define three levels: `ground`, `MLCT_1`, and `MLCT_3`. Each has an `energy`, which will by default be interpret as eV, a `multiplicity` which is either "singlet" or "triplet", and a default initial condition representing level population (which could optionally also have units). We then create a four transitions: `Absorption` represents photon absorption, which we give a ground state, an excited state and a rate, which represents the photon absorption cross section and must thus be in units of distance squared. There are three emissions: two non-radiative, `ReverseIntersystemCrossing` and `IntersystemCrossing`, and one radiative, `Phosphorescence`. This time the rate is the inverse of $\text{lifetime} \cdot \text{quantum yield}$, so it must be in units of inverse time. Transitions are later translated to ODEs (although stochastic simulation will be supported in the future). As an example the `Absorption` above translates to:

    $$
    \begin{aligned}
    \frac{d\,\text{ground}}{dt} &= -\text{rate} \cdot \text{pump} \cdot \text{ground} \\
    \frac{d\,\text{MLCT}_1}{dt} &= +\text{rate} \cdot \text{pump} \cdot \text{ground}
    \end{aligned}
    $$


    We can create a make a Jablonski diagram to visualize this easier.
    """)
    return


@app.cell
def _(Ruthenium, plt):
    from jablonski.plots import jablonski_diagram

    fig, ax = jablonski_diagram(Ruthenium, figsize=(7, 3), fontsize=8)
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To simulate it we define a pulse, which we  give an absorption which it excites, a height, which represents photon flux so it must be in units of $1/(\text{distance}^2 \cdot \text{time})$, and a duration and a start, which is 0 by default. Then we pass it to `piecewise` along with a `Simulator` for a system, and a series of times to save at, which are interpreted in seconds.
    """)
    return


@app.cell
def _(Ruthenium, Simulator, np, piecewise, pulse_excitation, u):
    sim_1 = Simulator(Ruthenium)
    pulse_1 = pulse_excitation(
        {Ruthenium.abs: 1e23 / (u.cm**2 * u.s)}, width=5 * u.us
    )  # Pulse in photon flux
    result_1 = piecewise(sim_1, events=pulse_1, save_at=np.linspace(0, 10, 100)* u.us)
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
    # Defiene Osmium Compound
    class Osmium(SpectroscopicSystem):
        ground: SingletState = initial(
            0 * u.eV, "singlet", default=0
        )  # Define ground state
        MLCT_3: TripletState = initial(1.7, "triplet", default=0)  # Singlet MLCT level
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
    To combine them we create instances of both, which will add both the levels and equations defined above. We can add more equations which represents the exchange of electrons. Simulation of the compound System is similar:
    """)
    return


@app.cell
def _(Ru_Os, Simulator, np, piecewise, pulse_excitation, u):
    pulse_2 = pulse_excitation(
        {Ru_Os.ru.abs: 1e25 / (u.cm**2 * u.s)}, width=5 * u.ns
    )

    sim_2 = Simulator(Ru_Os)
    result_2 = piecewise(sim_2, events=pulse_2, save_at=np.linspace(0, 30, 100)*u.ns)
    result_2.pint.dequantify().to_dataframe().plot()
    return pulse_2, sim_2


@app.cell
def _(Ru_Os, excitation, u):
    excitation[Ru_Os.ru.abs].to(1/ (u.cm**2 * u.s))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Apart from simulation Jablonski has a number of analysis tools. Instead of the population of each level we can get the spectral time resolved emission with the same pulse. We get the result by each line and  dictionary which gives us each line's energy.
    """)
    return


@app.cell
def _(np, plt, pulse_2, sim_2, spectral_time_resolved_emission, u):
    spectral = spectral_time_resolved_emission(
        sim_2, excitation=pulse_2, save_at=np.linspace(0, 30, 100) * u.ns
    )  # Time resolved emission
    spectral.pint.dequantify().to_dataframe().plot()
    plt.yscale("log")
    plt.show()
    spectral.attrs  # Print dict with energy associated with each line
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To get more realisic excitation we can use jablonki's helper functions. `pump_from_laser` allows us to calculate the resulting excitation whith the molecule in the center of a gaussian beam of a certatin width (as standard deviation), power, wavelength and linewidth.
    """)
    return


@app.cell
def _(Ru_Os, u):
    from jablonski.helpers import pump_from_laser, h, c

    wavelngth = (h*c)/Ru_Os.ru.abs.energy_difference
    excitation = pump_from_laser(Ru_Os, wavelength=wavelngth,linewidth=5*u.nm,  power= 1*u.W, width = 1*u.mm)
    excitation
    return excitation, pump_from_laser, wavelngth


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since jablonski doesn't support line profiles yet the linewidth is taken as a simple tolerance: anything within it is pumped at full power and anything outside is ignored.

    This can be combined with jablonski's sweep functions. We can get the emission spectra for different intensities by using the `sweep_emission_spectra` with pumps for different laser powers.
    """)
    return


@app.cell
def _(Ru_Os, np, plt, pump_from_laser, sim_2, u, wavelngth):
    from jablonski.sweeps import sweep_emission_spectra

    powers = np.logspace(-1,2, 20) * u.W # Powers to sweep 
    excitations = [pump_from_laser(Ru_Os, wavelength=wavelngth,linewidth=5*u.nm,  power=power, width = 1*u.mm) for power in powers]
    steady = sweep_emission_spectra(
        sim_2,
        excitations=excitations,
        keys = powers,
    )  # Sweep steady state emission spectra
    df = steady.pint.dequantify().to_dataframe().T
    df.index = df.index.map(lambda x: x.magnitude)
    df.plot()
    plt.xscale("log");plt.yscale("log"); plt.xlabel("Laser power [W]"); plt.ylabel("Emission [phtons / s]")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `keys` attribute gives us the keys for our Dataset. If the excitation affects only one transition it can be omitted and defaults to the pump intensity, but in this case it makes more sense to use the laser power.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Other utilities
    Jablonsi contains a number of other built-in classes to represent common photochemical transitions, found in `jablonski.transitions`, simulaton tools, found in `jablonski.simulation`, and graphical output tools, found in `jablonski.plots`, all of which are listed below. Since Jablonski is an extension of poincare most poincare utilities are also valid with jablonski and systems can interoperate (i.e. a system can be defined partly by transitions and partly by explicit ODEs). For more information see [poincare's documentation](https://dyscolab.github.io/poincare/#documentation).

    ### Transitions
    - `Absorption(ground: SingletState, excited: SingletState, rate: Parameter, pump: Parameter)`: A molecule absorbs a photon and is promoted from its ground (or any lower) singlet state to an excited singlet state, $S_0 \rightarrow S_1$. The effective rate is `rate * pump`.

    - `TripletTripletAbsorption(ground: TripletState, excited: TripletState, rate: Parameter, pump: Parameter)`: A molecule in a lower triplet state absorbs a photon and is promoted to a higher triplet state, $T_1 \rightarrow T_n$. The effective rate is `rate * pump`.

    - `VibrationalRelaxation(high: SpinState, low: SpinState, rate: Parameter)`: A non-radiative transition to a lower vibrational level within the same electronic state, $S_n(v') \rightarrow S_n(v)$.

    - `InternalConversion(high: SpinState, low: SpinState, rate: Parameter)`: A non-radiative transition between two electronic states of the same spin multiplicity, $S_2 \rightarrow S_1$.

    - `Fluorescence(excited: SpinState, ground: SpinState, rate: Parameter)`: A radiative transition between two electronic states of the same spin multiplicity, emitting a photon, $S_1 \rightarrow S_0 + h\nu$.

    - `IntersystemCrossing(source: SingletState, target: TripletState, rate: Parameter)`: A non-radiative transition between isoenergetic vibrational levels of electronic states with different spin multiplicity, crossing from singlet to triplet, $S_1 \rightarrow T_1$.

    - `ReverseIntersystemCrossing(source: TripletState, target: SingletState, rate: Parameter)`: A non-radiative transition between isoenergetic vibrational levels of electronic states with different spin multiplicity, crossing from triplet back to singlet, $T_1 \rightarrow S_1$.

    - `Phosphorescence(excited: SpinState, ground: SpinState, rate: Parameter)`: A radiative transition between two electronic states of different spin multiplicity, emitting a photon, $T_1 \rightarrow S_0 + h\nu$.

    - `EnergyTransferUpconversion(sensitizer: SingletState, activator: SingletState, relaxator: SingletState, rate: Parameter)`: Two sensitizer molecules in an excited state transfer their combined energy to promote an activator to a higher state while a relaxator relaxes to a lower one, $2 \cdot \text{Sensitizer} \rightarrow \text{Activator} + \text{Relaxator}$.

    - `EnergyTransferUpconversion4(sensitizer: SingletState, activator: SingletState, relaxator: SingletState, rate: Parameter)`: Two molecules in an excited state transfer their combined energy to promote an activator to a higher state while a sensitizer relaxes to a lower one, differen from `EnergyTransferUpconversion` in that both electrons can intially be in different levels, $\text{sensitizer\_high} + \text{activator\_low}  \rightarrow \text{sensitizer\_low} + \text{activator\_high}$.


    All transitions use `MassAction` kinetics. For example, `Fluorescence(excited=S1, ground=S0, rate=k)` gives:

    $$
    \begin{aligned}
    \frac{dS_1}{dt} &= -kS_1 \\
    \frac{dS_0}{dt} &= +kS_1
    \end{aligned}
    $$

    Jablonski contains a list of excpected ranges for all rates. To check them use `transitions.check_all(system: SpectroscopicSystem)`, which will raise warnings if any transitions have rates out of the expected range.

    ### Simulation tools
    **Time-resolved simulation** functions integrate the system dynamics over time:

    - `piecewise(sim: Simulator, events: dict[Time, Mapping], save_at: NDArray)`: Low-level solver that integrates a `Simulator` across a sequence of time segments, applying parameter changes at each event boundary. Returns an xarray `Dataset`.

    - `time_resolved_emission(sim: Simulator, excitation: dict, save_at: NDArray, kind: SpectraKind = "emission")`: Simulates total emission intensity over time for all radiative transitions in `system`, summed across all emission lines. Returns an xarray `Dataset` with a single `emission` variable.

    - `spectral_time_resolved_emission(sim: Simulator, excitation: dict, save_at: NDArray, kind: SpectraKind = "emission", join_by_energy: bool = False)`: Like `time_resolved_emission`, but returns each emission line separately. If `join_by_energy=True`, lines sharing the same energy difference are summed together. Returns an xarray `Dataset`.


    **Excitation creators** build event dictionaries to be passed to simulation functions:

    - `step_excitation(excitation: dict, start: Time = 0 * u.s)`: Creates a step excitation that turns on at `start` and remains on, setting `excitation_transition.pump` to `height`.

    - `pulse_excitation(excitation: dict, width: Time, start: Time = 0 * u.s)`: Creates a finite pulse excitation of duration `width` starting at `start`, setting `excitation_transition.pump` to `height` and then back to zero.

    - `delta_excitation(excitation_transition: Pumper, area: Time, start: Time = 0 * u.s)`: Creates an approximation of a $\delta$-function excitation with a given integrated `area`, implemented as a very short pulse of proportionally large height.

    **Steady-state simulation** functions compute the long-time equilibrium of the system under continuous excitation:

    - `steady_state_emission(sim: Simulator, excitation: dict, kind: SpectraKind = "emission")`: Computes total steady-state emission intensity, summed over all radiative transitions. Returns an xarray `Dataset` with a single `emission` variable.

    - `spectral_steady_state_emission(sim: Simulator, excitation: dict, kind: SpectraKind = "emission", join_by_energy: bool = False)`: Like `steady_state_emission`, but returns each emission line separately, optionally accepting multiple simultaneous excitation sources. If `join_by_energy=True`, lines sharing the same energy difference are summed together. Returns an xarray `Dataset` with a dictionary mapping the line's key to the actual pint quantity representing the enrgy in `Dataset.attrs`.

    **Spectra** functions compute emission and excitation spectra as a function of wavelength:

    - `emission_spectra(sim: Simulator, excitation: dict, unit: str | Unit = u.nm, kind: SpectraKind = "emission")`: Computes a CW emission spectrum, converting energy differences to wavelengths in the given `unit`. Returns an xarray `DataArray` indexed by wavelength.

    - `widened_emission_spectra(sim: Simulator, excitation: dict, unit: str | Unit = u.nm, kind: SpectraKind = "emission", samples: Iterable[float] = np.linspace(380, 700, 1000), width: float = 5)`: Like `emission_spectra`, but broadens each discrete emission line into a Gaussian of standard deviation `width` and evaluates the result over `samples`, mostly useful to generate plots manually (altough `plots.graph_spectra` can be uesd for that). Returns an xarray `DataArray` indexed by wavelength.

    - `excitation_emission_matrix(sim: Simulator, height: float | Iterable[float], unit: str | Unit = u.nm)`: Computes a emission spectra for every `Pumper` in `system`, returning the full excitation-emission matrix as an xarray `Dataset` keyed by pumper excited.

    - `excitation_spectra(sim: Simulator, emission: float | int | Quantity, height: float | Iterable[float], unit: str | Unit = u.nm)`: Extracts a slice of the excitation-emission matrix at a fixed emission wavelength, returning intensity as a function of excitation source. Returns an xarray `DataArray` indexed by pumper.

    ### Graphical output

    - `graph_spectra(sim: Simulator, excitation: dict, unit: str | Unit = u.nm, kind: SpectraKind = "emission", samples: Iterable[float] = np.linspace(380, 700, 1000), width: float = 5)`: Plots a widened emission spectrum as a wavelength-colored line, with wavelength on the x-axis and emission intensity in photons on the y-axis. Returns a `(fig, ax)` tuple.


    - `jablonski_diagram(system: SpectroscopicSystem, figsize: tuple[Number, Number] = (6.4, 4.8), fontsize: Number = 10, show_energy_axis: bool = True, unit: str | Unit = u.eV)`: Renders a Jablonski diagram for `system`, organizing states into singlet and triplet columns with energy on the y-axis in the given `unit`. Radiative and non-radiative transitions are drawn distinctly by a straight and wiggly line respectively. Returns a `(fig, ax)` tuple.

    - `model_report(model: type[SpectroscopicSystem], path: str | None = None, transform: dict | None = None, descriptions: dict | None = None, standalone: bool = True, replace_algebraics: bool = False)`: Generates a Latex report for `model`, including a Jablonski diagram section rendered as a PGF figure alongside the default model sections. If `path` is provided the report is written to disk; otherwise the source is returned as a string. `replace_algebraics` controls wheteher Parameters with algebraic dependence (i.e. that depend on ohter parameters or variables) are included as defined or replpaced by their dependence.
    """)
    return


if __name__ == "__main__":
    app.run()

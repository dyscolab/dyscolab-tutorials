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
            [
                "pint_pandas<=0.7",
                "typing_extensions>=4.15.0",
                "jablonski>=0.2.0",
                "matplotlib",
            ],
            verbose=False,
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Getting Started with Jablonski
    Jablonski is a python library for simulating photochemical systems. It extends [poincare](https://dyscolab.github.io/poincare/), a package for modelling dynamical systems. To get started let's implement the ruthenium tris(bipyridine) model described in https://doi.org/10.1016/j.ccr.2020.213758.
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
            source=MLCT_1, target=MLCT_3, rate=1 / (1 * u.ps) * 0.96
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
    Then we create a class which inherits from `SpectroscopicSystem` and we define three levels: `ground`, `MLCT_1`, and `MLCT_3`. Each has an `energy`, which will by default be interpret as eV, a `multiplicity` which is either "singlet" or "triplet", and a default initial condition. We then create a four transitions: `Absorption` represents photon absorption, which we give a ground state, an excited state and a rate, which represents the photon absorption cross section and must thus be in units of distance squared. There are three emissions: two non-radiative, `ReverseIntersystemCrossing` and `IntersystemCrossing`, and one radiative, `Phosphorescence`. This time the rate is the inverse of $\text{lifetime} \cdot \text{quantum yield}$, so it must be in units of inverse time. We can create a make a Jablonski diagram to visualize this easier.
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
        Ruthenium.abs, height=1e23 / (u.cm**2 * u.s), width=5 * u.us
    )  # Pulse in photon flux
    result_1 = piecewise(sim_1, events=pulse_1, save_at=np.linspace(0, 10e-6, 100))
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
    result_1.to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Jablonski Systems are composable, so to create the Compound with Osmium defined in the paper we can create the Osmium compound separately and then combine them.
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
        Ru_Os.ru.abs, height=1e25 / (u.cm**2 * u.s), width=5 * u.ns
    )
    sim_2 = Simulator(Ru_Os)
    result_2 = piecewise(sim_2, events=pulse_2, save_at=np.linspace(0, 30e-9, 100))
    result_2.to_dataframe().plot()
    return (pulse_2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Apart from simulation Jablonski has a number of analysis tools. Instead of the population of each level we can get the spectral time resolved emission with the same pulse. We get the result by each line and  dictionary which gives us each line's energy.
    """)
    return


@app.cell
def _(Ru_Os, np, plt, pulse_2, spectral_time_resolved_emission):
    spectral = spectral_time_resolved_emission(
        Ru_Os, excitation=pulse_2, save_at=np.linspace(0, 30e-9, 100)
    )  # Time resolved emission
    spectral.pint.dequantify().to_dataframe().plot()
    plt.yscale("log")
    plt.show()
    spectral.attrs  # Print dict with energy associated with each line
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can also get the spectra, which we can sweep for a range of excitation intensities. We give it which excitation transition to pump and an array of intensities to sweep.
    """)
    return


@app.cell
def _(Ru_Os, np, plt, u):
    from jablonski.sweeps import sweep_emission_spectra

    steady = sweep_emission_spectra(
        Ru_Os,
        excitation_transition=Ru_Os.ru.abs,
        heights=np.logspace(21, 24, 20) / (u.cm**2 * u.s),
    )  # Sweep steady state emission spectra
    steady.pint.dequantify().to_dataframe().T.plot()
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    return


if __name__ == "__main__":
    app.run()

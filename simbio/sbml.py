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
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "simbio>=1.2.0", "matplotlib"], verbose = False
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Importing SBML models

    [SBML](https://sbml.org/) (Systems Biology Markup Language) is a standard file format to exchange models of biological systems. Thousands of published models are available in SBML from repositories such as [BioModels](https://www.ebi.ac.uk/biomodels/). SimBio can import an SBML model as a `System`, which is then simulated and analysed like any model written in Python.

    SBML import needs two extra packages, `python-libsbml` and `biomodels`, which are installed with the `io` extra:

    ```sh
    pip install "simbio[io]"
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Loading an SBML file

    As a first example, we load a small model stored next to this notebook in `data/enzyme_pathway.xml`. A substrate `S` is supplied from a buffered `source`, converted into a product `P` by an enzyme with Michaelis-Menten kinetics, and `P` is degraded.

    `simbio.io.sbml.load()` reads an SBML file from a path, and `simbio.io.sbml.loads()` reads it from a string. Here we read the text ourselves, so that the notebook also works when it runs in the browser, where the data folder is a URL instead of a local folder:
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from urllib.request import urlopen

    from simbio.io import sbml

    location = mo.notebook_location() / "data" / "enzyme_pathway.xml"
    if isinstance(location, Path):
        # running locally: location is a file path
        text = location.read_text()
    else:
        # running in the browser: location is a URL
        with urlopen(str(location)) as _response:
            text = _response.read().decode()

    EnzymePathway = sbml.loads(text)
    print(EnzymePathway)
    return EnzymePathway, sbml, text, urlopen


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The result is a `System` class. Its components keep the ids they have in the SBML file, so they are accessed as `EnzymePathway.S`, `EnzymePathway.Vmax`, and so on. Species become variables, and SBML parameters and compartments become parameters.

    We simulate it with a `Simulator`, as any other model. Since we will simulate it more than once, we store the times on the simulator with `with_times()` (see [Simulation](https://dyscolab.github.io/static_notebooks/simulation.html)):
    """)
    return


@app.cell
def _(EnzymePathway):
    import numpy as np
    from simbio import Simulator

    sim = Simulator(EnzymePathway).with_times(save_at=np.linspace(0, 30, 100))
    sim.solve().to_dataframe().plot()
    return Simulator, np, sim


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Parameter values are changed with `with_values()`. With a slower enzyme, the substrate accumulates:
    """)
    return


@app.cell
def _(EnzymePathway, sim):
    sim.with_values({EnzymePathway.Vmax: 0.6}).solve().to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Function definitions

    SBML models can define functions in MathML and use them in rate laws. In this file the conversion rate is written with a function `michaelis_menten(vmax, km, s)`, defined as $\frac{v_{max}\, s}{k_m + s}$. On import, each call is replaced by the body of the function, as the report of the model shows:
    """)
    return


@app.cell
def _(EnzymePathway):
    from simbio import notebook_report

    notebook_report(EnzymePathway, level=3)
    return (notebook_report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The report also shows how the `source` species is handled. It is marked in the SBML file as a boundary condition (its value is set from outside the reactions), so it appears in the rate of the `supply` reaction but is not consumed by it, and it keeps its initial value.

    See [Getting started with SimBio](https://dyscolab.github.io/static_notebooks/getting_started_with_simbio.html) for more on model reports.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Renaming components

    SBML ids are often short or cryptic. The `identity_mapper` argument of `load()` and `loads()` is a function that receives each SBML id and returns the name to use in the `System`. Ids not in our dictionary keep their name:
    """)
    return


@app.cell
def _(sbml, text):
    names = {"S": "substrate", "P": "product", "k_deg": "degradation_rate"}

    RenamedPathway = sbml.loads(
        text, name="RenamedPathway", identity_mapper=lambda sbml_id: names.get(sbml_id, sbml_id)
    )
    print(RenamedPathway)
    return (RenamedPathway,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `name` argument sets the name of the model; by default it is taken from the SBML file. The new names are used everywhere, including the simulation output:
    """)
    return


@app.cell
def _(RenamedPathway, Simulator, np):
    Simulator(RenamedPathway).solve(save_at=np.linspace(0, 30, 100)).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Models from BioModels

    Models in BioModels can be downloaded by URL. Here we download Goldbeter's minimal model of the mitotic oscillator ([BIOMD0000000003](https://www.ebi.ac.uk/biomodels/BIOMD0000000003)), in which cyclin (`C`) activates the cdc2 kinase (`M`), which activates a protease (`X`) that degrades cyclin. We download the SBML text and load it with `loads()`:
    """)
    return


@app.cell
def _(urlopen):
    model_id = "BIOMD0000000003"
    url = f"https://www.ebi.ac.uk/biomodels/model/download/{model_id}?filename={model_id}_url.xml"
    try:
        with urlopen(url) as _response:
            goldbeter_text = _response.read().decode()
    except OSError as error:
        goldbeter_text = None
        print(f"Could not download {model_id}: {error}")
    return (goldbeter_text,)


@app.cell
def _(goldbeter_text, sbml):
    mo.stop(goldbeter_text is None, mo.md("The model could not be downloaded."))

    Goldbeter = sbml.loads(goldbeter_text, name="Goldbeter1991")
    print(Goldbeter)
    return (Goldbeter,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Parameters defined inside a reaction's kinetic law (local parameters in SBML) are added to the model with the reaction id as prefix, such as `reaction1__vi`, so they do not clash between reactions.

    The `biomodels` package gives a shortcut that downloads and loads a model in one step: `simbio.io.biomodels.load("BIOMD0000000003")`.

    Time in this model is in minutes. Over a few hours, the three components oscillate:
    """)
    return


@app.cell
def _(Goldbeter, Simulator, np):
    Simulator(Goldbeter).solve(save_at=np.linspace(0, 200, 1000)).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Imported models do not carry the units declared in the SBML file, so times and values are plain numbers in the model's own units.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What can be imported

    SimBio imports the parts of SBML that describe reaction networks and ODEs:

    - species, parameters and compartments,
    - reactions, including modifiers, local parameters, stoichiometries, and boundary or constant species,
    - function definitions,
    - initial assignments (with constant values),
    - assignment rules and rate rules.

    Loading a model that uses any of the following raises `NotImplementedError`:

    - events, constraints and algebraic rules,
    - fast reactions,
    - compartments with a size other than 1,
    - conversion factors,
    - the SBML packages `comp`, `fbc`, `groups`, `multi` and `qual`. Layout and rendering information is ignored. To load a model while skipping one of these packages, pass it in `ignore_namespaces`, for example `sbml.loads(text, ignore_namespaces=["fbc"])`.
    """)
    return


if __name__ == "__main__":
    app.run()

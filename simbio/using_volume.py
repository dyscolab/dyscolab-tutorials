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
            ["typing_extensions>=4.15.0", "simbio>=1.0.1", "matplotlib"]
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Using volume in simbio
    Simbio has the ability to keep track of a model's volume and whether variables represent concentrations or absolute amounts, varying equations accordingly. To use it we create a `Compartment`, which is similar to a `System` but it reacts to volume, and use `Species` instead of `Variable`.
    """)
    return


@app.cell
def _():
    import numpy as np
    from simbio import Compartment, System, Parameter, Species, RateLaw, MassAction, AbsoluteRateLaw, Simulator, Constant, assign, amount, concentration, volume, Volume


    import pint 
    u = pint.UnitRegistry()

    class Model(Compartment):
        V: Volume = volume(default=2 * u.L) # add volume to the compartment
        A: Species = concentration(default=2 * u.mol/u.L) # create Species A wich represents concentration
        B: Species = amount(default=3 * u.mol) # create Species B wich represents absolute amount
        AB: Species = concentration(default=0* u.mol/u.L) # create Species AB wich represents concentration


        r1 = RateLaw(reactants = [2*A, B], products = [AB], rate_law = 0.1*u.mol/u.L/u.s) # 2*A -> B at constant rate
        r2 = MassAction(reactants = [2*A, B], products = [AB], rate = 0.2*u.mol/u.L/u.s * (u.mol/u.L)**-3) # 2*A -> B at mass action rate

    return (
        AbsoluteRateLaw,
        Compartment,
        Constant,
        MassAction,
        Model,
        Parameter,
        Simulator,
        Species,
        Volume,
        amount,
        assign,
        concentration,
        np,
        u,
        volume,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's unpack all this: we created a `Compartment` called model and assinged to it a `Volume` named `V`. Inside `Model` we have 3 species: `A` and `AB`, which represent concentrations, and `B` which represents an absolute amount; their units are set accordingly. When we create the `RateLaw` simbio will take this into account, so the equations equivalent to r1 would be:
    $$\frac{dA}{dt} = -2 $$
    $$\frac{dB}{dt} =  -2V$$
    $$\frac{dAB}{dt} =  2$$

    `RateLaw` acts on concentrations by default, but `B` represents the absolute amount, so to accomplish a change of 2 mol/L to the conentration we need to change the amount by 2 mol/L $V$. The `MassAction` reaction r2 represents the equations:
    $$\frac{dA}{dt} = -3 A^2 \frac{B}{V} $$
    $$\frac{dB}{dt} = - 3 A^2 \frac{B}{V} V $$
    $$\frac{dAB}{dt} = 3 A^2 \frac{B}{V} $$
    Since `B` is an absolute amount, to get the concentration for the mass action law we must divide by the volume, and in the second equation we must again multiply by $V$ to represent the change in amount. This example uses units via [`pint`](https://github.com/hgrecco/pint) to make the difference between concentrations and amounts clearer, but it's not necessary.

    Simulation for Compartments is identical to systems:
    """)
    return


@app.cell
def _(Model, Simulator, np):
    sim = Simulator(Model)
    sim.solve(save_at=np.linspace(1,5, 50)).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Absolute rate laws
    To create reactions that act directly on concentrations we can create an `AbsoluteRateLaw`:
    """)
    return


@app.cell
def _(
    AbsoluteRateLaw,
    Compartment,
    Species,
    Volume,
    amount,
    concentration,
    u,
    volume,
):
    class Model2(Compartment):
        V: Volume = volume(default=2 * u.L) # add volume to the compartment
        A: Species = concentration(default=2 * u.mol/u.L) # create Species A wich represents concentration
        B: Species = amount(default=3 * u.mol) # create Species B wich represents absolute amount
        AB: Species = concentration(default=0* u.mol/u.L) # create Species AB wich represents concentration


        r1 = AbsoluteRateLaw(reactants = [2*A, B], products = [AB], rate_law = 0.1*u.mol/u.s) # 2*A -> B at constant rate

    return (Model2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In this case r1 represents the equations:
    $$\frac{dA}{dt} = -2/V $$
    $$\frac{dB}{dt} =  -2$$
    $$\frac{dAB}{dt} =  2/V$$
    since absolute rate laws work on absolute amounts, we now have to divide by $V$ in the equations for `A` and `AB` to represent the changes in concentration. Once again simulation works as normal
    """)
    return


@app.cell
def _(Model2, Simulator, np):
    sim_2 = Simulator(Model2)
    sim_2.solve(save_at=np.linspace(1,5, 50)).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dynamic volume
    `Volume` is a variable like any other, so we can give it equations to represent its change. We can represent a `compartment` expanding linearly with time as:
    """)
    return


@app.cell
def _(
    Compartment,
    MassAction,
    Parameter,
    Simulator,
    Species,
    Volume,
    amount,
    assign,
    np,
    u,
    volume,
):
    class Model3(Compartment):
        V: Volume = volume(default=2 * u.L) # add volume to the compartment
        A: Species = amount(default=2 * u.mol) # create Species A wich represents absolute amount
        B: Species = amount(default=3 * u.mol) # create Species B wich represents absolute amount
        AB: Species = amount(default=0* u.mol) # create Species AB wich represents absolute amount
        expansion_rate: Parameter = assign(default = 0.1 * u.L/u.s)

        r2 = MassAction(reactants = [2*A, B], products = [AB], rate = 0.2*u.mol/u.L/u.s * (u.mol/u.L)**-3) # 2*A -> B at mass action rate
        v_eq = V.derive() << expansion_rate # V' = 1 
    sim_3 = Simulator(Model3)
    sim_3.solve(save_at=np.linspace(0,5,50)).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Note that if a `Species` represents a conencentration in a comparment with changing volume it will, ignoring the reactions, stay constant as volume expands, resulting in an effective creation of the substance.  So unless this is the intended behaviour (such as a comparment expanding into a medium with a constant conenctration of a certain substance) it is not recommneded. If it is still desierd to input and output concentrations we can use `Constant` to jointly set inital conditions and transform to divide the output by volume:
    """)
    return


@app.cell
def _(
    Compartment,
    Constant,
    MassAction,
    Parameter,
    Simulator,
    Species,
    Volume,
    amount,
    assign,
    np,
    u,
    volume,
):
    class Model4(Compartment):
        V_0: Constant = assign(default=2 * u.L, constant=True)
        V: Volume = volume(default=2 * u.L) # add volume to the compartment
        A: Species = amount(default=2 * u.mol/u.L * V_0) # create Species A 
        B: Species = amount(default=3 * u.mol) # create Species B 
        AB: Species = amount(default=0* u.mol/u.L * V_0) # create Species AB 
        expansion_rate: Parameter = assign(default = 0.1 * u.L/u.s)

        r2 = MassAction(reactants = [2*A, B], products = [AB], rate = 0.2*u.mol/u.L/u.s * (u.mol/u.L)**-3) # 2*A -> B at mass action rate
        v_eq = V.derive() << expansion_rate # V' = 1
    sim_4 = Simulator(Model4, transform = {"A": Model4.A/Model4.V, "B": Model4.B, "AB": Model4.AB/Model4.V, "V": Model4.V })
    sim_4.solve(save_at=np.linspace(0,5,50)).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compartments and composition

     Poincare and simbio allow for the composition fo smaller models to make more complex ones; to learn more about this see the correspoinding tutorial in [poincare's documentation](https://dyscolab.github.io/poincare/). A `Compartment` can be composed into other models, but it has its quirks. `System` can represent either a logical units, when external variables are passed to it and it is used to assign interactions to them, or a physical units, when it is instantiated without external variables and let to it create its own. Since compartments are meant to represent physical compartments with a volume and a certain amount or concentration of different species it only ever makes sense to instantiate them as physiscal units. To ensure this the linking of internal `Species` or `Volume` to external `Species` or `Volume` is disallowed
    """)
    return


@app.cell
def _(Compartment, Species, Volume, amount, volume):
    class Internal(Compartment):
        V: Volume = volume(default=1)
        A: Species = amount(default=1)
    try:
        class External(Compartment):
            V: Volume = volume(default=1)
            ext_A: Species = amount(default=1)
            int = Internal(A = ext_A)
    except TypeError as err:
        print("TypeError: ", err)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This also prevents ambiguity: it's wouldn't be clear in a reaction wether `A` should use Internal's or External's volume. The passing of regular `Parameters` and `Variables` is allowed:
    """)
    return


@app.cell
def _(
    Compartment,
    Parameter,
    Simulator,
    Species,
    Volume,
    amount,
    assign,
    np,
    volume,
):
    from simbio.reactions import Destruction

    class Internal2(Compartment):
        V: Volume = volume(default=1)
        A: Species = amount(default=20)
        T: Parameter = assign(default=25)

        r = Destruction(A= A, rate = 0.01*T)

    class External2(Compartment):
        V: Volume = volume(default=1)
        ext_T: Parameter = assign(default=25)

        int = Internal2(T = ext_T)

    sim_5 = Simulator(External2)
    sim_5.solve(save_at=np.linspace(0,5,50)).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In this case `A` is destroyed at a rate proportional to temperatre, and to ensure uniform temperature across the entire system we can link the internal temperature `T` to the external temperature `ext_T`.
    """)
    return


if __name__ == "__main__":
    app.run()

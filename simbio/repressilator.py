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
            ["typing_extensions>=4.15.0", "simbio>=1.0.0b1", "matplotlib"]
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Implementing the repressilator in Simbio
    The [repressilator](https://en.wikipedia.org/wiki/Repressilator) is a genetic regulatory network where 3 mRNA species $m_1$, $m_2$, $m_3$ interact with 3 proteins $p_1$, $p_2$, $p_3$. We can model it using the equations:

    \begin{aligned}
    \frac{dm_1}{dt} &=  -m_1 + \frac{\alpha}{1+p_3^n} + \alpha_0 \quad \frac{dp_1}{dt} &= - \beta (p_1-m_1)  \\
    \frac{dm_2}{dt} &=  -m_2 + \frac{\alpha}{1+p_1^n} + \alpha_0 \quad \frac{dp_2}{dt} &= - \beta (p_2-m_2) \\
    \frac{dm_3}{dt} &=  -m_3 + \frac{\alpha}{1+p_2^n} + \alpha_0 \quad \frac{dp_2}{dt} &= - \beta (p_3-m_3)
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Simbio is desinged with composability in mind, so we can break this down into parts and combine them later:
    - Each $m_i$ is destroyed at a rate equal it itself.
    - Each $m_i$ is sinthesysed at a rate $\frac{\alpha}{1+p_{i+1}^n} + \alpha_0$.
    - Each $p_i$ is created at a rate equal $-\beta (p_i-m_i)$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can first implement the equation for each $m_i$. To do so we create a class wich inherits from `System`:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from simbio import (
        System,
        Parameter,
        RateLaw,
        Simulator,
        Variable,
        assign,
        initial,
    )
    from simbio.reactions.single import Destruction


    class mRNA(System):
        # Create the species
        m: Variable = initial(default=1)
        p: Variable = initial(default=1)

        # Create parameters
        alpha: Parameter = assign(default=1)
        alpha_0: Parameter = assign(default=1)
        n: Parameter = assign(default=1)
        creation_rate: Parameter = assign(default=alpha / (1 + p**n) + alpha_0)

        # Declare creation and destruction reactions for m
        destroy = Destruction(A=m, rate=1)

        create = RateLaw(reactants=[p], products=[m, p], rate_law=creation_rate)


    sim = Simulator(mRNA)
    result = sim.solve(save_at=np.linspace(0, 10, 1000))
    result.to_dataframe().plot()
    return (
        Parameter,
        RateLaw,
        Simulator,
        System,
        Variable,
        assign,
        initial,
        mRNA,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since the `Destruction` class creates a `MassAction` the reaction will already be proportional to the concentration of $m_i$, so we can set rate = 1. Simbio requires all Variables which influence a reaction to be involved as reactants, so we include `p` as both a reactant and a product so it doesn't have any net effect.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can also do the same for the proteins:
    """)
    return


@app.cell
def _(Parameter, RateLaw, Simulator, System, Variable, assign, initial, np):
    class Protein(System):
        m: Variable = initial(default=1)  # Create species
        p: Variable = initial(default=0)
        beta: Parameter = assign(default=1)

        create = RateLaw(reactants=[p, m], products=[2 * p, m], rate_law=-beta * (p - m))

    sim_1 = Simulator(Protein)  # Create parameter beta
    result_1 = sim_1.solve(save_at=np.linspace(0, 10, 1000))
    result_1.to_dataframe().plot()  # create_rate: Parameter =  #
    return (Protein,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here we are creating $p_i$ with a rate which can be negative, in which case it will destroy $p_i$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we can combine everything into the single Represillator:
    """)
    return


@app.cell
def _(Parameter, Protein, System, Variable, assign, initial, mRNA):
    class Repressilator(System):
        # Create all the species
        m1: Variable = initial(default=1)
        m2: Variable = initial(default=1)
        m3: Variable = initial(default=1)
        p1: Variable = initial(default=1)
        p2: Variable = initial(default=1)
        p3: Variable = initial(default=1)

        # Create all the paremeters
        alpha: Parameter = assign(default=150)
        alpha_0: Parameter = assign(default=0.5)
        n: Parameter = assign(default=2)
        beta: Parameter = assign(default=8)

        # Apply the creation and destruction laws to each species, in
        react_1 = mRNA(m=m1, p=p3, alpha=alpha, alpha_0=alpha_0, n=n)
        react_2 = mRNA(m=m2, p=p1, alpha=alpha, alpha_0=alpha_0, n=n)
        react_3 = mRNA(m=m3, p=p2, alpha=alpha, alpha_0=alpha_0, n=n)
        react_4 = Protein(p=p1, m=m1, beta=beta)
        react_5 = Protein(p=p2, m=m2, beta=beta)
        react_6 = Protein(p=p3, m=m3, beta=beta)

    return (Repressilator,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here we applied the reaction separately to each species, so for each $m_i$ we created an instance of `mRNA` where `m=mi` and set `p` to the corresponding protein. We must also pass the outer parameters in order to be able to set them all form `Repressilator`. Another more efficient way of doing the same thing would be to create the Proteins without passing external variable, so each one creates a protein and an mRNA. Then we need to apply the mRNA interactions to the variables it created.
    """)
    return


@app.cell
def _(Parameter, Protein, System, assign, mRNA):
    class EfficientRepressilator(System):
    
        # Create all the paremeters
        alpha: Parameter = assign(default=150)
        alpha_0: Parameter = assign(default=0.5)
        n: Parameter = assign(default=2)
        beta: Parameter = assign(default=8)
    
        # Create the protein reactions
        syst1 = Protein(beta=beta)
        syst2 = Protein(beta=beta)
        syst3 = Protein(beta=beta)



        # Apply the creation and destruction laws for the mRNA
        react_1 = mRNA(m=syst1.m, p=syst3.p, alpha=alpha, alpha_0=alpha_0, n=n)
        react_2 = mRNA(m=syst2.m, p=syst1.p, alpha=alpha, alpha_0=alpha_0, n=n)
        react_3 = mRNA(m=syst3.m, p=syst2.p, alpha=alpha, alpha_0=alpha_0, n=n)


    return (EfficientRepressilator,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can simulate them by creating an instance of `Simulator`:
    """)
    return


@app.cell
def _(EfficientRepressilator, Repressilator, Simulator, np, plt):
    sim_2 = Simulator(Repressilator)
    sim_2.solve(save_at=np.linspace(0, 10, 100)).to_dataframe().plot(title = "Normal")
    plt.show()
    sim_3 = Simulator(EfficientRepressilator)
    sim_3.solve(save_at=np.linspace(0, 10, 100)).to_dataframe().plot(title = "Effiecient")
    plt.show()
    return (sim_2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And we get the same output from both. We disappointingly don't get oscillations since all initial conditions are equal. We can change them by passing a `values` dictionary to `solve`; it can also include values for parameters such as $\alpha$.
    """)
    return


@app.cell
def _(Repressilator, np, sim_2):
    sim_2.solve(save_at=np.linspace(0, 100, 1000), values={Repressilator.m1: 1, Repressilator.m2: 2, Repressilator.m3: 3, Repressilator.p1: 4, Repressilator.p2: 5, Repressilator.p3: 6, Repressilator.alpha: 200, Repressilator.beta: 5}).to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analysis
    Poincare includes capabilities for a number of anlysis methods, including the search for stead states, bistability, or limit cycles.
    In this case we can use the `Oscillations` class to do a parameter sweep and see how the period changes.
    """)
    return


@app.cell
def _(Repressilator, Simulator, np):
    from poincare import Oscillations
    sim2 = Simulator(Repressilator(m1=1, m2=2, m3=3, p1=1, p2=2, p3=3, alpha=200))
    # Default inital conditions and values can also be changed at from Simulator creation
    osc = Oscillations()
    result_2 = osc.sweep(sim2, 
                         rel_time=60, # Estimated upper bound on relaxation time
                         T_min=1, # Minimum period expected
                         T_max=20, # Maximum period expected
                         variables=Repressilator.m1, # Variable to look at, can be an iterable with multiple variables
                         parameter=Repressilator.beta, 
                         values=np.linspace(5, 20, 15))

    result_2.sel(quantity = "period").to_dataframe().plot(style='--.')         # Parameter to sweep  # Values taken on by parameter
    return (result_2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We get a warning for a parameter value for which it couldn't be verified. The result also see includes the amplitude of the oscillation and the rms difference between periods to get another check on if the the system is actually oscillating.
    """)
    return


@app.cell
def _(result_2):
    result_2.sel(quantity = ["amplitude"]).to_dataarray().plot(marker = ".")
    result_2.sel(quantity = ["difference_rms"]).to_dataarray().plot(marker = ".")

    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For more information on `Oscillations` and other methods see poincare's notebook on [asymptotic behaviour and parameter sweeps](https://colab.research.google.com/github/dyscolab/poincare/blob/main/docs/Asymptotic_behaviour.ipynb).
    """)
    return


if __name__ == "__main__":
    app.run()

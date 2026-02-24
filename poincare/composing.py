import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model composition in Poincare
    One of Poincare's key features is being able to make larger models by composing smaller ones. Take as an example an oscillator:
    """)
    return


@app.cell
def _():
    from poincare import (
        Derivative,
        Parameter,
        Simulator,
        System,
        Variable,
        assign,
        initial,
    )

    class Oscillator(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)

        spring_constant: Parameter = assign(default=0)

        spring = vx.derive() << -spring_constant * x

    return (
        Derivative,
        Oscillator,
        Parameter,
        Simulator,
        System,
        Variable,
        assign,
        initial,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which represents the equations:
    $$ \frac{d^2x}{dt^2} = -kx $$
    If we wanted to add damping, instead of adding it directly in the model's code we could create a second class for the damping:
    """)
    return


@app.cell
def _(Derivative, Parameter, System, Variable, assign, initial):
    class Damping(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)

        damp_rate: Parameter = assign(default=0)

        dampening = vx.derive() << -damp_rate * vx

    return (Damping,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which represents the equations:
    $$ \frac{d^2x}{dt^2} = -\gamma\frac{dx}{dt} $$
    And add both to a third model:
    """)
    return


@app.cell
def _(
    Damping,
    Derivative,
    Oscillator,
    Parameter,
    System,
    Variable,
    assign,
    initial,
):
    class DampedOscillator(System):
        # Define new external variables for the sysyem
        x_ext: Variable = initial(default=1)
        vx_ext: Derivative = x_ext.derive(initial=0)

        spring_constant: Parameter = assign(default=1)
        damp_rate: Parameter = assign(default=0.1)

        # Apply the models to the external systems variables
        oscillator = Oscillator(x=x_ext, spring_constant=spring_constant)
        dampening = Damping(x=x_ext, damp_rate=damp_rate)

    return (DampedOscillator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here we are applying the systems to the external variable `x_ext`, adding both equations for it's second derivative:
    $$ \frac{d^2x}{dt^2} = - kx - \gamma \frac{dx}{dt}. $$
    We can run it using:
    """)
    return


@app.cell
def _(DampedOscillator, Simulator):
    import matplotlib.pyplot as plt
    import numpy as np

    sim_2 = Simulator(DampedOscillator)
    result_2 = sim_2.solve(save_at=np.linspace(0, 50, 1000))
    result_2.to_dataframe().plot()
    plt.show()
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple variables
    If we wanted to create a pair of coupled oscillators we can create a Coupling
    class to represent the interaction:
    """)
    return


@app.cell
def _(Derivative, Parameter, System, Variable, assign, initial):
    class Coupling(System):
        # Create the variables for both oscillators
        x_1: Variable = initial(default=0)
        v_1: Derivative = x_1.derive(initial=0)
        x_2: Variable = initial(default=0)
        v_2: Derivative = x_2.derive(initial=0)

        spring_constant: Parameter = assign(default=0.1)

        # Apply the force from the interaction to both springs
        force_1 = v_1.derive() << spring_constant * (x_2 - x_1)
        force_2 = v_2.derive() << spring_constant * (x_1 - x_2)

    return (Coupling,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And compose it with the damped oscillator class we defined before:
    """)
    return


@app.cell
def _(
    Coupling,
    DampedOscillator,
    Derivative,
    Parameter,
    System,
    Variable,
    assign,
    initial,
):
    class CoupledOscillators(System):
        # Create the variables for both oscillators
        x_1: Variable = initial(default=1)
        v_1: Derivative = x_1.derive(initial=0)
        x_2: Variable = initial(default=0)
        v_2: Derivative = x_2.derive(initial=0)

        # Define separate constants for the restoring force on each spring and the interaction
        own_constant: Parameter = assign(default=1)
        interaction_constant: Parameter = assign(default=0.05)
        damp_rate: Parameter = assign(default=0.01)

        # Apply Damped Oscillator model to each of the variables
        Damped_1 = DampedOscillator(
            x_ext=x_1, spring_constant=own_constant, damp_rate=damp_rate
        )
        Damped_2 = DampedOscillator(
            x_ext=x_2, spring_constant=own_constant, damp_rate=damp_rate
        )

        # Since the Coupling model includes efffect of the interaction for both we only need to apply it once
        Coupling = Coupling(x_1=x_1, x_2=x_2, spring_constant=interaction_constant)

    return (CoupledOscillators,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we can run it to get the expected beats:
    """)
    return


@app.cell
def _(CoupledOscillators, Simulator, np):
    sim_3 = Simulator(CoupledOscillators)
    result_3 = sim_3.solve(save_at=np.linspace(0, 500, 1000))
    result_3[["x_1", "x_2"]].to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can easily create larger systems this way; for a chain of 4 coupled oscillators:
    """)
    return


@app.cell
def _(
    Coupling,
    DampedOscillator,
    Derivative,
    Parameter,
    System,
    Variable,
    assign,
    initial,
):
    class CoupledOscillatorsFour(System):
        # Create the variables for all oscillators
        x_1: Variable = initial(default=1)
        v_1: Derivative = x_1.derive(initial=0)
        x_2: Variable = initial(default=0)
        v_2: Derivative = x_2.derive(initial=0)
        x_3: Variable = initial(default=0)
        v_3: Derivative = x_3.derive(initial=0)
        x_4: Variable = initial(default=0)
        v_4: Derivative = x_4.derive(initial=0)
        # Define separate constants for the restoring force on each spring and the interaction
        own_constant: Parameter = assign(default=1)
        interaction_constant: Parameter = assign(default=0.05)
        damp_rate: Parameter = assign(default=0.01)

        # Apply Damped Oscillator model to each of the variables
        Damped_1 = DampedOscillator(
            x_ext=x_1, spring_constant=own_constant, damp_rate=damp_rate
        )
        Damped_2 = DampedOscillator(
            x_ext=x_2, spring_constant=own_constant, damp_rate=damp_rate
        )
        Damped_3 = DampedOscillator(
            x_ext=x_3, spring_constant=own_constant, damp_rate=damp_rate
        )
        Damped_4 = DampedOscillator(
            x_ext=x_4, spring_constant=own_constant, damp_rate=damp_rate
        )

        # Since the Coupling model includes efffect of the interaction for all interacting paris 1-2, 2-3, 3-4
        Coupling_12 = Coupling(x_1=x_1, x_2=x_2, spring_constant=interaction_constant)
        Coupling_23 = Coupling(x_1=x_2, x_2=x_3, spring_constant=interaction_constant)
        Coupling_34 = Coupling(x_1=x_3, x_2=x_4, spring_constant=interaction_constant)

    return (CoupledOscillatorsFour,)


@app.cell
def _(CoupledOscillatorsFour, Simulator, np):
    sim_4 = Simulator(CoupledOscillatorsFour)
    result_4 = sim_4.solve(save_at=np.linspace(0, 500, 1000))
    result_4[["x_1", "x_2", "x_3", "x_4"]].to_dataframe().plot()
    return


@app.cell
def _():
    # Setup cell for marimo notebook, can be ignored
    import marimo as mo
    import sys

    # Import packages if running on mairmo playground
    if sys.executable == "/home/pyodide/this.program":
        import micropip

        await micropip.install(
            ["typing_extensions>=4.15.0", "poincare>=1.0.0b2", "matplotlib"]
        )
    return (mo,)


if __name__ == "__main__":
    app.run()

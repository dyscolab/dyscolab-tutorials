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
            ["typing_extensions>=4.15.0", "poincare>=1.0.0b3", "matplotlib"]
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Asymptotic behaviour in Poincare
    Poincare contains a number of features to  characterize the asymptotic behaviour of the system, including the search for steady states, bistability and limit cycles for different parameter values.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Steady States and parameter sweeps
    To find steady states, we must first create an instance of the `SteadyState` class.
    """)
    return


@app.cell
def _():
    import numpy as np

    from poincare import (
        Derivative,
        Parameter,
        Simulator,
        SteadyState,
        System,
        Variable,
        assign,
        initial,
    )

    steady = SteadyState(t_end=1000)
    return (
        Derivative,
        Parameter,
        Simulator,
        System,
        Variable,
        assign,
        initial,
        np,
        steady,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The solver can be changed from the default LSODA by passing a `solver` attribute in a way similar to `Simulator.solve()` (see above). The `SteadyState.solve()` method runs the simulator until it finds a steady states (the derivatives are small enough) or it reaches `t_end`. Since  the default for `t_end` is `np.inf`, not changing it may result in the `Simulator.solve()` running indefinitely.
    """)
    return


@app.cell
def _(Parameter, Simulator, System, Variable, assign, initial, steady):
    # Create a system
    class Pitchfork(System):
        x: Variable = initial(default=1)

        r: Parameter = assign(default=1)

        eq = x.derive() << r * x - x**3

    # Create a Simulator for Pitchfork
    sim = Simulator(Pitchfork)
    # Find steady state
    steady.solve(sim, values={Pitchfork.r: 2})
    return Pitchfork, sim


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It outputs an [xarray](https://docs.xarray.dev/en/stable/) `Dataset`  with the steady state reached for each variable and the time at which it terminated. `0` in the event column indicates it terminated correctly, if it reaches `t_end` without terminating it outputs `NA`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `SteadyState.sweep()` method allows for parameter sweeps, finding the steady state with each parameter:
    """)
    return


@app.cell
def _(Pitchfork, np, sim, steady):
    steady.sweep(sim, variable=Pitchfork.r, values=np.linspace(-10, 10, 100))[
        "x"
    ].to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since `values` only takes the values of the parameter to sweep, changes to initial conditions or values for other parameters in a sweep must be done at simulator creation level:
    """)
    return


@app.cell
def _(Pitchfork, Simulator, np, steady):
    sim_p = Simulator(Pitchfork(x=-1))
    steady.sweep(sim_p, variable=Pitchfork.r, values=np.linspace(-10, 10, 100))[
        "x"
    ].to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can also use `StadyState.sweep_up_and_down()`, which sweeps first up and then down through `values` setting the steady state found in each run as initial conditions for the next.
    """)
    return


@app.cell
def _(
    Derivative,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
    np,
    steady,
):
    import matplotlib.pyplot as plt

    class BiasedDouble(System):
        x: Variable = initial(default=-2)
        v: Derivative = x.derive(initial=0)

        c: Parameter = assign(default=1)

        eq = v.derive() << -v - x**3 + 2 * x + c

    sim_2 = Simulator(BiasedDouble)
    uad = steady.sweep_up_and_down(
        sim_2, variable=BiasedDouble.c, values=np.linspace(-3, 3, 30)
    )

    # Make bistability plot
    uad.sel(direction="up")["x"].plot()
    uad.sel(direction="down")["x"].plot()
    plt.title("")
    return BiasedDouble, plt, sim_2


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Poincare also allows for the search of bistability with different parameters:
    """)
    return


@app.cell
def _(BiasedDouble, np, sim_2, steady):
    steady.bistability(sim_2, variable=BiasedDouble.c, values=np.arange(-3, 3.1, 1))[
        "x"
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `SteadyState.bistability()` works by sweeping up and down and checking if the difference between the up and down value is either greater than the absolute tolerance or greater then the sum of the results times the relative tolerance. These can be changed manually by passing `atol` and `rtol` respectively as parameters:
    """)
    return


@app.cell
def _(BiasedDouble, np, sim_2, steady):
    steady.bistability(
        sim_2,
        variable=BiasedDouble.c,
        values=np.arange(-3, 3.1, 1),
        atol=0.1,
        rtol=0.01,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Limit cycle analysis
    If systems seem to stabilize around a limit cycle instead of reaching a steady state, we can use the `Oscillations` class to find it's period and amplitude. By default it uses the [Autoperiod](https://epubs.siam.org/doi/epdf/10.1137/1.9781611972757.40) method with an implementation based on the one in the [Periodicity Detection](https://pypi.org/project/periodicity-detection/) package.
    """)
    return


@app.cell
def _():
    from poincare import Oscillations

    osc = Oscillations()
    return (osc,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It allows for parameter sweeps using the `Oscillations.sweep()` method, which takes a simulator for the `System` as its first argument:
    """)
    return


@app.cell
def _(
    Derivative,
    Parameter,
    Simulator,
    System,
    Variable,
    assign,
    initial,
    np,
    osc,
):
    from symbolite import real

    from poincare import Independent

    # Create a System wich represents a sinusodally forced harmonic oscillator
    class ForcedDampedOscillator(System):
        t = Independent()
        # Define new external variables for the sysyem
        x: Variable = initial(default=1)
        vx: Derivative = x.derive(initial=0)

        spring_constant: Parameter = assign(default=2)
        damp_rate: Parameter = assign(default=0.3)
        omega: Parameter = assign(default=0.5)
        force: Parameter = assign(default=0.2 * real.sin(omega * t))

        # Apply the models to the external systems variables
        oscillator = vx.derive() << -spring_constant * x + force + 0.2
        dampening = vx.derive() << -damp_rate * vx

    # Create a simulator for the system
    sim_3 = Simulator(ForcedDampedOscillator)

    # Sweep parameters
    result_3 = osc.sweep(
        sim_3,
        T_min=1.5,
        T_max=20,
        rel_time=100,
        parameter=ForcedDampedOscillator.omega,
        values=np.linspace(1, 2, 20),
    )
    result_3
    return ForcedDampedOscillator, result_3, sim_3


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Besides the simulator, `Oscillations.sweep()` takes the following required keyword arguments:
    - `T_min`/`T_max`: minimum of maximum of range in which periods are expected to be found.
    - `rel_time`: upper bound for the system's relaxation time , after which it is expected to be close enough to it's limit cycle.
    - `parameter`: parameter to sweep in.
    - `values`: iterable of values which `parameter` will take.

    To determine `T_min`, `T_max` and `T_r` it is generally a reasonable first estimate to assume monotonicity with respect to the parameter, so we can simulate in it's maximum and minimum value and use the largest (or smallest for `T_min`) result obtained.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The output is a pandas `DataFrame` which contains period, amplitude and mean quadratic difference between consecutive periods indexed by the swept parameter's value. We can plot the results for period and amplitude:
    """)
    return


@app.cell
def _(plt, result_3):
    result_3.sel(quantity="period").to_dataframe().plot(style="--.", title="period")
    plt.legend()
    plt.show()
    result_3.sel(quantity="amplitude").to_dataframe().plot(
        style="--.", title="amplitude"
    )
    plt.legend()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And get an idea of how accurate the estimate is by plotting the mean quadratic difference between periods relative to the amplitude:
    """)
    return


@app.cell
def _(np, plt, result_3):
    relative = result_3.sel(quantity="difference_rms") / result_3.sel(
        quantity="amplitude"
    )
    relative.to_dataframe().plot()
    plt.hlines(
        1,
        np.min(relative.coords["omega"]),
        np.max(relative.coords["omega"]),
        linestyles="--",
        colors="maroon",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If the method fails to find or confirm a period it will raise warnings:
    """)
    return


@app.cell
def _(ForcedDampedOscillator, np, osc, sim_3):
    result_4 = osc.sweep(
        sim_3,
        T_min=2,
        T_max=35,
        rel_time=100,
        parameter=ForcedDampedOscillator.omega,
        values=np.linspace(0.2, 3, 20),
    )
    result_4.head(3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can try changing the minimum amount of periods it simulates after relaxation `T_after_rel` or the minimum amount of timesteps in  each period `timesteps_in_T` to get a better result:
    """)
    return


@app.cell
def _(ForcedDampedOscillator, np, osc, sim_3):
    result_5 = osc.sweep(
        sim_3,
        T_min=2,
        T_max=35,
        rel_time=100,
        parameter=ForcedDampedOscillator.omega,
        values=np.linspace(0.2, 3, 20),
        T_after_rel=15,
        timesteps_in_T=5,
    )
    result_5.head(3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    If the System has more than one variable `Oscillations.sweep`, will by default look at all of them, finding a period and amplitude for each:
    """)
    return


@app.cell
def _(Parameter, Simulator, System, Variable, assign, initial, np, osc):
    class LotkaVolterra(System):
        prey: Variable = initial(default=10)
        predator: Variable = initial(default=1)
        prey_birth_rate: Parameter = assign(default=1)
        prey_death_rate: Parameter = assign(default=1)
        predator_death_rate: Parameter = assign(default=1)
        predator_birth_rate: Parameter = assign(default=1)
        k: Parameter = assign(default=1)
        birth_prey = prey.derive() << prey_birth_rate * prey
        death_prey = prey.derive() << -prey_death_rate * prey * predator
        birth_predator = predator.derive() << predator_birth_rate * prey * predator
        death_predator = predator.derive() << -predator_death_rate * predator

    sim_4 = Simulator(LotkaVolterra)
    result_6 = osc.sweep(
        sim_4,
        T_min=5,
        T_max=20,
        rel_time=100,
        parameter=LotkaVolterra.prey_birth_rate,
        values=np.linspace(1, 3, 20),
        T_after_rel=10,
        timesteps_in_T=10,
    )
    result_6
    return LotkaVolterra, sim_4


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Or we can specifically pass a list `variables` specifying which should be considered:
    """)
    return


@app.cell
def _(LotkaVolterra, np, osc, sim_4):
    result_7 = osc.sweep(
        sim_4,
        T_min=5,
        T_max=20,
        rel_time=100,
        parameter=LotkaVolterra.prey_birth_rate,
        values=np.linspace(1, 3, 20),
        T_after_rel=10,
        timesteps_in_T=10,
        variables=[LotkaVolterra.prey],
    )
    result_7
    return


if __name__ == "__main__":
    app.run()

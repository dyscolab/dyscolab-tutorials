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
                "poincare>=1.1.0",
                "matplotlib",
            ],
            verbose=False,
        )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Excitable Cell-Cell interactions
    [Poincare](https://dyscolab.github.io/poincare/) offers a flexible suite of tools to simulate and analyze dynamical systems. As an example, we can implement the sensitivity analysis on Cell-Cell excitable intercations from [Excitability as a design principle in the immune system](10.1126/sciadv.aeb0921).
    To begin we define the system, including variables, parameters and equations.
    """)
    return


@app.cell
def _():
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from poincare import System, Variable, Parameter, initial, assign, Simulator

    class CellInteraction(System):  # Models are a class thath inherits from System
        X: Variable = initial(
            default=10
        )  # Variable X with default initial condition 10
        Y: Variable = initial(default=0.1)

        a: Parameter = assign(default=1)  # Variable a with default value 1
        b: Parameter = assign(default=1)
        c: Parameter = assign(default=100)
        d: Parameter = assign(default=2)
        f: Parameter = assign(default=1)
        g: Parameter = assign(default=1)

        eq_x = X.derive() << X * (
            a * X * (1 - X / c) - d - b * Y
        )  # Equation dX/dt = X(aX(1-X/c)-d-bY)
        eq_y = Y.derive() << Y * (f * X - g)

    return CellInteraction, Simulator, System, assign, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Which can be simulated with a `Simulator`
    """)
    return


@app.cell
def _(CellInteraction, Simulator, np):
    sim = Simulator(CellInteraction)  # Create Simulator for or
    result = sim.solve(save_at=np.linspace(0, 5, 100))  # Solve ODE
    result.to_dataframe().plot()  # Solve outputs an xarray Datset, convert to Dataframe to plot
    return (sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To get the refractory period we can use poincare's `Sweeper`  tool, which can do parameter sweeps and apply a custom function to analyze each run. We must first define the function
    """)
    return


@app.cell
def _():
    from poincare.analysis.sweeper import Sweeper

    def get_refractory(ds):  # Function to get refractory period
        max_time = ds["Y"].idxmax()
        max_val = ds["Y"].sel(time=max_time).item()

        data_after_max = ds["Y"].sel(time=slice(max_time, None))
        times_below_threshold = data_after_max.time.where(
            data_after_max < 0.01 * max_val, drop=True
        )
        return times_below_threshold.isel(time=0).item()

    return Sweeper, get_refractory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can then create a `Sweeper` for it
    """)
    return


@app.cell
def _(CellInteraction, Sweeper, get_refractory, np, plt, sim):
    sweeper = Sweeper(func=get_refractory)
    refractory_period = sweeper.sweep(
        sim=sim,
        save_at=np.linspace(0, 50, 1000),
        parameter=CellInteraction.g,
        values=np.linspace(0.1, 2, 50),
    )  # Do sweep, we give a save_at for each simulation run and values to sweep the parameter
    refractory_period.to_dataframe().plot(ls="--", marker=".")
    plt.ylabel("Refractory period")
    plt.show()
    return (refractory_period,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And calculate and plot the sensitivity $\frac{d\log(R)}{d\log(g)}$
    """)
    return


@app.cell
def _(np, plt, refractory_period):
    sensitiviy = np.gradient(
        np.log(refractory_period["result"].values),
        np.log(refractory_period["g"].values),
    )
    plt.plot(refractory_period["g"].values, sensitiviy, "--.")
    plt.xlabel("g")
    plt.ylabel("sensitivity")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can run a similar analysis from the pulse amplitude. Since the initial conditions cahnge depending on the parameters to make them comparable we can wrap our System and use `Constant` to ensure they are propperly linked
    """)
    return


@app.cell
def _(CellInteraction, Simulator, System, assign, np):
    from poincare import Constant
    from symbolite import real

    class LinkedCellInteraction(System):
        a: Constant = assign(
            default=1, constant=True
        )  # Constants to represnt initial conditions
        b: Constant = assign(default=1, constant=True)
        c: Constant = assign(default=100, constant=True)
        d: Constant = assign(default=2, constant=True)
        f: Constant = assign(default=1, constant=True)
        g: Constant = assign(default=1, constant=True)

        # X i.c. is set at 110% of threshold
        threshold_110: Constant = assign(
            default=1.1 * c / 2 * (1 - real.sqrt(1 - (4 * d) / (c * a))), constant=True
        )

        # Instancig a models adds all it's variables, equations and paremeters. We can pass values and initial conditions
        int = CellInteraction(a=a, b=b, c=c, d=d, f=f, g=g, X=threshold_110, Y=0.0025)

    lsim = Simulator(LinkedCellInteraction)
    lsim.solve(save_at=np.linspace(0, 10, 100))
    return LinkedCellInteraction, lsim


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then us another `Sweeper` to get the pulse amplitude on each run
    """)
    return


@app.cell
def _(LinkedCellInteraction, Sweeper, lsim, np):
    def get_pulse_amplitude(ds):
        max = ds["int.X"].max().item()
        last = ds["int.X"].isel(time=-1).item()
        verified = np.abs(last) < 0.9 * np.abs(
            max
        )  # Check X has decayed 10% from max to know ew seimlated long enough
        return {"max": max, "verified": verified}  # When we return a dictionary Sweep

    amp_sweeper = Sweeper(func=get_pulse_amplitude)
    pulse_amplitude = amp_sweeper.sweep(
        sim=lsim,
        save_at=np.arange(0, 3, 0.001),
        parameter=(LinkedCellInteraction.a),
        values=np.linspace(0.5, 2.5, 50),
    )
    pulse_amplitude
    return amp_sweeper, pulse_amplitude


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When a function returns a dictionary `Sweeper` will automatically unpcak it, altough this can be disabled by passing `unpack = False` to `sweep`. We can now plot the result
    """)
    return


@app.cell
def _(np, plt, pulse_amplitude):
    pulse_amplitude["max"].to_dataframe().plot(ls="--", marker=".")
    plt.ylabel("Pulse Amplitude")
    print(
        f"All simulations decayed at least 10% from maximum: {np.all(pulse_amplitude['verified']).values}"
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And calculate the sensitivity $\frac{d\log(A)}{d\log(a)}$
    """)
    return


@app.cell
def _(np, plt, pulse_amplitude):
    amp_sensitiviy = np.gradient(
        np.log(pulse_amplitude["max"].values), np.log(pulse_amplitude["a"].values)
    )
    plt.plot(pulse_amplitude["a"].values, amp_sensitiviy, "--.")
    plt.xlabel("a")
    plt.ylabel("sensitivity")
    return (amp_sensitiviy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This can be esaily generalized to the other parameters
    """)
    return


@app.cell
def _(LinkedCellInteraction, amp_sensitiviy, amp_sweeper, lsim, np, plt):
    ranges = {
        LinkedCellInteraction.b: np.linspace(0.1, 5, 50),
        LinkedCellInteraction.c: np.linspace(200, 500, 50),
        LinkedCellInteraction.d: np.linspace(
            0.2, 10, 50
        ),  # d range start changed from 0.01 to 0.1 since pulse wasn't seen
        LinkedCellInteraction.f: np.linspace(0.1, 2.5, 50),
        LinkedCellInteraction.g: np.linspace(0.1, 10, 50),
    }
    results = {LinkedCellInteraction.a: amp_sensitiviy}  # Include previous result
    all_verified = True
    for var, values in ranges.items():
        sweep_result = amp_sweeper.sweep(
            sim=lsim, save_at=np.arange(0, 25, 0.001), parameter=var, values=values
        )
        results[var] = np.gradient(
            np.log(sweep_result["max"].values), np.log(sweep_result[str(var)].values)
        )
        all_verified = all_verified and np.all(sweep_result["verified"]).values

    for var, res in results.items():
        plt.plot(np.linspace(0, 1, 50), res, "--.", label=str(var))
    plt.xlabel("Parameter value (normalized to 0-1)")
    plt.ylabel("Pulse amplitude sensitivity")
    plt.legend()
    print(f"All simulations decayed at least 10% from maximum: {all_verified}")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For more information on poincare and other dyscolab libraries, including [SimBio](https://dyscolab.github.io/simbio/) for chemical reaction neworks and [Jablonski](https://dyscolab.github.io/jablonski/) for photochemical systems, see the full [documentation](https://dyscolab.github.io/poincare/) on [dyscolab's homepage](https://dyscolab.github.io/).
    """)
    return


if __name__ == "__main__":
    app.run()

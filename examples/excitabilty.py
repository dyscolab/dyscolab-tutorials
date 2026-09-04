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
            ["pint_pandas<=0.7", "typing_extensions>=4.15.0", "poincare>=1.2.0", "matplotlib"],
            verbose=False,
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Excitable Cell-Cell interactions
    [Poincare](https://dyscolab.github.io/poincare/) offers a flexible suite of tools to simulate and analyze dynamical systems. As an example, we can implement the sensitivity analysis of cell-cell excitable interactions from _Yael Lebel, Uri Alon, Excitability as a design principle in the immune system.
    Sci. Adv.12, eaeb0921(2026).
    DOI: [10.1126/sciadv.aeb0921](https://www.science.org/doi/10.1126/sciadv.aeb0921)_. The results are analogous to those shown in the paper's Figure 6.

    To begin we define the system, including variables, parameters and equations.
    """)
    return


@app.cell
def _():
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    from poincare import System, Variable, Parameter, initial, assign, Simulator

    class CellInteraction(System):  # Models are a class that inherits from System
        X: Variable = initial(
            default=10
        )  # Variable X with default initial condition 10
        Y: Variable = initial(default=0.1)

        a: Parameter = assign(default=1)  # Parameter a with default value 1
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
def _():
    mo.md(r"""
    Which can be simulated with a `Simulator`.
    """)
    return


@app.cell
def _(CellInteraction, Simulator, np):
    sim = Simulator(CellInteraction)  # Create Simulator for CellInteraction
    result = sim.solve(save_at=np.linspace(0, 5, 100))  # Solve ODE
    result  # Display result (xarray Dataset)
    return result, sim


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `solve()` outputs an [xarray](https://docs.xarray.dev/en/stable/) `Dataset`, which we can plot by converting to a [pandas](https://pandas.pydata.org/) DataFrame and using the inbuilt method.
    """)
    return


@app.cell
def _(result):
    result.to_dataframe().plot()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Poincare's `latex_equations()` automatically generates Latex code for the model's equations so we can check they are defined correctly.
    """)
    return


@app.cell
def _(CellInteraction):
    from poincare.printing.latex import latex_equations

    equations = latex_equations(CellInteraction)
    mo.md(equations)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To get the refractory period we can use poincare's `Sweeper`  tool, which can do parameter sweeps and apply a custom function to analyze each run. We must first define the function.
    """)
    return


@app.cell
def _():
    from poincare.analysis.sweeper import Sweeper

    def get_refractory(simulation_result):
        """Calculates refractory period from simulation result by looking for
        the first time with Y concetration less than 1% of maximum"""
        max_time = simulation_result["Y"].idxmax()
        max_val = simulation_result["Y"].sel(time=max_time).item()

        data_after_max = simulation_result["Y"].sel(time=slice(max_time, None))
        times_below_threshold = data_after_max.time.where(
            data_after_max < 0.01 * max_val, drop=True
        )
        return times_below_threshold.isel(time=0).item()

    return Sweeper, get_refractory


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can then create a `Sweeper` for it.
    """)
    return


@app.cell
def _(CellInteraction, Sweeper, get_refractory, np, plt, sim):
    sweeper = Sweeper(func=get_refractory)
    refractory_period = sweeper.sweep(
        sim=sim,
        save_at=np.linspace(0, 50, 1000),
        parameter=CellInteraction.g,
        values=np.linspace(
            0.1, 2, 50
        ),  # Parameter sweep range from the paper's supplementary materials
    )  # Run the sweep, we pass a save_at for each simulation run and values to sweep the parameter
    refractory_period.to_dataframe().plot(ls="--", marker=".")
    plt.ylabel("Refractory period")
    plt.show()
    return (refractory_period,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And calculate and plot the sensitivity $\frac{d\log(R)}{d\log(g)}$.
    """)
    return


@app.cell
def _(np, plt, refractory_period):
    sensitivity = np.gradient(
        np.log(refractory_period["result"].values),
        np.log(refractory_period["g"].values),
    )
    plt.plot(refractory_period["g"].values, sensitivity, "--.")
    plt.xlabel("g")
    plt.ylabel("Sensitivity")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Although the numerical derivative shows some instability the sensitivity oscillates around -1, the value given in Figure 6.E. of the paper.

    We can run a similar analysis from the pulse amplitude. Since the initial conditions change depending on the parameters to make them comparable we can wrap our System and use `Constant` to ensure they are properly linked.
    """)
    return


@app.cell
def _(CellInteraction, Simulator, System, assign, np):
    from poincare import Constant
    from symbolite import real

    class LinkedCellInteraction(System):
        a: Constant = assign(
            default=1, constant=True
        )  # Constants to represent initial conditions
        b: Constant = assign(default=1, constant=True)
        c: Constant = assign(default=100, constant=True)
        d: Constant = assign(default=2, constant=True)
        f: Constant = assign(default=1, constant=True)
        g: Constant = assign(default=1, constant=True)

        # X i.c. is set at 110% of threshold
        threshold_110: Constant = assign(
            default=1.1 * c / 2 * (1 - real.sqrt(1 - (4 * d) / (c * a))), constant=True
        )

        # Instantiating a models adds all its variables, equations and parameters. We can pass values and initial conditions
        int = CellInteraction(a=a, b=b, c=c, d=d, f=f, g=g, X=threshold_110, Y=0.0025)

    lsim = Simulator(LinkedCellInteraction)
    lsim.solve(save_at=np.linspace(0, 10, 100))
    return LinkedCellInteraction, lsim


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Then use another `Sweeper` to get the pulse amplitude on each run.
    """)
    return


@app.cell
def _(LinkedCellInteraction, Sweeper, lsim, np):
    def get_pulse_amplitude(simulation_result):
        """Calculates pulse amplitude from simulation result by looking for maximum X concentration
        and checking it has decayed at least 10% from peak by the time the simulation ends"""
        max = simulation_result["int.X"].max().item()
        last = simulation_result["int.X"].isel(time=-1).item()

        # Check X has decayed 10% from max to know we simulated long enough
        verified = np.abs(last) < 0.9 * np.abs(max)
        return {"max": max, "verified": verified}

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
def _():
    mo.md(r"""
    When a function returns a dictionary `Sweeper` will automatically unpack it, although this can be disabled by passing `unpack = False` to `sweep`. We can now plot the result.
    """)
    return


@app.cell
def _(np, plt, pulse_amplitude):
    pulse_amplitude["max"].to_dataframe().plot(ls="--", marker=".")
    plt.ylabel("Pulse amplitude")
    print(
        f"All simulations decayed at least 10% from maximum: {np.all(pulse_amplitude['verified']).values}"
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And calculate the sensitivity $\frac{d\log(A)}{d\log(a)}$.
    """)
    return


@app.cell
def _(np, plt, pulse_amplitude):
    amp_sensitivity = np.gradient(
        np.log(pulse_amplitude["max"].values), np.log(pulse_amplitude["a"].values)
    )
    plt.plot(pulse_amplitude["a"].values, amp_sensitivity, "--.")
    plt.xlabel("a")
    plt.ylabel("Sensitivity")
    return (amp_sensitivity,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This can be easily generalized to the other parameters.
    """)
    return


@app.cell
def _(LinkedCellInteraction, amp_sensitivity, amp_sweeper, lsim, np, plt):
    ranges = {
        LinkedCellInteraction.b: np.linspace(0.1, 5, 50),
        LinkedCellInteraction.c: np.linspace(200, 500, 50),
        LinkedCellInteraction.d: np.linspace(
            0.2, 10, 50
        ),  # d range start changed from 0.01 to 0.1 since pulse wasn't seen
        LinkedCellInteraction.f: np.linspace(0.1, 2.5, 50),
        LinkedCellInteraction.g: np.linspace(0.1, 10, 50),
    }
    results = {LinkedCellInteraction.a: amp_sensitivity}  # Include previous result
    all_verified = True
    for var, values in ranges.items():
        sweep_result = amp_sweeper.sweep(
            sim=lsim, save_at=np.arange(0, 25, 0.001), parameter=var, values=values
        )
        results[var] = np.gradient(
            np.log(sweep_result["max"].values), np.log(sweep_result[str(var)].values)
        )
        all_verified = all_verified and np.all(sweep_result["verified"].values)

    for var, res in results.items():
        plt.plot(np.linspace(0, 1, 50), res, "--.", label=str(var))
    plt.xlabel("Parameter value (normalized to 0-1)")
    plt.ylabel("Pulse amplitude sensitivity")
    plt.legend()
    print(f"All simulations decayed at least 10% from maximum: {all_verified}")
    plt.show()
    return ranges, results


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can now get the sensitivity around the parameter values given in Figure 6.E. from the full result to make a similar bar chart.
    """)
    return


@app.cell
def _(LinkedCellInteraction, np, plt, pulse_amplitude, ranges, results):
    full_ranges = {
        LinkedCellInteraction.a: pulse_amplitude["a"].values
    } | ranges  # Parameter a wasn't included in ranges
    central_parameters = {
        LinkedCellInteraction.a: 1,
        LinkedCellInteraction.b: 1,
        LinkedCellInteraction.c: 100,
        LinkedCellInteraction.d: 2,
        LinkedCellInteraction.f: 1,
        LinkedCellInteraction.g: 1,
    }
    central_values = np.array(
        [
            results[param][np.argmin(np.abs(full_ranges[param] - val))]
            for param, val in central_parameters.items()
        ]
    )  # Get sensitvity at parameter value closest to central value
    plt.bar(range(len(central_values)), central_values)
    plt.xticks(
        range(len(central_values)), [str(param) for param in central_parameters.keys()]
    )
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For more information on poincare and other dyscolab libraries, including [SimBio](https://dyscolab.github.io/simbio/) for chemical reaction networks and [Jablonski](https://dyscolab.github.io/jablonski/) for photochemical systems, see the full [documentation](https://dyscolab.github.io/poincare/) on [dyscolab's homepage](https://dyscolab.github.io/).
    """)
    return


if __name__ == "__main__":
    app.run()

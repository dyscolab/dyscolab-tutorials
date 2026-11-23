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
    # Fitting spectroscopic data
    Once we have a model of a photochemical system we usually want to compare it with measurements and find the rates that best describe them. Jablonski's `fits` module fits the parameters of a `SpectroscopicSystem` to two kinds of measurements:

    - **Time resolved emission** (or absorption) after an excitation, with `fit_spectral_time_resolved`.
    - **Continuous wave (CW) spectra** under constant excitation, with `fit_spectra`.

    Both are built on Poincare's `Fitter`, which is explained in detail in the [fitting notebook](https://dyscolab.github.io/static_notebooks/fitting.html). It contains a more extensible fits API which can fit arbitrary simulations or combinations of multiple simulations. If you are new to Jablonski, start with [Getting started with Jablonski](https://dyscolab.github.io/static_notebooks/getting_started_with_jablonski.html).

    As an example we'll use 4CzIPN, a TADF (thermally activated delayed fluorescence) emitter used in OLEDs. The model has three levels and six transitions: absorption, fluorescence and internal conversion from $S_1$, intersystem crossing (ISC) $S_1 \rightarrow T_1$, RISC $T_1 \rightarrow S_1$, and a slow non-radiative decay of $T_1$ to the ground state (phosphorescence is negligible at room temperature). Uoyama et al. ([Nature 492, 234 (2012)](https://doi.org/10.1038/nature11687)) measured a photoluminescence quantum yield of 94% and a delayed lifetime of about 5 μs in toluene. The rates below are representative values that reproduce this behaviour.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pint
    from poincare import solvers

    from jablonski import Simulator, SingletState, SpectroscopicSystem, TripletState, initial
    from jablonski.simulation import pulse_excitation, spectra, spectral_time_resolved
    from jablonski.transitions import (
        Absorption,
        Fluorescence,
        InternalConversion,
        IntersystemCrossing,
        ReverseIntersystemCrossing,
    )

    u = pint.get_application_registry()

    class CzIPN(SpectroscopicSystem):
        S0: SingletState = initial(0 * u.eV, "singlet", default=10)
        S1: SingletState = initial(2.45 * u.eV, "singlet", default=0)
        T1: TripletState = initial(2.45 * u.eV, "triplet", default=0)

        abs = Absorption(ground=S0, excited=S1, rate=4e-17 * u.cm**2)
        fl = Fluorescence(excited=S1, ground=S0, rate=2e7 / u.s)
        ic = InternalConversion(high=S1, low=S0, rate=1e6 / u.s)
        isc = IntersystemCrossing(source=S1, target=T1, rate=3.5e7 / u.s)
        risc = ReverseIntersystemCrossing(source=T1, target=S1, rate=5e5 / u.s)
        # T1 -> S0 is also a crossing from a triplet to a singlet, here without emission
        triplet_decay = ReverseIntersystemCrossing(source=T1, target=S0, rate=1e4 / u.s)

    return (
        CzIPN,
        Simulator,
        np,
        plt,
        pulse_excitation,
        solvers,
        spectra,
        spectral_time_resolved,
        u,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The fit changes the parameters slightly and compares the simulations to estimate in which direction to move them, so the simulations must be much more precise than the changes. We create the `Simulator` with tight solver tolerances (if they are too loose the fit warns about it):
    """)
    return


@app.cell
def _(CzIPN, Simulator, solvers):
    sim = Simulator(CzIPN).with_solver(solvers.LSODA(rtol=1e-10, atol=1e-12))
    return (sim,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Time resolved emission
    ### Making synthetic data
    Instead of real measurements we'll generate synthetic data from the model itself, so we know which values the fit should find. We excite the molecules with a 1 ns pulse and record the fluorescence from 5 ns to 30 μs, adding 3% of random noise. To see both the prompt and the delayed components we space the measurement times logarithmically. The measurement times are set on the simulator with `with_times(save_at=...)` (for the difference between `save_at` and `t_span`, see [Simulation](https://dyscolab.github.io/static_notebooks/simulation.html)):
    """)
    return


@app.cell
def _(CzIPN, np, pulse_excitation, sim, spectral_time_resolved, u):
    pulse = pulse_excitation({CzIPN.abs: 1e23 / (u.cm**2 * u.s)}, width=1 * u.ns)
    save_at = np.geomspace(0.005, 30, 80) * u.us

    exact = spectral_time_resolved(sim.with_times(save_at=save_at), excitation=pulse)
    # The result also has the times of the excitation events, keep only save_at
    exact = exact.pint.sel(time=save_at, method="nearest")
    fluorescence = exact["emission_fl"].pint.dequantify().values

    rng = np.random.default_rng(seed=1)
    measured = fluorescence * (1 + 0.03 * rng.standard_normal(fluorescence.shape))
    return measured, pulse, rng, save_at


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Building the target
    The measured data has to be wrapped in an xarray `Dataset` shaped like the output of `spectral_time_resolved`. `make_time_resolved_target` does this from a dictionary `{transition: values}` and the measurement times. Only the transitions in the dictionary are fitted, so we can fit just the lines we measured.
    """)
    return


@app.cell
def _(CzIPN, measured, save_at, u):
    from jablonski.fits import fit_spectral_time_resolved, make_time_resolved_target

    target = make_time_resolved_target({CzIPN.fl: measured / u.s}, save_at=save_at)
    target
    return fit_spectral_time_resolved, target


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fitting
    We pass `fit_spectral_time_resolved` the `Simulator`, the target, the excitation used in the experiment and a dictionary `p0` with the parameters to fit and their initial guesses. A guess can also be a tuple `(initial guess, lower bound, upper bound)`. The fit simulates at the times of the target, so the simulator doesn't need any times set. Here we fit the two crossing rates between the singlet and the triplet, assuming the others are known: ISC mostly sets how fast the prompt fluorescence decays and how much of the emission is delayed, and RISC how fast the delayed fluorescence decays.

    The fitted values are returned in the units of `p0`. Choosing units where the values are of order one (here 1/μs rather than 1/s) also helps the optimizer.
    """)
    return


@app.cell
def _(CzIPN, fit_spectral_time_resolved, pulse, sim, target, u):
    p0 = {CzIPN.isc.rate: 10 / u.us, CzIPN.risc.rate: 1 / u.us}
    fit = fit_spectral_time_resolved(sim, target, excitation=pulse, p0=p0)
    fit
    return (p0,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The result is a `FitResult`, a dictionary `{parameter: best value}`, so it can be passed directly to `with_values`. The true values were $35\ \mu s^{-1}$ and $0.5\ \mu s^{-1}$.

    ### Uncertainties
    The covariance matrix of the fitted parameters is in `fit.pcov`. Indexing it with a pair of parameters gives their covariance with units, so the standard error of a parameter is the square root of its variance.

    If we know the experimental error we should pass it as `sigma`, in the same form as the target. Otherwise the fit assumes all points have the same error and estimates it from the residuals, which isn't true for our data: the noise is 3% of each value, so it is thousands of times larger at the peak of the prompt fluorescence than at the end of the delayed one. That's why the RISC rate above is off.
    """)
    return


@app.cell
def _(fit_spectral_time_resolved, np, p0, pulse, sim, target):
    sigma = 0.03 * target
    weighted_fit = fit_spectral_time_resolved(sim, target, excitation=pulse, p0=p0, sigma=sigma)

    for param in p0:
        error = np.sqrt(weighted_fit.pcov[param, param])
        print(f"{param} = {weighted_fit[param]:.4f} ± {error:.4f}")

    pcov = np.asarray(weighted_fit.pcov)  # plain matrix, ordered as pcov.parameters
    print(f"correlation = {pcov[0, 1] / np.sqrt(pcov[0, 0] * pcov[1, 1]):.2f}")
    return (weighted_fit,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both rates are positively correlated. The delayed fluorescence decays at roughly the RISC rate times the probability that a molecule which comes back to $S_1$ emits instead of crossing to the triplet again, so a faster ISC can be partly compensated by a faster RISC.

    To check the fit we simulate with the fitted values and compare it with the data:
    """)
    return


@app.cell
def _(
    measured,
    np,
    plt,
    pulse,
    save_at,
    sim,
    spectral_time_resolved,
    u,
    weighted_fit,
):
    fitted = spectral_time_resolved(
        sim.with_values(weighted_fit),
        excitation=pulse,
        save_at=np.geomspace(0.002, 30, 300) * u.us,
    )
    fitted_fl = fitted["emission_fl"].pint.to({"time": u.us})

    plt.errorbar(
        save_at.magnitude,
        measured,
        yerr=0.03 * measured,
        fmt=".",
        label="measured",
    )
    plt.plot(fitted_fl.time.values, fitted_fl.pint.dequantify().values, label="fit")
    plt.xscale("log"); plt.yscale("log"); plt.xlabel("time [μs]"); plt.ylabel("fluorescence [photons / s]"); plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## CW spectra
    Under constant excitation the system reaches a steady state, and its spectrum is the photon rate of each emission line. `spectra` gives it as an xarray `DataArray` indexed by wavelength. Since Jablonski lines don't have a shape, our model has a single line: the fluorescence at 506 nm.

    Its intensity is the number of photons absorbed per second times the quantum yield. The quantum yield depends on the rates we just fitted, so with them we can use the CW intensity under a known excitation to find the absorption cross section. As before, we make synthetic data with 3% noise:
    """)
    return


@app.cell
def _(CzIPN, rng, sim, spectra, u):
    cw_excitation = {CzIPN.abs: 1e20 / (u.cm**2 * u.s)}
    exact_spectrum = spectra(sim, excitation=cw_excitation).pint.dequantify()
    wavelengths = exact_spectrum.wavelength.values
    measured_spectrum = exact_spectrum.values * (
        1 + 0.03 * rng.standard_normal(exact_spectrum.shape)
    )
    return cw_excitation, measured_spectrum, wavelengths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `make_spectra_target` builds the target from the wavelengths and intensities, and `fit_spectra` fits it with the same arguments as before, but taking a CW excitation `{pumper: height}` instead of a time dependent one. We pass it the simulator with the rates from the time resolved fit, and give the cross section guess in Å² ($10^{-16}\ cm^2$) so that it is of order one:
    """)
    return


@app.cell
def _(
    CzIPN,
    cw_excitation,
    measured_spectrum,
    np,
    sim,
    u,
    wavelengths,
    weighted_fit,
):
    from jablonski.fits import fit_spectra, make_spectra_target

    spectrum_target = make_spectra_target(wavelengths * u.nm, measured_spectrum / u.s)

    spectrum_fit = fit_spectra(
        sim.with_values(weighted_fit),
        spectrum_target,
        excitation=cw_excitation,
        p0={CzIPN.abs.rate: 1 * u.angstrom**2},
        sigma=0.03 * spectrum_target,
    )
    cross_section = spectrum_fit[CzIPN.abs.rate]
    cross_section_error = np.sqrt(spectrum_fit.pcov[CzIPN.abs.rate, CzIPN.abs.rate])
    print(f"abs.rate = {cross_section:.3f} ± {cross_section_error:.3f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The true value was $0.4\ Å^2$ ($4 \cdot 10^{-17}\ cm^2$).

    ## More options
    - `kind` selects what is fitted, as in the simulation functions: `"emission"` (default), `"absorption"` (absorbed photon rate of each pumper) or a list of kinds.
    - `join_by_energy=True` in `fit_spectral_time_resolved` (and `make_time_resolved_target`) sums the lines with the same energy, for when the measurement can't tell them apart.
    - `scale` weighs each line in the fit, for example `scale={"emission_fl": 2}` counts the fluorescence error twice as much.
    - Any other keyword argument is passed to [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

    To fit several experiments together, for example decays measured with different excitation powers or a decay and a spectrum at once, use Poincare's `Fitter` directly with one simulation function per experiment, as explained in the [fitting notebook](https://dyscolab.github.io/static_notebooks/fitting.html). The functions in `jablonski.simulation` can be used as simulation functions.
    """)
    return


if __name__ == "__main__":
    app.run()

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

    Both are built on poincare's `Fitter`, which is explained in detail in the [fitting notebook](https://dyscolab.github.io/static_notebooks/fitting.html). If you are new to jablonski, start with [Getting started with Jablonski](https://dyscolab.github.io/static_notebooks/getting_started_with_jablonski.html).

    We'll use a version of the ruthenium tris(bipyridine) model from the getting started notebook, with a weak fluorescence from the singlet state added so that the CW spectrum has two lines.
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
        IntersystemCrossing,
        Phosphorescence,
        ReverseIntersystemCrossing,
    )

    u = pint.get_application_registry()

    class Ruthenium(SpectroscopicSystem):
        ground: SingletState = initial(0 * u.eV, "singlet", default=10)
        MLCT_3: TripletState = initial(2.12 * u.eV, "triplet", default=0)
        MLCT_1: SingletState = initial(2.74 * u.eV, "singlet", default=0)

        abs = Absorption(ground=ground, excited=MLCT_1, rate=5.6e-17 * u.cm**2)
        fl = Fluorescence(excited=MLCT_1, ground=ground, rate=1e9 / u.s)
        isc = IntersystemCrossing(source=MLCT_1, target=MLCT_3, rate=1e11 / u.s)
        ph = Phosphorescence(excited=MLCT_3, ground=ground, rate=1 / (0.6 * u.us) * 0.04)
        risc = ReverseIntersystemCrossing(
            source=MLCT_3, target=ground, rate=1 / (0.6 * u.us) * 0.96
        )

    return (
        Ruthenium,
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
def _(Ruthenium, Simulator, solvers):
    sim = Simulator(Ruthenium).with_solver(solvers.LSODA(rtol=1e-10, atol=1e-12))
    return (sim,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Time resolved emission
    ### Making synthetic data
    Instead of real measurements we'll generate synthetic data from the model itself, so we know which values the fit should find. We excite the molecules with a short pulse and record the phosphorescence every 50 ns from 50 ns to 3 μs, adding 3% of random noise. The measurement times are set on the simulator with `with_times(save_at=...)` (for the difference between `save_at` and `t_span`, see [Simulation](https://dyscolab.github.io/static_notebooks/simulation.html)):
    """)
    return


@app.cell
def _(Ruthenium, np, pulse_excitation, sim, spectral_time_resolved, u):
    pulse = pulse_excitation({Ruthenium.abs: 1e23 / (u.cm**2 * u.s)}, width=10 * u.ns)
    save_at = np.linspace(0.05, 3, 60) * u.us

    exact = spectral_time_resolved(sim.with_times(save_at=save_at), excitation=pulse)
    # The result also has the times of the excitation events, keep only save_at
    exact = exact.pint.sel(time=save_at, method="nearest")
    phosphorescence = exact["emission_ph"].pint.dequantify().values

    rng = np.random.default_rng(seed=1)
    measured = phosphorescence * (1 + 0.03 * rng.standard_normal(phosphorescence.shape))
    return measured, pulse, rng, save_at


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Building the target
    The measured data has to be wrapped in an xarray `Dataset` shaped like the output of `spectral_time_resolved`. `make_time_resolved_target` does this from a dictionary `{transition: values}` and the measurement times. Only the transitions in the dictionary are fitted, so we can fit just the lines we measured.
    """)
    return


@app.cell
def _(Ruthenium, measured, save_at, u):
    from jablonski.fits import fit_spectral_time_resolved, make_time_resolved_target

    target = make_time_resolved_target({Ruthenium.ph: measured / u.s}, save_at=save_at)
    target
    return fit_spectral_time_resolved, target


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fitting
    We pass `fit_spectral_time_resolved` the `Simulator`, the target, the excitation used in the experiment and a dictionary `p0` with the parameters to fit and their initial guesses. A guess can also be a tuple `(initial guess, lower bound, upper bound)`. The fit simulates at the times of the target, so the simulator doesn't need any times set. Here we fit both rates that empty the triplet state: the phosphorescence rate sets how bright the emission is, and the sum of both rates how fast it decays.

    The fitted values are returned in the units of `p0`. Choosing units where the values are of order one (here 1/μs rather than 1/s) also helps the optimizer.
    """)
    return


@app.cell
def _(Ruthenium, fit_spectral_time_resolved, pulse, sim, target, u):
    p0 = {Ruthenium.ph.rate: 0.1 / u.us, Ruthenium.risc.rate: 1 / u.us}
    fit = fit_spectral_time_resolved(sim, target, excitation=pulse, p0=p0)
    fit
    return (p0,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The result is a `FitResult`, a dictionary `{parameter: best value}`, so it can be passed directly to `with_values`. The true values were $0.0667\ \mu s^{-1}$ and $1.6\ \mu s^{-1}$.

    ### Uncertainties
    The covariance matrix of the fitted parameters is in `fit.pcov`. Indexing it with a pair of parameters gives their covariance with units, so the standard error of a parameter is the square root of its variance.

    If we know the experimental error we should pass it as `sigma`, in the same form as the target. Otherwise the fit assumes all points have the same error and estimates it from the residuals, which isn't true for our data: the noise is 3% of each value, so it is larger at the start of the decay.
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
    Both rates are positively correlated: a decay that is a bit too fast can be partly compensated by a brighter emission.

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
        save_at=np.linspace(0, 3, 300) * u.us,
    )
    fitted_ph = fitted["emission_ph"].pint.to({"time": u.us})

    plt.errorbar(
        save_at.magnitude,
        measured,
        yerr=0.03 * measured,
        fmt=".",
        label="measured",
    )
    plt.plot(fitted_ph.time.values, fitted_ph.pint.dequantify().values, label="fit")
    plt.xlabel("time [μs]"); plt.ylabel("phosphorescence [photons / s]"); plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## CW spectra
    Under constant excitation the system reaches a steady state, and its spectrum is the photon rate of each emission line. `spectra` gives it as an xarray `DataArray` indexed by wavelength. Since jablonski lines don't have a shape, our model has two lines: the fluorescence at 452 nm and the phosphorescence at 585 nm.

    The ratio between both lines depends on how fast the singlet crosses to the triplet, so we can use it to fit the intersystem crossing rate. As before, we make synthetic data with 3% noise:
    """)
    return


@app.cell
def _(Ruthenium, rng, sim, spectra, u):
    cw_excitation = {Ruthenium.abs: 1e20 / (u.cm**2 * u.s)}
    exact_spectrum = spectra(sim, excitation=cw_excitation).pint.dequantify()
    wavelengths = exact_spectrum.wavelength.values
    measured_spectrum = exact_spectrum.values * (
        1 + 0.03 * rng.standard_normal(exact_spectrum.shape)
    )
    return cw_excitation, measured_spectrum, wavelengths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `make_spectra_target` builds the target from the wavelengths and intensities, and `fit_spectra` fits it with the same arguments as before, but taking a CW excitation `{pumper: height}` instead of a time dependent one:
    """)
    return


@app.cell
def _(Ruthenium, cw_excitation, measured_spectrum, np, sim, u, wavelengths):
    from jablonski.fits import fit_spectra, make_spectra_target

    spectrum_target = make_spectra_target(wavelengths * u.nm, measured_spectrum / u.s)

    spectrum_fit = fit_spectra(
        sim,
        spectrum_target,
        excitation=cw_excitation,
        p0={Ruthenium.isc.rate: 30 / u.ns},
        sigma=0.03 * spectrum_target,
    )
    isc_rate = spectrum_fit[Ruthenium.isc.rate]
    isc_error = np.sqrt(spectrum_fit.pcov[Ruthenium.isc.rate, Ruthenium.isc.rate])
    print(f"isc.rate = {isc_rate:.1f} ± {isc_error:.1f}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The true value was $100\ ns^{-1}$.

    ## More options
    - `kind` selects what is fitted, as in the simulation functions: `"emission"` (default), `"absorption"` (absorbed photon rate of each pumper) or a list of kinds.
    - `join_by_energy=True` in `fit_spectral_time_resolved` (and `make_time_resolved_target`) sums the lines with the same energy, for when the measurement can't tell them apart.
    - `scale` weighs each line in the fit, for example `scale={"emission_ph": 2}` counts the phosphorescence error twice as much.
    - Any other keyword argument is passed to [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

    To fit several experiments together, for example decays measured with different excitation powers or a decay and a spectrum at once, use poincare's `Fitter` directly with one simulation function per experiment, as explained in the [fitting notebook](https://dyscolab.github.io/static_notebooks/fitting.html). The functions in `jablonski.simulation` can be used as simulation functions.
    """)
    return


if __name__ == "__main__":
    app.run()

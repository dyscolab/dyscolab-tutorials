# Setup cell for marimo notebook, can be ignored
import marimo as mo
import sys

# Import packages if running on marimo playground
if sys.executable == "/home/pyodide/this.program":
    import warnings
    with warnings.catch_warnings():
        import micropip
        micropip.uninstall("packaging")
        await micropip.install(
             ["pint_pandas <=0.7","poincare>=1.0.0b3","typing_extensions>=4.15.0", "matplotlib"]
        )
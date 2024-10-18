from pathlib import Path

# Compute local path to serve
serve_path = str(Path(__file__).with_name("serve").resolve())

# Serve directory for JS/CSS files
serve = {"__breasties_2": serve_path}

# List of JS files to load (usually from the serve path above)
scripts = ["__breasties_2/breasties_2.umd.js"]

# List of CSS files to load (usually from the serve path above)
# styles = ["__breasties_2/style.css"]

# List of Vue plugins to install/load
vue_use = ["breasties_2"]

# Uncomment to add entries to the shared state
# state = {}


# Optional if you want to execute custom initialization at module load
def setup(app, **kwargs):
    """Method called at initialization with possibly some custom keyword arguments"""
    pass

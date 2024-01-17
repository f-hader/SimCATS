"""Configuration file for the Sphinx documentation builder."""
import glob
import os
import shutil

project = "SimCATS"
copyright = "2023 Forschungszentrum Jülich GmbH - Central Institute of Engineering, Electronics and Analytics (ZEA) - Electronic Systems (ZEA-2)"
author = "Fabian Hader, Sarah Fleitmann, Fabian Fuchs"
release = "1.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # automatic documentation generation from docstrings
    "autoapi.extension",  # different automatic documentation generation from docstrings
    "sphinx_rtd_theme",  # readthedocs theme
    "sphinx.ext.napoleon",  # support google and numpy style docstrings
    "myst_nb",  # jupyter notebook support
]

exclude_patterns = []

# myst_nb
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# force notebook execution
nb_execution_mode = "off"

# autoapi
autoapi_dirs = ["../../simcats"]
autodoc_typehints = "description"
autodoc_typehints_format = "short"

python_use_unqualified_type_names = True
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    # "private-members",
    # "special-members",
    "show-inheritance",
    "show-inheritance-diagram",
    "show-module-summary",
    "imported-members",
]

# We don't need the autoapi toctree entry, as we add it ourselves
autoapi_add_toctree_entry = False

# inherit python class parameter description from the __init__ method
autoapi_python_class_content = "both"

# set template folder
templates_path = ["_templates"]
autoapi_template_dir = "_templates"

# graphviz
inheritance_alias = {
    "simcats.distortions._distortion_interfaces.DistortionInterface": "simcats.distortions.DistortionInterface",
    "simcats.distortions._distortion_interfaces.OccupationDistortionInterface": "simcats.distortions.OccupationDistortionInterface",
    "simcats.distortions._distortion_interfaces.SensorPotentialDistortionInterface": "simcats.distortions.SensorPotentialDistortionInterface",
    "simcats.distortions._distortion_interfaces.SensorResponseDistortionInterface": "simcats.distortions.SensorResponseDistortionInterface",
    "simcats.distortions._transition_blurring.OccupationTransitionBlurringGaussian": "simcats.distortions.OccupationTransitionBlurringGaussian",
    "simcats.distortions._transition_blurring.OccupationTransitionBlurringFermiDirac": "simcats.distortions.OccupationTransitionBlurringFermiDirac",
    "simcats.distortions._dot_jumps.OccupationDotJumps": "simcats.distortions.OccupationDotJumps",
    "simcats.distortions._pink_noise.SensorPotentialPinkNoise": "simcats.distortions.SensorPotentialPinkNoise",
    "simcats.distortions._random_telegraph_noise.RandomTelegraphNoise": "simcats.distortions.RandomTelegraphNoise",
    "simcats.distortions._random_telegraph_noise.SensorPotentialRTN": "simcats.distortions.SensorPotentialRTN",
    "simcats.distortions._random_telegraph_noise.SensorResponseRTN": "simcats.distortions.SensorResponseRTN",
    "simcats.distortions._white_noise.SensorResponseWhiteNoise": "simcats.distortions.SensorResponseWhiteNoise",
    "simcats.ideal_csd._ideal_csd_interface.IdealCSDInterface": "simcats.ideal_csd.IdealCSDInterface",
    "simcats.ideal_csd.geometric._ideal_csd_geometric.IdealCSDGeometric": "simcats.ideal_csd.IdealCSDGeometric",
    "simcats.sensor._sensor_interface.SensorPeakInterface": "simcats.sensor.SensorPeakInterface",
    "simcats.sensor._sensor_interface.SensorInterface": "simcats.sensor.SensorInterface",
    "simcats.sensor._gaussian_sensor_peak.SensorPeakGaussian": "simcats.sensor.SensorPeakGaussian",
    "simcats.sensor._lorentzian_sensor_peak.SensorPeakLorentzian": "simcats.sensor.SensorPeakLorentzian",
    "simcats.sensor._generic_sensor.SensorGeneric": "simcats.sensor.SensorGeneric",
    "simcats.support_functions._parameter_sampling.ParameterSamplingInterface": "simcats.support_functions.ParameterSamplingInterface",
    "simcats.support_functions._parameter_sampling.NormalSamplingRange": "simcats.support_functions.NormalSamplingRange",
    "simcats.support_functions._parameter_sampling.UniformSamplingRange": "simcats.support_functions.UniformSamplingRange",
    "simcats._simulation.Simulation": "simcats.Simulation",
}
graphviz_output_format = "svg"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = []

# for more options see https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
html_theme_options = {
    # Toc options
    "collapse_navigation": False,
    "navigation_depth": -1,
}

# copy notebooks and convert them
directory_path = "../../*.ipynb"
notebook_paths = glob.glob(directory_path)

if not os.path.isdir("./notebooks"):
    os.mkdir("./notebooks")

for path in notebook_paths:
    new_path = path.replace("../..", "./notebooks")
    if os.path.isfile(new_path):
        os.remove(new_path)
    shutil.copyfile(path, new_path)

# copy readme
if not os.path.isdir("./misc"):
    os.mkdir("./misc")

shutil.copyfile("../../README.md", "./misc/README.md")

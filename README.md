<h1 align="center">
  <img src="https://raw.githubusercontent.com/f-hader/SimCATS/main/SimCATS_symbol.svg" alt="SimCATS logo">
  <br>
</h1>

<div align="center">
  <a href="https://github.com/f-hader/SimCATS/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License: CC BY-NC-SA 4.0"/>
  </a>
  <a href="https://pypi.org/project/simcats/">
    <img src="https://img.shields.io/pypi/v/simcats.svg" alt="PyPi Latest Release"/>
  </a>
  <a href="https://simcats.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/readthedocs/simcats" alt="Read the Docs"/>
  </a>
  <a href="https://doi.org/10.1109/TQE.2024.3445967">
    <img src="https://img.shields.io/badge/DOI (Paper)-10.1109/TQE.2024.3445967-007ec6.svg" alt="DOI Paper"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.13805205">
    <img src="https://img.shields.io/badge/DOI (Code)-10.5281/zenodo.13805205-007ec6.svg" alt="DOI Code"/>
  </a>
</div>

# SimCATS

Simulation of CSDs for Automated Tuning Solutions (`SimCATS`) is a Python framework for simulating charge stability 
diagrams (CSDs) typically measured during the tuning process of qubits.

## Installation

The framework supports Python versions 3.7 - 3.11 and installs via pip:
```
pip install simcats
```

Alternatively, the `SimCATS` package can be installed by cloning the GitHub repository, navigating to the folder 
containing the `setup.py` file and executing

```
pip install .
```

For the installation in development/editable mode, use the option `-e`.

## Examples / Tutorials
After installing the package, a good starting point is a look into the Jupyter Notebook 
`example_SimCATS_simulation_class.ipynb`, which provides an overview of the usage of the simulation class offered by 
the framework. 
For more detailed examples and explanations of the geometric ideal CSD simulation using Total Charge Transitions (TCTs), look at the Jupyter Notebook `example_SimCATS_IdealCSDGeometric.ipynb`. This notebook also includes a hint
regarding the generation of required labels for training algorithms that might need line labels defined as start and
end points or require semantic information about particular transitions.

## Tests

The tests are written for the `PyTest` framework but should also work with the `unittest` framework.

To run the tests, install the packages `pytest`, `pytest-cov`, and `pytest-xdist` with

```
pip install pytest pytest-cov pytest-xdist
```

and run the following command:

```
pytest --cov=simcats -n auto --dist loadfile .\tests\
```

The argument 
- `--cov=simcats` enables a coverage summary of the `SimCATS` package,
- `-n auto` enables the test to run with multiple threads (auto will choose as many threads as possible, but can be replaced with a specific number of threads to use), and
- `--dist loadfile` specifies that each file should be executed only by one thread.

<!-- start sec:documentation -->
## Documentation

The official documentation is hosted on [ReadtheDocs](https://simcats.readthedocs.io), but can also be built locally.
To do this, first install the packages `sphinx`, `sphinx-rtd-theme`, `sphinx-autoapi`, `myst-nb `, and `jupytext` with

```
pip install sphinx sphinx-rtd-theme sphinx-autoapi myst-nb jupytext
```

and then, in the `docs` folder, execute the following command:

```
.\make html
```

To view the generated HTML documentation, open the file `docs\build\html\index.html`.
<!-- end sec:documentation -->

## Structure of SimCATS

The primary user interface for `SimCATS` is the class `Simulation`, which combines all the necessary functionalities to
measure (simulate) a CSD and adjust the parameters for the simulated measurement. The class `Simulation` and default
configurations for the simulation (`default_configs`) can be imported directly from `simcats`. Aside from that,
`SimCATS` contains the subpackages `ideal_csd`, `sensor`, `distortions`, and `support_functions`, described in
the following sections.

### Module `simulation`

An instance of the simulation class requires

-   an implementation of the `IdealCSDInterface` for the simulation of ideal CSD data,
-   an implementation of the `SensorInterface` for the simulation of the sensor (dot) reaction based on the ideal CSD
data, and
-   (optionally) implementations of the desired types of distortions, which can be implementations from `OccupationDistortionInterface`, `SensorPotentialDistortionInterface`, or `SensorResponseDistortionInterface`.

With an initialized instance of the `Simulation` class, it is possible to run simulations using the `measure` function
(see `example_SimCATS_simulation_class.ipynb`).

### Subpackage `ideal_csd`

This subpackage contains the `IdealCSDInterface` used by the `Simulation` class  and an implementation of
the `IdealCSDInterface` (`IdealCSDGeometric`) based on our geometric simulation approach.
Additionally, it contains in the subpackage `geometric` the functions used by `IdealCSDGeometric`, including the
implementation of the total charge transition (TCT) definition and functions for calculating the occupations using TCTs.

### Subpackage `distortions`

The distortions subpackage contains the `DistortionInterface` from which the `OccupationDistortionInterface`, the 
`SensorPotentialDistortionInterface`, and the `SensorResponseDistortionInterface` are derived. Distortion functions used
in the `Simulation` class have to implement these specific interfaces. Implemented distortions included in the
subpackage are:

-   white noise, generated by sampling from a normal distribution,
-   pink noise, generated using the package colorednoise ([https://github.com/felixpatzelt/colorednoise](https://github.com/felixpatzelt/colorednoise)),
-   random telegraph noise (RTN), generated using the algorithm described in ["Toward Robust Autotuning of Noisy Quantum Dot Devices" by Ziegler et al.](https://doi.org/10.1103/PhysRevApplied.17.024069) (RTN is called sensor jumps there),
-   dot jumps, simulated using the algorithm described in ["Toward Robust Autotuning of Noisy Quantum Dot Devices" by Ziegler et al.](https://doi.org/10.1103/PhysRevApplied.17.024069) (In the `Simulation` class, this is applied to a whole block of rows or columns, but there is also a function for applying it linewise.), and
-   lead transition blurring, simulated using Gaussian or Fermi-Dirac blurring.

The implementations also offer the option to set ratios (parameter `ratio`) for the occurrence of the distortion (e.g. dot jumps may only happen sometimes and not in every measurement). Moreover, it is also possible to sample the
noise parameters from a given sampling range using an object of type `ParameterSamplingInterface`.
Classes for randomly sampling from a normal distribution or a uniform distribution within a given range are available in
the subpackage `support_functions`.
In this case, the strength is randomly chosen from the given range for every measurement.
Additionally, it is possible to specify that this range should be a smaller subrange of the provided range.
This allows restricting distortion fluctuations during a simulation while enabling a large variety of different strengths
for the initialization of the objects. <br>
RTN, dot jumps, and lead transition blurring are applied in the pixel domain. However, the jump length or the blurring strength should be consistent in the voltage domain even if the resolution changes. Therefore, the parameters
are given in the voltage domain and adjusted according to the resolution in terms of pixel per voltage. <br>
For a simulated measurement with a continuous voltage sweep involving an averaging for each pixel, the noise strength of the
white and pink noise should be adjusted if the resolution (volt per pixel) changes, due to smoothing out the noise. This smoothing depends on the type of averaging used and is not incorporated in the default implementation.

### Subpackage `sensor`

This subpackage contains the `SensorInterface` that defines how a sensor simulation must be implemented to be used by the `Simulation` class. The `SensorPeakInterface` provides the desired representation for the definition of the Coulomb peaks the sensor uses. `SensorGeneric` implements the `SensorInterface` and offers functions for simulating the sensor response and potential. It offers the possibility to simulate with a single peak or multiple sensor peaks. Current implementations of the `SensorPeakInterface` are `SensorPeakGaussian` and `SensorPeakLorentzian`.

### Subpackage `support_functions`

This subpackage contains support functions, which are used by the end user and by different functions of the framework.  
- `fermi_filter1d` is an implementation of a one-dimensional Fermi-Dirac filter.
- `plot_csd` plots one and two-dimensional CSDs. The function can also plot ground truth data (see `example_SimCATS_simulation_class.ipynb` for examples).  
- `rotate_points` simply rotates coordinates (stored in a (n, 2) shaped array) by a given angle. It is especially used during the generation of the ideal data.
- `ParameterSamplingInterface` defines an interface for randomly sampled (fluctuated) strengths of distortions.
  - `NormalSamplingRange` and `UniformSamplingRange` are implementations of the `ParameterSamplingInterface`.

## Citations

```bibtex
@article{hader2024simcats,
  author={Hader, Fabian and Fleitmann, Sarah and Vogelbruch, Jan and Geck, Lotte and Waasen, Stefan van},
  journal={IEEE Transactions on Quantum Engineering}, 
  title={Simulation of Charge Stability Diagrams for Automated Tuning Solutions (SimCATS)}, 
  year={2024},
  volume={5},
  pages={1-14},
  doi={10.1109/TQE.2024.3445967}
}
```

## License, CLA, and Copyright

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

Contributions must follow the Contributor License Agreement. For more information, see the CONTRIBUTING.md file at the top of the GitHub repository.

Copyright © 2024 Forschungszentrum Jülich GmbH - Central Institute of Engineering, Electronics and Analytics (ZEA) - Electronic Systems (ZEA-2)

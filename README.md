# openCOSMO-RS(py) package

## About 

This package is a fully python based implementation of the model COSMO-RS, supporting multiple segment descriptors.
Papers related to openCOSMO-RS:
- original paper [here](https://www.sciencedirect.com/science/article/pii/S0378381222000954).
- version 24a paper[here](https://www.sciencedirect.com/science/article/pii/S0378381224002255).
- halocarbon paper [here](https://www.sciencedirect.com/science/article/pii/S0009250925002489).
- version 25a [here](https://www.sciencedirect.com/science/article/pii/S0009250925009935).


## Application 

The package can be installed from PyPI (Python 3.11 or newer):
```
pip install opencosmorspy
```
The 3D visualizations in `opencosmorspy.cosmo_visualizations` need additional packages (RDKit, scikit-image):
```
pip install "opencosmorspy[visualization]"
```
The development version can be installed directly from GitHub:
```
pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py
```

**Faster C++ implementation:** the same model is available as a C++ implementation with Python bindings, which is much faster. It is published on PyPI as `opencosmors` (`pip install opencosmors`, `import openCOSMORS`), see [openCOSMO-RS_cpp](https://github.com/TUHH-TVT/openCOSMO-RS_cpp). New model features are usually added there first. Note the similar names: `opencosmorspy` is this pure Python package, `opencosmors` is the C++ one.

A short introduction on its use can be found in the examples directory.

To generate the needed quantum chemical input files using RDKit and ORCA with the CPCM module, please have a look here: https://github.com/TUHH-TVT/openCOSMO-RS_conformer_pipeline

## Parameterizations and model extensions

The parameterizations are classes in `opencosmorspy.parameterization`:

| Class | Description |
| --- | --- |
| `Parameterization("default_turbomole")`, `Parameterization("default_orca")` | default COSMO-RS parameters |
| `openCOSMORS24a` | version 24a, including the parameters of the solvation free energy |
| `openCOSMORS25a` | version 25a (halocarbon paper, part 2): dispersion from atomic polarizabilities (`polariz_mode = 7`). Needs ORCA COSMO files with the section `#ATOMIC POLARIZABILITIES`. |

```python
from opencosmorspy.cosmors import COSMORS
from opencosmorspy.parameterization import openCOSMORS25a

crs = COSMORS(openCOSMORS25a())
```

The extensions of the model were ported from the
[C++ implementation](https://github.com/TUHH-TVT/openCOSMO-RS_cpp) and are
switched on by attributes of the parameterization (with the default
parameterizations the results did not change):

| Attribute (C++ switch) | Description |
| --- | --- |
| `polariz_mode` (`sw_SR_polarizabilities`) | 0 (off), 1, 3, 5, 6, 7, 8: projection of the atomic polarizabilities on the segments and the dispersion energy. Needs `"polariz"` in `descriptor_lst`, the parameters `vdw_m` (m_vdW) and `mf_e_f_corr` (E_F_corr). |
| `descriptor_lst` with `"sigma_corr"` | cluster `sigma_corr` instead of `sigma_orth` as in the C++ implementation |
| `mf_use_sigma_orth`, `mf_sigma_corr_exclude_monoatomic_ions` (`sw_SR_misfit` 0, 1, 2) | misfit correlation |
| `differentiate_hydrogens`, `differentiate_molecule_groups`, `ion_radii` (`sw_SR_differentiateHydrogens`, `sw_SR_differentiateMoleculeGroups`, `radii`) | hydrogens by bonding partner, molecule groups and monoatomic cations |
| `segment_reference_state` (`sw_SR_useSegmentReferenceStateForInteractionMatrix`) | `"conductor"` or `"pure_segment"` |
| `comb_term` | `"none"`, `"staverman_guggenheim"`, `"klamt"`, `"staverman_guggenheim_exponential_scaling"`, `"staverman_guggenheim_grensemann"`, `"franke_hannebauer"` |
| `radii_consistency` | check that the cavity radii of an element are identical in all COSMO files |

Reference states of `COSMORS.add_job`: `"cosmo"`, `"pure_component"`,
`"pure_component_only_neutral"`, `"reference_mixture"` and
`"solvation_energy"`. The solvation free energy (kcal/mol) is returned as
`res["dGsolv"]`. It needs the gas phase energy and the number of ring atoms of
every molecule: `crs.add_molecule([path], e_gas=-76.3, n_ring_atoms=0)`.

The interaction arrays contain the dispersion energy as `A_vdw`/`E_vdw`
(`A_int = A_mf + A_hb + A_vdw`), their contact statistics and partial molar
values are returned as `aim_*` and `pm_*`.

Note: the `aim_*` energies of the C++ implementation (`average_surface_energies`)
are resolved by molecule pairs, the values of this package equal the sum over
the partners. Both are given in J/mol.

## License

This project is licensed under the GNU General Public License v2.0

## Other COSMO-RS related projects

The reader might be also interested in the following related projects:
 - [openCOSMO-RS_cpp](https://github.com/TUHH-TVT/openCOSMO-RS_cpp)
 - [LVPP sigma profile database](https://github.com/lvpp/sigma)
 - [Benchmark COSMO-SAC implementation](https://github.com/usnistgov/COSMOSAC)
 - [Pysac](https://github.com/lvpp/pysac)

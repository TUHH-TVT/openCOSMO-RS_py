"""
Tests of the features ported from the openCOSMO-RS C++ implementation:
atomic polarizabilities (dispersion), further combinatorial terms, the
reference state of the segment interactions and solvation free energies.

The reference values were calculated with the C++ implementation (double
precision). The C++ implementation stops the COSMOSPACE iteration if the change
of ln(Gamma) is below 1e-5, therefore the activity coefficients are compared
with a tolerance of 1e-4.
"""

import os

import numpy as np
import pytest

from opencosmorspy.cosmors import COSMORS
from opencosmorspy.molecules import round_and_truncate, round_half_away
from opencosmorspy.parameterization import (
    IONIZATION_POTENTIAL_EV,
    Parameterization,
    get_ionization_potential_ev,
    openCOSMORS24a,
    openCOSMORS25a,
)

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COSMO_ORCA", "cpp_reference")
FILES = [
    os.path.join(DIR, "1112tetrachloroethane.orcacosmo"),
    os.path.join(DIR, "methanol.orcacosmo"),
    os.path.join(DIR, "water.orcacosmo"),
]
X = np.array([0.2, 0.5, 0.3])
N_AVOGADRO = 6.022140857e23  # used by the C++ implementation in its contact statistics

REFERENCE = {'comb0': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
 'comb1': [[-0.262550192, 0.001227582, -0.143122763], [-0.262550192, 0.001227582, -0.143122763]],
 'comb2': [[-0.148684097, 0.080908206, -0.035724279], [-0.148684097, 0.080908206, -0.035724279]],
 'comb3': [[0.003609219, 0.019312017, 0.023629238], [0.003609219, 0.019312017, 0.023629238]],
 'comb4': [[-0.150293049, -0.002192676, -0.103360021], [-0.150293049, -0.002192676, -0.103360021]],
 'comb5': [[-0.347664457, -0.118072154, -0.234704639], [-0.232438368, -0.010336655, -0.129417163]],
 'dgsolv': [[-292015.141049, -177.344966, 0.0]],
 'mode1_res': [[5.571005, 2.144029, 0.923257], [2.067645, -0.105986, 1.280491]],
 'mode3_res': [[5.561235, 2.13688, 0.922872], [2.064411, -0.107054, 1.280074]],
 'mode5_res': [[-5.542535, 2.686548, 1.250328], [4.422573, 0.552066, 1.62299]],
 'mode6_res': [[-6.157055, 2.854362, 1.327152], [4.96629, 0.675307, 1.684743]],
 'mode7_aim': [[7759.787, -1692.607, -6864.229], [9401.389, 4654.601, 3257.688],
               [-176.545, -6293.874, -10111.088], [-441.363, -15734.685, -25277.72],
               [-1465.057, -53.334, -10.828]],
 'mode7_pm': [[7502.846, -1166.276, -7570.112], [8545.01, 4924.039, 3379.543],
              [1107.949, -6251.182, -11038.53], [2769.873, -15627.954, -27596.324],
              [-2150.113, 160.867, 88.874]],
 'mode7_res': [[4.781831, 2.211788, 0.955943], [2.325362, -0.037878, 1.313109]],
 'mode8_res': [[5.214534, 2.179137, 0.937918], [2.191138, -0.072801, 1.295071]],
 'pure_segment_res': [[-2.257148, -11.784649, -18.835852], [2.32534, -0.037872, 1.313097]]}



def calculate(par, x_lst_refst, temperature=298.15, files=FILES, **molecule_kwargs):
    crs = COSMORS(par)
    for idx, file in enumerate(files):
        kwargs = {key: value[idx] for key, value in molecule_kwargs.items()}
        crs.add_molecule([file], **kwargs)
    for x, refst in x_lst_refst:
        crs.add_job(np.array(x, dtype=float), temperature, refst)
    return crs.calculate()


@pytest.mark.parametrize("mode", [1, 3, 5, 6, 7, 8])
def test_polarizability_modes_vs_cpp(mode):
    par = openCOSMORS25a()
    par.polariz_mode = mode
    res = calculate(par, [(X, "cosmo"), (X, "pure_component")])
    np.testing.assert_allclose(res["enth"]["lng"], REFERENCE[f"mode{mode}_res"], atol=1e-4)


def test_partial_interaction_matrices_vs_cpp():
    par = openCOSMORS25a()
    par.calculate_contact_statistics_molecule_properties = True
    res = calculate(par, [(X, "cosmo")])["enth"]

    keys = ["A_int", "E_mf", "A_hb", "E_hb", "E_vdw"]
    pm = np.array([res["pm_" + key][0] for key in keys])
    np.testing.assert_allclose(pm, REFERENCE["mode7_pm"], rtol=1e-4, atol=0.5)
    aim = np.array([res["aim_" + key][0] for key in keys])
    np.testing.assert_allclose(aim, REFERENCE["mode7_aim"], rtol=1e-4, atol=0.5)

    # A_int = E_mf + G_hb + E_vdW
    np.testing.assert_allclose(
        res["pm_A_int"], res["pm_E_mf"] + res["pm_A_hb"] + res["pm_E_vdw"], rtol=1e-8
    )


@pytest.mark.parametrize(
    "comb_term, number",
    [
        ("none", 0),
        ("staverman_guggenheim", 1),
        ("klamt", 2),
        ("staverman_guggenheim_exponential_scaling", 3),
        ("staverman_guggenheim_grensemann", 4),
        ("franke_hannebauer", 5),
    ],
)
def test_combinatorial_terms_vs_cpp(comb_term, number):
    par = openCOSMORS25a()
    par.comb_term = comb_term
    par.comb_sg_z_coord = 10.0
    par.comb_sg_a_std = 47.999
    res = calculate(par, [(X, "cosmo"), (X, "pure_component")])
    # the area and volume in the COSMO files are converted with slightly
    # different bohr radii in both implementations
    np.testing.assert_allclose(res["comb"]["lng"], REFERENCE[f"comb{number}"], atol=1e-6)


def test_pure_segment_reference_state_vs_cpp():
    par = openCOSMORS25a()
    par.segment_reference_state = "pure_segment"
    res = calculate(par, [(X, "cosmo"), (X, "pure_component")])
    np.testing.assert_allclose(res["enth"]["lng"], REFERENCE["pure_segment_res"], atol=1e-4)


def test_pure_segment_reference_state_cancels_for_pure_components():
    results = []
    for state in ("conductor", "pure_segment"):
        par = openCOSMORS25a()
        par.segment_reference_state = state
        results.append(calculate(par, [(X, "pure_component")])["tot"]["lng"])
    np.testing.assert_allclose(results[0], results[1], atol=1e-5)


def test_solvation_energy_vs_cpp():
    # model parameters of the reference calculation (without polarizabilities)
    # and the solvation energy parameters of the parameterization 24a
    par = openCOSMORS25a()
    par.polariz_mode = 0
    par24a = openCOSMORS24a()
    par.dgsolv_tau = par24a.dgsolv_tau
    par.dgsolv_eta = par24a.dgsolv_eta
    par.dgsolv_omega_ring = par24a.dgsolv_omega_ring
    # the gas phase energies are arbitrary numbers, as in the reference calculation
    res = calculate(
        par,
        [([0.0, 0.0, 1.0], "solvation_energy")],
        e_gas=[-1453.123, -115.5, -76.3],
        n_ring_atoms=[0, 0, 0],
    )
    np.testing.assert_allclose(res["dGsolv"], REFERENCE["dgsolv"], atol=1e-4)


def test_solvation_energy_needs_gas_energy():
    crs = COSMORS(openCOSMORS24a())
    for file in FILES:
        crs.add_molecule([file])
    crs.add_job(np.array([0.0, 0.0, 1.0]), 298.15, "solvation_energy")
    with pytest.raises(ValueError, match="e_gas"):
        crs.calculate()


def test_only_neutral_reference_state_is_pure_component_for_neutral_molecules():
    par = openCOSMORS25a()
    res = calculate(par, [(X, "pure_component"), (X, "pure_component_only_neutral")])
    np.testing.assert_allclose(res["tot"]["lng"][0], res["tot"]["lng"][1])


def test_polarizability_needs_file_with_polarizabilities():
    # this COSMO file does not contain the section #ATOMIC POLARIZABILITIES
    file = os.path.join(
        os.path.dirname(DIR),
        "H2O",
        "COSMO_TZVPD",
        "H2O_c000.orcacosmo",
    )
    crs = COSMORS(openCOSMORS25a())
    with pytest.raises(ValueError, match="atomic polarizabilities"):
        crs.add_molecule([file])

    # without polarizabilities the file can be used
    par = openCOSMORS25a()
    par.polariz_mode = 0
    COSMORS(par).add_molecule([file])


def test_parameterization_checks():
    par = openCOSMORS25a()
    par.polariz_mode = 2
    with pytest.raises(ValueError, match="polariz_mode"):
        par.check_parameterization()

    par = openCOSMORS25a()
    par.descriptor_lst = ["sigma", "sigma_corr", "elmnt_nr", "group", "mol_charge"]
    with pytest.raises(ValueError, match="polariz"):
        par.check_parameterization()

    par = openCOSMORS25a()
    par.comb_term = "unknown"
    with pytest.raises(ValueError, match="combinatorial"):
        par.check_parameterization()

    par = Parameterization("default_orca")
    par.check_parameterization()  # defaults of the extensions change nothing


def test_round_half_away_like_c():
    np.testing.assert_array_equal(
        round_half_away([0.5, 1.5, 2.5, -0.5, -2.5, 2.4, -2.6]), [1, 2, 3, -1, -3, 2, -3]
    )
    np.testing.assert_allclose(round_and_truncate([14.9, 15.0, 24.99], 1), [10, 20, 20])
    np.testing.assert_allclose(round_and_truncate([0.14, 0.17, 0.26], -1), [0.1, 0.2, 0.3])


def test_ionization_potentials():
    assert len(IONIZATION_POTENTIAL_EV) == 103
    assert get_ionization_potential_ev(9) == 17.423
    # differentiated hydrogens (100 + bonding partner) are hydrogens
    assert get_ionization_potential_ev(106) == 13.598
    assert get_ionization_potential_ev(1) == 13.598
    with pytest.raises(ValueError):
        get_ionization_potential_ev(0)
    with pytest.raises(ValueError):
        get_ionization_potential_ev(103, hydrogens_are_differentiated=False)

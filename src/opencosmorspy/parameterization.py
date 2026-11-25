#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of COSMO-RS parameterization class.

@author: Thomas Gerlach, 2021
"""

import numpy as np

# First ionization potentials in eV for the atomic numbers 1 to 102. Index 0 is
# not used. Taken from the openCOSMO-RS C++ implementation (types.hpp).
IONIZATION_POTENTIAL_EV = np.array(
    [np.nan,
    13.598, 24.587, 5.392, 9.323, 8.298, 11.26,
    14.534, 13.618, 17.423, 21.565, 5.139, 7.646,
    5.986, 8.152, 10.487, 10.36, 12.968, 15.76,
    4.341, 6.113, 6.561, 6.828, 6.746, 6.767,
    7.434, 7.902, 7.881, 7.64, 7.726, 9.394,
    5.999, 7.9, 9.789, 9.752, 11.814, 14.0,
    4.177, 5.695, 6.217, 6.634, 6.759, 7.092,
    7.28, 7.361, 7.459, 8.337, 7.576, 8.994,
    5.786, 7.344, 8.608, 9.01, 10.451, 12.13,
    3.894, 5.212, 5.577, 5.539, 5.473, 5.525,
    5.582, 5.644, 5.67, 6.15, 5.864, 5.939,
    6.022, 6.108, 6.184, 6.254, 5.426, 6.825,
    7.55, 7.864, 7.834, 8.438, 8.967, 8.959,
    9.226, 10.438, 6.108, 7.417, 7.286, 8.417,
    9.318, 10.749, 4.073, 5.279, 5.17, 6.307,
    5.89, 6.194, 6.266, 6.027, 5.974, 5.992,
    6.198, 6.3, 6.42, 6.5, 6.58, 6.65,
    ]
)


def get_ionization_potential_ev(atomic_number, hydrogens_are_differentiated=True):
    """
    Return the first ionization potential in eV of a segment atomic number.

    With differentiated hydrogens, hydrogen is stored as 100 + the atomic
    number of the bonding partner. These are mapped back to hydrogen.
    """

    element = int(atomic_number)
    if hydrogens_are_differentiated and element > 100:
        element = 1

    if element < 1 or element >= len(IONIZATION_POTENTIAL_EV):
        raise ValueError(
            f"No ionization potential available for atomic number "
            f"{atomic_number} (valid range: 1 to "
            f"{len(IONIZATION_POTENTIAL_EV) - 1})."
            + (
                " An atomic number of 0 occurs if 'elmnt_nr' is not part of "
                "the descriptors, which cannot be combined with "
                "polariz_mode 7 or 8."
                if element == 0
                else ""
            )
        )

    return float(IONIZATION_POTENTIAL_EV[element])


class Parameterization(object):
    def __init__(self, name):

        if name == "default_turbomole":
            self._get_default_turbomole()
        if name == "default_orca":
            self._get_default_orca()
        self._get_extension_defaults()
        self.name = name

    def _get_extension_defaults(self):
        """
        Defaults of the model extensions ported from the openCOSMO-RS C++
        implementation. With these defaults the behavior of the model is
        unchanged compared to the versions without the extensions.
        """

        # Dispersion from atomic polarizabilities (Grigorash et al.,
        # Chem. Eng. Sci. 319 (2026) 122170). Corresponds to the C++ switch
        # sw_SR_polarizabilities.
        #   0 : no polarizabilities
        #   1 : unscaled projection, scaled by 1 / (number of segments of the atom)
        #   3 : projection scaled by the area fraction of the segment on its atom
        #   5 : projection without scaling
        #   6 : like 5, spatially averaged with r_av
        #   7 : like 6, dispersion energy with ionization potentials (product)
        #   8 : like 6, dispersion energy with ionization potentials (sum)
        self.polariz_mode = 0
        self.vdw_m = 0.0  # m_vdW, prefactor of the dispersion energy
        self.mf_e_f_corr = 0.0  # E_F_corr, fluorine correction of the misfit energy in J/mol

        # Segment descriptors. Corresponds to the C++ switches
        # sw_SR_differentiateHydrogens and sw_SR_differentiateMoleculeGroups.
        self.differentiate_hydrogens = True
        self.differentiate_molecule_groups = False

        # Only possible if "sigma_corr" is used instead of "sigma_orth" as
        # descriptor. Corresponds to sw_SR_misfit = 2: no misfit correlation for
        # monoatomic ions.
        self.mf_sigma_corr_exclude_monoatomic_ions = False

        # Reference state of the segment interaction matrix.
        #   "conductor"       : COSMO-RS default
        #   "pure_segment"    : the interaction of a segment with itself is zero
        self.segment_reference_state = "conductor"

        # Check that the cavity radii of the elements are identical in all COSMO
        # files. "error" as in the C++ implementation, "warn", or "off".
        self.radii_consistency = "warn"

        # Radii of monoatomic cations in Angstrom as {atomic_number: radius},
        # needed if differentiate_molecule_groups is True.
        self.ion_radii = {}

        # Further combinatorial terms
        self.comb_sg_expsc_exponent = getattr(self, "comb_sg_expsc_exponent", 0.75)
        self.comb_lambda0 = 0.0  # Klamt (2003), Franke and Hannebauer (2011)
        self.comb_lambda1 = 0.0
        self.comb_lambda2 = 0.0
        self.comb_sgg_lambda = 0.0  # Grensemann and Gmehling (2005)
        self.comb_sgg_beta = 0.0

        # Solvation free energy. Energies in kcal/mol, areas in Angstrom**2.
        # dgsolv_tau : {atomic_number: tau}. Hydrogens bonded to other atoms are
        # treated as hydrogen.
        self.dgsolv_tau = {}
        self.dgsolv_eta = 0.0
        self.dgsolv_omega_ring = 0.0
        self.dgsolv_tau_approx = 0.0262  # median of the parameterized values
        # True: elements without tau raise an error
        # False: elements without tau are approximated with dgsolv_tau_approx
        self.dgsolv_strict = True

    def _get_default_turbomole(self):

        self.qc_program = "turbomole"

        self.descriptor_lst = ["sigma", "sigma_orth", "elmnt_nr", "group", "mol_charge"]

        self.a_eff = 6.25  # Eckert and Klamt, 2002
        self.r_av = 0.5  # Klamt and Eckert, 2000
        self.sigma_min = -0.15
        self.sigma_max = 0.15
        self.sigma_step = 0.001  # Klamt, 1995?
        self.sigma_grid = np.arange(
            self.sigma_min, self.sigma_max + self.sigma_step, self.sigma_step
        )
        self.sigma_orth_min = -0.15
        self.sigma_orth_max = 0.15
        self.sigma_orth_step = 0.001
        self.sigma_orth_grid = np.arange(
            self.sigma_orth_min,
            self.sigma_orth_max + self.sigma_orth_step,
            self.sigma_orth_step,
        )

        self.mf_use_sigma_orth = True
        self.mf_alpha = 5950.0e3  # Eckert and Klamt, 2002
        self.mf_r_av_corr = 1.0  # Klamt et al., 1998
        self.mf_f_corr = 2.4  # Klamt et al., 1998

        self.hb_term = "default"
        self.hb_c = 36700.0e3  # Eckert and Klamt, 2002
        self.hb_c_T = 1.5  # Klamt and Eckert, 2000
        self.hb_sigma_thresh = 0.0085  # Eckert and Klamt, 2002
        self.hb_don_elmnt_nr_arr = np.array(
            [1] + [entry for entry in range(100, 151, 1)]
        )
        self.hb_acc_elmnt_nr_arr = np.array([6, 7, 8, 9, 15, 16, 17, 35, 53])
        self.hb_pref_arr = np.ones(201, dtype="float")

        self.comb_term = "staverman_guggenheim"
        self.comb_sg_z_coord = 10.0
        self.comb_sg_a_std = 79.53  # Lin and Sandler, 2002
        self.comb_sg_expsc_exponent = 0.75

        self.calculate_contact_statistics_molecule_properties = False

        self.cosmospace_max_iter = 1000
        self.cosmospace_conv_thresh = 1e-6

    def _get_default_orca(self):

        self.qc_program = "orca"

        self.descriptor_lst = ["sigma", "sigma_orth", "elmnt_nr", "group", "mol_charge"]

        self.a_eff = 6.226
        self.r_av = 0.5  # Klamt and Eckert, 2000
        self.sigma_min = -0.15
        self.sigma_max = 0.15
        self.sigma_step = 0.001  # Klamt, 1995?
        self.sigma_grid = np.arange(
            self.sigma_min, self.sigma_max + self.sigma_step, self.sigma_step
        )
        self.sigma_orth_min = -0.15
        self.sigma_orth_max = 0.15
        self.sigma_orth_step = 0.001
        self.sigma_orth_grid = np.arange(
            self.sigma_orth_min,
            self.sigma_orth_max + self.sigma_orth_step,
            self.sigma_orth_step,
        )

        self.mf_use_sigma_orth = True
        self.mf_alpha = 7.579075e6
        self.mf_r_av_corr = 1.0  # Klamt et al., 1998
        self.mf_f_corr = 2.4  # Klamt et al., 1998

        self.hb_term = "default"
        self.hb_c = 2.7488747e7
        self.hb_c_T = 1.5  # Klamt and Eckert, 2000
        self.hb_sigma_thresh = 7.686e-03
        self.hb_don_elmnt_nr_arr = np.array(
            [1] + [entry for entry in range(100, 151, 1)]
        )
        self.hb_acc_elmnt_nr_arr = np.array([6, 7, 8, 9, 15, 16, 17, 35, 53])
        self.hb_pref_arr = np.ones(201, dtype="float")

        self.comb_term = "staverman_guggenheim"
        self.comb_sg_z_coord = 10.0
        self.comb_sg_a_std = 47.999
        # self.comb_sg_expsc_exponent = 0.75

        self.calculate_contact_statistics_molecule_properties = False

        self.cosmospace_max_iter = 1000
        self.cosmospace_conv_thresh = 1e-6

    def check_parameterization(self):
        """Check the consistency of the parameterization."""

        if self.polariz_mode not in (0, 1, 3, 5, 6, 7, 8):
            raise ValueError(
                f"polariz_mode = {self.polariz_mode} is not supported. "
                f"Possible values are: 0, 1, 3, 5, 6, 7, 8."
            )

        if self.polariz_mode > 0:
            if "polariz" not in self.descriptor_lst:
                raise ValueError(
                    "polariz_mode > 0 requires 'polariz' in descriptor_lst."
                )
            if "elmnt_nr" not in self.descriptor_lst:
                raise ValueError(
                    "polariz_mode > 0 requires 'elmnt_nr' in descriptor_lst, as "
                    "the fluorine correction and the ionization potentials "
                    "depend on the element."
                )

        if "sigma_orth" in self.descriptor_lst and "sigma_corr" in self.descriptor_lst:
            raise ValueError("Use either sigma_orth or sigma_corr as descriptor.")

        if (
            self.mf_sigma_corr_exclude_monoatomic_ions
            and "sigma_corr" not in self.descriptor_lst
        ):
            raise ValueError(
                "mf_sigma_corr_exclude_monoatomic_ions requires 'sigma_corr' in "
                "descriptor_lst."
            )

        if self.segment_reference_state not in ("conductor", "pure_segment"):
            raise ValueError(
                f"Unknown segment_reference_state {self.segment_reference_state}"
            )

        if self.comb_term not in (
            "none",
            "staverman_guggenheim",
            "klamt",
            "staverman_guggenheim_exponential_scaling",
            "staverman_guggenheim_grensemann",
            "franke_hannebauer",
        ):
            raise ValueError(f"Unknown combinatorial term {self.comb_term}")

    def __str__(self):
        attributes = vars(self)
        max_key_length = max(len(key) for key in attributes)
        formatted_attributes = "\n".join(
            f"{key.rjust(max_key_length)} : {value}"
            for key, value in attributes.items()
        )
        return f"{self.name}:\n{formatted_attributes}"


class openCOSMORS24a(Parameterization):
    def __init__(self):
        super().__init__("default_orca")
        self.name = "24a"

        self.hb_don_elmnt_nr_arr = np.array([1] + [entry for entry in range(1, 151, 1)])
        self.hb_acc_elmnt_nr_arr = np.array([1] + [entry for entry in range(1, 151, 1)])
        self.hb_pref_arr = np.ones(201, dtype="float")

        # parameters different then default_orca
        self.a_eff = 5.9248470  # Å²
        self.mf_alpha = 7.2847361e06  # J/mol Å² e²
        self.hb_c = 4.3311555e07  # J/mol Å² e²
        self.hb_sigma_thresh = 9.6112460e-03  # e/Å²

        self.comb_sg_a_std = 4.1623570e01  # Å²

        # the solvation energy parameters
        # in the original optimization were in kcal/mol
        # as all of the data was in kcal/mol
        # values in the paper were given in kJ/mol
        # for consistency purposes
        self.tau_1 = 2.933803e-02  # kcal/mol Å²
        self.tau_6 = 2.287904e-02  # kcal/mol Å²
        self.tau_7 = 7.007681e-04  # kcal/mol Å²
        self.tau_8 = 3.545052e-03  # kcal/mol Å²
        self.tau_9 = 5.608829e-03  # kcal/mol Å²
        self.tau_17 = 3.414282e-02  # kcal/mol Å²
        self.tau_35 = 4.085111e-02  # kcal/mol Å²
        self.tau_14 = 4.215503e-03  # kcal/mol Å²
        self.tau_15 = 3.607977e-03  # kcal/mol Å²
        self.tau_16 = 3.498700e-02  # kcal/mol Å²
        self.eta = -4.448499e+00 # kcal/mol
        self.omega_ring = 2.6302510E-01  # kcal/mol

        self.dgsolv_tau = {
            1: self.tau_1,
            6: self.tau_6,
            7: self.tau_7,
            8: self.tau_8,
            9: self.tau_9,
            14: self.tau_14,
            15: self.tau_15,
            16: self.tau_16,
            17: self.tau_17,
            35: self.tau_35,
        }
        self.dgsolv_eta = self.eta
        self.dgsolv_omega_ring = self.omega_ring


# Hydrogen bond classes of the C++ implementation: potential donors are
# hydrogen, some metals and all hydrogens that are differentiated by their
# bonding partner (100 + atomic number). All other elements are potential
# acceptors.
_CPP_HB_DONOR_ELEMENTS = [1, 3, 4, 11, 12, 13, 19, 20, 24, 26, 27, 29, 30,
                          37, 38, 48, 55, 56]


class openCOSMORS25a(Parameterization):
    """
    openCOSMO-RS version 25a: dispersion from atomic polarizabilities.

    Latest parameterization of the openCOSMO-RS C++ implementation, which is
    Eq.(24)+w5(Eq.(11)) in Grigorash et al., Chem. Eng. Sci. 319 (2026) 122170
    (polariz_mode = 7). Requires ORCA COSMO files including the section
    #ATOMIC POLARIZABILITIES.

    The segments are clustered with sigma and sigma_corr (not sigma_orth), the
    hydrogen bonding classes of the elements and the combinatorial term (the
    Flory-Huggins term) are chosen as in the C++ implementation.
    """

    def __init__(self):
        super().__init__("default_orca")
        self.name = "25a"

        self.descriptor_lst = [
            "sigma",
            "sigma_corr",
            "elmnt_nr",
            "group",
            "mol_charge",
            "polariz",
        ]

        self.differentiate_hydrogens = False
        self.mf_sigma_corr_exclude_monoatomic_ions = True

        self.hb_don_elmnt_nr_arr = np.array(
            _CPP_HB_DONOR_ELEMENTS + list(range(101, 219))
        )
        self.hb_acc_elmnt_nr_arr = np.array(
            [i for i in range(1, 101) if i not in _CPP_HB_DONOR_ELEMENTS]
        )
        self.hb_pref_arr = np.ones(301, dtype="float")

        self.a_eff = 4.90825  # Å²
        self.r_av = 0.5  # Å
        self.mf_alpha = 7.876e6  # J/mol Å² e²
        self.mf_r_av_corr = 1.0  # Å
        self.mf_f_corr = 2.4
        self.hb_c = 4.9318e7  # J/mol Å² e²
        self.hb_c_T = 1.5
        self.hb_sigma_thresh = 9.953e-03  # e/Å²

        # Flory-Huggins
        self.comb_term = "staverman_guggenheim"
        self.comb_sg_z_coord = 0.0
        self.comb_sg_a_std = 1.0
        self.comb_sg_expsc_exponent = 2.0 / 3.0
        self.comb_lambda0 = 0.463
        self.comb_lambda1 = 0.42
        self.comb_lambda2 = 0.065
        self.comb_sgg_lambda = 0.773
        self.comb_sgg_beta = 0.778

        self.polariz_mode = 7
        self.vdw_m = 29.567
        self.mf_e_f_corr = 346.82  # J/mol

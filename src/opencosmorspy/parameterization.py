#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of COSMO-RS parameterization class.

@author: Thomas Gerlach, 2021
"""

import numpy as np


class Parameterization(object):
    def __init__(self, name):

        if name == "default_turbomole":
            self._get_default_turbomole()
        if name == "default_orca":
            self._get_default_orca()
        self.name = name

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
        pass

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

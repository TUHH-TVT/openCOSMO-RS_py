#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of COSMO-RS parameterization class.

@author: Thomas Gerlach, 2021
"""

import numpy as np


class Parameterization(object):
    def __init__(self, name):

        if name == 'default_turbomole':
            self._get_default_turbomole()
        if name == 'default_orca':
            self._get_default_orca()

    def _get_default_turbomole(self):

        self.qc_program = 'turbomole'

        self.descriptor_lst = ['sigma', 'sigma_orth', 'elmnt_nr', 'group',
                               'mol_charge']

        self.a_eff = 6.25   # Eckert and Klamt, 2002
        self.r_av = 0.5   # Klamt and Eckert, 2000
        self.sigma_min = -0.15
        self.sigma_max = 0.15
        self.sigma_step = 0.001   # Klamt, 1995?
        self.sigma_grid = (np.arange(self.sigma_min,
                                     self.sigma_max+self.sigma_step,
                                     self.sigma_step))
        self.sigma_orth_min = -0.15
        self.sigma_orth_max = 0.15
        self.sigma_orth_step = 0.001
        self.sigma_orth_grid = (
            np.arange(self.sigma_orth_min,
                      self.sigma_orth_max+self.sigma_orth_step,
                      self.sigma_orth_step))

        self.mf_use_sigma_orth = True
        self.mf_alpha = 5950.0e3   # Eckert and Klamt, 2002
        self.mf_r_av_corr = 1.0   # Klamt et al., 1998
        self.mf_f_corr = 2.4   # Klamt et al., 1998

        self.hb_term = 'default'
        self.hb_c = 36700.0e3   # Eckert and Klamt, 2002
        self.hb_c_T = 1.5   # Klamt and Eckert, 2000
        self.hb_sigma_thresh = 0.0085   # Eckert and Klamt, 2002
        self.hb_don_elmnt_nr_arr = np.array(
            [1]+[entry for entry in range(100, 151, 1)])
        self.hb_acc_elmnt_nr_arr = np.array([6, 7, 8, 9,
                                             15, 16, 17,
                                             35, 53])
        self.hb_pref_arr = np.ones(201, dtype='float')

        self.comb_term = 'staverman_guggenheim'
        self.comb_sg_z_coord = 10.0
        self.comb_sg_a_std = 79.53   # Lin and Sandler, 2002
        self.comb_sg_expsc_exponent = 0.75

        self.calculate_contact_statistics_molecule_properties = False

        self.cosmospace_max_iter = 1000
        self.cosmospace_conv_thresh = 1e-6


    def _get_default_orca(self):

        self.qc_program = 'orca'

        self.descriptor_lst = ['sigma', 'sigma_orth', 'elmnt_nr', 'group',
                               'mol_charge']

        self.a_eff = 6.226
        self.r_av = 0.5   # Klamt and Eckert, 2000
        self.sigma_min = -0.15
        self.sigma_max = 0.15
        self.sigma_step = 0.001   # Klamt, 1995?
        self.sigma_grid = (np.arange(self.sigma_min,
                                     self.sigma_max+self.sigma_step,
                                     self.sigma_step))
        self.sigma_orth_min = -0.15
        self.sigma_orth_max = 0.15
        self.sigma_orth_step = 0.001
        self.sigma_orth_grid = (
            np.arange(self.sigma_orth_min,
                      self.sigma_orth_max+self.sigma_orth_step,
                      self.sigma_orth_step))

        self.mf_use_sigma_orth = True
        self.mf_alpha = 7.579075e6
        self.mf_r_av_corr = 1.0   # Klamt et al., 1998
        self.mf_f_corr = 2.4   # Klamt et al., 1998

        self.hb_term = 'default'
        self.hb_c = 2.7488747e7
        self.hb_c_T = 1.5   # Klamt and Eckert, 2000
        self.hb_sigma_thresh = 7.686e-03
        self.hb_don_elmnt_nr_arr = np.array(
            [1]+[entry for entry in range(100, 151, 1)])
        self.hb_acc_elmnt_nr_arr = np.array([6, 7, 8, 9,
                                             15, 16, 17,
                                             35, 53])
        self.hb_pref_arr = np.ones(201, dtype='float')

        self.comb_term = 'staverman_guggenheim'
        self.comb_sg_z_coord = 10.0
        self.comb_sg_a_std = 47.999
        # self.comb_sg_expsc_exponent = 0.75

        self.calculate_contact_statistics_molecule_properties = False

        self.cosmospace_max_iter = 1000
        self.cosmospace_conv_thresh = 1e-6


    def check_parameterization(self):
        pass

    def __str__(self):

        text = 'qc_program             : {}\n'.format(self.qc_program)
        text += ('descriptors            : ' +
                 ', '.join(self.descriptor_lst)+' \n')
        text += 'a_eff                  : {:.3f}\n'.format(self.a_eff)
        text += 'r_av                   : {:.3f}\n'.format(self.r_av)
        text += 'sigma_step             : {}\n'.format(self.sigma_step)
        text += 'sigma_orth_step        : {}\n'.format(self.sigma_orth_step)
        text += 'mf_use_sigma_orth      : {}\n'.format(self.mf_use_sigma_orth)
        text += 'mf_alpha               : {:13.6e}\n'.format(self.mf_alpha)
        text += 'mf_r_av_corr           : {:.3f}\n'.format(self.mf_r_av_corr)
        text += 'mf_f_corr              : {:.3f}\n'.format(self.mf_f_corr)
        text += 'hb_term                : {}\n'.format(self.hb_term)
        text += 'hb_c                   : {:13.6e}\n'.format(self.hb_c)
        text += 'hb_c_T                 : {:.3f}\n'.format(self.hb_c_T)
        text += 'hb_sigma_thresh        : {:.6f}\n'.format(
            self.hb_sigma_thresh)
        text += 'hb_don_elmnt_nr_arr    : {}\n'.format(
            self.hb_don_elmnt_nr_arr)
        text += 'hb_acc_elmnt_nr_arr    : {}\n'.format(
            self.hb_acc_elmnt_nr_arr)
        text += 'hb_pref_arr            : {}\n'.format(self.hb_pref_arr)
        text += 'comb_term              : {}\n'.format(self.comb_term)
        text += 'comb_sg_z_coord        : {:3f}\n'.format(self.comb_sg_z_coord)
        text += 'comb_sg_a_std          : {:4f}\n'.format(self.comb_sg_a_std)
        if self.comb_term == 'staverman_guggenheim_exponential_scaling':
            text += ('comb_sg_expsc_exponent : {:4f}\n'.
                     format(self.comb_sg_expsc_exponent))
    
        text += 'calculate_contact_statistics_molecule_properties : {}\n'.format(
            self.calculate_contact_statistics_molecule_properties)
        
        text += 'cosmospace_max_iter : {}\n'.format(
            self.cosmospace_max_iter)
        text += 'cosmospace_conv_thresh : {:.6e}\n'.format(
            self.cosmospace_conv_thresh)
        
        return text

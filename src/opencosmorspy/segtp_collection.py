#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 22:07:53 2021

@author: lergos
"""
import numpy as np
import pandas as pd


from opencosmorspy.parameterization import Parameterization


class SegtpCollection(object):
    def __init__(self, par):

        self.par = par
        self.segtp_lst = []

    def __iter__(self):
        return iter(self.segtp_lst)

    def __len__(self):
        return self.segtp_lst.__len__()

    def __getitem__(self, key):
        return self.segtp_lst.__getitem__(key)

    def cluster_cosmo_struct(self, cosmo_struct, sigma_grid, sigma_orth_grid):

        seg_segtp_assignment_lst = []

        for idx_seg, (sigma, sigma_orth, elmnt_nr, group) in enumerate(
            zip(
                cosmo_struct.seg_sigma,
                cosmo_struct.seg_sigma_orth,
                cosmo_struct.seg_elmnt_nr,
                cosmo_struct.seg_group,
            )
        ):

            if sigma < 0 and elmnt_nr in self.par.hb_don_elmnt_nr_arr:
                hbd_sw = True
            else:
                hbd_sw = False

            if sigma >= 0 and elmnt_nr in self.par.hb_acc_elmnt_nr_arr:
                hba_sw = True
            else:
                hba_sw = False

            segment = {
                "sigma": sigma,
                "sigma_orth": sigma_orth,
                "elmnt_nr": elmnt_nr,
                "hbd_sw": hbd_sw,
                "hba_sw": hba_sw,
                "group": group,
                "mol_charge": round(-cosmo_struct.screen_charge),
            }

            segtp_lst_new, segtp_frac_lst = cluster_segment(
                segment, sigma_grid, sigma_orth_grid, self.par.descriptor_lst
            )

            seg_segtp_assignment = {}

            for segtp, segtp_frac in zip(segtp_lst_new, segtp_frac_lst):
                try:
                    idx_segtp = self.segtp_lst.index(segtp)
                except Exception:
                    idx_segtp = len(self.segtp_lst)
                    self.segtp_lst.append(segtp)

                seg_segtp_assignment[idx_segtp] = segtp_frac

            seg_segtp_assignment_lst.append(seg_segtp_assignment)

        cosmo_struct.associate_segtps(seg_segtp_assignment_lst)

    def get_segtps_as_array_dct(self):

        segtp_arr_dct = {}
        descriptors = self.segtp_lst[0].keys()
        for descriptor in descriptors:
            segtp_arr_dct[descriptor] = np.array(
                [segtp[descriptor] for segtp in self.segtp_lst]
            )

        return segtp_arr_dct


def cluster_segment(segment, sigma_grid, sigma_orth_grid, descriptor_lst):
    """

    segment = {
        'sigma': 0.003,
        'sigma_orth': 0.004,
        'elmnt_nr': 0.002,
        'group': None,
        'mol_charge': 0}
    """

    handled_descriptors = ["sigma", "sigma_orth", "elmnt_nr", "group", "mol_charge"]

    unhandled_descriptors = list(set(descriptor_lst) - set(handled_descriptors))

    if unhandled_descriptors:
        raise ValueError("Unhandled descriptors encountered")

    if "sigma" not in descriptor_lst:
        raise ValueError("Sigma must be included as descriptor")

    segtp_frac_lst = [1.0]
    segtp_lst = [segment]

    if "sigma" in descriptor_lst:
        segtp_lst, segtp_frac_lst = _cluster_float_var(
            "sigma", sigma_grid, segtp_lst, segtp_frac_lst
        )

    if "sigma_orth" in descriptor_lst:
        segtp_lst, segtp_frac_lst = _cluster_float_var(
            "sigma_orth", sigma_orth_grid, segtp_lst, segtp_frac_lst
        )

    # Delete unused descriptors
    unused_descriptors = list(set(descriptor_lst) - set(handled_descriptors))
    for descriptor in unused_descriptors:
        for segtp in segtp_lst:
            del segtp[descriptor]

    return segtp_lst, segtp_frac_lst


def _cluster_float_var(var_name, var_grid, segtp_lst, segtp_frac_lst):

    if var_name not in segtp_lst[0]:
        raise ValueError("Cannot cluster unknown variable {:s}".format(var_name))

    gridstep = np.abs(var_grid[1] - var_grid[0])

    segtp_lst_new = []
    segtp_frac_lst_new = []

    for segtp_frac, segtp in zip(segtp_frac_lst, segtp_lst):

        val = segtp[var_name]
        idx_right = np.searchsorted(var_grid, val, side="left")
        idx_left = idx_right - 1
        val_right = var_grid[idx_right]
        val_left = var_grid[idx_left]

        if idx_left < 0 or idx_right > len(var_grid) - 1:
            raise ValueError("Encountered {} outside of bins".format(var_name))

        frac_left = (val_right - val) / gridstep
        segtp_frac_lst_new.append(segtp_frac * frac_left)
        segtp_lst_new.append(segtp.copy())
        segtp_lst_new[-1][var_name] = val_left

        segtp_frac_lst_new.append(segtp_frac * (1 - frac_left))
        segtp_lst_new.append(segtp.copy())
        segtp_lst_new[-1][var_name] = val_right

    return segtp_lst_new, segtp_frac_lst_new


if __name__ == "__main__":

    pass

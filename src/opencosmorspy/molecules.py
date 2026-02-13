#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of molecule classes

@author: Thomas Gerlach, 2021
"""
import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

import opencosmorspy.helper_functions as hf
from opencosmorspy.input_parsers import SigmaProfileParser
from opencosmorspy.parameterization import Parameterization
from opencosmorspy.segtp_collection import SegtpCollection


class Molecule(object):
    def __init__(self, filepath_lst):

        if len(filepath_lst) > 1:
            raise NotImplementedError("More than one conformer not supported")

        self.cosmo_struct_lst = []
        for path in filepath_lst:
            cosmo_info = SigmaProfileParser(path)
            self.cosmo_struct_lst.append(COSMOStruct(cosmo_info))
            self.cosmo_struct_lst[-1]

    def convert_properties(self, r_av, mf_r_av_corr):

        for cosmo_struct in self.cosmo_struct_lst:
            cosmo_struct.convert_properties(r_av, mf_r_av_corr)

    def get_segtp_area_dct(self):

        return self.cosmo_struct_lst[0].segtp_area_dct

    def get_segtp_nseg_dct(self, a_eff):

        segtp_nseg_dct = {}

        for idxstp, area in self.cosmo_struct_lst[0].segtp_area_dct.items():
            segtp_nseg_dct[idxstp] = area / a_eff

        return segtp_nseg_dct

    def get_segtp_nseg_arr(self, a_eff, n_segtp):

        segtp_nseg_arr = np.zeros(n_segtp, dtype="float64")

        segtp_nseg_dct = self.get_segtp_nseg_dct(a_eff)

        for idxstp, n_seg in segtp_nseg_dct.items():
            segtp_nseg_arr[idxstp] = n_seg

        return segtp_nseg_arr

    def get_area(self):

        return self.cosmo_struct_lst[0].area

    def get_volume(self):

        return self.cosmo_struct_lst[0].volume


class COSMOStruct(object):
    def __init__(self, cosmo_info: dict):
        """
        self.cosmo_info = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'qc_program': qc_program,
                'method': '',
                'area': None,
                'volume': None,
                'energy_tot': None,
                'energy_dielectric': None,
                'atm_nr': [],
                'atm_pos': [],
                'atm_elmnt': [],
                'atm_rad': [],
                'seg_nr': [],
                'seg_atm_nr': [],
                'seg_pos': [],
                'seg_charge': [],
                'seg_area': [],
                'seg_sigma_raw': [],
                'seg_potential': []
                }
        """

        self.filepath = cosmo_info["filepath"]
        self.filename = cosmo_info["filename"]

        self.energy_tot = cosmo_info["energy_tot"]
        self.energy_dielectric = cosmo_info["energy_dielectric"]

        self.area = cosmo_info["area"]
        self.volume = cosmo_info["volume"]

        self.atm_elmnt = np.array([elmnt.lower() for elmnt in cosmo_info["atm_elmnt"]])

        self.atm_pos = cosmo_info["atm_pos"]
        self.atm_rad = cosmo_info["atm_rad"]
        self.seg_nr = cosmo_info["seg_nr"]
        self.seg_atm_nr = cosmo_info["seg_atm_nr"]
        self.seg_pos = cosmo_info["seg_pos"]
        self.seg_charge = cosmo_info["seg_charge"]
        self.seg_area = cosmo_info["seg_area"]
        self.seg_sigma_raw = (
            cosmo_info["seg_sigma_raw"]
            if len(cosmo_info["seg_sigma_raw"]) != 0
            else cosmo_info["seg_sigma_raw_uncorrected"]
        )
        self.seg_potential = cosmo_info["seg_potential"]

        # Define missing properties
        self.seg_sigma = None
        self.seg_sigma_orth = None
        self.atm_elmnt_nr = None
        self.seg_elmnt_nr = None
        self.seg_group = None
        self.screen_charge = None

        self.seg_segtp_assignment_lst = []
        self.segtp_area_dct = {}

    def convert_properties(self, r_av, mf_r_av_corr):

        self.atm_elmnt_nr = _convert_element_symbols(self.atm_elmnt)

        # Modify element numbers of hydrogen to account for bond
        self._convert_hydrogen_element_number()

        # Add element numbers to segments
        self.seg_elmnt_nr = self.atm_elmnt_nr[self.seg_atm_nr]

        # Assign groups to segment
        self.seg_group = np.array(["default"] * len(self.seg_nr))

        # Calculate total screening charge
        self.screen_charge = (self.seg_area * self.seg_sigma_raw).sum()

        # Calculate averaged sigma and sigma_orth
        self.seg_sigma, dist_sq_arr = sigma_averaging(
            r_av, self.seg_pos, self.seg_area, self.seg_sigma_raw
        )

        # Calculate sigma_orthogonal
        seg_sigma_corr, _ = sigma_averaging(
            mf_r_av_corr,
            self.seg_pos,
            self.seg_area,
            self.seg_sigma_raw,
            dist_sq_arr=dist_sq_arr,
        )
        self.seg_sigma_orth = seg_sigma_corr - 0.816 * self.seg_sigma

        # print('WARNING: OVERWRITING SIGMA ORTH FOR TESTS')
        # self.seg_sigma_orth = seg_sigma_corr

    def _convert_hydrogen_element_number(self):

        dist_sq_arr = hf.calculate_squared_distances(
            self.atm_pos, diagonal_element=np.nan
        )

        for idx, elmnt in enumerate(self.atm_elmnt_nr):
            if elmnt == 1:

                idx_bnd = np.nanargmin(dist_sq_arr[idx, :])

                elmnt_nr_new = 100 + self.atm_elmnt_nr[idx_bnd]

                assert elmnt_nr_new != 101

                self.atm_elmnt_nr[idx] = elmnt_nr_new

    def associate_segtps(self, seg_segtp_assignment_lst):
        """


        Parameters
        ----------
        seg_segtp_assignment_lst : List of dict. required
            Each dictionary corresponds to a segment and contains
            index_of_segtp_in_collection:fraction_of_seg_in_segtp
            pairs

        Returns
        -------
        None.

        """
        self.seg_segtp_assignment_lst = seg_segtp_assignment_lst

        segtp_area_dct = {}

        assert len(seg_segtp_assignment_lst) == len(self.seg_area)

        for area, assignment in zip(self.seg_area, self.seg_segtp_assignment_lst):
            for key, value in assignment.items():
                if key in segtp_area_dct:
                    segtp_area_dct[key] += area * value
                else:
                    segtp_area_dct[key] = area * value

        self.segtp_area_dct = segtp_area_dct


def sigma_averaging(r_av, seg_pos, seg_area, seg_sigma_raw, dist_sq_arr=None):
    """
    Creates averaged sigmas from raw sigmas

    Arguments:
    ---------
    r_av: float. required.
        Averaging radius in Angstrom

    Returns:
    --------
    None

    """

    r_seg_sq = (1 / np.pi) * seg_area
    r_av_sq = r_av**2

    inv_rad_arr = 1 / (r_seg_sq + r_av_sq)

    if dist_sq_arr is None:
        dist_sq_arr = hf.calculate_squared_distances(seg_pos)

    exp_arr = np.exp(-dist_sq_arr * inv_rad_arr)

    buff_arr = (r_av_sq * inv_rad_arr * exp_arr).T

    # sigma_av = ((seg_sigma_raw*fac_arr).sum(axis=1) /
    #             fac_arr.sum(axis=1))
    sigma_av = (seg_sigma_raw * r_seg_sq).dot(buff_arr) / r_seg_sq.dot(buff_arr)

    do_test = False
    if do_test:
        n_seg = seg_pos.shape[0]
        sigma_av_test = np.zeros(n_seg)
        dist_sq_arr_test = np.zeros((n_seg, n_seg))
        for idx1 in range(n_seg):
            buffdb1 = 0
            sigma_av_test[idx1] = 0
            # r_seg_sq_idx1_test = 1/np.pi*seg_area[idx1]
            for idx2 in range(n_seg):
                dist_sq_arr_test[idx1, idx2] = (
                    (seg_pos[idx1, 0] - seg_pos[idx2, 0]) ** 2
                    + (seg_pos[idx1, 1] - seg_pos[idx2, 1]) ** 2
                    + (seg_pos[idx1, 2] - seg_pos[idx2, 2]) ** 2
                )

                r_seg_sq_idx2_test = 1 / np.pi * seg_area[idx2]

                buffdb3 = r_seg_sq_idx2_test + r_av_sq

                av_fac = (
                    r_seg_sq_idx2_test
                    * r_av_sq
                    / buffdb3
                    * np.exp(-dist_sq_arr_test[idx1, idx2] / buffdb3)
                )

                buffdb1 = buffdb1 + av_fac

                sigma_av_test[idx1] += seg_sigma_raw[idx2] * av_fac

            sigma_av_test[idx1] /= buffdb1

            sigma_av = sigma_av_test

    return sigma_av, dist_sq_arr


def _convert_element_symbols(elmnt_lst):

    elmnt_dct = {
        "h": 1, "he": 2, "li": 3, "be": 4, "b": 5, "c": 6, "n": 7, "o": 8, "f": 9, "ne": 10,
        "na": 11, "mg": 12, "al": 13, "si": 14, "p": 15, "s": 16, "cl": 17, "ar": 18, "k": 19, "ca": 20,
        "sc": 21, "ti": 22, "v": 23, "cr": 24, "mn": 25, "fe": 26, "co": 27, "ni": 28, "cu": 29, "zn": 30,
        "ga": 31, "ge": 32, "as": 33, "se": 34, "br": 35, "kr": 36, "rb": 37, "sr": 38, "y": 39, "zr": 40,
        "nb": 41, "mo": 42, "tc": 43, "ru": 44, "rh": 45, "pd": 46, "ag": 47, "cd": 48, "in": 49, "sn": 50,
        "sb": 51, "te": 52, "i": 53, "xe": 54, "cs": 55, "ba": 56, "la": 57, "ce": 58, "pr": 59, "nd": 60,
        "pm": 61, "sm": 62, "eu": 63, "gd": 64, "tb": 65, "dy": 66, "ho": 67, "er": 68, "tm": 69, "yb": 70,
        "lu": 71, "hf": 72, "ta": 73, "w": 74, "re": 75, "os": 76, "ir": 77, "pt": 78, "au": 79, "hg": 80,
        "tl": 81, "pb": 82, "bi": 83, "po": 84, "at": 85, "rn": 86, "fr": 87, "ra": 88, "ac": 89, "th": 90,
        "pa": 91, "u": 92, "np": 93, "pu": 94, "am": 95, "cm": 96, "bk": 97, "cf": 98, "es": 99, "fm": 100,
        "md": 101, "no": 102, "lr": 103, "rf": 104, "db": 105, "sg": 106, "bh": 107, "hs": 108, "mt": 109,
        "ds": 110, "rg": 111, "cn": 112, "nh": 113, "fl": 114, "mc": 115, "lv": 116, "ts": 117, "og": 118
    }

    elmnt_lst_lower = [elmnt.lower() for elmnt in elmnt_lst]

    missing_elements = set(elmnt_lst_lower) - set(elmnt_dct.keys())
    if missing_elements:
        raise ValueError("Unknown elements")

    elmnt_nr_lst = [elmnt_dct[elmnt] for elmnt in elmnt_lst]

    return np.array(elmnt_nr_lst, dtype="int_")


def helper_create_extsp(path_inp, qc_program):

    if qc_program == "turbomole":
        par = Parameterization("default_turbomole")
    else:
        par = Parameterization("default_orca")

    mol = Molecule([path_inp], qc_program)
    mol.convert_properties(par.r_av, par.mf_r_av_corr)

    segtp_col = SegtpCollection(par)

    segtp_col.cluster_cosmo_struct(
        mol.cosmo_struct_lst[0], par.sigma_grid, par.sigma_orth_grid
    )
    print("N_clusters", len(segtp_col.segtp_lst))

    # cosmo_struct = mol.cosmo_struct_lst[0]
    df_esp = pd.DataFrame(segtp_col.get_segtps_as_array_dct())
    df_esp["area"] = 0.0
    df_esp.loc[list(mol.cosmo_struct_lst[0].segtp_area_dct.keys()), "area"] = list(
        mol.cosmo_struct_lst[0].segtp_area_dct.values()
    )

    return df_esp


def helper_print_segment_clusters(crs):

    segtp_col = crs.enth.segtp_collection
    print("N_clusters", len(segtp_col.segtp_lst))
    df_esp = pd.DataFrame(segtp_col.get_segtps_as_array_dct())

    return df_esp


if __name__ == "__main__":

    import os

    filepath_lst = []
    filepath_lst.append(
        os.path.abspath(
            r"./../../000_publication/COSMO_ORCA"
            r"/C2H6O_001_ethanol/COSMO_TZVPD"
            r"/C2H6O_001_ethanol_CnfS1_c000.orcacosmo"
        )
    )
    # qc_program_dct[filepath_lst[-1]] = 'orca'
    # plot_label_dct[filepath_lst[-1]] = 'ORCA'

    mol_orca = Molecule(filepath_lst, qc_program="orca")
    struct_orca = mol_orca.cosmo_struct_lst[0]
    print("ORCA ETOH", (struct_orca.seg_area * struct_orca.seg_sigma_raw).sum())

    filepath_lst = []
    filepath_lst.append(
        os.path.abspath(
            r"./../../000_publication/COSMO_TMOLE"
            r"/C2H6O_001_ethanol/COSMO_TZVP"
            r"/C2H6O_001_ethanol_CnfS1_c000.cosmo"
        )
    )
    mol_tmole = Molecule(filepath_lst, qc_program="turbomole")
    struct_tmole = mol_tmole.cosmo_struct_lst[0]
    print("TMOLE ETOH", (struct_tmole.seg_area * struct_tmole.seg_sigma_raw).sum())

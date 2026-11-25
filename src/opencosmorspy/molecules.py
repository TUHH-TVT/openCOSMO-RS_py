#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of molecule classes

@author: Thomas Gerlach, 2021
"""
import collections
import warnings

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
from opencosmorspy.input_parsers import angstrom_per_bohr

# Constants of the C++ implementation, used for the projection of the atomic
# polarizabilities, because the projections are rounded to a grid.
BOHR_CPP = 0.529177249  # to convert atom positions from Angstrom to bohr
BOHR_DISTANCE_CPP = 0.52917721067121  # to convert distances in the averaging

POLARIZ_MODES = (0, 1, 3, 5, 6, 7, 8)


def round_half_away(values):
    """Round like C round(): halfway cases are rounded away from zero."""

    values = np.asarray(values, dtype="float64")
    truncated = np.trunc(values)
    return truncated + np.where(
        np.abs(values - truncated) >= 0.5, np.sign(values), 0.0
    )


def round_and_truncate(values, n):
    """Round to multiples of 10**n (as round_and_truncate of the C++ code)."""

    factor = 10.0**n
    return round_half_away(np.asarray(values, dtype="float64") / factor) * factor


class Molecule(object):
    def __init__(self, filepath_lst):

        if len(filepath_lst) > 1:
            raise NotImplementedError("More than one conformer not supported")

        self.cosmo_struct_lst = []
        for path in filepath_lst:
            cosmo_info = SigmaProfileParser(path)
            self.cosmo_struct_lst.append(COSMOStruct(cosmo_info))
            self.cosmo_struct_lst[-1]

    def convert_properties(self, r_av, mf_r_av_corr, par=None):

        for cosmo_struct in self.cosmo_struct_lst:
            cosmo_struct.convert_properties(r_av, mf_r_av_corr, par)

    def get_molar_volume_25c(self, strict=True):
        """Molar volume at 25 C in cm**3/mol from the QSPR model (for dGsolv)."""

        return self.cosmo_struct_lst[0].calculate_molar_volume_25c(strict)

    def get_energy_tot(self):

        return self.cosmo_struct_lst[0].energy_tot

    def get_charge(self):

        return self.cosmo_struct_lst[0].mol_charge

    def get_group(self):

        return self.cosmo_struct_lst[0].mol_group

    def get_n_atoms(self):

        return len(self.cosmo_struct_lst[0].atm_elmnt)

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

        self.method = cosmo_info["method"] if "method" in cosmo_info else ""
        self.polarizability_tensor = (
            cosmo_info["polarizability_tensor"]
            if "polarizability_tensor" in cosmo_info
            else None
        )

        # Define missing properties
        self.seg_sigma = None
        self.seg_sigma_orth = None
        self.atm_elmnt_nr = None
        self.seg_elmnt_nr = None
        self.seg_group = None
        self.screen_charge = None
        self.seg_sigma_corr = None
        self.seg_polariz = None
        self.seg_corr_unbinned = None
        self.mol_charge = None
        self.mol_group = None
        self.area_cosmo_file = self.area
        self.volume_cosmo_file = self.volume

        self.seg_segtp_assignment_lst = []
        self.segtp_area_dct = {}

        self._store_original_segments()

    def _store_original_segments(self):
        """Keep the data that is replaced for monoatomic cations."""

        self._original_segment_data = {
            key: getattr(self, key)
            for key in (
                "area",
                "volume",
                "seg_nr",
                "seg_atm_nr",
                "seg_pos",
                "seg_charge",
                "seg_area",
                "seg_sigma_raw",
                "seg_potential",
            )
        }

    def _restore_original_segments(self):

        for key, value in self._original_segment_data.items():
            setattr(self, key, value)

    def convert_properties(self, r_av, mf_r_av_corr, par=None):
        """
        Calculate the segment properties needed for the clustering

        Parameters
        ----------
        r_av : float
            Averaging radius for sigma in Angstrom.
        mf_r_av_corr : float
            Averaging radius for the sigma correlation in Angstrom.
        par : Parameterization, optional
            If given, the options of the extensions (hydrogen differentiation,
            molecule groups, polarizabilities, misfit correlation of ions) are
            used. If not given, the defaults of the original implementation
            are used.
        """

        self._restore_original_segments()

        differentiate_hydrogens = (
            True if par is None else par.differentiate_hydrogens
        )
        differentiate_molecule_groups = (
            False if par is None else par.differentiate_molecule_groups
        )
        polariz_mode = 0 if par is None else par.polariz_mode
        use_corr = True if par is None else par.mf_use_sigma_orth
        exclude_ion_corr = (
            False if par is None else par.mf_sigma_corr_exclude_monoatomic_ions
        )

        self.atm_elmnt_nr = _convert_element_symbols(self.atm_elmnt)
        n_atoms = len(self.atm_elmnt_nr)
        n_seg = len(self.seg_nr)

        # Calculate total screening charge and the charge of the molecule
        self.screen_charge = (self.seg_area * self.seg_sigma_raw).sum()
        self.mol_charge = int(round_half_away(-self.screen_charge))

        # Assign groups to segments
        #   0 monoatomic neutral, 1 polyatomic neutral, 2 water,
        #   3 monoatomic cation, 4 polyatomic cation,
        #   5 monoatomic anion, 6 polyatomic anion
        if differentiate_molecule_groups:
            self.mol_group = self._get_molecule_group(
                self.atm_elmnt_nr, self.mol_charge
            )
            self.seg_group = np.array([self.mol_group] * n_seg)
        else:
            self.mol_group = None
            self.seg_group = np.array(["default"] * n_seg)

        # Modify element numbers of hydrogen to account for bond
        if differentiate_hydrogens:
            self._convert_hydrogen_element_number()

        # Add element numbers to segments
        self.seg_elmnt_nr = self.atm_elmnt_nr[self.seg_atm_nr]

        # Calculate averaged sigma and sigma_orth
        self.seg_sigma, dist_sq_arr = sigma_averaging(
            r_av, self.seg_pos, self.seg_area, self.seg_sigma_raw
        )

        # Calculate sigma_corr and sigma_orthogonal
        no_corr = (not use_corr) or (
            exclude_ion_corr and self.mol_charge != 0 and n_atoms == 1
        )
        if not no_corr:
            self.seg_sigma_corr, _ = sigma_averaging(
                mf_r_av_corr,
                self.seg_pos,
                self.seg_area,
                self.seg_sigma_raw,
                dist_sq_arr=dist_sq_arr,
            )
        else:
            self.seg_sigma_corr = np.zeros(n_seg)
        self.seg_sigma_orth = self.seg_sigma_corr - 0.816 * self.seg_sigma

        # sigma_corr of monoatomic ions and without misfit correlation is not
        # clustered, it stays zero
        self.seg_corr_unbinned = np.full(
            n_seg,
            bool(
                no_corr
                or (differentiate_molecule_groups and self.mol_group in (3, 5))
            ),
        )

        # Projection of the atomic polarizabilities on the segments
        if polariz_mode != 0:
            self.seg_polariz = self.calculate_polariz_projection(polariz_mode, r_av)
        else:
            self.seg_polariz = np.zeros(n_seg)

        # Monoatomic cations are represented by a sphere of the given radius
        if differentiate_molecule_groups and self.mol_group == 3:
            self._rescale_monoatomic_cation(par)

    @staticmethod
    def _get_molecule_group(atm_elmnt_nr, mol_charge):

        n_atoms = len(atm_elmnt_nr)
        if (
            n_atoms == 3
            and np.sum(atm_elmnt_nr == 1) == 2
            and np.sum(atm_elmnt_nr == 8) == 1
        ):
            return 2
        if mol_charge == 0:
            return 0 if n_atoms == 1 else 1
        if mol_charge > 0:
            return 3 if n_atoms == 1 else 4
        return 5 if n_atoms == 1 else 6

    def _rescale_monoatomic_cation(self, par):

        n_atoms = len(self.atm_elmnt_nr)
        atomic_number = int(self.atm_elmnt_nr[0])
        radius = par.ion_radii.get(atomic_number, 0.0) if n_atoms == 1 else 0.0
        if not radius > 0:
            raise ValueError(
                f"No radius was given for the monoatomic cation {self.filename} "
                f"(atomic number {atomic_number}). With "
                f"differentiate_molecule_groups the radius of every monoatomic "
                f"cation has to be given in the parameter ion_radii."
            )

        new_area = 4 * np.pi * radius**2
        self.area = new_area
        self.volume = 4.0 / 3 * np.pi * radius**3

        # one segment with the screening charge of the ion on the sphere
        new_sigma = -self.mol_charge / new_area
        self.seg_nr = np.array([0])
        self.seg_atm_nr = np.array([0])
        self.seg_pos = self.seg_pos[:1]
        self.seg_area = np.array([new_area])
        self.seg_sigma_raw = np.array([new_sigma])
        self.seg_sigma = np.array([new_sigma])
        self.seg_sigma_corr = np.zeros(1)
        self.seg_sigma_orth = -0.816 * self.seg_sigma
        self.seg_corr_unbinned = np.array([True])
        self.seg_elmnt_nr = self.seg_elmnt_nr[:1]
        self.seg_group = self.seg_group[:1]
        self.seg_polariz = self.seg_polariz[:1]

    def calculate_polariz_projection(self, mode, r_av):
        """
        Project the atomic polarizability tensors on the segments

        Port of projectSegments of the C++ implementation. The projections are
        rounded to grids on purpose, the parameters were fitted with these
        grids (Grigorash et al., Chem. Eng. Sci. 319 (2026) 122170).

        Parameters
        ----------
        mode : int
            Projection mode, see Parameterization.polariz_mode.
        r_av : float
            Averaging radius for the modes 6 to 8 in Angstrom.

        Returns
        -------
        np.array(n_seg)
            Projected polarizabilities (a.u.) of all segments.
        """

        if mode not in POLARIZ_MODES:
            raise ValueError(
                f"polariz_mode = {mode} is not supported. Possible values are: "
                f"{POLARIZ_MODES}."
            )

        n_atoms = len(self.atm_elmnt)
        tensors = self.polarizability_tensor
        if tensors is None or np.shape(tensors) != (n_atoms, 6):
            raise ValueError(
                f"The COSMO file {self.filename} does not contain atomic "
                f"polarizabilities for all {n_atoms} atoms, which are needed "
                f"for polariz_mode > 0. Please use ORCA COSMO files containing "
                f"the section '#ATOMIC POLARIZABILITIES'."
            )

        # positions and areas in atomic units (as the segments in the file)
        seg_au = self.seg_pos / angstrom_per_bohr
        seg_area_au = self.seg_area / angstrom_per_bohr**2
        atm_au = np.asarray(self.atm_pos) / BOHR_CPP
        seg_area_cpp = seg_area_au * BOHR_CPP**2
        seg_radius_sq = seg_area_cpp / np.pi

        idx_atm = self.seg_atm_nr
        area_of_atom = np.bincount(idx_atm, weights=seg_area_au, minlength=n_atoms)
        segments_per_atom = np.bincount(idx_atm, minlength=n_atoms).astype(float)

        # symmetric tensors, columns of the file: xx, yy, zz, xy, xz, yz
        tensor_3x3 = np.zeros((n_atoms, 3, 3))
        tensor_3x3[:, 0, 0] = tensors[:, 0]
        tensor_3x3[:, 1, 1] = tensors[:, 1]
        tensor_3x3[:, 2, 2] = tensors[:, 2]
        tensor_3x3[:, 0, 1] = tensor_3x3[:, 1, 0] = tensors[:, 3]
        tensor_3x3[:, 0, 2] = tensor_3x3[:, 2, 0] = tensors[:, 4]
        tensor_3x3[:, 1, 2] = tensor_3x3[:, 2, 1] = tensors[:, 5]

        distance = np.abs(seg_au - atm_au[idx_atm])
        distance = distance / distance.sum(axis=1).reshape(-1, 1)
        vector = np.einsum("si,sij->sj", distance, tensor_3x3[idx_atm])
        projection = np.linalg.norm(vector, axis=1)

        if mode == 5:
            projection = round_and_truncate(projection, 1)
        elif mode == 1:
            projection = round_and_truncate(
                projection / segments_per_atom[idx_atm], -1
            )
        elif mode == 3:
            projection = round_and_truncate(
                projection * seg_area_au / area_of_atom[idx_atm], -1
            )
        else:
            # modes 6, 7, 8: spatial averaging of the unrounded projections
            r_av_sq = r_av * r_av
            distance_sq = BOHR_DISTANCE_CPP**2 * hf.calculate_squared_distances(
                seg_au
            )
            denominator = seg_radius_sq + r_av_sq
            factors = ((seg_radius_sq * r_av_sq) / denominator) * np.exp(
                -distance_sq / denominator
            )
            factors = factors / factors.sum(axis=1).reshape(-1, 1)
            projection = round_and_truncate(factors.dot(projection), 1)

        return projection

    def calculate_molar_volume_25c(self, strict=True):
        """
        QSPR model of the molar volume at 25 C in cm**3/mol

        Needed for the calculation of solvation free energies with a solvent
        (port of the C++ implementation).
        """

        method = self.method.replace("COSMO", "CPCM")
        if method not in (
            "DFT_CPCM_BP86_def2-TZVP+def2-TZVPD_SP",
            "DFT_BP86_def2-TZVPD_SP",
        ):
            if strict:
                raise ValueError(
                    "The QSPR model for the molar volume only works for the "
                    "quantum chemistry method DFT_BP86_def2-TZVPD_SP"
                )
            warnings.warn(
                "The QSPR model for the molar volume was parametrized using a "
                "different quantum chemistry method than the one you are using. "
                "Recommended method: DFT_BP86_def2-TZVPD_SP"
            )

        elmnt_nr = self.atm_elmnt_nr
        n_atoms = len(elmnt_nr)
        n_si = int(np.sum(elmnt_nr == 14))
        n_h = int(np.sum((elmnt_nr == 1) | (elmnt_nr > 100)))
        n_o = int(np.sum(elmnt_nr == 8))

        if n_atoms == 3 and n_h == 2 and n_o == 1:
            return 18.06863632

        sigma_sq = self.seg_sigma**2
        second_moment = (sigma_sq * self.seg_area).sum() * 10000
        fourth_moment = (sigma_sq * sigma_sq * self.seg_area).sum() * 100000000

        return (
            0.9430785419976806 * n_atoms
            + 0.6977322963011842 * self.area_cosmo_file
            - 0.3161763939689293 * second_moment
            + 0.032441059832647084 * fourth_moment
            + 8.113026329415828 * n_si
            - 0.07066832029215675
        )

    def _convert_hydrogen_element_number(self):
        """
        Hydrogen gets the element number 100 + the element number of the
        closest atom that is not hydrogen (as in the C++ implementation). The
        closest atom including hydrogens is used for H2.
        """

        dist_sq_arr = hf.calculate_squared_distances(
            self.atm_pos, diagonal_element=np.nan
        )

        original_elmnt_nr = self.atm_elmnt_nr.copy()
        n_atoms = len(original_elmnt_nr)

        for idx, elmnt in enumerate(original_elmnt_nr):
            if elmnt != 1:
                continue

            dist_sq = dist_sq_arr[idx, :].copy()
            if n_atoms != 2:
                dist_sq[original_elmnt_nr == 1] = np.nan

            if np.all(np.isnan(dist_sq)):
                raise ValueError(
                    f"No bonding partner was found for hydrogen atom {idx} of "
                    f"{self.filename}."
                )

            idx_bnd = np.nanargmin(dist_sq)
            self.atm_elmnt_nr[idx] = 100 + original_elmnt_nr[idx_bnd]

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

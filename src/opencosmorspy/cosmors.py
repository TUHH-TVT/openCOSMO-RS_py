#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of COSMO-RS including multiple segment descriptors

@author: Thomas Gerlach, 2021
"""

import collections
import re

# import sys
import time

import numpy as np
import scipy.constants as spcon

from opencosmorspy.molecules import Molecule
from opencosmorspy.parameterization import Parameterization
from opencosmorspy.segtp_collection import SegtpCollection

import logging

logger = logging.getLogger(__name__)


class COSMORS(object):
    """
    COSMO-RS class containing submodels and managing reference states etc
    """

    def __init__(self, par="default_turbomole"):
        """
        Constructor

        Parameters
        ----------
        par : string or Parameter, optional
            Either a Parameterization object is passed or an indicator string.
            'default' : Standard parameter chosen

        Raises
        ------
        ValueError
            Unknown value for parameterization.

        Returns
        -------
        None.

        """

        self.par = None
        if type(par) == str:
            self.par = Parameterization(par)
        elif isinstance(par, Parameterization):
            par.check_parameterization()
            self.par = par
        else:
            raise ValueError(
                "Do not know how to handle passed " f"parameterization {par} x"
            )

        self.enth = COSMORSEnthalpic(self.par)
        self.comb = COSMORSCombinatorial(self.par)

        self.mix_lst = []

        self.job_lst = []

        # Temporary lists, later converted to arrays
        self._refmix_idxs_lst = []
        self._jobmix_idxs_lst = []
        self._x_lst = []
        self._T_lst = []

        self.jobmix_idxs = None  # Job mixture index array [n_job]
        self.refmix_idxs = None  # Ref state mixture index array [n_job, n_mol]

    def add_molecule(self, filepath_lst):
        """
        Add a molecule to the model class

        Parameters
        ----------
        filepath_lst : list
            List of filepaths of all cosmo files of a molecule

        Returns
        -------
        None.

        """

        self.clear_jobs()

        self.enth.add_molecule(filepath_lst)

        A_arr = np.array([mol.get_area() for mol in self.enth.mol_lst])
        self.comb.set_area_array(A_arr)

        V_arr = np.array([mol.get_volume() for mol in self.enth.mol_lst])
        self.comb.set_volume_array(V_arr)

    def clear_molecules(self):
        """
        Clear all molecules

        Returns
        -------
        None.

        """

        self.enth.clear_molecules()
        self.comb.clear_molecules()

        self.clear_jobs()

    def add_job(self, x, T, refst, x_refst=None):
        """
        Add a calculation job

        Parameters
        ----------
        x : np.array
            Mole fraction array.
        T : float
            Temperature in Kelvin
        refst : string
            reference_state type "cosmo", "pure_component", "reference_mixture"
        x_refst : np.array, optional
            Mole fraction array for reference state if
            refst=="reference_mixture"

        Returns
        -------
        None.

        """

        self._x_lst.append(x)
        self._T_lst.append(T)

        n_mol = len(x)

        if refst == "cosmo":
            n_mol = len(self.enth.mol_lst)
            refst_idx = -1 * np.ones(n_mol, dtype="int_")
            self._refmix_idxs_lst.append(refst_idx)

        elif refst == "pure_component":

            refst_idx = np.nan * np.ones(n_mol, dtype="int_")
            for idxm, mol in enumerate(self.enth.mol_lst):
                x_refst = np.zeros(n_mol)
                x_refst[idxm] = 1.0
                mix_ref, idx_mix_ref = self._get_mixture(x_refst, T)
                if mix_ref is None:
                    self.mix_lst.append(Mixture(x_refst, T, self.enth, self.comb))
                    mix_ref = self.mix_lst[-1]
                    idx_mix_ref = len(self.mix_lst) - 1
                refst_idx[idxm] = idx_mix_ref
            self._refmix_idxs_lst.append(refst_idx)

        if refst == "reference_mixture":
            mix_ref, idx_mix_ref = self._get_mixture(x_refst, T)
            if mix_ref is None:
                self.mix_lst.append(Mixture(x_refst, T, self.enth, self.comb))
                mix_ref = self.mix_lst[-1]
                refst_idx = (len(self.mix_lst) - 1) * np.ones(n_mol, dtype="int_")
            else:
                refst_idx = idx_mix_ref * np.ones(n_mol, dtype="int_")
            self._refmix_idxs_lst.append(refst_idx)

        self.mix_lst.append(Mixture(x, T, self.enth, self.comb))
        self._jobmix_idxs_lst.append(len(self.mix_lst) - 1)

    def calculate(self):
        """
        Calculate for all submodels and combine

        Returns
        -------
        res : dictionary
            Results dictionary:
                x : np.array(n_job, n_mol)
                T : np.array()
                tot : dictionary
                    lng : np.array(n_job, n_mol)
                    ...
                enth : dictionary
                    lng : np.array(n_job, n_mol)
                    ...
                comb : dictionary
                    lng : np.array(n_job, n_mol)
                    ...

        """

        n_mol = len(self._x_lst[0])
        n_job = len(self._T_lst)

        x = np.array(self._x_lst)
        T = np.array(self._T_lst)
        self.jobmix_idxs = np.array(self._jobmix_idxs_lst, dtype="int_")
        self.refmix_idxs = np.array(self._refmix_idxs_lst, dtype="int_")

        for mix in self.mix_lst:
            mix.calculate()

        tot_res_odct = collections.OrderedDict({})
        keys = self.mix_lst[0].tot_res.keys()
        keys = [key for key in keys if re.match("(lng|aim_|pm_).*", key)]
        for key in keys:
            tot_res_odct[key] = []
            for idx_job, idx_jobm in enumerate(self.jobmix_idxs):
                assert idx_job == len(tot_res_odct[key])
                tot_res_odct[key].append(self.mix_lst[idx_jobm].tot_res[key].copy())
                for idx_mol in range(n_mol):
                    idx_refm = self.refmix_idxs[idx_job, idx_mol]
                    if idx_refm != -1:
                        tot_res_odct[key][idx_job][idx_mol] -= self.mix_lst[
                            idx_refm
                        ].tot_res[key][idx_mol]
            tot_res_odct[key] = np.array(tot_res_odct[key], dtype="float64")

        enth_res_odct = collections.OrderedDict({})
        keys = self.mix_lst[0].enth_res.keys()
        keys = [key for key in keys if re.match("(lng|aim_|pm_).*", key)]
        for key in keys:
            enth_res_odct[key] = []
            for idx_job, idx_jobm in enumerate(self.jobmix_idxs):
                assert idx_job == len(enth_res_odct[key])
                enth_res_odct[key].append(self.mix_lst[idx_jobm].enth_res[key].copy())
                for idx_mol in range(n_mol):
                    idx_refm = self.refmix_idxs[idx_job, idx_mol]
                    if idx_refm != -1:
                        enth_res_odct[key][idx_job][idx_mol] -= self.mix_lst[
                            idx_refm
                        ].enth_res[key][idx_mol]
            enth_res_odct[key] = np.array(enth_res_odct[key], dtype="float64")

        comb_res_odct = collections.OrderedDict({})
        keys = self.mix_lst[0].comb_res.keys()
        keys = [key for key in keys if re.match("(lng|aim_|pm_).*", key)]
        for key in keys:
            comb_res_odct[key] = []
            for idx_job, idx_jobm in enumerate(self.jobmix_idxs):
                assert idx_job == len(comb_res_odct[key])
                comb_res_odct[key].append(self.mix_lst[idx_jobm].comb_res[key].copy())
                for idx_mol in range(n_mol):
                    idx_refm = self.refmix_idxs[idx_job, idx_mol]
                    if idx_refm != -1:
                        comb_res_odct[key][idx_job][idx_mol] -= self.mix_lst[
                            idx_refm
                        ].comb_res[key][idx_mol]
            comb_res_odct[key] = np.array(comb_res_odct[key], dtype="float64")

        res = collections.OrderedDict()
        res["x"] = x
        res["T"] = T
        res["tot"] = tot_res_odct
        res["enth"] = enth_res_odct
        res["comb"] = comb_res_odct

        return res

    def _get_mixture(self, x, T):
        """
        Find mixture with same composition and temperature in list of mixtures

        Parameters
        ----------
        x : np.array
            Mole fraction array.
        T : float
            Temperature in K.

        Returns
        -------
        Mixture
            Mixture object of searched mixture. None if not found.
        int
            Index of mixture object in list of mixtures. None if not found.

        """

        for idx_mix, mix in enumerate(self.mix_lst):
            if mix.equal(x, T):
                return mix, idx_mix

        return None, None

    def clear_jobs(self):
        """
        Clear all jobs.

        Returns
        -------
        None.

        """

        self.mix_lst = []

        self._jobmix_idxs_lst = []
        self._refmix_idxs_lst = []
        self._x_lst = []
        self._T_lst = []

        self.jobmix_idxs = None
        self.refmix_idxs = None


class Mixture(object):
    """
    Manages state of a single mixture including the interdependence between
    submodels
    """

    def __init__(self, x, T, enth, comb):
        """
        Constructor

        Parameters
        ----------
        x : np.array(n_mol)
            Mole fraction array.
        T : float
            Temperature in Kelvin
        enth : COSMORSEnthalpic object
            Controls segment ensemble submodel
        comb : COSMORSCombinatorial object
            Controls combinatorial submodel

        Raises
        ------
        ValueError
            Non-accepted mole fractions

        Returns
        -------
        None.

        """

        if len(x) != len(enth.mol_lst):
            raise ValueError(
                "Number of mole fractions inconsistent "
                "with number of model components."
            )

        if np.abs(1 - x.sum()) > 1e-16:
            raise ValueError("Mole fraction sum unequal unity")
        if np.any(x < -1e-16):
            raise ValueError("Negative mole fraction encountered")

        # Normalize and remove resiudal negatives
        x[x < 0] = 0
        x = x / x.sum()

        self.x = x
        self.T = T

        self.enth = enth
        self.comb = comb

        self.enth_res = None
        self.comb_res = None
        self.tot_res = None

    def calculate(self):
        """
        Solve mixture calculation for all submodels

        Returns
        -------
        None.

        """

        self.enth_res = self.enth.calculate(self.x, self.T)
        self.comb_res = self.comb.calculate(self.x, self.T)
        self.tot_res = collections.OrderedDict()
        self.tot_res["lng"] = self.enth_res["lng"] + self.comb_res["lng"]

    def equal(self, x, T):
        """
        Check if mixture object represents wanted conditions

        Parameters
        ----------
        x : np.array(n_mol)
            Mole fraction array
        T : float
            Temperature in Kelvin

        Returns
        -------
        boolean
            Mixture is equal to input composition

        """

        x_equal = np.all(np.abs(self.x - x) < 1e-16)
        T_equal = np.abs(self.T - T) < 1e-16

        return x_equal and T_equal


class COSMORSCombinatorial(object):
    """Combinatorial part of COSMO-RS"""

    def __init__(self, par):
        """
        Constructor

        Parameters
        ----------
        par : Parameterization object
            COSMO-RS parameters

        Returns
        -------
        None.

        """

        self.par = par

        self.A_arr = None
        self.V_arr = None

        self.lng_comb = None

    def set_area_array(self, A_arr):
        """
        Set molecule areas

        Parameters
        ----------
        A_arr : np.array(n_mol)
            Area in Angstrom**2 for each molecule.

        Returns
        -------
        None.

        """

        self.A_arr = A_arr

    def set_volume_array(self, V_arr):
        """
        Set molecule volume

        Parameters
        ----------
        V_arr : np.array(n_mol)
            Volume in Angstrom**3 for each molecule.

        Returns
        -------
        None.

        """

        self.V_arr = V_arr

    def clear_molecules(self):
        """
        Clear all molecules

        Returns
        -------
        None.

        """

        self.A_arr = None
        self.V_arr = None

    def calculate(self, x, T):
        """
        Solve combinatorial model

        Staverman-Guggenheim is implemented

        Parameters
        ----------
        x : np.array(n_mol)
            Mole fractions of all species.
        T : float
            Temperature in Kelvin.

        Returns
        -------
        result_odct : TYPE
            DESCRIPTION.

        """

        phi_dash_pxi = self.V_arr / (x * self.V_arr).sum()
        theta_dash_pxi = self.A_arr / (x * self.A_arr).sum()
        rel_area = self.A_arr / self.par.comb_sg_a_std
        buffarr1 = phi_dash_pxi / theta_dash_pxi

        if self.par.comb_term == "staverman_guggenheim":

            lng = (
                np.log(phi_dash_pxi)
                + 1
                - phi_dash_pxi
                - self.par.comb_sg_z_coord
                * 0.5
                * rel_area
                * (np.log(buffarr1) + 1 - buffarr1)
            )

        elif self.par.comb_term == "staverman_guggenheim_exponential_scaling":
            # Soares (2011), Kikic (1980), Donohue Prausnitz (1975) */

            buffarr1 = self.V_arr**self.par.comb_sg_expsc_exponent
            phi_dash_pxi_scale = buffarr1 / (buffarr1 * x).sum()

            lng = (
                np.log(phi_dash_pxi_scale)
                + 1
                - phi_dash_pxi_scale
                - self.par.comb_sg_z_coord
                * 0.5
                * rel_area
                * (np.log(buffarr1) + 1 - buffarr1)
            )

        else:
            raise ValueError("Unknown combinatorial term " f"{self.par.comb_term}")

        result_odct = collections.OrderedDict({"lng": lng})

        return result_odct


class COSMORSEnthalpic(object):
    """Segment interaction submodel of COSMO-RS"""

    def __init__(self, par):
        """
        Constructor

        Parameters
        ----------
        par : Parameterization object
            Contains COSMO-RS parameters.

        Returns
        -------
        None.

        """

        self.par = par

        self.mol_lst = []

        self.segtp_collection = SegtpCollection(self.par)

        self.int_arr_store_dct = {}

        self.molecules_cache = {}

    def add_molecule(self, filepath_lst):
        """
        Add a molecule to the object

        Parameters
        ----------
        filepath_lst : list
            Lost of filepaths of COSMO-structures

        Returns
        -------
        None.

        """

        if tuple(filepath_lst) not in self.molecules_cache:
            self.molecules_cache[tuple(filepath_lst)] = Molecule(filepath_lst)

        self.mol_lst.append(self.molecules_cache[tuple(filepath_lst)])
        mol = self.mol_lst[-1]
        mol.convert_properties(self.par.r_av, self.par.mf_r_av_corr)

        for cosmo_struct in mol.cosmo_struct_lst:
            self.segtp_collection.cluster_cosmo_struct(
                cosmo_struct, self.par.sigma_grid, self.par.sigma_orth_grid
            )

    def clear_molecules(self):
        """
        Clear all molecules from the model

        Returns
        -------
        None.

        """

        self.mol_lst = []
        self.segtp_collection = SegtpCollection(self.par)
        self.int_arr_store_dct = {}

    def get_interaction_arrays(self, T):
        """
        Calculate all interaction arrays

        Parameters
        ----------
        T : float
            Temperature in Kelvin.

        Returns
        -------
        dict
            Contains all interaction array np.array(n_mol, n_mol).

        """

        if T in self.int_arr_store_dct:
            return self.int_arr_store_dct[T]

        A_mf, E_mf, sigma_mf = self._calculate_A_mf_E_mf(T)
        A_hb, E_hb = self._calculate_A_hb_E_hb(T)

        E_int = E_mf + E_hb
        A_int = A_mf + A_hb
        tau = np.exp(-A_int * (1 / (spcon.R * T)))

        int_arr_dct = {
            "sigma_mf": sigma_mf,
            "sigma_mf_abs": np.abs(sigma_mf),
            "A_mf": A_mf,
            "E_mf": E_mf,
            "A_hb": A_hb,
            "E_hb": E_hb,
            "A_int": A_int,
            "E_int": E_int,
            "A_hb": A_hb,
            "tau": tau,
        }

        # Save temperature dependent array states for later reuse
        self.int_arr_store_dct = {T: int_arr_dct}

        return int_arr_dct

    def calculate(self, x, T):
        """
        Calculate a COSMO-RS enthalpic calculation

        Parameters
        ----------
        x : np.array(n_mol)
            Mole fraction array
        T : float
            Temperature.

        Returns
        -------
        result : dict
            Contains calculation results.
            - lng : np.array(n_mol) - Logarithmic molecule activity coefficient
            - Gamma : np.array(n_segtp) - Segment activity coefficients
            - X : np.array(n_segtp) - Segment mole fraction
            - n_iter : int. Number of iterations for COSMOspace
            Optional:
            - aim_A_int : np.array(n_mol) - Average interaction energy molecule
                ...
            - pm_A_int : np.array(n_mol) - Partial molar energy
                ...

        """

        int_arr_dct = self.get_interaction_arrays(T)

        Gamma, X, n_iter = self._calculate_cosmospace_succ_subst(x, int_arr_dct["tau"])

        lng = self._calculate_lng(Gamma)

        result = collections.OrderedDict()
        result["lng"] = lng
        result["Gamma"] = Gamma
        result["X"] = X
        result["n_iter"] = n_iter

        if self.par.calculate_contact_statistics_molecule_properties:
            cp_stp_stp = self._calculate_cont_prob_pss(X, Gamma, int_arr_dct["tau"])

            aim_dct = self._calculate_av_int_en_mol_aim(cp_stp_stp, int_arr_dct)
            result.update(aim_dct)

            ndGamma_dni = self._calculate_ndGamma_dni(x, X, Gamma, int_arr_dct["tau"])

            ndpss_dni = self._calculate_ndpss_dni(
                x, X, Gamma, int_arr_dct["tau"], ndGamma_dni
            )

            pm_dct = self._calculate_part_mol_pm(
                x, X, Gamma, cp_stp_stp, int_arr_dct, ndpss_dni
            )
            result.update(pm_dct)

        return result

    def _calculate_A_mf_E_mf(self, T):
        """
        Calculate misfit energy

        Parameters
        ----------
        T : float
            Temperature in Kelvin.

        Returns
        -------
        A_mf : np.array(n_segtp, n_segtp)
            Free misfit interaction energies between all segments.
        E_mf : np.array(n_segtp, n_segtp)
            Misfit interaction energies between all segments.
        sigma_mf : np.array(n_segtp, n_segtp)
            Misfit screening charges between all segments

        """

        segtp_arr_dct = self.segtp_collection.get_segtps_as_array_dct()
        sigma_arr = segtp_arr_dct["sigma"]

        mf_pref = self.par.mf_alpha * self.par.a_eff * 0.5
        sigma_mf = sigma_arr + sigma_arr.reshape(-1, 1)

        if self.par.mf_use_sigma_orth:
            sigma_orth_arr = segtp_arr_dct["sigma_orth"]

            # print('WARNING: CORRECTING SIGMA_ORTH IN COSMORS FOR TESTS')
            # sigma_orth_arr = sigma_orth_arr-0.816*sigma_arr

            buffarr1 = sigma_orth_arr + sigma_orth_arr.reshape(-1, 1)
            E_mf = mf_pref * sigma_mf * (sigma_mf + self.par.mf_f_corr * buffarr1)
        else:
            E_mf = mf_pref * sigma_mf * sigma_mf

        A_mf = E_mf.copy()

        return A_mf, E_mf, sigma_mf

    def _calculate_A_hb_E_hb(self, T):
        """
        Calculate hydrogen bond energy

        Parameters
        ----------
        T : float
            Temperature in Kelvin.

        Returns
        -------
        A_hb : np.array(n_segtp, n_segtp)
            Free hydrogen bond interaction energies between all segments.
        E_hb : np.array(n_segtp, n_segtp)
            Hydrogen bond interaction energies between all segments.

        """

        # Get segment type arrays
        segtp_arr_dct = self.segtp_collection.get_segtps_as_array_dct()
        sigma_arr = segtp_arr_dct["sigma"]
        elmnt_nr_arr = segtp_arr_dct["elmnt_nr"]
        hbd_sw_arr = segtp_arr_dct["hbd_sw"]
        hba_sw_arr = segtp_arr_dct["hba_sw"]

        # Calculate interaction matrix. Identify donor, acceptor arrays first
        # and get hb prefactor corresponding to each element of each
        # segment type
        seg_hb_pref_arr = np.zeros_like(elmnt_nr_arr)
        elmnt_it = np.nditer(elmnt_nr_arr, flags=["c_index"])
        for elmnt_nr in elmnt_it:
            idx = elmnt_it.index
            if elmnt_nr in self.par.hb_don_elmnt_nr_arr:
                seg_hb_pref_arr[idx] = self.par.hb_pref_arr[elmnt_nr]
            if elmnt_nr in self.par.hb_acc_elmnt_nr_arr:
                seg_hb_pref_arr[idx] = self.par.hb_pref_arr[elmnt_nr]

        # Get difference sigma+sigma_th for donors, zero if no donor
        del_sigma_hbd_arr = (
            seg_hb_pref_arr * hbd_sw_arr * (sigma_arr + self.par.hb_sigma_thresh)
        )
        del_sigma_hbd_arr[del_sigma_hbd_arr > 0] = 0.0

        # Get difference sigma-sigma_th for acceptors, zero if no donor
        del_sigma_hba_arr = (
            seg_hb_pref_arr * hba_sw_arr * (sigma_arr - self.par.hb_sigma_thresh)
        )
        del_sigma_hba_arr[del_sigma_hba_arr < 0] = 0.0

        # Convert to 2 dimensional hb delta_sigma*prefactor array
        del_sigma_hbd_mult_hba = del_sigma_hba_arr * del_sigma_hbd_arr.reshape(-1, 1)
        del_sigma_hbd_mult_hba += del_sigma_hbd_mult_hba.T

        # Calculate temperature dependent prefactor
        buffdb1 = 1.0 - self.par.hb_c_T + self.par.hb_c_T * (298.15 / T)
        if buffdb1 > 0:
            hb_c_at_T = self.par.hb_c * buffdb1
            hb_ge_conv_factor = (1 / buffdb1) * (
                1.0 - self.par.hb_c_T + 2 * self.par.hb_c_T * 298.15 / T
            )
        else:
            hb_c_at_T = 0.0
            hb_ge_conv_factor = 0.0

        # Full hb interaction matrix
        A_hb = hb_c_at_T * self.par.a_eff * del_sigma_hbd_mult_hba
        E_hb = hb_ge_conv_factor * A_hb

        return A_hb, E_hb

    def _calculate_X(self, x):
        """
        Calculate segment type mole fraction array

        Parameters
        ----------
        x : np.array(n_mol)
            Mole fractions.

        Returns
        -------
        np.array(n_segment_types)
            Segment mole fractions.

        """

        areas = np.zeros(len(self.segtp_collection), dtype="float64")
        for x_mol, mol in zip(x, self.mol_lst):
            for idx_stp, area_stp in mol.get_segtp_area_dct().items():
                areas[idx_stp] += x_mol * area_stp

        return areas / areas.sum()

    def _calculate_cosmospace_succ_subst(self, x, tau):
        """
        Solve COSMOspace using successive substitution algorithm

        Parameters
        ----------
        x : np.array(n_mol)
            Mole fraction array.
        tau : np.array(n_segtp, n_segtp)
            Interaction parameter array.

        Raises
        ------
        ValueError
            Convergence error.
            TODO: Introduce proper arrays

        Returns
        -------
        Gamma : np.array(n_segtp)
            Segment activity coefficient array.
        X : np.array(n_segtp)
            Segment mole fraction array.
        n_iter : int
            Number of iterations used.

        """

        X = self._calculate_X(x)

        Gamma = np.ones(len(self.segtp_collection))

        converged = False
        n_iter = 0
        while True:
            n_iter += 1
            Gamma_new = 1 / ((X * Gamma).dot(tau.T))

            # print('TEST', self.n_iter, np.max((Gamma_new-self.Gamma)/(self.Gamma)))

            if np.all(
                np.abs(Gamma_new - Gamma) / Gamma < self.par.cosmospace_conv_thresh
            ):
                converged = True
                break
            elif n_iter > self.par.cosmospace_max_iter:
                break

            Gamma_mid = 0.7 * (Gamma_new - Gamma) + Gamma
            Gamma = Gamma_mid

        if not converged:
            raise ValueError("COSMOspace did not converge")

        return Gamma, X, n_iter

    def _calculate_lng(self, Gamma):
        """
        Calculate logarithmic activity coefficients

        Parameters
        ----------
        Gamma : np.array(n_segtp)
            Segment activity coefficient array.

        Returns
        -------
        lng : np.array(n_mol)
            Logarithmic molecule activity coefficients.

        """

        lng = np.zeros_like(self.mol_lst)
        for idxm, mol in enumerate(self.mol_lst):
            mol_segtp_nseg_dct = mol.get_segtp_nseg_dct(self.par.a_eff)
            for idxstp, n_seg in mol_segtp_nseg_dct.items():
                lng[idxm] += n_seg * np.log(Gamma[idxstp])

        return lng

    def _calculate_ndGamma_dni(self, x, X, Gamma, tau):
        """
        Calculate segment activity coefficient derivatives w.r.t. molecule
        amount

        Gradient is unitless: [n_molecules*d(Gamma)/(dni)]

        Parameters
        ----------
        x : np.array(n_mol)
            Mole fraction array.
        X : np.array(n_segtp)
            Segment mole fraction array.
        Gamma : np.array(n_segtp)
            Segment activity coefficient array.
        tau : np.array(n_segtp, n_segtp)
            Interaction parameter array.

        Raises
        ------
        ValueError
            Input errors.

        Returns
        -------
        ndGamma_dni : np.array(n_mol, n_segtp)
            Partial molar gradient array of segment activity coefficients

        """

        for mol in self.mol_lst:
            if len(mol.cosmo_struct_lst) > 1:
                raise ValueError(
                    "Gamma derivatives available only for single " "conformers"
                )

        n_seg = Gamma.size

        dGamma_ndni_les_arr = Gamma.reshape(-1, 1) ** 2 * X * tau + np.eye(n_seg)

        N_i_I = np.array(
            [mol.get_segtp_nseg_arr(self.par.a_eff, len(X)) for mol in self.mol_lst]
        )

        # Set up linear equation set vector and array for exact Gamma molecule
        # number gradient
        # Solve linear equation set to calculate exact LnGamma gradient towards
        # molecule mole numbers
        ndGamma_dni_lst = []
        for idxm, mol in enumerate(self.mol_lst):

            buffarr1 = Gamma / (x.dot(N_i_I)).sum()
            buffarr2 = N_i_I[idxm, :].sum()
            buffarr3 = Gamma * (N_i_I[idxm, :] * Gamma).dot(tau)
            dGamma_ndni_les_vec = buffarr1 * (buffarr2 - buffarr3)

            ###print("dGamma_ndnk_les_vec", dGamma_ndni_les_vec)
            ###pprint("dGamma_ndnk_les_arr", dGamma_ndni_les_arr)
            ndGamma_dni_lst.append(
                np.linalg.solve(dGamma_ndni_les_arr, dGamma_ndni_les_vec)
            )

        ndGamma_dni = np.array(ndGamma_dni_lst)

        test = False
        if test:
            print("NUMERICAL GRADIENT BASED Gamma derivative EVAL")
            N_molecules = len(x)
            # Define concentration difference for numerical gradient
            ndGamma_dni_NG_lst = []
            n_total = 10000000
            for i_mol in range(len(x)):
                # Calculate Gammas at numerical gradient temperatures
                n_var = np.copy(x) * n_total
                n_var[i_mol] = n_var[i_mol] + 1
                x_var = n_var / n_var.sum()
                Gamma_var, X_var, n_iter = self._calculate_cosmospace_succ_subst(x, tau)
                # Calculate with numerical gradient
                ndGamma_dni_NG_lst.append((Gamma_var - Gamma) / 1 * n_total)

            # Save in numpy array
            ndGamma_dni_NG = np.array(ndGamma_dni_NG_lst)
            print("dGamma_dnk_norm", ndGamma_dni)
            print("dGamma_dnk_norm_NG", ndGamma_dni_NG)
            print(ndGamma_dni - ndGamma_dni_NG)

        return ndGamma_dni

    def _calculate_cont_prob_pss(self, X, Gamma, tau):
        """
        Calculate contact probability between segment types

        The contact probabilities are normalized with respect to segment type
        I, meaning the resulting matrix is not symmetric. Get element [I, J]
        of the resulting matrix to get the average number of contacts I-J
        divided by the number of segments of type I.

        Parameters
        ----------
        X : np.array(n_segtp)
            Segment mole fraction array.
        Gamma : np.array(n_segtp)
            Segment activity coefficients.
        tau : np.array(n_segtp, n_segtp)
            Interaction parameter array.

        Returns
        -------
        ndpss_dni : np.array(n_segtp, n_segtp).
            Non-symmetric contact probability array

        """

        return Gamma.reshape(-1, 1) * ((X * Gamma) * tau)

    def _calculate_ndpss_dni(self, x, X, Gamma, tau, ndGamma_dni):
        """
        Calculate partial derivative of segment_type-segment_type
        contact probabilities with respect to amount of molecules

        Contact probabilities are calculated as probability of a contact
        of segment type I with segment type J, normalized by number of segments
        of type I.  (pss[I-J] = <N_{I-J}>/N_I)
        Molar derivatives are multiplied with total number of segments
        for dimensionlessness:
        [n_segtp*d(propability I-J)/d(n_i)]

        Parameters
        ----------
        x : np.array(n_mol)
            Molecule mole fraction array.
        X : np.array(n_segtp)
            Segment mole fraction array.
        Gamma : np.array(n_segtp)
            Segment activity coefficients.
        tau : np.array(n_segtp, n_segtp)
            Interaction parameter array.
        ndGamma_dni : np.array(n_mol, n_segtp)
            Partial molar gradient array of segment activity coefficients

        Returns
        -------
        ndpss_dni : np.array(n_mol, n_segtp, n_segtp)
            Partial molar gradient array of segment activity coefficients
            1st index: Molecule, for which gradient is calculated
            2nd index: Segment type I, which is also used for contact
                       probability normalization
            3rd index: Segment type J, with which I is in contact

        """

        cp_stp_stp_ndni = None

        ndX_dni = self._calculate_ndX_dni(x, X)

        buffarr2 = Gamma * Gamma.reshape(-1, 1)
        buffarr3 = buffarr2 * tau
        ndpss_dni_lst = []
        for idxm in range(len(x)):
            buffarr4 = buffarr3 * ndX_dni[idxm, :]
            buffarr4 += X * Gamma * tau * ndGamma_dni[idxm, :].reshape(-1, 1)
            buffarr4 += X * Gamma.reshape(-1, 1) * tau * ndGamma_dni[idxm, :]
            ndpss_dni_lst.append(buffarr4)

        ndpss_dni = np.array(ndpss_dni_lst)

        test = False
        if test:
            print("NUMERICAL GRADIENT BASED MOL CONTACT PROB DER EVAL")
            N_molecules = len(x)
            pss = self._calculate_cont_prob_pss(X, Gamma, tau)
            # Define concentration difference for numerical gradient
            ndpss_dni_NG_lst = []
            n_total = 10000000
            for i_mol in range(len(x)):
                # Calculate Gammas at numerical gradient temperatures
                n_var = np.copy(x) * n_total
                n_var[i_mol] = n_var[i_mol] + 1
                x_var = n_var / n_var.sum()
                X_var = self._calculate_X(x_var)
                Gamma_var, X_var, n_iter = self._calculate_cosmospace_succ_subst(
                    x_var, tau
                )
                pss_var = self._calculate_cont_prob_pss(X_var, Gamma_var, tau)
                # Calculate with numerical gradient
                ndpss_dni_NG_lst.append((pss_var - pss) / 1 * n_total)

            # Save in numpy array
            ndpss_dni_NG = np.array(ndpss_dni_NG_lst)
            print("ndcp_stp_stp_dni", ndpss_dni)
            print("ndcp_stp_stp_dni_NG", ndpss_dni_NG)

        return ndpss_dni

    def _calculate_part_mol_pm(self, x, X, Gamma, pss, int_arr_dct, ndpss_dni):
        """
        Calculate partial molar properties with respect to molecules

        Parameters
        ----------
        x : np.array(n_mol)
            Molecule mole fraction array.
        X : np.array(n_segtp)
            Segment mole fraction array.
        Gamma : np.array(n_segtp)
            Segment activity coefficients.
        pss : np.array(n_segtp, n_segtp)
            Non-symmetric contact probabilities between segment types.
        int_arr_dct : dict of np.array(n_segtp, n_segtp)
            Contains all interaction energy, interaction free energy, and
            tau arrays. See interaction array calculation function for
            reference.
        ndpss_dni : np.array(n_mol, n_segtp, n_segtp)
            Partial molar gradient array of segment activity coefficients

        Returns
        -------
        pm_dct : dict of np.array(n_mol)
            Arrays containing partial molar properties. Calculated for all
            interaction energy arrays, e.g.:
             - pm_A_int : Partial molar interaction free energies
                ...

        """

        tau = int_arr_dct["tau"]

        N_i_I = np.array(
            [mol.get_segtp_nseg_arr(self.par.a_eff, len(X)) for mol in self.mol_lst]
        )

        ndX_dni = self._calculate_ndX_dni(x, X)

        n_mol = x.size

        pm_dct = {}

        int_keys = set(int_arr_dct.keys()) - set("tau")

        buffflt1 = x.dot(N_i_I).sum()
        buffarr1 = X * Gamma * Gamma.reshape(-1, 1) * tau
        for idxm in range(n_mol):
            buffarr2 = buffflt1 * (
                X.reshape(-1, 1) * ndpss_dni[idxm, :, :]
                + ndX_dni[idxm, :].reshape(-1, 1) * buffarr1
            )
            N_i = N_i_I[idxm, :].sum()
            buffarr3 = 0.5 * (X.reshape(-1, 1) * pss * N_i + buffarr2)

            for int_key in int_keys:
                new_key = "pm_" + int_key

                if new_key not in pm_dct:
                    pm_dct[new_key] = np.zeros(len(self.mol_lst))

                pm_dct[new_key][idxm] = (int_arr_dct[int_key] * buffarr3).sum()

        test2 = False
        if test2:
            print("LOOP BASED BASED MOL PART MOL PROP EVAL")
            ndGamma_dni = self._calculate_ndGamma_dni(x, X, Gamma, int_arr_dct["tau"])
            N_test = 0.0
            for idxm in range(n_mol):
                for idxs1 in range(len(X)):
                    N_test += N_i_I[idxm, idxs1]

            for idxm in range(n_mol):
                Ni_test = 0.0
                for idxs1 in range(len(X)):
                    Ni_test += N_i_I[idxm, idxs1]
                dP_sym_arr_dn_test = np.zeros((len(X), len(X)))
                for idxs1 in range(len(X)):
                    for idxs2 in range(len(X)):
                        dP_sym_arr_dn_test[idxs1, idxs2] = (
                            X[idxs2]
                            * Gamma[idxs1]
                            * Gamma[idxs2]
                            * tau[idxs1, idxs2]
                            * ndX_dni[idxm, idxs1]
                        )
                        dP_sym_arr_dn_test[idxs1, idxs2] += (
                            X[idxs1]
                            * Gamma[idxs1]
                            * Gamma[idxs2]
                            * tau[idxs1, idxs2]
                            * ndX_dni[idxm, idxs2]
                        )
                        dP_sym_arr_dn_test[idxs1, idxs2] += (
                            X[idxs1]
                            * X[idxs2]
                            * Gamma[idxs2]
                            * tau[idxs1, idxs2]
                            * ndGamma_dni[idxm, idxs1]
                        )
                        dP_sym_arr_dn_test[idxs1, idxs2] += (
                            X[idxs1]
                            * X[idxs2]
                            * Gamma[idxs1]
                            * tau[idxs1, idxs2]
                            * ndGamma_dni[idxm, idxs2]
                        )
                buffarr2_test = dP_sym_arr_dn_test * N_test
                P_sym_test = np.zeros((len(X), len(X)))
                for idxs1 in range(len(X)):
                    for idxs2 in range(len(X)):
                        P_sym_test[idxs1, idxs2] = (
                            X[idxs1]
                            * X[idxs2]
                            * Gamma[idxs1]
                            * Gamma[idxs2]
                            * tau[idxs1, idxs2]
                        )
                buffarr3_test = 0.5 * (buffarr2_test + P_sym_test * Ni_test)
                pm_A_int_test = 0.0
                for idxs1 in range(len(X)):
                    for idxs2 in range(len(X)):
                        pm_A_int_test += (
                            int_arr_dct["A_int"][idxs1, idxs2]
                            * buffarr3_test[idxs1, idxs2]
                        )
                print("FINISHED PARTIAL MOLAR PROPERTIES TEST")

        test = False
        if test:
            print("NUMERICAL GRADIENT BASED MOL PART MOL PROP EVAL")
            pm_dct_NG = {}
            for int_key in int_keys:
                new_key = "partmol_" + int_key
                pm_dct_NG[new_key] = []

                # Define concentration difference for numerical gradient
                # ndcp_stp_stp_dni_NG_lst = []
                n_total = 10000000
                N_per_n = x.dot(N_i_I).sum()
                buffval0 = (
                    N_per_n
                    * n_total
                    * 0.5
                    * (int_arr_dct[int_key] * pss * X.reshape(-1, 1)).sum()
                )
                for i_mol in range(len(x)):
                    # Calculate Gammas at numerical gradient temperatures
                    n_var = np.copy(x) * n_total
                    n_var[i_mol] = n_var[i_mol] + 1
                    x_var = n_var / n_var.sum()
                    X_var = self._calculate_X(x_var)
                    Gamma_var, X_var, n_iter = self._calculate_cosmospace_succ_subst(
                        x_var, int_arr_dct["tau"]
                    )
                    cp_stp_stp_var = self._calculate_cont_prob_pss(
                        X_var, Gamma_var, int_arr_dct["tau"]
                    )
                    N_per_n_var = x_var.dot(N_i_I).sum()
                    buffval1 = (
                        N_per_n_var
                        * n_var.sum()
                        * 0.5
                        * int_arr_dct[int_key]
                        * cp_stp_stp_var
                        * X_var.reshape(-1, 1)
                    ).sum()
                    # Calculate with numerical gradient
                    pm_dct_NG[new_key].append((buffval1 - buffval0) / 1)
                pm_dct_NG[new_key] = np.array(pm_dct_NG[new_key])

            # Save in numpy array
            print("part_mol_dct", pm_dct["partmol_A_int"])
            print("part_mol_dct_NG", pm_dct_NG["partmol_A_int"])

        return pm_dct

    def _calculate_av_int_en_mol_aim(self, pss, int_arr_dct):
        """
        Calculate average interaction energies of all molecules

        Parameters
        ----------
        pss : np.array(n_segtp, n_segtp)
            Non-symmetric contact probabilities between segment types.
        int_arr_dct : dict of np.array(n_segtp, n_segtp)
            Contains all interaction energy, interaction free energy, and
            tau arrays. See interaction array calculation function for
            reference.

        Returns
        -------
        aim_dct : dict of np.array(n_mol)
            Arrays containing average interaction energies of molecules.
            Calculated for all interaction energy arrays, e.g.:
             - aim_A_int : Partial molar interaction free energies
                ...

        """

        aim_dct = {}

        keys = set(int_arr_dct.keys()) - set("tau")

        factor = 0.5  # Avvogadro in paper, but included in energy units
        for int_key in keys:
            new_key = "aim_" + int_key
            aim_dct[new_key] = np.zeros(len(self.mol_lst))
            buffarr = pss * int_arr_dct[int_key]
            for idxm, mol in enumerate(self.mol_lst):
                mol_segtp_nseg_arr = mol.get_segtp_nseg_arr(
                    self.par.a_eff, pss.shape[0]
                )
                aim_dct[new_key][idxm] = (
                    factor * (mol_segtp_nseg_arr.dot(buffarr)).sum()
                )

        return aim_dct

    def _calculate_ndX_dni(self, x, X):
        """
        Calculate partial derivative of segment type mole fractions

        Calculated as [n_molecules*d(X_segtp)/d(n_i)]

        Parameters
        ----------
        x : np.array(n_mol)
            Molecule mole fractions.
        X : np.array(n_segtp)
            Segment type mole fractions.

        Returns
        -------
        ndX_dni : np.array(n_mol, n_segtp)
            Segment type mole fraction derivatives.

        """

        N_i_I = np.array(
            [mol.get_segtp_nseg_arr(self.par.a_eff, len(X)) for mol in self.mol_lst]
        )

        buffarr1 = N_i_I - X * N_i_I.sum(axis=1).reshape(-1, 1)
        ndX_dni = buffarr1 / x.dot(N_i_I).sum()

        test = False
        if test:
            print("NUMERICAL GRADIENT BASED ndX_dni")
            N_molecules = len(x)
            # Define concentration difference for numerical gradient
            ndX_dni_lst_NG = []
            n_total = 10000000
            for i_mol in range(len(x)):
                # Calculate Gammas at numerical gradient temperatures
                n_var = np.copy(x) * n_total
                n_var[i_mol] = n_var[i_mol] + 1
                x_var = n_var / n_var.sum()
                X_var = self._calculate_X(x_var)

                # Calculate with numerical gradient
                ndX_dni_lst_NG.append((X_var - X) / 1 * n_total)

            # Save in numpy array
            ndX_dni_NG = np.array(ndX_dni_lst_NG)
            print("ndX_dni", ndX_dni)
            print("ndX_dni_NG", ndX_dni_NG)
            print(ndX_dni_NG - ndX_dni)

        return ndX_dni


if __name__ == "__main__":

    pass

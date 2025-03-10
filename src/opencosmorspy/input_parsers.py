#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from collections import UserDict

import numpy as np

kJ_per_kcal = 4.184  # kJ/kcal
angstrom_per_bohr = 0.52917721092  # angstrom/bohr
kJdivmol_per_hartree = 2625.499639479  # (kJ/mol)/hartree


class SigmaProfileParser(UserDict):
    def __init__(self, filepath, qc_program=None, *, calculate_averaged_sigmas=False):

        if qc_program is None:
            ext = os.path.splitext(filepath)[-1]
            if ext.lower().startswith(".cosmo"):
                qc_program = "turbomole"
            else:
                qc_program = "orca"

        self.data = {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "qc_program": qc_program,
            "method": "",
            "area": None,
            "dipole_moment": None,  # Debye
            "volume": None,
            "energy_tot": None,  # kJ/mol
            "energy_dielectric": None,  # kJ/mol
            "energy_tot_uncorrected": None,  # kJ/mol
            "energy_dielectric_uncorrected": None,  # kJ/mol
            "atm_nr": [],
            "atm_pos": [],  # angstrom
            "atm_elmnt": [],
            "atm_rad": [],
            "seg_nr": [],
            "seg_atm_nr": [],
            "seg_pos": [],
            "seg_charge": [],  # in e
            "seg_charge_uncorrected": [],  # in e
            "seg_area": [],  # angstrom²
            "seg_sigma_raw": [],  # e/angstrom²
            "seg_sigma_raw_uncorrected": [],  # e/angstrom²
            "seg_potential": [],  # #kJ/(e*mol)
            "seg_potential_uncorrected": [],  # #kJ/(e*mol)
            "version": 1,
        }

        if qc_program == "turbomole":
            self._read_turbomolesp()
        elif qc_program == "orca":
            self._read_orcasp()
        else:
            raise ValueError(f"Unknown QC file format: {qc_program}")

        if calculate_averaged_sigmas:
            self.calculate_averaged_sigmas()

    def save_to_xyz(self, comment=""):
        lines = []
        n_atoms = len(self["atm_nr"])
        lines.append(f"{n_atoms}")
        lines.append(f"{comment}")
        for i_atom in range(n_atoms):
            lines.append(
                "{:s}  {:.16f}  {:.16f}  {:.16f}".format(
                    self["atm_elmnt"][i_atom],
                    self["atm_pos"][i_atom][0],
                    self["atm_pos"][i_atom][1],
                    self["atm_pos"][i_atom][2],
                )
            )
        return "\n".join(lines)

    def save_to_xyz_file(self, filepath_xyz, comment=""):
        with open(filepath_xyz, "w") as xyzf:
            xyzf.write(self.save_to_xyz(comment))

    def cluster_segments_into_segmenttypes(self, descriptors, descriptor_ranges):
        areas = self['seg_area']
        if len(descriptors) != len(descriptor_ranges):
            raise ValueError('descriptors and descriptor_ranges have to have same length.')
        
        # descriptor_ranges are assumed to be float, equidistant and sorted or integer and sorted
        descriptor_types = [descriptor_range.dtype for descriptor_range in descriptor_ranges]
        if not all([dt == float or dt == int for dt in descriptor_types]):
            raise TypeError('Only int or float descriptors are supported right now.')

        descriptors = np.transpose(np.array(descriptors))
        n_descriptors = descriptors.shape[1]

        def cluster_segments_to_segment_types_for_one_descriptor(segments_descriptors, segments_area, i_descriptor_to_cluster):
            segment_types_descriptors = []
            segment_type_areas = []
            i_descriptor_range = descriptor_ranges[i_descriptor_to_cluster]
            for segment_descriptors, segment_area in zip(segments_descriptors, segments_area):
                descriptor_value = segment_descriptors[i_descriptor_to_cluster]

                ind_left = np.flatnonzero(i_descriptor_range <= descriptor_value)[-1]
                left_clustered_descriptor_value = i_descriptor_range[ind_left]
                left_new_segment_type_descriptors = segment_descriptors.copy()
                left_new_segment_type_descriptors[i_descriptor_to_cluster] = left_clustered_descriptor_value
                segment_types_descriptors.append(left_new_segment_type_descriptors)

                if left_clustered_descriptor_value == descriptor_value:
                    segment_type_areas.append(segment_area)
                else:
                    right_new_segment_type_descriptors = segment_descriptors.copy()
                    right_clustered_descriptor_value = i_descriptor_range[ind_left + 1]
                    clustered_delta = right_clustered_descriptor_value - left_clustered_descriptor_value
                    right_new_segment_type_descriptors[i_descriptor_to_cluster] = right_clustered_descriptor_value
                    segment_types_descriptors.append(right_new_segment_type_descriptors)
                    segment_type_areas.append(segment_area * ((right_clustered_descriptor_value - descriptor_value)/clustered_delta))
                    segment_type_areas.append(segment_area * ((descriptor_value - left_clustered_descriptor_value)/clustered_delta))

            return segment_types_descriptors, segment_type_areas

        all_segment_types = []
        all_segment_type_areas = []

        for i_segment_descriptors, i_segment_area in zip(descriptors, areas):

            segments_types, segments_types_areas = [i_segment_descriptors.tolist()], [i_segment_area]

            for i_descriptor in range(n_descriptors):
                segments_types, segments_types_areas = cluster_segments_to_segment_types_for_one_descriptor(segments_types, segments_types_areas, i_descriptor)

            for segment_type, segment_type_area in zip(segments_types, segments_types_areas):
                if segment_type not in all_segment_types:
                    all_segment_types.append(segment_type)
                    all_segment_type_areas.append(0)

                segment_type_index = all_segment_types.index(segment_type)
                all_segment_type_areas[segment_type_index] += segment_type_area

        all_segment_types = np.array(all_segment_types)
        all_segment_type_areas = np.array(all_segment_type_areas)

        # sort
        descriptors = [all_segment_types[:, i].tolist() for i in range(n_descriptors)]
        descriptors.reverse()
        sorted_indices = np.lexsort(descriptors)

        all_segment_types = all_segment_types[sorted_indices]
        all_segment_type_areas = all_segment_type_areas[sorted_indices]
        
        return all_segment_types, all_segment_type_areas

    def cluster_and_create_sigma_profile(
        self, sigmas="seg_sigma_averaged", sigmas_range=np.arange(-0.03, 0.03, 0.001)
    ):
        if sigmas == "seg_sigma_averaged":
            if "seg_sigma_averaged" not in self:
                self.calculate_averaged_sigmas()

        descriptors = [self[sigmas]]
        descriptor_ranges = [sigmas_range]
        sp_areas = []
        sp_sigmas = []
        clustered_sigmas, clustered_areas = self.cluster_segments_into_segmenttypes(
            descriptors, descriptor_ranges
        )
        clustered_sigmas = clustered_sigmas.reshape(-1)
        
        for sigma in sigmas_range:
            area = 0.0
            if sigma in clustered_sigmas:
                area = clustered_areas[clustered_sigmas == sigma][0]
            sp_areas.append(area)
            sp_sigmas.append(sigma)
        return np.array(sp_sigmas), np.array(sp_areas)

    def calculate_averaged_sigmas(self, *, sigmas_raw=None, averaging_radius=0.5):

        if sigmas_raw is None:
            sigmas_raw = self["seg_sigma_raw"]

        areas = self["seg_area"]
        seg_radii_squared = areas / np.pi

        averaging_radius_squared = averaging_radius**2
        sigmas_averaged = np.zeros_like(sigmas_raw)

        for i_segment in range(len(sigmas_raw)):
            d_ij_squared = np.power(
                self["seg_pos"] - self["seg_pos"][i_segment, :], 2
            ).sum(1)

            radii_squared_plus_r_av_squared = (
                seg_radii_squared + averaging_radius_squared
            )

            sigma_weights = (
                seg_radii_squared
                * averaging_radius_squared
                / radii_squared_plus_r_av_squared
            ) * np.exp(-1 * d_ij_squared / radii_squared_plus_r_av_squared)

            sigmas_averaged[i_segment] = np.sum(sigmas_raw * sigma_weights) / np.sum(
                sigma_weights
            )

        self["seg_sigma_averaged"] = sigmas_averaged

    def calculate_sigma_moments(self, *, sigmas=None, sigma_hb_threshold=0.0085):

        if sigmas is None:
            if "seg_sigma_averaged" not in self:
                self.calculate_averaged_sigmas()
            sigmas = self["seg_sigma_averaged"]

        # Zeroth Moment (total surface)
        # First Moment (charge)
        # Second Moment (polarity)
        # Third Moment (sigma profile skewness)
        # Fourth Moment (no physical meaning)
        # Fifth Moment (no physical meaning...)
        # Sixth Moment (no physical meaning...)

        n_moments = 7
        areas = self["seg_area"]

        sigma_moments = np.zeros((n_moments))

        for i in range(n_moments):
            sigma_moments[i] = np.sum(np.power(sigmas, i) * areas)
            if i > 1:
                sigma_moments[i] *= 100**i

        self["sigma_moments"] = sigma_moments

        sigma_hydrogen_bond_acceptor_moments = np.zeros((n_moments))
        sigma_hydrogen_bond_donor_moments = np.zeros((n_moments))

        # first step adjusting to manual
        # only calculate:
            # Second Moment
            # Third Moment
            # Fourth Moment
        for i in [2, 3, 4]:
            current_HB_threshold = 0.006 + i * 0.002

            sigma_hydrogen_bond_acceptor_moments[i] = np.sum(
                np.maximum(sigmas - current_HB_threshold, 0) * areas
            )
            sigma_hydrogen_bond_donor_moments[i] = np.sum(
                np.maximum(-1 * sigmas - current_HB_threshold, 0) * areas
            )

        sigma_hydrogen_bond_acceptor_moments = 100 * np.abs(
            sigma_hydrogen_bond_acceptor_moments
        )
        sigma_hydrogen_bond_donor_moments = 100 * np.abs(
            sigma_hydrogen_bond_donor_moments
        )

        self["sigma_hydrogen_bond_acceptor_moments"] = (
            sigma_hydrogen_bond_acceptor_moments
        )
        self["sigma_hydrogen_bond_donor_moments"] = sigma_hydrogen_bond_donor_moments

    def _read_single_float(self, line, variable, regex, scaling_factor):

        re_match = re.match(regex, line)
        if re_match:
            self[variable] = float(re_match.groups()[0])
            self[variable] *= scaling_factor

    def _read_turbomole_atom_section(self, cosmofile):

        line = True

        mode = None
        while line:
            line = next(cosmofile).strip()
            line_splt = line.split()
            if len(line_splt) != 4 and len(line_splt) != 6:
                if mode:
                    break
                else:
                    continue

            if not mode:
                mode = len(line_splt)

            if len(line_splt) == 6:
                atm_nr = int(line_splt[0]) - 1
                atm_pos = [float(val) for val in line_splt[1:4]]
                atm_elmnt = line_splt[4].title()
                atm_rad = float(line_splt[5])
            elif len(line_splt) == 4:
                atm_nr = len(self["atm_nr"])
                atm_pos = [float(val) for val in line_splt[1:4]]
                atm_elmnt = line_splt[0].title()
                atm_rad = None
            else:
                raise ValueError("Lines shoud either have a length of 4 or 6.")

            self["atm_nr"].append(atm_nr)
            self["atm_pos"].append(atm_pos)
            self["atm_elmnt"].append(atm_elmnt)
            self["atm_rad"].append(atm_rad)

        atm_pos_multiplier = angstrom_per_bohr
        if mode == 4:
            atm_pos_multiplier = 1

        self["atm_nr"] = np.array(self["atm_nr"], dtype="int64")
        self["atm_pos"] = (
            np.array(self["atm_pos"], dtype="float64") * atm_pos_multiplier
        )
        self["atm_rad"] = np.array(self["atm_rad"], dtype="float64")

    def _read_turbomole_seg_section(self, cosmofile):

        line = next(cosmofile)

        while line:
            try:
                line = next(cosmofile).strip()
                line_splt = line.split()
                if len(line_splt) != 9:
                    if self["seg_nr"]:
                        break
                    else:
                        continue

            except StopIteration:
                break
            if not line:
                break

            self["seg_nr"].append(int(line_splt[0]) - 1)
            self["seg_atm_nr"].append(int(line_splt[1]) - 1)
            self["seg_pos"].append([float(val) for val in line_splt[2:5]])
            self["seg_charge"].append(float(line_splt[5]))
            self["seg_area"].append(float(line_splt[6]))
            self["seg_sigma_raw"].append(float(line_splt[7]))
            self["seg_potential"].append(float(line_splt[8]))

        self["seg_nr"] = np.array(self["seg_nr"], dtype="int64")
        self["seg_atm_nr"] = np.array(self["seg_atm_nr"], dtype="int64")
        self["seg_pos"] = np.array(self["seg_pos"], dtype="float64") * angstrom_per_bohr
        self["seg_charge"] = np.array(self["seg_charge"], dtype="float64")
        self["seg_area"] = np.array(self["seg_area"], dtype="float64")
        self["seg_sigma_raw"] = np.array(self["seg_sigma_raw"], dtype="float64")
        self["seg_potential"] = (
            np.array(self["seg_potential"], dtype="float64") * kJdivmol_per_hartree * angstrom_per_bohr
        )

    def _read_turbomolesp(self):

        with open(self["filepath"], "r") as cosmofile:

            for i_line, line in enumerate(cosmofile):

                line = line.strip()

                if line == "$info":
                    line = next(cosmofile).strip()
                    self["method"] = (
                        f'{line.split(";")[-2]}_{line.split(";")[-1]}'.lower()
                    )

                self._read_single_float(
                    line, "area", r"area\s*=\s*([0-9+-.eE]+)", angstrom_per_bohr**2
                )

                self._read_single_float(
                    line, "volume", r"volume\s*=\s*([0-9+-.eE]+)", angstrom_per_bohr**3
                )

                self._read_single_float(
                    line,
                    "energy_tot",
                    r"Total\s+energy\s+corrected.*=\s*([0-9+-.eE]+)",
                    kJdivmol_per_hartree,
                )

                self._read_single_float(
                    line,
                    "energy_dielectric",
                    r"Dielectric\s+energy\s+\[a\.u\.\]\s*=\s*([0-9+-.eE]+)",
                    kJdivmol_per_hartree,
                )

                if line == "$coord_rad" or i_line == 0 and line == "$coord_car":
                    self._read_turbomole_atom_section(cosmofile)

                if line == "$segment_information":
                    self._read_turbomole_seg_section(cosmofile)

    def _read_orca_atom_coordinates(self, orcasp_file):

        next(orcasp_file)
        line = next(orcasp_file).strip()

        atm_nr = 0
        while line:
            try:
                line = next(orcasp_file).strip()
            except StopIteration:
                line = False

            if not line or "###" in line:
                break
            line_splt = line.split()

            self["atm_nr"].append(atm_nr)
            atm_nr += 1

            self["atm_pos"].append([float(val) for val in line_splt[1:]])
            self["atm_elmnt"].append(line_splt[0].title())

        self["atm_nr"] = np.array(self["atm_nr"], dtype="int64")
        self["atm_pos"] = np.array(self["atm_pos"], dtype="float64")

    def _read_orca_atom_radii(self, orcasp_file):

        line = next(orcasp_file)

        while line:
            line = next(orcasp_file).strip()
            if not line or "---" in line:
                break
            line_splt = line.split()

            self["atm_rad"].append(line_splt[3])

        self["atm_rad"] = np.array(self["atm_rad"], dtype="float64") * angstrom_per_bohr

    def _read_orca_seg_section(self, orcasp_file):

        next(orcasp_file)
        line = next(orcasp_file)

        seg_nr = 0
        while line:
            try:
                line = next(orcasp_file).strip()
            except StopIteration:
                break
            if not line:
                break
            line_splt = line.split()

            self["seg_nr"].append(seg_nr)
            seg_nr += 1
            self["seg_atm_nr"].append(int(line_splt[-1]))
            self["seg_pos"].append([float(val) for val in line_splt[0:3]])
            self["seg_charge_uncorrected"].append(float(line_splt[5]))
            self["seg_area"].append(float(line_splt[3]))
            self["seg_potential_uncorrected"].append(float(line_splt[4]))

        self["seg_nr"] = np.array(self["seg_nr"], dtype="int64")
        self["seg_atm_nr"] = np.array(self["seg_atm_nr"], dtype="int64")
        self["seg_pos"] = np.array(self["seg_pos"], dtype="float64") * angstrom_per_bohr
        self["seg_charge_uncorrected"] = np.array(
            self["seg_charge_uncorrected"], dtype="float64"
        )
        self["seg_area"] = (
            np.array(self["seg_area"], dtype="float64") * angstrom_per_bohr**2
        )
        self["seg_sigma_raw_uncorrected"] = (
            self["seg_charge_uncorrected"] / self["seg_area"]
        )
        self["seg_potential_uncorrected"] = (
            np.array(self["seg_potential_uncorrected"], dtype="float64")
            * kJdivmol_per_hartree
        )

    def _read_orca_cpcm_correction_section(self, orcasp_file):

        line = next(orcasp_file)
        self._read_single_float(
            line,
            "energy_dielectric",
            r"Corrected\s+dielectric\s+energy\s+=\s*([0-9+-.eE]+)",
            kJdivmol_per_hartree,
        )

        self["energy_tot_uncorrected"] = self["energy_tot"]
        self["energy_tot"] = (
            self["energy_tot_uncorrected"]
            - self["energy_dielectric_uncorrected"]
            + self["energy_dielectric"]
        )
        next(orcasp_file)
        next(orcasp_file)

        corrected_charges = []
        while line:
            try:
                line = next(orcasp_file).strip()
            except StopIteration:
                break
            if not line:
                break

            corrected_charge = float(line)
            corrected_charges.append(corrected_charge)

        assert self["seg_charge_uncorrected"].size == len(corrected_charges)
        self["seg_charge"] = np.array(corrected_charges, dtype="float64")
        self["seg_sigma_raw"] = self["seg_charge"] / self["seg_area"]

    def _read_orca_adjacency_matrix(self, orcasp_file):

        atm_nr = self["atm_pos"].shape[0]
        adjacency_marix = []
        for line_nr in range(atm_nr):
            line = next(orcasp_file)
            line = [int(entry.strip()) for entry in line.split()]
            adjacency_marix.append(line)

        self["adjacency_matrix"] = np.array(adjacency_marix, dtype="int")

    def calculate_molecular_dipole(self):
        if "dipole_moment" not in self:
            raise ValueError(
                "The specified input file did not include dipole moment information."
            )
        return float(np.linalg.norm(self["dipole_moment"]))

    def _read_orcasp(self):

        with open(self["filepath"], "r") as orcasp_file:

            line = next(orcasp_file).strip()
            self["name"], self["method"] = (entry.strip() for entry in line.split(":"))

            for line in orcasp_file:

                line = line.strip()

                self._read_single_float(
                    line, "area", r"\s*([0-9+-.eE]+)\s+#\s*Area", angstrom_per_bohr**2
                )

                self._read_single_float(
                    line,
                    "volume",
                    r"\s*([0-9+-.eE]+)\s+#\s*Volume",
                    angstrom_per_bohr**3,
                )

                self._read_single_float(
                    line,
                    "energy_tot",
                    r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s*([0-9+-.eE]+)",
                    kJdivmol_per_hartree,
                )

                self._read_single_float(
                    line,
                    "energy_dielectric_uncorrected",
                    r"\s*([0-9+-.eE]+)\s+#\s*CPCM\s+dielectric\s+energy",
                    kJdivmol_per_hartree,
                )

                if line == "#XYZ_FILE":
                    self._read_orca_atom_coordinates(orcasp_file)

                if "CARTESIAN COORDINATES (A.U.)" in line:
                    self._read_orca_atom_radii(orcasp_file)

                if "SURFACE POINTS (A.U.)" in line:
                    self._read_orca_seg_section(orcasp_file)

                if "#COSMO_corrected" in line or "#CPCM_corrected" in line:
                    self._read_orca_cpcm_correction_section(orcasp_file)

                if "#ADJACENCY_MATRIX" in line:
                    self._read_orca_adjacency_matrix(orcasp_file)

                if "DIPOLE MOMENT (Debye)" in line:
                    line = next(orcasp_file).strip()
                    self["dipole_moment"] = np.array(
                        [float(v) for v in line.strip().split()[-3:]]
                    )


class PyCrsError(Exception):
    pass


if __name__ == "__main__":
    main()

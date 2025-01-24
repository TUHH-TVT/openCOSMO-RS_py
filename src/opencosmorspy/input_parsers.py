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
            if ext.lower().startswith('.cosmo'):
                qc_program = 'turbomole'
            else:
                qc_program = 'orca'

        self.data = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'qc_program': qc_program,
            'method': '',
            'area': None,
            'dipole_moment': None,  # Debye
            'volume': None,
            'energy_tot': None,  # kJ/mol
            'energy_dielectric': None,  # kJ/mol
            'energy_tot_uncorrected': None,  # kJ/mol
            'energy_dielectric_uncorrected': None,  # kJ/mol
            'atm_nr': [],
            'atm_pos': [],  # angstrom
            'atm_elmnt': [],
            'atm_rad': [],
            'seg_nr': [],
            'seg_atm_nr': [],
            'seg_pos': [],
            'seg_charge': [],  # in e
            'seg_charge_uncorrected': [],  # in e
            'seg_area': [],  # angstrom²
            'seg_sigma_raw': [],  # e/angstrom²
            'seg_sigma_raw_uncorrected': [],  # e/angstrom²
            'seg_potential': [],  # #kJ/(e*mol)
            'seg_potential_uncorrected': [],  # #kJ/(e*mol)
            'version': 1
        }

        if qc_program == 'turbomole':
            self._read_turbomolesp()
        elif qc_program == 'orca':
            self._read_orcasp()
        else:
            raise ValueError(f'Unknown QC file format: {qc_program}')

        if calculate_averaged_sigmas:
            self.calculate_averaged_sigmas()

    def save_to_xyz(self, comment=''):
        lines = []
        n_atoms = len(self['atm_nr'])
        lines.append(f'{n_atoms}')
        lines.append(f'{comment}')
        for i_atom in range(n_atoms):
            lines.append(
                '{:s}  {:.16f}  {:.16f}  {:.16f}'.format(
                    self['atm_elmnt'][i_atom],
                    self['atm_pos'][i_atom][0],
                    self['atm_pos'][i_atom][1],
                    self['atm_pos'][i_atom][2],
                )
            )
        return '\n'.join(lines)

    def save_to_xyz_file(self, filepath_xyz, comment=''):
        with open(filepath_xyz, 'w') as xyzf:
            xyzf.write(self.save_to_xyz(comment))

    def cluster_segments_into_segmenttypes(self, descriptors, descriptor_ranges, molecule_index = -1, add_bounds_to_first_descriptor = True, segment_type_areas = None, segment_types = None):
        # descriptors needs to be a list of [ndarray of shape (n_segments, 1) or (n_segments,)]
        # descriptor_ranges needs to be a list of [ndarray of shape (n_segments, 1) or (n_segments,)]

        areas = self['seg_area']
        n_segments = areas.size
        n_descriptors = len(descriptor_ranges)

        descriptors_temp = np.zeros((n_segments, 0))

        for i_descriptor in range(0, n_descriptors):
            descriptors_temp = np.hstack((descriptors_temp, descriptors[i_descriptor].reshape((n_segments, 1))))
            descriptor_range = descriptor_ranges[i_descriptor]
            if descriptor_range is None:
                descriptor_ranges[i_descriptor] = np.unique(descriptors_temp[:, i_descriptor])

        descriptors = descriptors_temp

        if segment_types is None:
            segment_types = np.zeros((0, n_descriptors))

        n_segment_types = segment_types.shape[0]

        n_max_molecules = 1

        if molecule_index != -1:
            n_max_molecules = molecule_index + 1
        else:
            molecule_index = 0

        if segment_type_areas is None:
            segment_type_areas = np.zeros((0, n_max_molecules))
        else:
            segment_type_areas.resize((n_segment_types, n_max_molecules))

        for i_area in range(0, n_segments):
            
            A = areas[i_area]

            segmentsFor_i_area = np.zeros((1, n_descriptors + 1))

            segmentsFor_i_area[0, 0] = A

            for i_descriptor in range(0, n_descriptors):

                descriptor_value = descriptors[i_area, i_descriptor]
                descriptor_range = descriptor_ranges[i_descriptor]

                ind_left = np.flatnonzero(descriptor_range <= descriptor_value).max()

                n_segmentsFor_i_area = segmentsFor_i_area.shape[0]
                if descriptor_range[ind_left] == descriptor_value:
                    for i_segment in range(0, n_segmentsFor_i_area):
                        segmentsFor_i_area[i_segment, i_descriptor + 1] = descriptor_range[ind_left]
                else:
                    newSegmentsFor_i_area = np.zeros((1, n_descriptors + 1))

                    n_newSegmentsFor_i_area = 0

                    for i_segment in range(0, n_segmentsFor_i_area):
                        
                        newSegmentsFor_i_area.resize((n_newSegmentsFor_i_area + 2, n_descriptors + 1), refcheck=False)

                        segmentsFor_i_area_A = segmentsFor_i_area[i_segment, 0]

                        right_factor = (descriptor_value - descriptor_range[ind_left]) / (descriptor_range[ind_left + 1] - descriptor_range[ind_left])
                        left_factor = (descriptor_range[ind_left + 1] - descriptor_value) / (descriptor_range[ind_left + 1] - descriptor_range[ind_left])

                        # left
                        newSegmentsFor_i_area[n_newSegmentsFor_i_area, :] = segmentsFor_i_area[i_segment, :]
                        newSegmentsFor_i_area[n_newSegmentsFor_i_area, 0] = left_factor * segmentsFor_i_area_A
                        newSegmentsFor_i_area[n_newSegmentsFor_i_area, i_descriptor + 1] = descriptor_range[ind_left]
                        
                        # right
                        n_newSegmentsFor_i_area += 1
                        newSegmentsFor_i_area[n_newSegmentsFor_i_area, :] = segmentsFor_i_area[i_segment, :]
                        newSegmentsFor_i_area[n_newSegmentsFor_i_area, 0] = right_factor * segmentsFor_i_area_A
                        newSegmentsFor_i_area[n_newSegmentsFor_i_area, i_descriptor + 1] = descriptor_range[ind_left + 1]

                    segmentsFor_i_area = newSegmentsFor_i_area

            n_segmentsFor_i_area = segmentsFor_i_area.shape[0]

            for i_segment in range(0, n_segmentsFor_i_area):

                i_segment_type_new = -1
                A = segmentsFor_i_area[i_segment, 0]

                for i_segment_type in range(0, n_segment_types):
                    
                    if np.all(segment_types[i_segment_type, :] == segmentsFor_i_area[i_segment, np.arange(1, segmentsFor_i_area.shape[1])]):
                        i_segment_type_new = i_segment_type    
                        break

                if i_segment_type_new == -1:

                    i_segment_type_new = n_segment_types
                    n_segment_types += 1

                    segment_types.resize((n_segment_types, n_descriptors), refcheck=False)
                    segment_type_areas.resize((n_segment_types, n_max_molecules), refcheck=False)

                    segment_types[i_segment_type_new, :] = segmentsFor_i_area[i_segment, np.arange(1, segmentsFor_i_area.shape[1])]
            
                segment_type_areas[i_segment_type_new, molecule_index] += A

        if n_descriptors > 1:
            new_column_order = np.arange(1, segment_types.shape[1])
            new_column_order = np.append(new_column_order, 0)
            new_column_order = np.flip(new_column_order) # necessary because lexsort somehow searches the columns in reverse

            sorted_indices = np.lexsort(segment_types.transpose()[new_column_order])
        else:
            sorted_indices = np.argsort(segment_types, axis=0)

        segment_types = segment_types[sorted_indices]
        segment_type_areas = segment_type_areas[sorted_indices]

        n_segment_types = segment_type_areas.size

        if add_bounds_to_first_descriptor:

            step_between_first_descriptor_values = segment_types[1, 0] - segment_types[0, 0] # this assumed at least 2 segment types

            if n_descriptors > 1:
                n_different_descriptor_combinations = np.unique(segment_types[:, np.arange(1, n_descriptors)], axis=0).shape[0]
                n_segment_types_with_bounds = segment_type_areas.size + (2 * n_different_descriptor_combinations)

                new_segment_types = np.zeros((n_segment_types_with_bounds, n_descriptors))
                new_segment_type_areas = np.zeros((n_segment_types_with_bounds, 1))

                last_second_descriptor_value = np.inf
                n_extra_segment_types_due_to_bounds = 0
                
                
                for i_segment_type in range(0, n_segment_types):

                    # add starting value
                    if i_segment_type == 0:
                        last_second_descriptor_value = np.nan
                    else:
                        last_second_descriptor_value = segment_types[i_segment_type - 1, 1]

                    if last_second_descriptor_value != segment_types[i_segment_type, 1]:
                        new_segment_types[i_segment_type + n_extra_segment_types_due_to_bounds, :] = segment_types[i_segment_type, :]
                        new_segment_types[i_segment_type + n_extra_segment_types_due_to_bounds, 0] -= step_between_first_descriptor_values
                        new_segment_type_areas[i_segment_type + n_extra_segment_types_due_to_bounds, 0] = 0
                        n_extra_segment_types_due_to_bounds += 1

                    # add normal values
                    new_segment_types[i_segment_type + n_extra_segment_types_due_to_bounds, :] = segment_types[i_segment_type, :]
                    new_segment_type_areas[i_segment_type + n_extra_segment_types_due_to_bounds, 0] = segment_type_areas[i_segment_type, 0]

                    # add ending value
                    if i_segment_type < n_segment_types - 1:
                        next_second_descriptor_value = segment_types[i_segment_type + 1, 1]
                    else:
                        next_second_descriptor_value = np.nan

                    if segment_types[i_segment_type, 1] != next_second_descriptor_value:
                        n_extra_segment_types_due_to_bounds += 1               
                        new_segment_types[i_segment_type + n_extra_segment_types_due_to_bounds, :] = segment_types[i_segment_type, :] 
                        new_segment_types[i_segment_type + n_extra_segment_types_due_to_bounds, 0] += step_between_first_descriptor_values
                        new_segment_type_areas[i_segment_type + n_extra_segment_types_due_to_bounds, 0] = 0

            else:  
                n_segment_types_with_bounds = segment_type_areas.size + 2
                new_segment_types = np.zeros((n_segment_types_with_bounds, n_descriptors))
                new_segment_type_areas = np.zeros((n_segment_types_with_bounds, 1))

                new_segment_types[0, :] = segment_types[0, :]
                new_segment_types[0, 0] -= step_between_first_descriptor_values

                new_segment_types[1:-1, :] = segment_types[:, 0]
                new_segment_type_areas[1:-1, :] = segment_type_areas[:, 0]

                new_segment_types[-1, :] = segment_types[-1, :]
                new_segment_types[-1, 0] += step_between_first_descriptor_values


            segment_type_areas = new_segment_type_areas
            segment_types = new_segment_types


        return segment_types, segment_type_areas

    def cluster_and_create_sigma_profile(self, sigmas='seg_sigma_averaged', sigmas_range=np.arange(-0.03, 0.03, 0.001)):
        if sigmas == 'seg_sigma_averaged':
            if 'seg_sigma_averaged' not in self:
                self.calculate_averaged_sigmas()

        descriptors = [self[sigmas]]
        descriptor_ranges = [sigmas_range]
        sp_areas = []
        sp_sigmas = []
        clustered_sigmas, clustered_areas = self.cluster_segments_into_segmenttypes(descriptors, descriptor_ranges)
        for sigma in sigmas_range:
            area = 0.0
            if sigma in clustered_sigmas:
                area = clustered_areas[clustered_sigmas == sigma][0]
            sp_areas.append(area)
            sp_sigmas.append(sigma)
        return np.array(sp_sigmas), np.array(sp_areas)

    def calculate_averaged_sigmas(self, *, sigmas_raw=None, averaging_radius=0.5):

        if sigmas_raw is None:
            sigmas_raw = self['seg_sigma_raw']

        areas = self['seg_area']
        seg_radii_squared = areas / np.pi

        averaging_radius_squared = averaging_radius**2
        sigmas_averaged = np.zeros_like(sigmas_raw)

        for i_segment in range(len(sigmas_raw)):
            d_ij_squared = np.power(
                self['seg_pos'] - self['seg_pos'][i_segment, :], 2
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

        self['seg_sigma_averaged'] = sigmas_averaged

    def calculate_sigma_moments(self, *, sigmas=None, sigma_hb_threshold=0.0085):

        if sigmas is None:
            if 'seg_sigma_averaged' not in self:
                self.calculate_averaged_sigmas()
            sigmas = self['seg_sigma_averaged']

        # Zeroth Moment (total surface)
        # First Moment (charge)
        # Second Moment (polarity)
        # Third Moment (sigma profile skewness)
        # Fourth Moment (no physical meaning)
        # Fifth Moment (no physical meaning...)
        # Sixth Moment (no physical meaning...)

        n_moments = 7
        areas = self['seg_area']

        sigma_moments = np.zeros((n_moments))

        for i in range(n_moments):
            sigma_moments[i] = np.sum(np.power(sigmas, i) * areas)
            if i > 1:
                sigma_moments[i] *= 100**i

        self['sigma_moments'] = sigma_moments

        sigma_hydrogen_bond_acceptor_moments = np.zeros((n_moments))
        sigma_hydrogen_bond_donor_moments = np.zeros((n_moments))

        # first step adjusting to manual
        for i in [2, 3, 4]:
            current_HB_threshold = 0.006 + i * 0.002

            sigma_hydrogen_bond_acceptor_moments[i] = np.sum(
                np.maximum(sigmas - current_HB_threshold, 0) * areas
            )
            sigma_hydrogen_bond_donor_moments[i] = np.sum(
                np.maximum(-1 * sigmas - current_HB_threshold, 0) * areas
            )

        sigma_hydrogen_bond_acceptor_moments = 100 * np.abs(sigma_hydrogen_bond_acceptor_moments)
        sigma_hydrogen_bond_donor_moments = 100 * np.abs(sigma_hydrogen_bond_donor_moments)

        self['sigma_hydrogen_bond_acceptor_moments'] = (
            sigma_hydrogen_bond_acceptor_moments
        )
        self['sigma_hydrogen_bond_donor_moments'] = (
            sigma_hydrogen_bond_donor_moments
        )

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
                atm_nr = len(self['atm_nr'])
                atm_pos = [float(val) for val in line_splt[1:4]]
                atm_elmnt = line_splt[0].title()
                atm_rad = None
            else:
                raise ValueError('Lines shoud either have a length of 4 or 6.')

            self['atm_nr'].append(atm_nr)
            self['atm_pos'].append(atm_pos)
            self['atm_elmnt'].append(atm_elmnt)
            self['atm_rad'].append(atm_rad)

        atm_pos_multiplier = angstrom_per_bohr
        if mode == 4:
            atm_pos_multiplier = 1

        self['atm_nr'] = np.array(self['atm_nr'], dtype='int64')
        self['atm_pos'] = (
            np.array(self['atm_pos'], dtype='float64') * atm_pos_multiplier
        )
        self['atm_rad'] = np.array(self['atm_rad'], dtype='float64')

    def _read_turbomole_seg_section(self, cosmofile):

        line = next(cosmofile)

        while line:
            try:
                line = next(cosmofile).strip()
                line_splt = line.split()
                if len(line_splt) != 9:
                    if self['seg_nr']:
                        break
                    else:
                        continue
                    
            except StopIteration:
                break
            if not line:
                break

            self['seg_nr'].append(int(line_splt[0]) - 1)
            self['seg_atm_nr'].append(int(line_splt[1]) - 1)
            self['seg_pos'].append([float(val) for val in line_splt[2:5]])
            self['seg_charge'].append(float(line_splt[5]))
            self['seg_area'].append(float(line_splt[6]))
            self['seg_sigma_raw'].append(float(line_splt[7]))
            self['seg_potential'].append(float(line_splt[8]))

        self['seg_nr'] = np.array(self['seg_nr'], dtype='int64')
        self['seg_atm_nr'] = np.array(
            self['seg_atm_nr'], dtype='int64'
        )
        self['seg_pos'] = (
            np.array(self['seg_pos'], dtype='float64') * angstrom_per_bohr
        )
        self['seg_charge'] = np.array(
            self['seg_charge'], dtype='float64'
        )
        self['seg_area'] = np.array(
            self['seg_area'], dtype='float64'
        )
        self['seg_sigma_raw'] = np.array(
            self['seg_sigma_raw'], dtype='float64'
        )
        self['seg_potential'] = (
            np.array(self['seg_potential'], dtype='float64')
            * kJdivmol_per_hartree
        )

    def _read_turbomolesp(self):

        with open(self['filepath'], 'r') as cosmofile:

            for i_line, line in enumerate(cosmofile):

                line = line.strip()

                if line == '$info':
                    line = next(cosmofile).strip()
                    self['method'] = (
                        f'{line.split(";")[-2]}_{line.split(";")[-1]}'.lower()
                    )

                self._read_single_float(
                    line, 'area', r'area\s*=\s*([0-9+-.eE]+)', angstrom_per_bohr**2
                )

                self._read_single_float(
                    line, 'volume', r'volume\s*=\s*([0-9+-.eE]+)', angstrom_per_bohr**3
                )

                self._read_single_float(
                    line,
                    'energy_tot',
                    r'Total\s+energy\s+corrected.*=\s*([0-9+-.eE]+)',
                    kJdivmol_per_hartree,
                )

                self._read_single_float(
                    line,
                    'energy_dielectric',
                    r'Dielectric\s+energy\s+\[a\.u\.\]\s*=\s*([0-9+-.eE]+)',
                    kJdivmol_per_hartree,
                )

                if line == '$coord_rad' or i_line == 0 and line == '$coord_car':
                    self._read_turbomole_atom_section(cosmofile)

                if line == '$segment_information':
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

            if not line or '###' in line:
                break
            line_splt = line.split()

            self['atm_nr'].append(atm_nr)
            atm_nr += 1

            self['atm_pos'].append([float(val) for val in line_splt[1:]])
            self['atm_elmnt'].append(line_splt[0].title())

        self['atm_nr'] = np.array(self['atm_nr'], dtype='int64')
        self['atm_pos'] = np.array(self['atm_pos'], dtype='float64')

    def _read_orca_atom_radii(self, orcasp_file):

        line = next(orcasp_file)

        while line:
            line = next(orcasp_file).strip()
            if not line or '---' in line:
                break
            line_splt = line.split()

            self['atm_rad'].append(line_splt[3])

        self['atm_rad'] = (
            np.array(self['atm_rad'], dtype='float64') * angstrom_per_bohr
        )

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

            self['seg_nr'].append(seg_nr)
            seg_nr += 1
            self['seg_atm_nr'].append(int(line_splt[-1]))
            self['seg_pos'].append([float(val) for val in line_splt[0:3]])
            self['seg_charge_uncorrected'].append(float(line_splt[5]))
            self['seg_area'].append(float(line_splt[3]))
            self['seg_potential_uncorrected'].append(float(line_splt[4]))

        self['seg_nr'] = np.array(self['seg_nr'], dtype='int64')
        self['seg_atm_nr'] = np.array(
            self['seg_atm_nr'], dtype='int64'
        )
        self['seg_pos'] = (
            np.array(self['seg_pos'], dtype='float64') * angstrom_per_bohr
        )
        self['seg_charge_uncorrected'] = np.array(
            self['seg_charge_uncorrected'], dtype='float64'
        )
        self['seg_area'] = (
            np.array(self['seg_area'], dtype='float64') * angstrom_per_bohr**2
        )
        self['seg_sigma_raw_uncorrected'] = (
            self['seg_charge_uncorrected'] / self['seg_area']
        )
        self['seg_potential_uncorrected'] = (
            np.array(self['seg_potential_uncorrected'], dtype='float64')
            * kJdivmol_per_hartree
        )

    def _read_orca_cpcm_correction_section(self, orcasp_file):

        line = next(orcasp_file)
        self._read_single_float(
            line,
            'energy_dielectric',
            r'Corrected\s+dielectric\s+energy\s+=\s*([0-9+-.eE]+)',
            kJdivmol_per_hartree,
        )

        self['energy_tot_uncorrected'] = self['energy_tot']
        self['energy_tot'] = (
            self['energy_tot_uncorrected']
            - self['energy_dielectric_uncorrected']
            + self['energy_dielectric']
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

        assert self['seg_charge_uncorrected'].size == len(corrected_charges)
        self['seg_charge'] = np.array(corrected_charges, dtype='float64')
        self['seg_sigma_raw'] = (
            self['seg_charge'] / self['seg_area']
        )

    def _read_orca_adjacency_matrix(self, orcasp_file):

        atm_nr = self['atm_pos'].shape[0]
        adjacency_marix = []
        for line_nr in range(atm_nr):
            line = next(orcasp_file)
            line = [int(entry.strip()) for entry in line.split()]
            adjacency_marix.append(line)

        self['adjacency_matrix'] = np.array(adjacency_marix, dtype='int')
    
    def calculate_molecular_dipole(self):
        if 'dipole_moment' not in self:
            raise ValueError('The specified input file did not include dipole moment information.')
        return float(np.linalg.norm(self['dipole_moment']))
            
    def _read_orcasp(self):

        with open(self['filepath'], 'r') as orcasp_file:

            line = next(orcasp_file).strip()
            self['name'], self['method'] = (
                entry.strip() for entry in line.split(':')
            )

            for line in orcasp_file:

                line = line.strip()

                self._read_single_float(
                    line, 'area', r'\s*([0-9+-.eE]+)\s+#\s*Area', angstrom_per_bohr**2
                )

                self._read_single_float(
                    line,
                    'volume',
                    r'\s*([0-9+-.eE]+)\s+#\s*Volume',
                    angstrom_per_bohr**3,
                )

                self._read_single_float(
                    line,
                    'energy_tot',
                    r'FINAL\s+SINGLE\s+POINT\s+ENERGY\s*([0-9+-.eE]+)',
                    kJdivmol_per_hartree,
                )

                self._read_single_float(
                    line,
                    'energy_dielectric_uncorrected',
                    r'\s*([0-9+-.eE]+)\s+#\s*CPCM\s+dielectric\s+energy',
                    kJdivmol_per_hartree,
                )

                if line == '#XYZ_FILE':
                    self._read_orca_atom_coordinates(orcasp_file)

                if 'CARTESIAN COORDINATES (A.U.)' in line:
                    self._read_orca_atom_radii(orcasp_file)

                if 'SURFACE POINTS (A.U.)' in line:
                    self._read_orca_seg_section(orcasp_file)

                if '#COSMO_corrected' in line or '#CPCM_corrected' in line:
                    self._read_orca_cpcm_correction_section(orcasp_file)

                if '#ADJACENCY_MATRIX' in line:
                    self._read_orca_adjacency_matrix(orcasp_file)

                if 'DIPOLE MOMENT (Debye)' in line:
                    line = next(orcasp_file).strip()
                    self['dipole_moment'] = np.array(
                        [float(v) for v in line.strip().split()[-3:]]
                    )


class PyCrsError(Exception):
    pass

if __name__ == '__main__':
    main()

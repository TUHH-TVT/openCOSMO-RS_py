 
import pytest
from opencosmorspy.input_parsers import SigmaProfileParser


bohr_per_angstrom = 0.52917721092

hartree_per_kJdivmol = 2625.499639479
elpotatmu_per_volt = 27.211386245988


class TestTurbomole:

    @pytest.fixture(autouse=True)
    def sigma_profile_parser_instance(self):
        filepath = 'tests/COSMO_TURBOMOLE/tBuOH_turbomole.cosmo'
        self.spp = SigmaProfileParser(filepath, 'turbomole')


    def test_method(self):

        assert self.spp['method'] == 'b-p_def2-tzvp'


    def test_area(self):

        assert abs(self.spp['area'] - 434.24*bohr_per_angstrom**2) < 1e-12


    def test_volume(self):

        assert abs(self.spp['volume'] - 764.08*bohr_per_angstrom**3) < 1e-12


    def test_energy_tot_corr(self):

        assert abs(self.spp['energy_tot'] -
                   (-233.7935299418)*hartree_per_kJdivmol) < 1e-12


    def test_energy_tot_uncorr(self):

        assert abs(self.spp['energy_tot_uncorrected'] -
                   (-233.7935049908)*hartree_per_kJdivmol) < 1e-12


    def test_energy_dielectric_corr(self):

        assert abs(self.spp['energy_dielectric'] -
                    (-0.0093272823)*hartree_per_kJdivmol) < 1e-12


    def test_energy_dielectric_uncorr(self):

        assert abs(self.spp['energy_dielectric_uncorrected'] -
                    (-0.0093023312)*hartree_per_kJdivmol) < 1e-12


    def test_atm_nr(self):

        assert self.spp['atm_nr'].shape == (15,)


    def test_atm_pos(self):

        assert abs(self.spp['atm_pos'][-2, 0] - 3.22439434711559*bohr_per_angstrom) < 1e-6
        assert abs(self.spp['atm_pos'][-2, 1] - 2.39663244387626*bohr_per_angstrom) < 1e-6
        assert abs(self.spp['atm_pos'][-2, 2] - (-0.03042937804566)*bohr_per_angstrom) < 1e-6


    def test_atm_elmnt(self):

        assert len(self.spp['atm_elmnt']) == 15
        assert self.spp['atm_elmnt'][0] == 'H'
        assert self.spp['atm_elmnt'][1] == 'O'
        assert self.spp['atm_elmnt'][2] == 'C'


    def test_atm_rad(self):

        assert len(self.spp['atm_rad']) == 15
        assert abs(self.spp['atm_rad'][0] - 1.30000) < 1e-6
        assert abs(self.spp['atm_rad'][1] - 1.72000) < 1e-6
        assert abs(self.spp['atm_rad'][2] - 2.00000) < 1e-6


    def test_seg_nr(self):
        
        assert len(self.spp['seg_nr']) == 469
        assert self.spp['seg_nr'][-1] + 1 == 469


    def test_seg_atm_nr(self):
        
        assert len(self.spp['seg_atm_nr']) == 469
        assert self.spp['seg_atm_nr'][-1] + 1 == 13
        assert self.spp['seg_atm_nr'][49] + 1 == 4


    def test_seg_pos(self):

        assert self.spp['seg_pos'].shape == (469, 3)
        assert abs(self.spp['seg_pos'][-8, 0] - (-0.037084018)*bohr_per_angstrom) < 1e-8
        assert abs(self.spp['seg_pos'][-8, 1] - 5.956613275*bohr_per_angstrom) < 1e-8
        assert abs(self.spp['seg_pos'][-8, 2] - (-1.204782040)*bohr_per_angstrom) < 1e-8


    def test_seg_charge(self):

        assert len(self.spp['seg_charge']) == 469
        assert abs(self.spp['seg_charge'][-5] - -0.000023266) < 1e-8


    def test_seg_area(self):

        assert len(self.spp['seg_area']) == 469
        assert abs(self.spp['seg_area'][-5] - 0.144657251) < 1e-8


    def test_seg_sigma_raw(self):

        assert len(self.spp['seg_sigma_raw']) == 469
        assert abs(self.spp['seg_sigma_raw'][-5] -
                   (-0.000023266)/0.144657251) < 1e-8


    def test_seg_potential(self):

        assert len(self.spp['seg_potential']) == 469
        assert abs(self.spp['seg_potential'][-2] -
                   0.000680544*hartree_per_kJdivmol*bohr_per_angstrom) < 1e-8



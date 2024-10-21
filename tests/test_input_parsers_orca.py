 
import pytest
from opencosmorspy.input_parsers import SigmaProfileParser


bohr_per_angstrom = 0.52917721092

hartree_per_kJdivmol = 2625.499639479
elpotatmu_per_volt = 27.211386245988


class TestOrca:
    
    @pytest.fixture(autouse=True)
    def sigma_profile_parser_instance(self):
        filepath = (r'tests/COSMO_ORCA/C2H5NO_002_Acetamide'
                    r'/COSMO_TZVPD'
                    r'/C2H5NO_002_Acetamide_CnfS1_c000.orcacosmo')
        self.sigma_profile_parser = SigmaProfileParser(filepath, 'orca')

    def test_method(self):
        
        assert ((self.sigma_profile_parser['method'])=='DFT_COSMO_BP86_def2-TZVP+def2-TZVPD_SP')
    
    def test_area(self):
        
        assert abs(self.sigma_profile_parser['area']-346.719451230*bohr_per_angstrom**2) < 1e-12
        
    def test_volume(self):
        
        assert abs(self.sigma_profile_parser['volume']-512.628916151*bohr_per_angstrom**3) < 1e-12
        
    def test_energy_tot(self):
        
        assert abs(self.sigma_profile_parser['energy_tot'] -
                   (-209.329723227055)*hartree_per_kJdivmol) < 1e-12
    
    def test_energy_dielectric(self):
        
        assert abs(self.sigma_profile_parser['energy_dielectric'] -
                    (-0.022773896)*hartree_per_kJdivmol) < 1e-12
        
    def test_atm_nr(self):
        
        assert self.sigma_profile_parser['atm_nr'][-2] == 7 and len(info['atm_nr'])==9
    
    
    def test_atm_pos(self):
        
        assert abs(self.sigma_profile_parser['atm_pos'][-2, 0]-(-3.086454081)*bohr_per_angstrom) < 1e-6
        assert abs(self.sigma_profile_parser['atm_pos'][-2, 1]-1.235399813*bohr_per_angstrom) < 1e-6
        assert abs(self.sigma_profile_parser['atm_pos'][-2, 2]-(0.123316481)*bohr_per_angstrom) < 1e-6
    
    def test_atm_elmnt(self):
        
        assert len(self.sigma_profile_parser['atm_elmnt']) == 9
        assert self.sigma_profile_parser['atm_elmnt'][0] == 'C'
        assert self.sigma_profile_parser['atm_elmnt'][2] == 'O'
        assert self.sigma_profile_parser['atm_elmnt'][3] == 'N'
        assert self.sigma_profile_parser['atm_elmnt'][4] == 'H'
        
    
    def test_atm_rad(self):
        
        assert len(self.sigma_profile_parser['atm_rad']) == 9
        assert abs(self.sigma_profile_parser['atm_rad'][0]-3.779452268*bohr_per_angstrom) < 1e-6
        assert abs(self.sigma_profile_parser['atm_rad'][2]-3.250328950*bohr_per_angstrom) < 1e-6
        assert abs(self.sigma_profile_parser['atm_rad'][3]-3.458198825*bohr_per_angstrom) < 1e-6
        assert abs(self.sigma_profile_parser['atm_rad'][4]-2.456643974*bohr_per_angstrom) < 1e-6

    
    def test_seg_nr(self):
        
        assert len(self.sigma_profile_parser['seg_nr']) == 590
        assert self.sigma_profile_parser['seg_nr'][-1] == 589
            
    
    def test_seg_atm_nr(self):
        
        assert len(self.sigma_profile_parser['seg_atm_nr']) == 590
        assert self.sigma_profile_parser['seg_atm_nr'][-1] == 8
        assert self.sigma_profile_parser['seg_atm_nr'][-60] == 8
        assert self.sigma_profile_parser['seg_atm_nr'][-61] == 7
        
    def test_seg_pos(self):
        
        assert self.sigma_profile_parser['seg_pos'].shape == (590, 3)
        assert self.sigma_profile_parser['seg_pos'][-8, 0] == -4.952453963*bohr_per_angstrom
        assert self.sigma_profile_parser['seg_pos'][-8, 1] == -1.872670855*bohr_per_angstrom
        assert self.sigma_profile_parser['seg_pos'][-8, 2] ==  -2.883912107*bohr_per_angstrom
        
            
    def test_seg_charge(self):
        
        assert len(self.sigma_profile_parser['seg_charge']) == 590
        assert self.sigma_profile_parser['seg_charge'][-5] == -0.000157636
                    
    
    def test_seg_area(self):
        
        assert len(self.sigma_profile_parser['seg_area']) == 590
        assert self.sigma_profile_parser['seg_area'][-5] == 0.086357755*bohr_per_angstrom**2
    
    
    def test_seg_sigma_raw(self):
        
        assert len(self.sigma_profile_parser['seg_sigma_raw']) == 590
        assert self.sigma_profile_parser['seg_sigma_raw'][-5] == (
            -0.000157636/(0.086357755*bohr_per_angstrom**2))
    
    
    # def test_seg_potential(self):
        
    #     assert len(self.sigma_profile_parser['seg_potential']) == 320
    #     assert self.sigma_profile_parser['seg_potential'][-2] == -0.000271805*elpotatmu_per_volt



    

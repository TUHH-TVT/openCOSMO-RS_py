 
import pytest
from opencosmorspy.input_parsers import COSMOParser


bohr_per_angstrom = 0.52917721092

hartree_per_kJdivmol = 2625.499639479
elpotatmu_per_volt = 27.211386245988


class TestOrca:
    
    @pytest.fixture(autouse=True)
    def cosmoparser_instance(self):
        filepath = (r'tests/COSMO_ORCA/C2H5NO_002_Acetamide'
                    r'/COSMO_TZVPD'
                    r'/C2H5NO_002_Acetamide_CnfS1_c000.orcacosmo')
        self.cosmoparser = COSMOParser(filepath, 'orca')

    def test_method(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert ((info['method'])=='DFT_COSMO_BP86_def2-TZVP+def2-TZVPD_SP')
    
    def test_area(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert abs(info['area']-346.719451230*bohr_per_angstrom**2) < 1e-12
        
    def test_volume(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert abs(info['volume']-512.628916151*bohr_per_angstrom**3) < 1e-12
        
    def test_energy_tot(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert abs(info['energy_tot'] -
                   (-209.329723227055)*hartree_per_kJdivmol) < 1e-12
    
    def test_energy_dielectric(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert abs(info['energy_dielectric'] -
                    (-0.022773896)*hartree_per_kJdivmol) < 1e-12
        
    def test_atm_nr(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert info['atm_nr'][-2] == 7 and len(info['atm_nr'])==9
    
    
    def test_atm_pos(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert abs(info['atm_pos'][-2, 0]-(-3.086454081)*bohr_per_angstrom) < 1e-6
        assert abs(info['atm_pos'][-2, 1]-1.235399813*bohr_per_angstrom) < 1e-6
        assert abs(info['atm_pos'][-2, 2]-(0.123316481)*bohr_per_angstrom) < 1e-6
    
    def test_atm_elmnt(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert len(info['atm_elmnt']) == 9
        assert info['atm_elmnt'][0] == 'C'
        assert info['atm_elmnt'][2] == 'O'
        assert info['atm_elmnt'][3] == 'N'
        assert info['atm_elmnt'][4] == 'H'
        
    
    def test_atm_rad(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert len(info['atm_rad']) == 9
        assert abs(info['atm_rad'][0]-3.779452268*bohr_per_angstrom) < 1e-6
        assert abs(info['atm_rad'][2]-3.250328950*bohr_per_angstrom) < 1e-6
        assert abs(info['atm_rad'][3]-3.458198825*bohr_per_angstrom) < 1e-6
        assert abs(info['atm_rad'][4]-2.456643974*bohr_per_angstrom) < 1e-6

    
    def test_seg_nr(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert len(info['seg_nr']) == 590
        assert info['seg_nr'][-1] == 589
            
    
    def test_seg_atm_nr(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert len(info['seg_atm_nr']) == 590
        assert info['seg_atm_nr'][-1] == 8
        assert info['seg_atm_nr'][-60] == 8
        assert info['seg_atm_nr'][-61] == 7
        
    def test_seg_pos(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert info['seg_pos'].shape == (590, 3)
        assert info['seg_pos'][-8, 0] == -4.952453963*bohr_per_angstrom
        assert info['seg_pos'][-8, 1] == -1.872670855*bohr_per_angstrom
        assert info['seg_pos'][-8, 2] ==  -2.883912107*bohr_per_angstrom
        
            
    def test_seg_charge(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert len(info['seg_charge']) == 590
        assert info['seg_charge'][-5] == -0.000157636
                    
    
    def test_seg_area(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert len(info['seg_area']) == 590
        assert info['seg_area'][-5] == 0.086357755*bohr_per_angstrom**2
    
    
    def test_seg_sigma_raw(self):
        
        info = self.cosmoparser.get_cosmo_info()
        assert len(info['seg_sigma_raw']) == 590
        assert info['seg_sigma_raw'][-5] == (
            -0.000157636/(0.086357755*bohr_per_angstrom**2))
    
    
    # def test_seg_potential(self):
        
    #     info = self.cosmoparser.get_cosmo_info()
    #     assert len(info['seg_potential']) == 320
    #     assert info['seg_potential'][-2] == -0.000271805*elpotatmu_per_volt



    

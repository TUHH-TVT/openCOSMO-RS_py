import numpy as np
import pytest

import scipy.constants as spcon

from opencosmorspy.cosmors import COSMORS
from opencosmorspy.parameterization import Parameterization

@pytest.fixture
def parameterization_orca():

    par_dct = {}
    par_dct['qc_program'] = 'orca'

    par_dct['descriptor_lst'] = ['sigma', 'sigma_orth', 'elmnt_nr', 'group',
                                 'mol_charge']

    par_dct['a_eff'] = 6.226
    par_dct['r_av'] = 0.5   # Klamt and Eckert, 2000
    par_dct['sigma_min'] = -0.15
    par_dct['sigma_max'] = 0.15
    par_dct['sigma_step'] = 0.001   # Klamt, 1995?
    par_dct['sigma_grid'] = (np.arange(par_dct['sigma_min'],
                                 par_dct['sigma_max']+par_dct['sigma_step'],
                                 par_dct['sigma_step']))
    par_dct['sigma_orth_min'] = -0.15
    par_dct['sigma_orth_max'] = 0.15
    par_dct['sigma_orth_step'] = 0.001
    par_dct['sigma_orth_grid'] = (
        np.arange(par_dct['sigma_orth_min'],
                  par_dct['sigma_orth_max']+par_dct['sigma_orth_step'],
                  par_dct['sigma_orth_step']))

    par_dct['mf_use_sigma_orth'] = True
    par_dct['mf_alpha'] = 7.579075e6
    par_dct['mf_r_av_corr'] = 1.0   # Klamt et al., 1998
    par_dct['mf_f_corr'] = 2.4   # Klamt et al., 1998

    par_dct['hb_term'] = 'default'
    par_dct['hb_c'] = 2.7488747e7
    par_dct['hb_c_T'] = 1.5   # Klamt and Eckert, 2000
    par_dct['hb_sigma_thresh'] = 7.686e-03
    par_dct['hb_don_elmnt_nr_arr'] = np.array(
        [1]+[entry for entry in range(100, 151, 1)])
    par_dct['hb_acc_elmnt_nr_arr'] = np.array([6, 7, 8, 9,
                                         15, 16, 17,
                                         35, 53])
    par_dct['hb_pref_arr'] = np.ones(201, dtype='float')

    par_dct['comb_term'] = 'staverman_guggenheim'
    par_dct['comb_sg_z_coord'] = 10.0
    par_dct['comb_sg_a_std'] = 47.999
    # par_dct['comb_sg_expsc_exponent'] = 0.75

    par_dct['calculate_contact_statistics_molecule_properties'] = False

    par_dct['cosmospace_max_iter'] = 1000
    par_dct['cosmospace_conv_thresh'] = 1e-6
        
    par = Parameterization('default_orca')
    
    for key, value in par_dct.items():
        par.key = value

    return par



def test_cosmors_lng_1(parameterization_orca):
    print(type(parameterization_orca))
    
    crs = COSMORS(par=parameterization_orca)
    crs.add_molecule(['./tests/COSMO_ORCA/C2H2Cl4_001_1112tetrachloroethane'
                      r'/COSMO_TZVPD/C2H2Cl4_001_1112tetrachloroethane_CnfS1_c000.orcacosmo'])
    crs.add_molecule(['./tests/COSMO_ORCA/H2O_001/COSMO_TZVPD/H2O_c000.orcacosmo'])

    x = np.array([0.0, 1.0])
    T = 298.15

    crs.add_job(x, T, refst='pure_component')
    results = crs.calculate()

    lng_res_validated = np.array([[10.02243494, 0.]])
    lng_comb_validated = np.array([[-0.7728501, 0.        ]])
    
    assert np.all(np.abs(results['enth']['lng']-lng_res_validated)<1e-8)
    assert np.all(np.abs(results['comb']['lng']-lng_comb_validated)<1e-8)
    assert np.all(np.abs(results['tot']['lng']-
                         lng_res_validated-lng_comb_validated)<1e-8)
    
def test_cosmors_refstate_conversion(parameterization_orca):

    par = parameterization_orca
    par.calculate_contact_statistics_molecule_properties = True

    crs = COSMORS(par)
    crs.add_molecule([r'./tests/COSMO_ORCA/C2H2Cl4_001_1112tetrachloroethane'
                      r'/COSMO_TZVPD/C2H2Cl4_001_1112tetrachloroethane_CnfS1_c000.orcacosmo'])
    crs.add_molecule([r'./tests/COSMO_ORCA/CH4O_001_methanol/COSMO_TZVPD'
                      r'/CH4O_001_methanol_c000.orcacosmo'])
    crs.add_molecule([r'./tests/COSMO_ORCA/H2O_001/COSMO_TZVPD/H2O_c000.orcacosmo'])

    x = np.array([0.2, 0.5, 0.3])
    T = 298.15
    crs.clear_jobs()
    crs.add_job(x, T, refst='cosmo')
    results_cosmo_mix = crs.calculate()
    
    x = np.array([0.0, 1.0, 0.0])
    T = 298.15
    crs.clear_jobs()
    crs.add_job(x, T, refst='cosmo')
    results_cosmo_ref = crs.calculate()
    
    x = np.array([0.2, 0.5, 0.3])
    T = 298.15
    crs.clear_jobs()
    crs.add_job(x, T, refst='pure_component')
    results_pure_comp = crs.calculate()
    
    x = np.array([0.2, 0.5, 0.3])
    T = 298.15
    crs.clear_jobs()
    crs.add_job(x, T, refst='reference_mixture',
                x_refst=np.array([0, 1.0, 0.]))
    results_refmix = crs.calculate()
    
    
    print('COSMO reference state - MIXTURE')
    print('Partial molar           - pm_A_int ',
          results_cosmo_mix['enth']['pm_A_int'])
    print('Average interaction mol - aim_A_int ',
          results_cosmo_mix['enth']['aim_A_int'])
    print('')
    
    print('COSMO reference state - PURE METHANOL')
    print('Partial molar           - pm_A_int ',
          results_cosmo_ref['enth']['pm_A_int'])
    print('Average interaction mol - aim_A_int ',
          results_cosmo_ref['enth']['aim_A_int'])
    print('')
    
    print('PURE COMPONENT reference state')
    print('Partial molar           - pm_A_int ',
          results_pure_comp['enth']['pm_A_int'])
    print('Average interaction mol - aim_A_int ',
          results_pure_comp['enth']['aim_A_int'])
    # print('Manual comparison for component 2')
    # print('Partial molar           - pm_A_int ',
    #       pm_A_int_mix[0, 1]-pm_A_int_ref[0, 1])
    # print('Average interaction mol - aim_A_int ',
    #       aim_A_int_mix[0, 1]-aim_A_int_ref[0, 1])
    # print('')

    print('REFERENCE MIXTURE reference state - PURE METHANOL')
    print('Partial molar           - pm_A_int ',
          results_refmix['enth']['pm_A_int'])
    print('Average interaction mol - aim_A_int ',
          results_refmix['enth']['aim_A_int'])
    # print('Manual comparison for component 2')
    # print('Partial molar           - pm_A_int ',
    #       pm_A_int_mix[0, 1]-pm_A_int_ref[0, 1])
    # print('Average interaction mol - aim_A_int ',
    #       aim_A_int_mix[0, 1]-aim_A_int_ref[0, 1])
    # print('')

    # Check identity of pure component ref
    prop_lst = ['lng', 'pm_A_int', 'pm_A_hb',
                'pm_E_int']
    for prop in prop_lst:
        assert(np.abs(results_cosmo_mix['enth'][prop][0, 1] -
                      results_cosmo_ref['enth'][prop][0, 1] -
                      results_pure_comp['enth'][prop][0, 1]) < 1e-10)

    assert (np.abs(results_cosmo_mix['tot']['lng'][0, 1] -
                  results_cosmo_ref['tot']['lng'][0, 1] -
                  results_pure_comp['tot']['lng'][0, 1]) < 1e-10)

    # Check identity of reference mixture ref
    prop_lst = ['lng', 'pm_A_int', 'pm_A_hb',
                'pm_E_int']
    for prop in prop_lst:
        assert(np.abs(results_cosmo_mix['enth'][prop][0, 1] -
                      results_cosmo_ref['enth'][prop][0, 1] -
                      results_refmix['enth'][prop][0, 1]) < 1e-10)


def test_pm_vs_aim(parameterization_orca):

    par = parameterization_orca
    par.calculate_contact_statistics_molecule_properties = True

    crs = COSMORS(par)
    crs.add_molecule([r'./tests/COSMO_ORCA/C2H2Cl4_001_1112tetrachloroethane'
                      r'/COSMO_TZVPD/C2H2Cl4_001_1112tetrachloroethane_CnfS1_c000.orcacosmo'])
    crs.add_molecule([r'./tests/COSMO_ORCA/CH4O_001_methanol/COSMO_TZVPD'
                      r'/CH4O_001_methanol_c000.orcacosmo'])
    crs.add_molecule([r'./tests/COSMO_ORCA/H2O_001/COSMO_TZVPD/H2O_c000.orcacosmo'])


    x = np.array([0.2, 0.5, 0.3])
    T = 298.15

    crs.add_job(x, T, refst='cosmo')
    results = crs.calculate()

    pm_A_int_mix = results['enth']['pm_A_int']
    aim_A_int_mix = results['enth']['aim_A_int']

    # Cross-validate pm vs aim energy (works in cosmo refst):
    print((x*aim_A_int_mix).sum())
    print((x*pm_A_int_mix).sum())
    assert (np.abs(((x*aim_A_int_mix).sum() -
                    (x*pm_A_int_mix).sum()) /
                   (x*pm_A_int_mix).sum()) < 1e-6)


def test_pm_gibbs_helmholtz(parameterization_orca):

    par = parameterization_orca
    par.calculate_contact_statistics_molecule_properties = True

    crs = COSMORS(par)
    crs.add_molecule([r'./tests/COSMO_ORCA/C2H2Cl4_001_1112tetrachloroethane'
                      r'/COSMO_TZVPD/C2H2Cl4_001_1112tetrachloroethane_CnfS1_c000.orcacosmo'])
    crs.add_molecule([r'./tests/COSMO_ORCA/CH4O_001_methanol/COSMO_TZVPD'
                      r'/CH4O_001_methanol_c000.orcacosmo'])
    crs.add_molecule([r'./tests/COSMO_ORCA/H2O_001/COSMO_TZVPD/H2O_c000.orcacosmo'])


    x = np.array([0.2, 0.5, 0.3])
    T = 298.15

    crs.add_job(x, T, refst='cosmo')
    results = crs.calculate()

    del_T = 0.001
    crs.clear_jobs()
    crs.add_job(x, T+del_T, refst='cosmo')
    results_T2 = crs.calculate()

    pm_E_int_mix = results['enth']['pm_E_int']

    # Numerical Gibbs Helmholtz calculation of interaction energy
    pm_E_int_mix_GH = -spcon.R*T**2/del_T * (results_T2['tot']['lng'] -
                                              results['tot']['lng'])

    # Check partial molar interaction energy against gibbs.helmholtz
    assert np.all(np.abs((pm_E_int_mix-pm_E_int_mix_GH)/pm_E_int_mix) < 1e-4)

    



# def test_misfit_array():
#     test = False
#     if test:
#         n_seg = len(sigma_arr)
#         A_mf_test = np.nan*np.ones((n_seg, n_seg), dtype='float64')
#         for idx1 in range(n_seg):
#             for idx2 in range(n_seg):
#                 A_mf_test[idx1, idx2] = (
#                     self.par.a_eff*self.par.mf_alpha*0.5 *
#                     (sigma_arr[idx1]+sigma_arr[idx2]) *
#                     ((sigma_arr[idx1]+sigma_arr[idx2])+
#                      self.par.mf_f_corr *
#                      (sigma_orth_arr[idx1]+sigma_orth_arr[idx2])))
#         print('Calculated A_mf with slow test method')
#         A_mf = A_mf_test
    

# def test_hb_array():
    
#     test = False
#     if test:
#         n_seg = len(sigma_arr)
#         A_hb_test = np.nan*np.ones((n_seg, n_seg), dtype='float64')
#         del_sigma_hb_don_mult_acc_test = np.nan*np.ones((n_seg, n_seg),
#                                                         dtype='float64')
#         for idx1 in range(n_seg):
#             for idx2 in range(n_seg):
                
#                 if sigma_arr[idx1] < sigma_arr[idx2]:
#                     idxd = idx1
#                     idxa = idx2
#                 else:
#                     idxd = idx2
#                     idxa = idx1
                
#                 ENd = elmnt_nr_arr[idxd]
#                 ENa = elmnt_nr_arr[idxa]
                
#                 if ENd in self.par.hb_don_elmnt_nr_arr:
#                     fd = self.par.hb_pref_arr[ENd]
#                 else:
#                     fd = 0.0
#                 if ENa in self.par.hb_acc_elmnt_nr_arr:
#                     fa = self.par.hb_pref_arr[ENa]
#                 else:
#                     fa = 0.0
                
               
#                 buffdb1 = 1.0 - self.par.hb_c_T + self.par.hb_c_T * (298.15 / (T));
#                 if buffdb1 > 0:
#                     CHB_T = self.par.hb_c * buffdb1
#                 else:
#                     CHB_T = 0
                     
#                 buffdb1 = 0
#                 buffdb2 = 0
#                 if sigma_arr[idxa] - self.par.hb_sigma_thresh > 0:
#                     buffdb1 = sigma_arr[idxa] - self.par.hb_sigma_thresh
                
#                 if sigma_arr[idxd] + self.par.hb_sigma_thresh < 0:
#                     buffdb2 = sigma_arr[idxd] + self.par.hb_sigma_thresh
                    
#                 del_sigma_hb_don_mult_acc_test[idx1, idx2] = fd*fa*buffdb1*buffdb2

#                 A_hb_test[idx1, idx2] = fd*fa * CHB_T * self.par.a_eff * buffdb1 * buffdb2
                
#         print('Calculated A_hb with slow test method')
#         print('STOP')
#         A_hb = A_hb_test
    

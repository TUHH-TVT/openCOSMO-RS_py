from abc import ABC, abstractmethod
import copy
import os
import re
import shutil
import subprocess as spr
import sys
import time
import zipfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import scipy.constants as spcon

import opencosmorspy.input_parsers as ip

class ConformerGenerator(object):

    @staticmethod
    def get_embedded_mol(smiles):

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError('Could not generate mol struct from smiles: {}'.
                             format(smiles))

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)

        return mol

    # modified from https://github.com/dkoes/rdkit-scripts
    @staticmethod
    def getDihedralMatches(mol):
        '''return list of atom indices of dihedrals'''
        #this is rdkit's "strict" pattern
        pattern = r'*~[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]~*'
        qmol = Chem.MolFromSmarts(pattern)
        matches = mol.GetSubstructMatches(qmol)
        #these are all sets of 4 atoms, uniquify by middle two
        uniqmatches = []
        seen = set()
        for (a,b,c,d) in matches:
            if (b,c) not in seen:
                seen.add((b,c))
                uniqmatches.append((a,b,c,d))
        return uniqmatches

    def __init__(self, n_cores, settings_nondef, dir_job, name, smiles,
                 charge):

        self._n_cores = n_cores

        mol = Chem.MolFromSmiles(smiles)
        dihedral_matches = self.getDihedralMatches(mol)
        n_rotmatches = len(dihedral_matches)

        # heuristic taken from https://doi.org/10.1021/ci2004658
        if n_rotmatches <= 7:
            n_confs = 50
        if n_rotmatches >= 8 and n_rotmatches <= 12:
            n_confs = 200
        if n_rotmatches >= 13:
            n_confs = 300

        settings = {
            'rdkconfgen_n_conf_generated': n_confs,
            'rdkconfgen_rms_threshhold': 1.0,
            'orca_keywords_scf': []}
        if settings_nondef:
            settings.update(settings_nondef)

        print('- settings used (default and non-default): ', settings)

        self.dir_job = dir_job
        self._setup_folder()

        self.name = name
        self.charge = charge
        self.settings = settings

        self.smiles = smiles

        self.step_names = []
        self.step_folders = []

    def execute(self):

        mol = ConformerGenerator.get_embedded_mol(self.smiles)

        charge_from_molecule = Chem.rdmolops.GetFormalCharge(mol)

        if self.charge != charge_from_molecule:
            raise ValueError('The charge of following molecule does not agree with the smiles: ' + self.name)

        self._save_initial_state(mol)

        self._generate_rdkit_conformers(mol)

        method = ORCA_AM1_COSMO(1)
        self._calculate_orca(method)

        self._sort_energy(method)

        self._filter(lambda orcafiles: [orcafiles[0]])

        method = ORCA_DFT_COSMO(self._n_cores)
        self._calculate_orca(method)

        self._log_results(True)

        self._cleanup()

    def _log_results(self, last=False):

        # try logging the previous step results
        index_to_log = -2
        if last:
            index_to_log = -1
        try:
            print('- {:s}, n_confs: {:s}'.format(
                self.step_names[index_to_log],
                len(os.listdir(os.path.join(self.dir_job,
                                            self.step_folders[index_to_log],
                                            'res_cosmo')))))
        except Exception:
            pass

    def _save_initial_state(self, mol):

        dir_step = self._add_subdir_step('init')

        os.mkdir(os.path.join(dir_step, 'res_cosmo'))
        Chem.MolToXYZFile(mol, os.path.join(dir_step, 'res_cosmo',
                                            self.name+'.xyz'))

    def _add_subdir_step(self, step_name, copy_files_from_previous_step=False):

        if self.step_folders:
            nr = int(self.step_folders[-1].split('_')[0]) + 1
        else:
            nr = 0

        subdir = f'{nr:02d}_{step_name}'

        dir_step = os.path.join(self.dir_job, subdir)
        os.mkdir(dir_step)

        self.step_names.append(step_name)
        self.step_folders.append(subdir)

        if copy_files_from_previous_step:
            dir_step_prev = os.path.join(self.dir_job, self.step_folders[-2])
            shutil.copytree(os.path.join(dir_step_prev, 'res_cosmo'),
                            os.path.join(dir_step, 'inp_cosmo'))

        self._log_results()

        return dir_step

    def _setup_folder(self):

        if os.path.isdir(self.dir_job):
            while True:
                inp = input(f'Should dir "{self.dir_job}" be removed [y/n] ')
                # inp = 'y'
                if inp.lower() == 'n':
                    sys.exit()
                elif inp.lower() == 'y':
                    shutil.rmtree(self.dir_job)
                    time.sleep(1)
                    break
        os.makedirs(self.dir_job)

    def _generate_rdkit_conformers(self, mol):

        dir_step = self._add_subdir_step('rdkconfgen')

        _ = Chem.AllChem.EmbedMultipleConfs(
            mol, numConfs=self.settings['rdkconfgen_n_conf_generated'],
            randomSeed=0xf00d)

        mmff_optimized = Chem.AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=4, maxIters=2000)

        energies = np.array([res[1] for res in mmff_optimized if res[0] == 0])
        energies *= 4.187*1E3
        rel_probs = (np.exp(-energies/(spcon.R*298.15)) /
                     (np.exp(-(energies/(spcon.R*298.15)).min())))

        cnfs = mol.GetConformers()

        n_conf = len(energies)
        sw_retain = np.ones(n_conf, dtype='bool')
        sw_retain[rel_probs < 1e-2] = False
        for idx0 in range(n_conf):
            for idx1 in range(idx0+1, n_conf, 1):

                rms = None
                if sw_retain[idx0] and sw_retain[idx1]:
                    rms = Chem.AllChem.GetConformerRMS(mol, idx0, idx1,
                                                       prealigned=False)
                    if rms < self.settings['rdkconfgen_rms_threshhold']:
                        if energies[idx1] > energies[idx0]:
                            sw_retain[idx1] = False
                        else:
                            sw_retain[idx0] = False

        cnfs = np.array(cnfs)[sw_retain]
        energies = energies[sw_retain]
        rel_probs = rel_probs[sw_retain]

        idx_arr = energies.argsort()
        cnfs = cnfs[idx_arr]
        for idx, cnf in enumerate(cnfs):
            cnf.SetId(idx)
        energies = energies[idx_arr]
        rel_probs = rel_probs[idx_arr]

        mol2 = copy.deepcopy(mol)
        mol2.RemoveAllConformers()

        for cnf in cnfs:
            mol2.AddConformer(cnf)

        os.mkdir(os.path.join(dir_step, 'res_cosmo'))
        for idx in range(len(cnfs)):
            Chem.MolToXYZFile(mol2,
                              os.path.join(dir_step, 'res_cosmo',
                                           f'{self.name}_c{idx:03d}.xyz'), idx)

    def _calculate_orca(self, method):

        dir_step = self._add_subdir_step(method.step_name, True)

        dir_calc = os.path.join(dir_step, 'calculate')
        os.mkdir(dir_calc)

        files = os.listdir(os.path.join(dir_step, 'inp_cosmo'))

        # create xyz files from orcacosmo files if not available
        xyz_files = []
        for file in files:
            filename, file_extension = os.path.splitext(file)

            if file_extension == '.xyz':
                xyz_files.append(file)
                continue

            if filename + '.xyz' not in files:
                filepath_xyz = os.path.join(dir_step, 'inp_cosmo',
                                            filename+'.xyz')
                orcacosmofile = ip.COSMOParser(
                    os.path.join(dir_step, 'inp_cosmo', file),
                    'orca')
                orcacosmofile.save_xyz_file(filepath_xyz)
                xyz_files.append(filename + '.xyz')

        for file in xyz_files:
            # if not file.endswith('.xyz'):
            #     continue

            filename = os.path.splitext(file)[0]
            dir_struct = os.path.join(os.path.join(dir_calc, filename))
            os.mkdir(dir_struct)
            shutil.copy2(os.path.join(dir_step, 'inp_cosmo', file),
                         os.path.join(os.path.join(dir_calc, filename, file)))

            path_res = method.execute(
                os.path.join(dir_calc, filename, file), dir_step, self.charge,
                self.settings)

    def _sort_energy(self, method, step_name='sort_energy'):

        dir_step = self._add_subdir_step(step_name, True)

        energies = []
        cosmo_files = list(os.listdir(os.path.join(dir_step, 'inp_cosmo')))
        for orcacosmo_file in cosmo_files:
            orcacosmo = ip.COSMOParser(
                 os.path.join(dir_step, 'inp_cosmo', orcacosmo_file), 'orca')
            cosmo_info = orcacosmo.get_cosmo_info()
            energies.append(cosmo_info['energy_tot'])

        cosmo_files = [x for _, x in sorted(zip(energies, cosmo_files))]

        _, file_ext = os.path.splitext(os.path.join(dir_step,
                                                          'inp_cosmo',
                                                          cosmo_files[0]))

        os.mkdir(os.path.join(dir_step, 'res_cosmo'))
        for idx, cosmo_file in enumerate(cosmo_files):
            shutil.copy2(os.path.join(dir_step, 'inp_cosmo', cosmo_file),
                         os.path.join(dir_step, 'res_cosmo',
                                      f'{self.name}_c{idx:03d}'+file_ext))

    def _filter(self, filtering_function, step_name='filter'):

        dir_step = self._add_subdir_step(step_name, True)

        cosmo_files = list(os.listdir(os.path.join(dir_step, 'inp_cosmo')))
        cosmo_files.sort()
        cosmo_files = filtering_function(cosmo_files)
        os.mkdir(os.path.join(dir_step, 'res_cosmo'))
        for cosmo_file in cosmo_files:
            shutil.copy2(os.path.join(dir_step, 'inp_cosmo', cosmo_file),
                         os.path.join(dir_step, 'res_cosmo', cosmo_file))

    def _cleanup(self):

        # Zip results
        zipname = 'calculation_files.zip'
        zipfile_path = os.path.join(self.dir_job, zipname)
        with zipfile.ZipFile(zipfile_path, 'w', compression=zipfile.ZIP_LZMA) \
                as zipf:
            for root, _, files in os.walk(self.dir_job):
                for file in files:
                    if file != zipname:
                        zipf.write(os.path.join(root, file),
                                   os.path.relpath(os.path.join(root, file),
                                                   os.path.join(self.dir_job,
                                                                '..')))

        # Copy final results to appropriate folder
        shutil.copytree(os.path.join(self.dir_job,
                                     self.step_folders[-1],
                                     'res_cosmo'),
                        os.path.join(self.dir_job,
                                     'COSMO_TZVPD'))

        # Delete zipped folders
        for step_folder in self.step_folders:
            shutil.rmtree(os.path.join(self.dir_job, step_folder))

class ORCA(ABC):
    def __init__(self, n_cores):

        self.filepath_inp = ''
        self.charge = None

        self.method = ''
        self.step_name = 'orca'

        self.filename_final_log = 'log_output.dat'
        self.filename_final_xyz = ''
        self.filename_final_cpcm = ''

        self._n_cores = n_cores

        output = spr.run(["whereis", "orca"], capture_output=True)
        self._orca_full_path = output.stdout.decode('utf-8').split()[1].strip()


    def execute(self, filepath_inp, dir_step, charge, settings):

        self.dir_step = dir_step
        self.dir_old = os.getcwd()
        self.dir_work = os.path.dirname(filepath_inp)

        self.charge = charge
        self.filepath_inp = filepath_inp
        self.filename = os.path.basename(filepath_inp)
        self.structname = os.path.splitext(self.filename)[0]

        try:
            os.chdir(self.dir_work)

            self._write_input(settings)

            self._call_orca()

            self._concatenate_output()

            os.chdir(self.dir_old)

            self._assort_result('geo_opt.xyz')

        except Exception:
            os.chdir(self.dir_old)
            raise

    @abstractmethod
    def _write_input(self, settings):
        pass

    def _call_orca(self):

        with open('log_output.dat', 'w') as out:
            spr.run([self._orca_full_path, 'input.dat'], stdout=out, stderr=out)

    def _concatenate_output(self):

        with open(f'{self.structname}.orcacosmo', 'w') as file:

            file.write(f'{self.structname} : {self.method}\n')

            file.write('\n'+'#'*50+'\n')
            file.write('#ENERGY\n')
            line_final_energy = ''
            with open(self.filename_final_log, 'r') as log_file:
                for line in log_file:
                    re_match = re.match(
                        r'.*FINAL\s+SINGLE\s+POINT\s+ENERGY(.+)', line)
                    if re_match:
                        # According to manual and full energy contributions
                        # output in log, this can be expected to include the
                        # difelectric energy
                        line_final_energy = (
                            'FINAL TOTAL SINGLE POINT ENERGY{}'.format(
                                re_match.groups()[0]))
            file.write(line_final_energy)

            file.write('\n'+'#'*50+'\n')
            file.write('#XYZ_FILE\n')
            with open(self.filename_final_xyz, 'r') as xyz_file:
                for line in xyz_file:
                    file.write(line)

            file.write('\n'+'#'*50+'\n')
            file.write('#COSMO\n')
            with open(self.filename_final_cpcm, 'r') as cpcm_file:
                for line in cpcm_file:
                    file.write(line)

    def _assort_result(self, filename_xyz):

        if not os.path.isdir(os.path.join(self.dir_step, 'res_cosmo')):
            os.mkdir(os.path.join(self.dir_step, 'res_cosmo'))

        # Copy concatenated orcacosmo file
        shutil.copy2(os.path.join(self.dir_work,
                                  f'{self.structname}.orcacosmo'),
                     os.path.join(self.dir_step, 'res_cosmo',
                                  f'{self.structname}.orcacosmo'))


class ORCA_AM1_COSMO(ORCA):
    def __init__(self, n_cores):

        assert(n_cores == 1) # this method does not support parallelization

        super().__init__(n_cores)

        self.method = 'AM1_COSMO'
        self.step_name = 'orca_am1_cosmo'

        self.filename_final_xyz = 'geo_opt.xyz'
        self.filename_final_cpcm = 'geo_opt.cpcm'

    def _write_input(self, settings):

        lines = []
        lines.append('%MaxCore 4000')

        lines.append('')
        lines.append('%cpcm')
        lines.append('radius[1]  1.30')  # H
        lines.append('radius[6]  2.00')  # C
        lines.append('radius[7]  1.83')  # N
        lines.append('radius[8]  1.72')  # O
        lines.append('radius[9]  1.72')  # F
        #lines.append('radius[14]  ???')  # Si
        #lines.append('radius[15]  ???')  # P
        lines.append('radius[16]  2.16')  # S
        lines.append('radius[17]  2.05')  # Cl
        lines.append('radius[35]  2.16')  # Br
        lines.append('radius[53]  2.32')  # I
        lines.append('radius[3]  1.57')  # Li
        lines.append('radius[11]  1.80')  # Na
        lines.append('radius[19]  2.29')    # K
        lines.append('end')

        nondef_keyword_str = (
            ' '.join(settings['orca_keywords_scf']))

        lines.append('')
        lines.append(f'! AM1 OPT CPCM {nondef_keyword_str} ')

        lines.append('')
        lines.append('%base "geo_opt"')

        lines.append('')
        lines.append(f'* xyzfile {self.charge} 1 {self.filename} ')

        with open('input.dat', 'w') as file:
            file.write('\n'.join(lines))


class ORCA_DFT_COSMO(ORCA):
    def __init__(self, n_cores):

        super().__init__(n_cores)

        self.method = 'DFT_COSMO_BP86_def2-TZVP+def2-TZVPD_SP'
        self.step_name = 'orca_dft_cosmo'

        self.filename_final_xyz = 'geo_opt_tzvp.xyz'
        self.filename_final_cpcm = 'single_point_tzvpd.cpcm'

    def _write_input(self, settings):

        lines = []
        lines.append('%MaxCore 4000')
        lines.append('')

        lines.append('%cpcm')
        lines.append('radius[1]  1.30')  # H
        lines.append('radius[6]  2.00')  # C
        lines.append('radius[7]  1.83')  # N
        lines.append('radius[8]  1.72')  # O
        lines.append('radius[9]  1.72')  # F
        #lines.append('radius[14]  ???')  # Si
        #lines.append('radius[15]  ???')  # P
        lines.append('radius[16]  2.16')  # S
        lines.append('radius[17]  2.05')  # Cl
        lines.append('radius[35]  2.16')  # Br
        lines.append('radius[53]  2.32')  # I
        lines.append('radius[3]  1.57')  # Li
        lines.append('radius[11]  1.80')  # Na
        lines.append('radius[19]  2.29')    # K
        lines.append('end')
        lines.append('')

        parallel_string = ''
        if self._n_cores > 1:
            parallel_string = f' PAL{self._n_cores}'

        nondef_keyword_str = (
            ' '.join(settings['orca_keywords_scf']))
        lines.append(f'! DFT OPT CPCM BP86 def2-TZVP{parallel_string} '
                     f'{nondef_keyword_str}')
        lines.append('')

        lines.append('%base "geo_opt_tzvp"')
        lines.append('')

        lines.append(f'* xyzfile {self.charge} 1 {self.filename}')
        lines.append('')

        lines.append('$new_job')
        lines.append('')

        lines.append(f'! def2-TZVPD SP{parallel_string}')
        lines.append('')

        lines.append('%base "single_point_tzvpd"')
        lines.append('')

        lines.append(f'* xyzfile {self.charge} 1 geo_opt_tzvp.xyz')
        lines.append('')

        with open('input.dat', 'w') as file:
            file.write('\n'.join(lines))


if __name__ == "__main__":

    dir_work = r'test_job'

    n_cores = 1

    name_smiles_dct = {}
    # name_smiles_dct['C9H18O_002_2Nonanone_CnfS1'] = ('CCCCCCCC(=O)C', 0)
    # name_smiles_dct['C2H6O2_001_ethyleneglycol'] = ('OCCO', 0)
    name_smiles_dct['CH4O_001_methanol'] = ('CO', 0)

    # Non-default settings
    # (- Set 'orca_keywords_scf': ['NOSOSCF'] in case of scf converg. issues)
    settings_nondef = {
            'orca_keywords_scf': ['NOSOSCF']}
    # settings_nondef = None

    for name, (smiles, charge) in name_smiles_dct.items():
        print(f'starting: {name}')
        dir_job = os.path.join(dir_work, name)
        cg = ConformerGenerator(n_cores, settings_nondef, dir_job,
                                name, smiles, charge)
        cg.execute()
        print(f'finished: {name}')

    print('\nFinished everything')

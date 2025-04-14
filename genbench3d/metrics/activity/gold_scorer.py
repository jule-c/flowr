import os
import tempfile
import numpy as np

from rdkit import Chem
from rdkit.Chem import Mol
from genbench3d.utils import rdkit_conf_to_ccdc_mol
from typing import List
try:
    from ccdc.docking import Docker
    from ccdc.io import Entry, EntryWriter
    CCDC_IMPORTED = True
except ImportError:
    class Docker: pass
    class Entry: pass
    class EntryWriter: pass
    CCDC_IMPORTED = False

class GoldScorer():
    
    def __init__(self,
                 protein_path: str,
                 native_ligand: Mol,
                 ) -> None:
        
        if not CCDC_IMPORTED:
            raise ImportError('CCDC is not available')
        
        self.docker = Docker()
        settings = self.docker.settings
        self.temp_dir = tempfile.mkdtemp()
        self.settings_conf_file = os.path.join(self.temp_dir, 'api_gold.conf')
        settings.output_directory = self.temp_dir
        settings.fitness_function = None
        settings.rescore_function = 'plp'
        
        settings.add_protein_file(protein_path)

        protein = settings.proteins[0]
        native_ligand_ccdc = rdkit_conf_to_ccdc_mol(native_ligand)
        native_ligand_ccdc.identifier = 'test_ligand_gold'
        settings.binding_site = settings.BindingSiteFromLigand(protein=protein, 
                                                               ligand=native_ligand_ccdc, 
                                                               distance=10.)
        
    def score_mols(self,
                   mols: List[Mol]):
        
        self.docker.settings.clear_ligand_files()
        
        results_file = os.path.join(self.temp_dir, 'rescore.mol2')
        if os.path.exists(results_file):
            os.remove(results_file)
        
        assert len(mols) > 0, 'You must test at least one molecule'
            
        ligand_names = []
        for i, gen_mol in enumerate(mols):
            ligand = rdkit_conf_to_ccdc_mol(gen_mol)
            ligand_name = f'Ligand_{i}'
            ligand_names.append(ligand_name)
            ligand.identifier = ligand_name
            ligand_entry = Entry.from_molecule(ligand)

            ligand_preparation = Docker.LigandPreparation()
            prepared_ligand = ligand_preparation.prepare(ligand_entry)
            prepared_ligand_filename = os.path.join(self.temp_dir, f'test_ligand_{i}.mol2')
            with EntryWriter(prepared_ligand_filename) as writer:
                writer.write(prepared_ligand)
            self.docker.settings.add_ligand_file(file_name=prepared_ligand_filename)

        self.docker.dock(file_name=self.settings_conf_file)
        
        scores = []
        current_i = -1 # to start at 0 in the loop
        for docked_ligand in self.docker.results.ligands:
            current_i += 1
            current_name = f'Ligand_{current_i}'
            while docked_ligand.identifier != current_name:
                # Detect missing scores
                current_i += 1
                current_name = f'Ligand_{current_i}'
                scores.append(np.nan)
            score = docked_ligand.attributes['Gold.PLP.Fitness']
            scores.append(float(score))
            
        while current_i + 1 < len(mols):
            # Detect missing scores
            current_i += 1
            current_name = f'Ligand_{current_i}'
            scores.append(np.nan)
            
        if len(scores) != len(mols):
            import pdb;pdb.set_trace()
            
        return scores
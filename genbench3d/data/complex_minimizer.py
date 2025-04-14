import logging

from rdkit import Chem
from rdkit.Chem import (AllChem, 
                        Mol)
from .structure import Pocket


class ComplexMinimizer():
    
    def __init__(self,
                 pocket: Pocket,
                 config: dict,
                 ) -> None:
        """
        The protein must be only the protein atoms, with hydrogens
        """
        self.pocket = pocket
        self.n_steps = config['n_steps']
        self.distance_constraint = config['distance_constraint']
    
    
    def minimize_ligand(self,
                        ligand_mol: Mol,
                        minimized_ligand_filepath: str = None,
                        ignore_pocket: bool = False,
                        ):
        
        ligand = Chem.AddHs(ligand_mol, addCoords=True)
        if not ignore_pocket:
            complx = Chem.CombineMols(self.pocket.mol, ligand)
        else:
            complx = ligand
            
        Chem.SanitizeMol(complx)
            
        try:
            mol_properties = AllChem.MMFFGetMoleculeProperties(complx, 
                                                                mmffVariant='MMFF94s')
            mmff = AllChem.MMFFGetMoleculeForceField(complx, 
                                                    mol_properties,
                                                    confId=0,
                                                    nonBondedThresh=10.0,
                                                    ignoreInterfragInteractions=False)
            mmff.Initialize()
            
            for idx in range(self.pocket.mol.GetNumAtoms(), complx.GetNumAtoms()):
                atom = complx.GetAtomWithIdx(idx)
                if atom.GetSymbol() != 'H':
                    mmff.MMFFAddPositionConstraint(idx, self.distance_constraint, 999.0)

            # get the initial energy
            E_init = mmff.CalcEnergy()

        except Exception as e:
            logging.warning(f'MMFF minimization exception: {e}')
            return None
            
        else:
            
            if not ignore_pocket :
                for i in range(self.pocket.mol.GetNumAtoms()):
                    mmff.AddFixedPoint(i)
                
            with Chem.SDWriter('test_complex_before.sdf') as w:
                w.write(complx)
                
            results = mmff.Minimize(maxIts=self.n_steps)
            not_converged = results

            with Chem.SDWriter('test_complex_after.sdf') as w:
                w.write(complx)

            E_final = mmff.CalcEnergy()
            if not_converged:
                print('Not converged')
                
            # print(E_final - E_init)
                
            # Chem.MolToPDBFile(complx, 'test_opti_complx.pdb')
            minimized_frags = Chem.GetMolFrags(complx, asMols=True)
            minimized_ligand = minimized_frags[-1]
            if minimized_ligand_filepath is not None:
                with Chem.SDWriter(minimized_ligand_filepath) as writer:
                    writer.write(minimized_ligand)
                    
            # import pdb;pdb.set_trace()
                    
            return minimized_ligand #, E_init - E_final
    
    
    # Inspired from LiGAN code by Ragoza et al.
    
    # def uff_minimize_ligand(self,
    #                        ligand_mol: Mol,
    #                        pocket_filepath: str = None,
    #                        minimized_ligand_filepath: str = None,
    #                        n_steps: int = 1000):
    #     ligand = Chem.AddHs(ligand_mol, addCoords=True)
        
    #     if pocket_filepath is not None:
    #         if not os.path.exists(pocket_filepath):
    #             pdb_filepath = pocket_filepath.replace('_pocket', '_protein_only')
    #             self.extract_pocket(ligand_mol, pdb_filepath, pocket_filepath)
    #         pocket = mda.Universe(pocket_filepath)
    #         # pocket_selection = pocket.select_atoms('protein')
    #         pocket = pocket.atoms.convert_to("RDKIT")
    #         complx = Chem.CombineMols(pocket, ligand)
    #     else:
    #         complx = ligand
            
    #     Chem.SanitizeMol(complx)
            
    #     try:
    #         # initialize force field
    #         uff = AllChem.UFFGetMoleculeForceField(
    #             complx, confId=0, ignoreInterfragInteractions=False
    #         )
    #         uff.Initialize()
            
    #         for idx in range(pocket.GetNumAtoms(), complx.GetNumAtoms()):
    #             atom = complx.GetAtomWithIdx(idx)
    #             if atom.GetSymbol() != 'H':
    #                 uff.UFFAddPositionConstraint(idx, 0.5, 9999.0)
                
    #         # for bond in complx.GetBonds():
    #         #     id1 = bond.GetBeginAtomIdx()
    #         #     id2 = bond.GetEndAtomIdx()
    #         #     uff.UFFAddDistanceConstraint(id1, id2, True, -0.5, 0.5, 99999.0)

    #         # get the initial energy
    #         E_init = uff.CalcEnergy()

    #     except Exception as e:
    #         if 'getNumImplicitHs' in str(e):
    #             print('No implicit valence')
    #         if 'bad params pointer' in str(e):
    #             print('Invalid atom type')
    #         print('UFF1 exception: ', e)
    #         import pdb;pdb.set_trace()
            
    #     if pocket_filepath is not None:
    #         for i in range(pocket.GetNumAtoms()):
    #             uff.AddFixedPoint(i)
            
    #     with Chem.SDWriter('test_complex_before.sdf') as w:
    #         w.write(complx)
            
    #     results = uff.Minimize(maxIts=n_steps)
    #     not_converged = results

    #     with Chem.SDWriter('test_complex_after.sdf') as w:
    #         w.write(complx)

    #     # import pdb;pdb.set_trace()

    #     # get the final energy
    #     E_final = uff.CalcEnergy()
    #     if not_converged:
    #         print('Not converged')
            
    #     # print(E_final - E_init)
            
    #     # Chem.MolToPDBFile(complx, 'test_opti_complx.pdb')
    #     minimized_frags = Chem.GetMolFrags(complx, asMols=True)
    #     minimized_ligand = minimized_frags[-1]
    #     if minimized_ligand_filepath is not None:
    #         with Chem.SDWriter(minimized_ligand_filepath) as writer:
    #             writer.write(minimized_ligand)
                
    #     return minimized_ligand, E_init - E_final
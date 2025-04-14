from molvs import Standardizer
from rdkit import Chem
from rdkit.Chem import Mol

# TODO: use the RDKit standardization (they incorporated molvs stategy in RDKit)
class MolStandardizer() :
    """Standardize molecules using molvs and a neutralization function
    """

    def __init__(self) -> None:
        self.standardizer = Standardizer()
        
    def standardize(self,
                    mol: Mol,
                    neutralize: bool = True) -> Mol:
        """Standardize a molecule using molvs

        :param mol: Input molecule
        :type mol: Mol
        :param neutralize: Set to True to neutralize the molecule, defaults to True
        :type neutralize: bool, optional
        :return: Standardized molecule
        :rtype: Mol
        """
        
        new_mol = Mol(mol)
        standard_mol = self.standardizer.standardize(new_mol)
        
        # Uncharge for later comparison, because some functional groups might
        # be protonated. PDBbind stores a neutral pH protonated version of 
        # the ligand
        if neutralize:
            standard_mol = self.neutralize_mol(standard_mol)
            
        prop_names = mol.GetPropNames()
        for prop_name in prop_names:
            value = mol.GetProp(prop_name)
            standard_mol.SetProp(prop_name, str(value))
        for conf in mol.GetConformers():
            conf_id = conf.GetId()
            new_conf = standard_mol.GetConformer(conf_id)
            prop_names = conf.GetPropNames(includePrivate=True, 
                                            includeComputed=True)
            for prop_name in prop_names:
                value = conf.GetProp(prop_name)
                new_conf.SetProp(prop_name, str(value))
            
        return standard_mol
    
    def neutralize_mol(self, 
                       mol: Mol) -> Mol:
        """Neutralize molecule (setting number of H of each atom) based on 
        https://www.rdkit.org/docs/Cookbook.html#neutralizing-molecules

        :param mol: Input molecule
        :type mol: Mol
        :return: Neutralized molecule
        :rtype: Mol
        """

        neutral_mol = Mol(mol)
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = neutral_mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = neutral_mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return neutral_mol
        
       
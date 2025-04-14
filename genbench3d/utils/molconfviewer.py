"""
molconfviewer - Visualize molecule conformations in Jupyter
"""

__version__ = "0.1.0"

import ipywidgets
import nglview

from IPython.display import display
from ipywidgets import interact, fixed
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDepictor import Compute2DCoords
from typing import Tuple, List, Dict, Any
from rdkit.Chem.rdMolAlign import AlignMolConformers
from nglview import NGLWidget

class MolConfViewer():
    """Class to generate views of molecule conformations
    
    :param widget_size: canvas size
    :type widget_size: tuple(int, int)
    :param style: type of drawing molecule, see 3dmol.js
    :type style: str in ['line', 'stick', 'sphere', 'cartoon']
    :param draw_surface: display SAS
    :type draw_surface: bool
    :param opacity: opacity of surface, ranging from 0 to 1
    :type opacity: float
    """
    
    def __init__(self, 
                 widget_size: Tuple[int, int]=(300, 300), 
                 styles: List[str] = ['stick', 'sphere'], 
                 draw_surface: bool = False, 
                 opacity: float = 0.5) -> None:
        """Setup the viewer
        """
        self.widget_size = widget_size
        for style in styles :
            assert style in ('line', 'stick', 'sphere', 'cartoon')
        self.styles = styles
        self.draw_surface = draw_surface
        self.opacity = opacity
    
    
    def view_mols(self,
                  mols: List[Mol],
                  properties: List[Dict[str, List[float]]] = None,
                  align: bool = True) -> None:
        """View all molecules from a list. There will be one slider for the 
        molecule, and one slider for the conformation in the molecule

        :param mols: Input molecules
        :type mols: List[Mol]
        :param properties: List of Dict of properties for each conformation of each 
            molecule, defaults to None
        :type properties: List[Dict[str, List[float]]], optional
        :param align: Set to True to align conformers of a molecule, defaults to True
        :type align: bool, optional
        """
        n_mols = len(mols)
        max_mol_i = n_mols - 1
        mol_i_inttext = ipywidgets.BoundedIntText(min=0, 
                                                max=max_mol_i, 
                                                step=1,
                                                description='Molecule number:')
        interact(self.view_mol_from_list, 
                 mols=fixed(mols),
                 mol_i=mol_i_inttext,
                 properties=fixed(properties),
                 align=fixed(align))
        
        
    def view_mol_from_list(self,
                           mols: List[Mol],
                           mol_i: int = 0,
                           properties: List[Dict[str, List[Any]]] = None,
                           align: bool = True) -> None:
        """View one molecule from the input list

        :param mols: List of molecules
        :type mols: List[Mol]
        :param mol_i: Index of the molecule in the list to view, defaults to 0
        :type mol_i: int, optional
        :param properties: List of conf properties of each mol, defaults to None
        :type properties: List[Dict[str, List[float]]], optional
        :param align: Set to True to align conformers for the molecule, defaults to True
        :type align: bool, optional
        """
        mol = mols[mol_i]
        mol_2d = Mol(mol)
        Compute2DCoords(mol_2d)
        display(mol_2d)
        if properties is not None:
            mol_properties = properties[mol_i]
        else:
            mol_properties = None
        self.view(mol,
                  properties=mol_properties,
                  align=align)
    
    
    def view(self, 
             mol: Mol,
             properties: Dict[str, Any] = None,
             align: str = True) -> None:
        """View a RDKit molecule in 3D, with a slider to explore conformations.
        
        :param mol: molecule to show conformers for
        :type mol: Mol
        :param properties: Dict of properties for each conformer
        :type properties: Dict[str, Any]
        """
        n_confs = mol.GetNumConformers()
        max_conf_id = n_confs - 1
        # conf_id_slider = ipywidgets.IntSlider(min=0, 
        #                                       max=max_conf_id, 
        #                                       step=1)
        conf_i_inttext = ipywidgets.BoundedIntText(min=0, 
                                                    max=max_conf_id, 
                                                    step=1,
                                                    description='Conformer ID:')
        
        if properties :
            assert isinstance(properties, dict), \
                'Type of properties must be a dict'
            for property_name, values in properties.items() :
                n_values = len(values)
                assert n_values == n_confs, \
                    f"""Length of property {property_name} = {n_values} must 
                    be same as number of conformers = {n_confs}"""
                    
        new_mol = Mol(mol)
        if align:
            AlignMolConformers(new_mol)
        
        interact(self.get_viewer, 
                 mol=fixed(new_mol), 
                 conf_i=conf_i_inttext,
                 properties=fixed(properties))
        # return conf_i_inttext
        
    def get_viewer(self, 
                    mol: Mol, 
                    conf_i: int = -1,
                    properties: Dict[str, Any] = None,
                    ) -> NGLWidget:
        """Draw a given conformation for a molecule in 3D 
        Largely inspired from
        https://birdlet.github.io/2019/10/02/py3dmol_example/
        initially coded with py3dmol, but it stopped working in 2023
        so it was changed to nglview
        
        :param mol: molecule to show conformers for
        :type mol: Mol
        :param conf_id: id of the RDKit Conformer in the Mol to visualize
        :type conf_id: int
        :param properties: Dict of properties for each conformer
        :type properties: Dict[str, Any]
        :return: molecule viewer for given conf_id
        :rtype: py3Dmol.view
        """
        
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        conf_id = conf_ids[conf_i]
        print(f'Conformer ID = {conf_id}')
        
        view = nglview.show_rdkit(mol, conf_id=conf_id, fmt='sdf')
        view.clear_representations()
        view.add_ball_and_stick(multipleBond=True, bondScale=0.5, bondSpacing=1.5)
        # view.add_ball_and_stick()
        
        if properties :
            for property_name, values in properties.items() :
                print(property_name, '=', values[conf_i])
        else:
            conf = mol.GetConformer(conf_id)
            for prop_name in conf.GetPropNames():
                print(prop_name, conf.GetProp(prop_name))
                
        return view
    
    
    # was used when py3dmol was working
    # def set_viewer_style(self, viewer) :
    #     style_d = {}
    #     for style in self.styles :
    #         if style == 'sphere' :
    #             style_d[style] = {'scale' : 0.25}
    #         else :
    #             style_d[style] = {}
    #     viewer.setStyle(style_d)
    #     return viewer
    
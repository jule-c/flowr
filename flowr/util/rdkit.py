import threading
from typing import Optional, Union

import biotite.structure.io.pdb as io_pdb
import numpy as np
import rdkit
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

x_map = {
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
    "hybridization": [
        rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        rdkit.Chem.rdchem.HybridizationType.S,
        rdkit.Chem.rdchem.HybridizationType.SP,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        rdkit.Chem.rdchem.HybridizationType.SP3,
        rdkit.Chem.rdchem.HybridizationType.SP2D,
        rdkit.Chem.rdchem.HybridizationType.SP3D,
        rdkit.Chem.rdchem.HybridizationType.SP3D2,
        rdkit.Chem.rdchem.HybridizationType.OTHER,
    ],
    "is_h_donor": [False, True],
    "is_h_acceptor": [False, True],
}


ArrT = np.ndarray


# *************************************************************************************************
# ************************************ Periodic Table class ***************************************
# *************************************************************************************************


class PeriodicTable:
    """Singleton class wrapper for the RDKit periodic table providing a neater interface"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        self._table = Chem.GetPeriodicTable()

        # Just to be certain that vocab objects are thread safe
        self._pt_lock = threading.Lock()

    def atomic_from_symbol(self, symbol: str) -> int:
        with self._pt_lock:
            symbol = symbol.upper() if len(symbol) == 1 else symbol
            atomic = self._table.GetAtomicNumber(symbol)

        return atomic

    def symbol_from_atomic(self, atomic_num: int) -> str:
        with self._pt_lock:
            token = self._table.GetElementSymbol(atomic_num)

        return token

    def valence(self, atom: Union[str, int]) -> int:
        with self._pt_lock:
            valence = self._table.GetDefaultValence(atom)

        return valence


# *************************************************************************************************
# ************************************* Global Declarations ***************************************
# *************************************************************************************************


PT = PeriodicTable()

IDX_BOND_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
BOND_IDX_MAP = {bond: idx for idx, bond in IDX_BOND_MAP.items()}

IDX_CHARGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: -1, 5: -2, 6: -3}
CHARGE_IDX_MAP = {charge: idx for idx, charge in IDX_CHARGE_MAP.items()}


# *************************************************************************************************
# *************************************** Util Functions ******************************************
# *************************************************************************************************

# TODO merge these with check functions in other files


def _check_shape_len(arr, allowed, name="object"):
    num_dims = len(arr.shape)
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed:
        raise RuntimeError(
            f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}"
        )


def _check_dim_shape(arr, dim, allowed, name="object"):
    shape = arr.shape[dim]
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if shape not in allowed:
        raise RuntimeError(
            f"Shape of {name} for dim {dim} must be in {allowed}, got {shape}"
        )


# *************************************************************************************************
# ************************************* External Functions ****************************************
# *************************************************************************************************


def mol_is_valid(
    mol: Chem.rdchem.Mol,
    with_hs: bool = True,
    connected: bool = True,
    add_hs=False,
) -> bool:
    """Whether the mol can be sanitised and, optionally, whether it's fully connected

    Args:
        mol (Chem.Mol): RDKit molecule to check
        with_hs (bool): Whether to check validity including hydrogens (if they are in the input mol), default True
        connected (bool): Whether to also assert that the mol must not have disconnected atoms, default True

    Returns:
        bool: Whether the mol is valid
    """

    if mol is None:
        return False

    mol_copy = Chem.Mol(mol)
    if not with_hs:
        mol_copy = Chem.RemoveAllHs(mol_copy)

    if add_hs:
        mol_copy = Chem.AddHs(mol_copy)

    try:
        AllChem.SanitizeMol(mol_copy)
    except Exception:
        return False

    n_frags = len(AllChem.GetMolFrags(mol_copy))
    if connected and n_frags != 1:
        return False

    return True


def remove_radicals(mol: Chem.Mol, sanitize: bool = True) -> Chem.Mol:
    """Remove free radicals from a molecule."""

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the current atom has any radical electrons
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals > 0:
            # Saturate the atom with hydrogen atoms
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radicals)
            atom.SetNumRadicalElectrons(0)

    if sanitize:
        AllChem.SanitizeMol(mol)

    return mol


def has_explicit_hydrogens(mol: Chem.rdchem.Mol) -> bool:
    """Check whether an RDKit molecule has explicit hydrogen atoms

    Args:
        mol (Chem.Mol): RDKit molecule object

    Returns:
        bool: True if the molecule has explicit hydrogen atoms, False otherwise
    """
    if mol is None:
        return False

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Hydrogen has atomic number 1
            return True

    return False


def has_radicals(mol: Chem.Mol) -> bool:
    """Check if a molecule has any free radicals."""

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the current atom has any radical electrons
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals > 0:
            return True

    return False


def calc_energy(mol: Chem.rdchem.Mol, per_atom: bool = False) -> float:
    """Calculate the energy for an RDKit molecule using the MMFF forcefield

    The energy is only calculated for the first (0th index) conformer within the molecule. The molecule is copied so
    the original is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        per_atom (bool): Whether to normalise by number of atoms in mol, default False

    Returns:
        float: Energy of the molecule or None if the energy could not be calculated
    """

    mol_copy = Chem.Mol(mol)

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props, confId=0)
        energy = ff.CalcEnergy()
        energy = energy / mol.GetNumAtoms() if per_atom else energy
    except:
        energy = None

    return energy


def optimise_mol(mol: Chem.rdchem.Mol, max_iters: int = 1000) -> Chem.rdchem.Mol:
    """Optimise the conformation of an RDKit molecule

    Only the first (0th index) conformer within the molecule is optimised. The molecule is copied so the original
    is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        max_iters (int): Max iterations for the conformer optimisation algorithm

    Returns:
        Chem.Mol: Optimised molecule or None if the molecule could not be optimised within the given number of
                iterations
    """

    mol_copy = Chem.Mol(mol)
    try:
        exitcode = AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
    except:
        exitcode = -1

    if exitcode == 0:
        return mol_copy

    return None


def conf_distance(
    mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol, fix_order: bool = True
) -> float:
    """Approximately align two molecules and then calculate RMSD between them

    Alignment and distance is calculated only between the default conformers of each molecule.

    Args:
        mol1 (Chem.Mol): First molecule to align
        mol2 (Chem.Mol): Second molecule to align
        fix_order (bool): Whether to fix the atom order of the molecules

    Returns:
        float: RMSD between molecules after approximate alignment
    """

    assert len(mol1.GetAtoms()) == len(mol2.GetAtoms())

    if not fix_order:
        raise NotImplementedError()

    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())

    # Firstly, centre both molecules
    coords1 = coords1 - (coords1.sum(axis=0) / coords1.shape[0])
    coords2 = coords2 - (coords2.sum(axis=0) / coords2.shape[0])

    # Find the best rotation alignment between the centred mols
    rotation, _ = Rotation.align_vectors(coords1, coords2)
    aligned_coords2 = rotation.apply(coords2)

    sqrd_dists = (coords1 - aligned_coords2) ** 2
    rmsd = np.sqrt(sqrd_dists.sum(axis=1).mean())
    return rmsd


# TODO could allow more args
def smiles_from_mol(
    mol: Chem.rdchem.Mol,
    canonical: bool = True,
    include_stereocenters: bool = True,
    remove_hs: bool = False,
    explicit_hs: bool = False,
) -> Union[str, None]:
    """Create a SMILES string from a molecule

    Args:
        mol (Chem.Mol): RDKit molecule object
        canonical (bool): Whether to create a canonical SMILES, default True
        explicit_hs (bool): Whether to embed hydrogens in the mol before creating a SMILES, default False. If True
                this will create a new mol with all hydrogens embedded. Note that the SMILES created by doing this
                is not necessarily the same as creating a SMILES showing implicit hydrogens.

    Returns:
        str: SMILES string which could be None if the SMILES generation failed
    """

    if mol is None:
        return None

    if explicit_hs:
        mol = Chem.AddHs(mol)

    if remove_hs:
        mol = Chem.RemoveHs(mol)

    try:
        smiles = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=include_stereocenters
        )
    except:
        smiles = None

    return smiles


def mol_from_smiles(
    smiles: str, explicit_hs: bool = False
) -> Union[Chem.rdchem.Mol, None]:
    """Create a RDKit molecule from a SMILES string

    Args:
        smiles (str): SMILES string
        explicit_hs (bool): Whether to embed explicit hydrogens into the mol

    Returns:
        Chem.Mol: RDKit molecule object or None if one cannot be created from the SMILES
    """

    if smiles is None:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) if explicit_hs else mol
    except Exception:
        mol = None

    return mol


def mol_from_atoms(
    coords: ArrT,
    tokens: list[str],
    bonds: Optional[ArrT] = None,
    charges: Optional[ArrT] = None,
    sanitise=True,
    remove_hs=False,
    kekulize=False,
):
    """Create RDKit mol from atom coords and atom tokens (and optionally bonds)

    If any of the atom tokens are not valid atoms (do not exist on the periodic table), None will be returned.

    If bonds are not provided this function will create a partial molecule using the atomics and coordinates and then
    infer the bonds based on the coordinates using OpenBabel. Otherwise the bonds are added to the molecule as they
    are given in the bond array.

    If bonds are provided they must not contain any duplicates.

    If charges are not provided they are assumed to be 0 for all atoms.

    Args:
        coords (np.ndarray): Coordinate tensor, shape [n_atoms, 3]
        atomics (list[str]): Atomic numbers, length must be n_atoms
        bonds (np.ndarray, optional): Bond indices and types, shape [n_bonds, 3]
        charges (np.ndarray, optional): Charge for each atom, shape [n_atoms]
        sanitise (bool): Whether to apply RDKit sanitization to the molecule, default True

    Returns:
        Chem.rdchem.Mol: RDKit molecule or None if one cannot be created
    """

    _check_shape_len(coords, 2, "coords")
    _check_dim_shape(coords, 1, 3, "coords")

    if coords.shape[0] != len(tokens):
        raise ValueError(
            "coords and atomics tensor must have the same number of atoms."
        )

    if bonds is not None:
        _check_shape_len(bonds, 2, "bonds")
        _check_dim_shape(bonds, 1, 3, "bonds")

    if charges is not None:
        _check_shape_len(charges, 1, "charges")
        _check_dim_shape(charges, 0, len(tokens), "charges")

    try:
        atomics = [PT.atomic_from_symbol(token) for token in tokens]
    except Exception:
        # print(f"Error: {e}")
        return None

    charges = charges.tolist() if charges is not None else [0] * len(tokens)

    # Add atom types and charges
    mol = Chem.EditableMol(Chem.Mol())
    for idx, atomic in enumerate(atomics):
        atom = Chem.Atom(atomic)
        atom.SetFormalCharge(charges[idx])
        mol.AddAtom(atom)

    # Add 3D coords
    conf = Chem.Conformer(coords.shape[0])
    for idx, coord in enumerate(coords.tolist()):
        conf.SetAtomPosition(idx, coord)

    mol = mol.GetMol()
    mol.AddConformer(conf)

    if bonds is None:
        return _infer_bonds(mol)

    # Add bonds if they have been provided
    mol = Chem.EditableMol(mol)
    for bond in bonds.astype(np.int32).tolist():
        start, end, b_type = bond

        if b_type not in IDX_BOND_MAP:
            # print(f"Invalid bond type {b_type}")
            return None

        # Don't add self connections
        if start != end:
            b_type = IDX_BOND_MAP[b_type]
            mol.AddBond(start, end, b_type)

    try:
        mol = mol.GetMol()
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception:
        # print("Error building the molecule")
        return None

    if remove_hs:
        mol = Chem.RemoveHs(mol)

    if kekulize:
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            try:
                mol = Chem.RemoveHs(mol)
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                # print("Error kekulizing the molecule")
                return None

    if sanitise:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # print("Error sanitising the molecule")
            return None

    return mol


def _infer_bonds(mol: Chem.rdchem.Mol):
    coords = mol.GetConformer().GetPositions().tolist()
    coord_strs = ["\t".join([f"{c:.6f}" for c in cs]) for cs in coords]
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    xyz_str_header = f"{str(mol.GetNumAtoms())}\n\n"
    xyz_strs = [
        f"{str(atom)}\t{coord_str}" for coord_str, atom in zip(coord_strs, atom_symbols)
    ]
    xyz_str = xyz_str_header + "\n".join(xyz_strs)

    try:
        pybel_mol = pybel.readstring("xyz", xyz_str)
    except Exception:
        pybel_mol = None

    if pybel_mol is None:
        return None

    mol_str = pybel_mol.write("mol")
    mol = Chem.MolFromMolBlock(mol_str, removeHs=False, sanitize=True)
    return mol


def generate_conformer(mol: Chem.rdchem.Mol, explicit_hs=True) -> Chem.rdchem.Mol:
    """Generate a conformer for an RDKit molecule

    Args:
        mol (Chem.Mol): RDKit molecule

    Returns:
        Chem.Mol: Molecule with generated conformer
    """
    # Copy the molecule to avoid modifying the original
    mol_copy = Chem.Mol(mol)
    mol_copy = Chem.AddHs(mol_copy) if explicit_hs else mol_copy

    # Generate the conformer
    AllChem.EmbedMolecule(mol_copy)

    return mol_copy


def write_sdf_file(sdf_path, molecules, name=""):
    w = Chem.SDWriter(str(sdf_path))
    for i, m in enumerate(molecules):
        m.SetProp("Name", f"{name}_{i}")
        if m is not None:
            w.write(m)
    w.close()


def canonicalize(smiles: str, include_stereocenters=True, remove_hs=False):
    mol = Chem.MolFromSmiles(smiles)
    if remove_hs:
        mol = Chem.RemoveHs(mol)

    if mol is not None:
        return Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=include_stereocenters
        )
    else:
        return None


def canonicalize_list(
    smiles_list,
    include_stereocenters=True,
    remove_hs=False,
):
    canonicalized_smiles = [
        canonicalize(smiles, include_stereocenters, remove_hs=remove_hs)
        for smiles in smiles_list
    ]
    # Remove None elements
    canonicalized_smiles = [s for s in canonicalized_smiles]

    return remove_duplicates(canonicalized_smiles)


def remove_duplicates(list_with_duplicates):
    unique_set = set()
    unique_list = []
    ids = []
    for i, element in enumerate(list_with_duplicates):
        if element not in unique_set and element is not None:
            unique_set.add(element)
            unique_list.append(element)
        else:
            ids.append(i)

    return unique_list, ids


def canonicalize_mol_list(
    mols: list[Chem.rdchem.Mol],
    ref_smiles: list[str],
    include_stereocenters=True,
    remove_hs=False,
):
    smiles_list = [
        smiles_from_mol(
            mol,
            canonical=True,
            include_stereocenters=include_stereocenters,
            remove_hs=remove_hs,
        )
        for mol in mols
    ]
    unique_smiles = []
    unique_mols = []
    for i, (smiles, mol) in enumerate(zip(smiles_list, mols)):
        if (
            smiles is not None
            and smiles not in unique_smiles
            and smiles not in ref_smiles
        ):
            unique_smiles.append(smiles)
            unique_mols.append(mol)
    return unique_mols


def sanitize_list(
    mols: list[Chem.rdchem.Mol],
    ref_mols: Optional[list[Chem.rdchem.Mol]] = None,
    ref_mols_with_hs: Optional[list[Chem.rdchem.Mol]] = None,
    pdbs: Optional[list[str]] = None,
    pdbs_with_hs: Optional[list[str]] = None,
    sanitize: bool = False,
    filter_uniqueness: bool = False,
    filter_pdb: bool = False,
):

    rdkit_valid = [mol_is_valid(mol, connected=True) for mol in mols]
    valid_mols = [mol for mol, valid in zip(mols, rdkit_valid) if valid]
    if sanitize:
        for mol in valid_mols:
            AllChem.SanitizeMol(mol)
    if ref_mols is not None:
        valid_ref_mols = [mol for mol, valid in zip(ref_mols, rdkit_valid) if valid]
    if ref_mols_with_hs is not None:
        valid_ref_mols_with_hs = [
            mol for mol, valid in zip(ref_mols_with_hs, rdkit_valid) if valid
        ]
    if pdbs is not None:
        valid_pdbs = [pdb for pdb, valid in zip(pdbs, rdkit_valid) if valid]
    if pdbs_with_hs is not None:
        valid_pdbs_with_hs = [
            pdb for pdb, valid in zip(pdbs_with_hs, rdkit_valid) if valid
        ]

    if filter_uniqueness:
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        unique_valid_smiles, duplicate_ids = canonicalize_list(valid_smiles)
        valid_mols = [mol for i, mol in enumerate(valid_mols) if i not in duplicate_ids]
        if ref_mols is not None:
            valid_ref_mols = [
                mol for i, mol in enumerate(valid_ref_mols) if i not in duplicate_ids
            ]
        if ref_mols_with_hs is not None:
            valid_ref_mols_with_hs = [
                mol
                for i, mol in enumerate(valid_ref_mols_with_hs)
                if i not in duplicate_ids
            ]
        if pdbs is not None:
            valid_pdbs = [
                pdb for i, pdb in enumerate(valid_pdbs) if i not in duplicate_ids
            ]
        if pdbs_with_hs is not None:
            valid_pdbs_with_hs = [
                pdb
                for i, pdb in enumerate(valid_pdbs_with_hs)
                if i not in duplicate_ids
            ]

    if len(valid_mols) == 0:
        return []

    if filter_pdb:
        read_fn = io_pdb.get_structure
        if pdbs is not None or pdbs_with_hs is not None:
            if pdbs is not None:
                assert pdbs_with_hs is None, "Cannot filter both pdbs and pdbs_with_hs"
                pdb_valid = [
                    (
                        read_fn(
                            io_pdb.PDBFile.read(str(pdb)), model=1, include_bonds=True
                        )
                        if pdb is not None
                        else None
                    )
                    for pdb in valid_pdbs
                ]
            elif pdbs_with_hs is not None:
                assert pdbs is None, "Cannot filter both pdbs and pdbs_with_hs"
                pdb_valid = [
                    (
                        read_fn(
                            io_pdb.PDBFile.read(str(pdb)), model=1, include_bonds=True
                        )
                        if pdb is not None
                        else None
                    )
                    for pdb in valid_pdbs_with_hs
                ]
            valid_pdbs = [pdb for pdb, valid in zip(valid_pdbs, pdb_valid) if valid]
            valid_mols = [mol for mol, valid in zip(valid_mols, pdb_valid) if valid]
            if ref_mols is not None:
                valid_ref_mols = [
                    mol for mol, valid in zip(valid_ref_mols, pdb_valid) if valid
                ]
            if ref_mols_with_hs is not None:
                valid_ref_mols_with_hs = [
                    mol
                    for mol, valid in zip(valid_ref_mols_with_hs, pdb_valid)
                    if valid
                ]
        else:
            raise ValueError("No PDB files provided to filter")

    if len(valid_mols) == 0:
        return []

    result = [valid_mols]
    if ref_mols is not None:
        result.append(valid_ref_mols)
    if ref_mols_with_hs is not None:
        result.append(valid_ref_mols_with_hs)
    if pdbs is not None:
        result.append(valid_pdbs)
    if pdbs_with_hs is not None:
        result.append(valid_pdbs_with_hs)
    if len(result) == 1:
        return result[0]
    return tuple(result)

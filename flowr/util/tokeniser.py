from __future__ import annotations

import pickle
import threading
from abc import ABC, abstractmethod
from typing import Optional, Union

import flowr.util.functional as smolF

indicesT = Union[list[int], list[list[int]]]


PICKLE_PROTOCOL = 4


# *** Util functions ***


def _check_unique(obj_list, name="objects"):
    if len(obj_list) != len(set(obj_list)):
        raise RuntimeError(f"{name} cannot contain duplicates")


def _check_type_all(obj_list, exp_type, name="list"):
    for obj in obj_list:
        if not isinstance(obj, exp_type):
            raise TypeError(f"all objects in {name} must be instances of {exp_type}")


# *** Tokeniser Interface ***


class Tokeniser(ABC):
    """Interface for tokeniser classes"""

    @abstractmethod
    def tokenise(self, sentences: list[str]) -> Union[list[str], list[int]]:
        pass

    @classmethod
    @abstractmethod
    def from_vocabulary(cls, vocab: Vocabulary) -> Tokeniser:
        pass


# *** Tokeniser Implementations ***

# TODO


# *** Vocabulary Implementations ***


class Vocabulary:
    """Vocabulary class which maps tokens <--> indices"""

    def __init__(self, tokens: list[str]):
        _check_unique(tokens, "tokens list")

        token_idx_map = {token: idx for idx, token in enumerate(tokens)}
        idx_token_map = {idx: token for idx, token in enumerate(tokens)}

        self.token_idx_map = token_idx_map
        self.idx_token_map = idx_token_map

        # Just to be certain that vocab objects are thread safe
        self._vocab_lock = threading.Lock()

        # So that we can save this object without assuming the above dictionaries are ordered
        self._tokens = tokens

    @property
    def size(self) -> int:
        return len(self)

    def __len__(self) -> int:
        with self._vocab_lock:
            length = len(self.token_idx_map)

        return length

    def contains(self, token: str) -> bool:
        with self._vocab_lock:
            contains = token in self.token_idx_map

        return contains

    def tokens_from_indices(self, indices: list[int]) -> list[str]:
        _check_type_all(indices, int, "indices list")
        with self._vocab_lock:
            tokens = [self.idx_token_map[idx] for idx in indices]

        return tokens

    def indices_from_tokens(
        self, tokens: list[str], one_hot: Optional[bool] = False
    ) -> indicesT:
        _check_type_all(tokens, str, "tokens list")

        with self._vocab_lock:
            indices = [self.token_idx_map[token] for token in tokens]

        if not one_hot:
            return indices

        one_hots = smolF.one_hot_encode(indices, len(self)).tolist()
        return one_hots

    def to_bytes(self) -> bytes:
        with self._vocab_lock:
            obj_bytes = pickle.dumps(self._tokens, protocol=PICKLE_PROTOCOL)

        return obj_bytes

    @staticmethod
    def from_bytes(data: bytes) -> Vocabulary:
        tokens = pickle.loads(data)
        return Vocabulary(tokens)


atom_encoder = {
    "H": 0,
    "Li": 1,
    "B": 2,
    "C": 3,
    "N": 4,
    "O": 5,
    "F": 6,
    "Na": 7,
    "Mg": 8,
    "Al": 9,
    "Si": 10,
    "P": 11,
    "S": 12,
    "Cl": 13,
    "K": 14,
    "Ca": 15,
    "Ti": 16,
    "V": 17,
    "Cr": 18,
    "Mn": 19,
    "Fe": 20,
    "Co": 21,
    "Ni": 22,
    "Cu": 23,
    "Zn": 24,
    "Ge": 25,
    "As": 26,
    "Se": 27,
    "Br": 28,
    "Zr": 29,
    "Mo": 30,
    "Ru": 31,
    "Rh": 32,
    "Pd": 33,
    "Ag": 34,
    "Cd": 35,
    "In": 36,
    "Sn": 37,
    "Sb": 38,
    "Te": 39,
    "I": 40,
    "Ba": 41,
    "Nd": 42,
    "Gd": 43,
    "Yb": 44,
    "Pt": 45,
    "Au": 46,
    "Hg": 47,
    "Pb": 48,
    "Bi": 49,
}
atom_decoder = {v: k for k, v in atom_encoder.items()}
atomic_nb = [
    1,
    3,
    5,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    19,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    34,
    35,
    40,
    42,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    53,
    56,
    60,
    64,
    70,
    78,
    79,
    80,
    82,
    83,
]


pocket_residue_names_apo_holo = [
    "00C",
    "2CO",
    "ALA",
    "ALY",
    "ARG",
    "ASN",
    "ASP",
    "CME",
    "CSD",
    "CSO",
    "CSS",
    "CSX",
    "CYS",
    "DM0",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "HOX",
    "HYP",
    "ILE",
    "KCX",
    "KPI",
    "LEU",
    "LLP",
    "LYS",
    "MEN",
    "MET",
    "MLY",
    "NEP",
    "OAS",
    "OCS",
    "OCY",
    "ORN",
    "PCA",
    "PHD",
    "PHE",
    "PRO",
    "PTR",
    "QPA",
    "SCY",
    "SEP",
    "SER",
    "SGB",
    "SNN",
    "SUN",
    "SVX",
    "SXE",
    "THR",
    "TIS",
    "TPO",
    "TRP",
    "TYR",
    "UNK",
    "VAL",
    "XCN",
    "YCM",
    "YOF",
]

res_name_encoder_apo_holo = {
    res: idx + 1 for idx, res in enumerate(pocket_residue_names_apo_holo)
}  # add 1 to account for the ligand
res_name_decoder_apo_holo = {
    idx + 1: res for idx, res in enumerate(pocket_residue_names_apo_holo)
}  # add 1 to account for the ligand

pocket_atom_names_apo_holo = [
    "C",
    "C1",
    "C10",
    "C1A",
    "C1T",
    "C2",
    "C2'",
    "C2A",
    "C2T",
    "C3",
    "C4",
    "C4'",
    "C5",
    "C5'",
    "C6",
    "C8",
    "C9",
    "CA",
    "CB",
    "CD",
    "CD1",
    "CD2",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "CG",
    "CG1",
    "CG2",
    "CH",
    "CH1",
    "CH2",
    "CH3",
    "CM1",
    "CM2",
    "CS",
    "CX",
    "CX1",
    "CX2",
    "CZ",
    "CZ2",
    "CZ3",
    "F",
    "N",
    "N1",
    "NC",
    "ND1",
    "ND2",
    "NE",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "NZ2",
    "O",
    "O1",
    "O10",
    "O11",
    "O12",
    "O13",
    "O14",
    "O1P",
    "O1T",
    "O2",
    "O2P",
    "O2T",
    "O3",
    "O3P",
    "O5",
    "O6",
    "O9",
    "OAC",
    "OCD",
    "OD",
    "OD1",
    "OD2",
    "OD3",
    "OE",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "OP1",
    "OP2",
    "OP3",
    "OP4",
    "OQ1",
    "OQ2",
    "OXT",
    "OZ",
    "OZ1",
    "P",
    "P1",
    "P2",
    "SD",
    "SG",
]

atom_name_encoder_apo_holo = {
    atom: idx + 1 for idx, atom in enumerate(pocket_atom_names_apo_holo)
}  # add 1 to account for padding and ligand atoms. Add len(res_name_encoder) to account for residues
atom_name_decoder_apo_holo = {
    idx + 1: atom for idx, atom in enumerate(pocket_atom_names_apo_holo)
}  # add 1 to account for padding and ligand atoms. Add len(res_name_encoder) to account for residues

pocket_atom_names_plinder = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG1",
    "CG2",
    "CD1",
    "H",
    "HA",
    "HB",
    "HG13",
    "HG12",
    "HG21",
    "HG22",
    "HG23",
    "HD11",
    "HD12",
    "HD13",
    "CG",
    "CD2",
    "HB3",
    "HB2",
    "HG",
    "HD21",
    "HD22",
    "HD23",
    "HB1",
    "CD",
    "OE1",
    "OE2",
    "HG3",
    "HG2",
    "HG11",
    "CE1",
    "CE2",
    "CZ",
    "HD1",
    "HD2",
    "HE1",
    "HE2",
    "HZ",
    "SG",
    "NE2",
    "HE22",
    "HE21",
    "OG1",
    "HG1",
    "OG",
    "OH",
    "HH",
    "SD",
    "CE",
    "HE3",
    "NZ",
    "HD3",
    "HZ1",
    "HZ2",
    "HZ3",
    "ND1",
    "HA3",
    "HA2",
    "NE",
    "NH1",
    "NH2",
    "HE",
    "HH12",
    "HH11",
    "HH22",
    "HH21",
    "OD1",
    "OD2",
    "ND2",
    "OXT",
    "NE1",
    "CE3",
    "CZ2",
    "CZ3",
    "CH2",
    "HH2",
    "N1",
    "C2",
    "C2'",
    "C3",
    "O3",
    "C4",
    "C4'",
    "C5",
    "C6",
    "C5'",
    "OP4",
    "P",
    "OP1",
    "OP2",
    "OP3",
    "H2'1",
    "H2'2",
    "H2'3",
    "H3",
    "H4'",
    "H6",
    "H5'2",
    "H5'3",
    "HP2",
    "HP3",
    "H1",
    "H2",
    "C2A",
    "C1A",
    "OAC",
    "H2A1",
    "H2A2",
    "H2A3",
    "HZ4",
    "HXT",
    "O1",
    "P1",
    "C1",
    "O2",
    "HN",
    "H1_1",
    "H1_2",
    "H1_3",
    "HC2",
    "H4_1",
    "H4_2",
    "H4_3",
    "H3_1",
    "H3_2",
    "H3_3",
    "O1P",
    "O2P",
    "O3P",
    "HA1",
    "HA4",
    "1HXT",
    "2HXT",
    "3HXT",
    "4HXT",
    "HB4",
    "HG4",
    "HD4",
    "HE4",
    "HH1",
    "HH13",
    "OZ",
    "HOZ",
    "C1T",
    "O1T",
    "O2T",
    "C2T",
    "HO1T",
    "HO2T",
    "H2T1",
    "H2T2",
    "H2T3",
    "1HX3",
    "2HX2",
    "OCD",
    "OD3",
    "P2",
    "C8",
    "C9",
    "O9",
    "C10",
    "O10",
    "O11",
    "O12",
    "O13",
    "O14",
    "H9_1",
    "H9_2",
    "H9_3",
    "CX1",
    "CX2",
    "HNZ",
    "HE23",
    "H2_1",
    "H2_2",
    "H2_3",
    "OD",
    "HD",
    "HH23",
    "HH24",
    "OZ1",
    "NZ2",
    "HZ21",
    "HZ22",
    "CH",
    "CH3",
    "HH31",
    "HH32",
    "HH33",
    "O6",
    "O5",
    "CX",
    "OQ1",
    "OQ2",
    "HQ2",
    "HZ_1",
    "HZ_2",
    "OE",
    "CH1",
    "CM1",
    "CM2",
    "HM11",
    "HM12",
    "HM13",
    "HM21",
    "HM22",
    "HM23",
    "NC",
    "CS",
    "HN1",
    "HC1",
    "H5_1",
    "H5_2",
    "H5_3",
    "HE_1",
    "HE_2",
    "HE_3",
    "HOE",
    "H1P",
    "H2P",
    "F",
    "HOH",
    "HD14",
    "HD24",
    "HE11",
    "HE12",
    "HE13",
    "HE14",
    "HE24",
    "HO",
    "HOXT",
]

pocket_residue_names_plinder = [
    "ILE",
    "LEU",
    "ALA",
    "GLU",
    "VAL",
    "PHE",
    "CYS",
    "GLN",
    "THR",
    "SER",
    "TYR",
    "MET",
    "LYS",
    "HIS",
    "GLY",
    "PRO",
    "ARG",
    "ASP",
    "ASN",
    "CME",
    "TRP",
    "LLP",
    "OAS",
    "SGB",
    "CSD",
    "SEP",
    "OCY",
    "TIS",
    "SCY",
    "OCS",
    "QPA",
    "KPI",
    "PHD",
    "MEN",
    "SUN",
    "TPO",
    "CSO",
    "YCM",
    "UNK",
    "ALY",
    "SVX",
    "PTR",
    "KCX",
    "HOX",
    "PCA",
    "00C",
    "DM0",
    "XCN",
    "SXE",
    "HYP",
    "CSX",
    "CSS",
    "ORN",
    "MLY",
    "2CO",
    "NEP",
    "YOF",
    "SNN",
]

atom_name_encoder_plinder = {
    atom: idx + 2 for idx, atom in enumerate(pocket_atom_names_plinder)
}  # add 1 to account for padding. Add len(res_name_encoder) to account for residues
atom_name_decoder_plinder = {
    v: k for k, v in atom_name_encoder_plinder.items()
}  # add 1 to account for padding. Add len(res_name_encoder) to account for residues
res_name_encoder_plinder = {
    res: idx + 2 for idx, res in enumerate(pocket_residue_names_plinder)
}  # add 1 to account for padding
res_name_decoder_plinder = {v: k for k, v in res_name_encoder_plinder.items()}

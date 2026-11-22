"""Partial charge utilities for the electrostatic field (申请书公式(5)).

The FLOWR data pipeline stores pocket atom charges as the PDB formal charge
annotation (columns 79-80 of the PDB format), which is empty -- and therefore
zero -- for almost all protein atoms. A meaningful protein electrostatic field
needs force field partial charges instead (OPLS4 in the proposal).

This module provides:

  * `read_pqr`: parse a PQR file (PDB-like format where the occupancy and
    B-factor columns are replaced by partial charge and radius) as produced
    by Schrodinger (prepwizard + structconvert) or PDB2PQR.
  * `match_charges_to_system`: align the per-atom partial charges of a PQR
    file to the pocket atoms of a `PocketComplex` system, matching atoms by
    (chain, residue number, residue name, atom name). The returned tensor is
    in the exact pocket atom order of the system and can be injected into the
    model via the `pocket_charge_values` interface, where it takes precedence
    over the one-hot formal charges of the data pipeline.

All code here is additive and only used when the caller passes a charge file
(e.g. `--pocket_charge_file` in flowr/gen/generate_from_pdb.py).
"""

from typing import Optional

import numpy as np
import torch


def _norm(s) -> str:
    """Normalise a chain/residue/atom name for matching."""
    return str(s).strip().upper()


class PQRAtom(dict):
    """A single parsed PQR atom record."""


def read_pqr(pqr_path) -> list:
    """Parse a PQR file into a list of atom records.

    Each record is a dict with the keys: chain, res_num, res_name, atom_name,
    element, charge, radius.

    Both fixed-column PDB layout (charge in columns 55-62, radius in 63-70,
    as written by PDB2PQR) and whitespace-separated layouts (as written by
    Schrodinger structconvert) are supported.
    """
    atoms = []
    with open(pqr_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            record = {}
            try:
                # Fixed-column layout first (PDB2PQR style)
                record["atom_name"] = line[12:16].strip()
                record["res_name"] = line[17:20].strip()
                record["chain"] = line[21:22].strip()
                record["res_num"] = int(line[22:26].strip())
                record["charge"] = float(line[54:62].strip())
                record["radius"] = float(line[62:70].strip())
                # Columns can be shifted in some writers, so sanity check the
                # coordinates parse as well before accepting the result
                float(line[30:38].strip())
                float(line[38:46].strip())
                float(line[46:54].strip())
            except (ValueError, IndexError):
                # Whitespace fallback (Schrödinger structconvert style):
                # the last two tokens are charge and radius
                tokens = line.split()
                if len(tokens) < 9:
                    continue
                try:
                    charge = float(tokens[-2])
                    radius = float(tokens[-1])
                except ValueError:
                    continue
                # tokens: ATOM serial name resname [chain] resnum x y z q r
                if len(tokens) >= 11:
                    # with chain
                    record["atom_name"] = tokens[2]
                    record["res_name"] = tokens[3]
                    record["chain"] = tokens[4]
                    record["res_num"] = _parse_res_num(tokens[5])
                    coords = tokens[6:9]
                else:
                    # without chain
                    record["atom_name"] = tokens[2]
                    record["res_name"] = tokens[3]
                    record["chain"] = ""
                    record["res_num"] = _parse_res_num(tokens[4])
                    coords = tokens[5:8]
                try:
                    [float(c) for c in coords]
                except ValueError:
                    continue
                record["charge"] = charge
                record["radius"] = radius

            record["element"] = _guess_element(record["atom_name"])
            atoms.append(record)

    return atoms


def _parse_res_num(token: str) -> int:
    """Extract the integer residue number from a (possibly icode'd) token."""
    digits = "".join(ch for ch in token if ch.isdigit())
    return int(digits) if digits else -1


def _guess_element(atom_name: str) -> str:
    """Best-effort element guess from the atom name (H detection is what
    matters here: hydrogen charges are matched by name, not element)."""
    name = atom_name.strip().upper()
    if not name:
        return ""
    if name[0].isdigit() and len(name) > 1:
        name = name[1:]
    if name.startswith("H") or name.startswith("D"):
        return "H"
    return name[0]


def match_charges_to_system(
    pqr_path,
    system,
    state: str = "holo",
    verbose: bool = True,
) -> torch.Tensor:
    """Match PQR partial charges to the pocket atoms of a `PocketComplex`.

    Atoms are matched by (chain, residue number, residue name, atom name);
    the first PQR record wins for duplicated keys (alternate locations). The
    returned tensor is ordered exactly like the pocket atoms of the system,
    so it can be used directly as `pocket_charge_values`.

    Args:
        pqr_path: Path to the PQR file with partial charges.
        system: A `PocketComplex` (with a `holo`/`apo` `ProteinPocket`).
        state: Which pocket of the system to match ("holo" or "apo").
        verbose: Print a match summary.

    Returns:
        torch.Tensor: Partial charges, shape [n_pocket_atoms].
    """
    pocket = getattr(system, state)
    atoms = pocket.atoms

    pqr_atoms = read_pqr(pqr_path)
    if len(pqr_atoms) == 0:
        raise ValueError(f"No ATOM/HETATM records parsed from {pqr_path}")

    charge_index = {}
    for record in pqr_atoms:
        key = (
            _norm(record["chain"]),
            record["res_num"],
            _norm(record["res_name"]),
            _norm(record["atom_name"]),
        )
        if key not in charge_index:
            charge_index[key] = record["charge"]

    chain_ids = np.asarray(atoms.chain_id).astype(str)
    res_ids = np.asarray(atoms.res_id).astype(int)
    res_names = np.asarray(atoms.res_name).astype(str)
    atom_names = np.asarray(atoms.atom_name).astype(str)

    charges = []
    n_missing = 0
    missing_examples = []
    for i in range(len(atoms)):
        key = (
            _norm(chain_ids[i]),
            int(res_ids[i]),
            _norm(res_names[i]),
            _norm(atom_names[i]),
        )
        q = charge_index.get(key, None)
        if q is None:
            n_missing += 1
            if len(missing_examples) < 10:
                missing_examples.append(key)
            q = 0.0
        charges.append(float(q))

    if verbose:
        n_total = len(atoms)
        n_matched = n_total - n_missing
        charges_t = torch.tensor(charges)
        print(
            f"Partial charges from {pqr_path}: matched {n_matched}/{n_total} "
            f"pocket atoms "
            f"(total charge {charges_t.sum().item():+.2f} e, "
            f"min {charges_t.min().item():+.3f}, max {charges_t.max().item():+.3f})"
        )
        if n_missing > 0:
            frac = n_missing / max(n_total, 1)
            warning = (
                "WARNING: the PQR file may not match this structure"
                if frac > 0.1
                else "Unmatched atoms set to charge 0"
            )
            print(
                f"{warning} ({n_missing} unmatched, e.g. {missing_examples}). "
                "Check that the PQR was generated from the same protein "
                "(same residue numbering) and that hydrogens are named "
                "consistently."
            )

    return torch.tensor(charges, dtype=torch.float32)


def pqr_charge_summary(pqr_path) -> dict:
    """Summarise the charges of a PQR file (used by the prep scripts)."""
    atoms = read_pqr(pqr_path)
    if not atoms:
        return {"n_atoms": 0}
    charges = np.array([a["charge"] for a in atoms])
    return {
        "n_atoms": len(atoms),
        "total": float(charges.sum()),
        "n_nonzero": int((charges != 0).sum()),
        "min": float(charges.min()),
        "max": float(charges.max()),
    }


def partial_charges_to_dict(pqr_path) -> Optional[dict]:
    """Convenience: build a {(chain, res_num, res_name, atom_name): charge}
    lookup from a PQR file."""
    return {
        (
            _norm(rec["chain"]),
            rec["res_num"],
            _norm(rec["res_name"]),
            _norm(rec["atom_name"]),
        ): rec["charge"]
        for rec in read_pqr(pqr_path)
    }

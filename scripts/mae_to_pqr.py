"""Write a PQR file with OPLS partial charges from a Schrodinger .mae file.

Schrodinger's ``structconvert`` has no PQR writer (it rejects the .pqr
extension), so the OPLS partial charges assigned by prepwizard cannot be
exported with it. This helper reads the prepared structure through mmshare
and writes a PDB2PQR-style PQR: the partial charge and radius replace the
occupancy / B-factor columns, which is the layout parsed by
``flowr.util.partial_charges.read_pqr``.

Run it with the mmshare python wrapper (the conda env has no mmshare):

    $SCHRODINGER/run scripts/mae_to_pqr.py prepared.mae out.pqr
"""

import sys

from schrodinger import structure


def _record(serial, atom):
    try:
        q = float(atom.partial_charge)
    except Exception:
        q = 0.0
    try:
        r = float(atom.radius)
    except Exception:
        r = 1.0

    name = (atom.pdbname or "").strip()[:4] or "X"
    resname = (atom.pdbres or "").strip()[:3] or "UNK"
    chain = (atom.chain_name or "").strip()[:1] or " "
    resnum = int(atom.resnum)

    # Fixed-column PDB layout (read_pqr parses line[17:20] for resName,
    # line[21:22] for chain, line[22:26] for resSeq, line[54:62] for the
    # charge and line[62:70] for the radius).
    return (
        f"ATOM  {serial:>5} {name:<4} {resname:>3} {chain:1}"
        f"{resnum:>4}    {atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}"
        f"{q:>8.4f}{r:>8.4f}"
    )


def main():
    if len(sys.argv) != 3:
        sys.exit(
            "usage: $SCHRODINGER/run scripts/mae_to_pqr.py input.mae output.pqr"
        )
    inp, outp = sys.argv[1], sys.argv[2]

    n_atoms = 0
    with open(outp, "w") as f:
        f.write(f"REMARK   1 PQR written by mae_to_pqr.py from {inp}\n")
        for st in structure.StructureReader(inp):
            for serial, atom in enumerate(st.atom, start=1):
                f.write(_record(serial, atom) + "\n")
                n_atoms += 1
    print(f"Wrote {n_atoms} atoms to {outp}")


if __name__ == "__main__":
    main()

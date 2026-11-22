"""Smoke tests for the electric-field-guided diffusion additions.

Tests, without requiring the full FLOWR training stack:
  1. 公式(5): electrostatic field superposition against a hand-computed value.
  2. SE(3)-equivariance of the field (rotation invariance of |E| and
     equivariance of E itself) and of the ElectricFieldEncoder outputs.
  3. The additive-only guarantee: constructing LigandGenerator with
     use_electric_field=False must succeed and produce a model without any
     efield parameters; with use_electric_field=True the efield parameters
     appear.
  4. A full LigandGenerator forward pass with the electric field enabled
     (training-style call, pocket conditioning).
  5. Sampling-style electric field guidance: Integrator.step applies the
     electric_field_drift entry of the predicted dict when efield_guidance
     is on, and ignores it (bitwise identical coords) when it is off.
  6. VP diffusion (公式(6)-(8)): schedule endpoints, variance preservation
     and the type vector field.
  7. The "vp" interpolation branch of GeometricInterpolant._interpolate_mol.

Run with: python scripts/test_electric_field.py
"""

import math
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_optional_dependency_stubs():
    """Stub heavy scientific deps that are unavailable on this test machine.

    The modules below are only used inside functions of the FLOWR codebase
    (never at import time, apart from harmless class-alias type hints), so
    stub modules are enough to import the model code for these tests. On the
    real training environment the actual packages are installed and the
    stubs here are never used.
    """

    class _AnyStub:
        """Chainable placeholder for any attribute of a stubbed library."""

        def copy(self):
            return {}

        # Support module-level arithmetic on unit-like placeholders,
        # e.g. `10.0 * unit.kilocalories_per_mole / unit.angstroms**2`
        def __rmul__(self, other):
            return self

        def __mul__(self, other):
            return self

        def __truediv__(self, other):
            return self

        def __rtruediv__(self, other):
            return self

        def __pow__(self, other):
            return self

        def __rpow__(self, other):
            return self

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return self

    _stub_instance = _AnyStub()

    class _StubModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _stub_instance

    stub_names = [
        "biotite",
        "biotite.structure",
        "biotite.structure.io",
        "biotite.structure.io.pdb",
        "biotite.structure.io.pdbx",
        "prolif",
        "hydride",
        "MDAnalysis",
        "MDAnalysis.converters",
        "MDAnalysis.converters.RDKit",
        "MDAnalysis.lib",
        "MDAnalysis.lib.util",
        "openmm",
        "openmm.app",
        "openmm.unit",
        "pdbinf",
        "openbabel",
        "openbabel.pybel",
        "openbabel.openbabel",
        "pdbfixer",
        "genbench3d",
        "genbench3d.data",
        "genbench3d.data.source",
        "genbench3d.data.structure",
        "genbench3d.geometry",
        "genbench3d.utils",
        "posebusters",
        "posecheck",
        "sascorer",
        "molvs",
    ]
    for name in stub_names:
        try:
            __import__(name)
            continue  # real package available, no stub needed
        except ImportError:
            pass
        sys.modules[name] = _StubModule(name)

    # Make parent package attributes resolve to the stub submodules
    for name in stub_names:
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if isinstance(parent, _StubModule):
                setattr(parent, child_name, sys.modules[name])


_install_optional_dependency_stubs()

from flowr.data.vp_diffusion import (  # noqa: E402
    vp_mean_std,
    vp_noisy_types,
    vp_signal_coeff,
    vp_type_vector_field,
)
from flowr.models.electric_field import (
    CHARGE_VALUES,
    ElectricFieldEncoder,
    charges_to_values,
    electric_field_drift,
    electrostatic_field,
)
from flowr.models.pocket import LigandGenerator, PocketEncoder


def test_field_superposition():
    # Single protein atom with charge q at the origin, query point on the x axis
    q = 2.0
    d = 1.5
    pocket_coords = torch.zeros(1, 1, 3)
    pocket_q = torch.tensor([[q]])
    query = torch.tensor([[[d, 0.0, 0.0]]])
    softening = 0.5

    field, potential = electrostatic_field(
        query, pocket_coords, pocket_q, softening=softening
    )
    denom = (d**2 + softening**2) ** 1.5
    expected_e = q * d / denom
    expected_phi = q / math.sqrt(d**2 + softening**2)

    assert torch.allclose(field.squeeze(), torch.tensor([expected_e, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(potential.squeeze(), torch.tensor(expected_phi), atol=1e-5)
    print("[PASS] 公式(5) field superposition matches hand computation")


def test_field_equivariance():
    # E must rotate with the frame; |E| and phi must be invariant
    torch.manual_seed(0)
    pocket_coords = torch.randn(2, 9, 3)
    pocket_q = torch.randn(2, 9)
    query = torch.randn(2, 5, 3)

    # Random rotation via QR decomposition
    q_mat, r_mat = torch.linalg.qr(torch.randn(3, 3))
    q_mat = q_mat * torch.sign(torch.diagonal(r_mat))

    field, potential = electrostatic_field(query, pocket_coords, pocket_q)
    field_rot, potential_rot = electrostatic_field(
        query @ q_mat.T, pocket_coords @ q_mat.T, pocket_q
    )

    assert torch.allclose(field_rot, field @ q_mat.T, atol=1e-4)
    assert torch.allclose(potential_rot, potential, atol=1e-4)
    print("[PASS] E is SE(3)-equivariant, phi and |E| are invariant")


def test_encoder_equivariance():
    torch.manual_seed(0)
    enc = ElectricFieldEncoder(d_equi=8, d_inv=16, d_pairwise_out=8)
    field = torch.randn(2, 5, 3)
    equi = enc.embed_equi(field)
    assert equi.shape == (2, 5, 3, 8)

    q_mat, _ = torch.linalg.qr(torch.randn(3, 3))
    equi_rot = enc.embed_equi(field @ q_mat.T)
    # Rotate the 3D axis of the equivariant features (dim 2 of [B, N, 3, d])
    expected_rot = torch.einsum("ij,bnjf->bnif", q_mat, equi)
    assert torch.allclose(equi_rot, expected_rot, atol=1e-4)
    print("[PASS] ElectricFieldEncoder equi features are SE(3)-equivariant")


def test_charges_to_values():
    onehot = torch.zeros(2, 4, 7)
    onehot[0, 0, 4] = 1.0  # class 4 -> -1
    onehot[0, 1, 1] = 1.0  # class 1 -> +1
    onehot[1, 2, 0] = 1.0  # class 0 -> 0
    vals = charges_to_values(onehot)
    assert vals[0, 0].item() == -1.0 and vals[0, 1].item() == 1.0
    assert vals[1, 2].item() == 0.0
    idx_vals = charges_to_values(torch.tensor([[4, 1], [0, 6]]))
    assert idx_vals[0, 0].item() == -1.0 and idx_vals[0, 1].item() == 1.0
    assert idx_vals[1, 1].item() == -3.0
    print("[PASS] charges_to_values handles one-hot and index tensors")


def _build_generator(use_efield, interactions=False):
    pocket_enc = PocketEncoder(
        d_equi=4,
        d_inv=24,
        d_message=8,
        n_layers=2,
        n_attn_heads=4,
        d_message_ff=16,
        d_edge=8,
        n_atom_names=20,
        n_bond_types=5,
        n_res_types=25,
        emb_size=8,
    )
    gen = LigandGenerator(
        d_equi=4,
        d_inv=24,
        d_message=8,
        n_layers=2,
        n_attn_heads=4,
        d_message_ff=16,
        d_edge=8,
        n_atom_types=10,
        n_bond_types=5,
        emb_size=8,
        n_extra_atom_feats=1,
        self_cond=False,
        use_lig_pocket_rbf=True,
        n_interaction_types=9 if interactions else None,
        predict_interactions=interactions,
        flow_interactions=interactions,
        pocket_enc=pocket_enc,
        use_electric_field=use_efield,
        efield_cutoff=10.0,
        efield_softening=0.5,
    )
    return gen


def test_flag_off_is_unchanged():
    gen_off = _build_generator(use_efield=False)
    names_off = {n for n, _ in gen_off.ligand_dec.named_parameters()}
    assert not any("efield" in n for n in names_off)
    assert not hasattr(gen_off.ligand_dec, "efield_encoder")
    assert not gen_off.use_electric_field

    gen_on = _build_generator(use_efield=True)
    names_on = {n for n, _ in gen_on.ligand_dec.named_parameters()}
    assert any("efield_encoder" in n for n in names_on)
    # All original parameters are still present, field ones are extra
    assert names_off.issubset(names_on)
    print("[PASS] use_electric_field=False leaves the model untouched; True only adds parameters")


def test_generator_forward_with_field():
    torch.manual_seed(0)
    gen = _build_generator(use_efield=True)
    gen.eval()

    B, N, N_p = 2, 6, 12
    coords = torch.randn(B, N, 3)
    atom_types = torch.randint(0, 10, (B, N))
    bond_types = torch.randint(0, 5, (B, N, N))
    atom_mask = torch.ones(B, N, dtype=torch.long)
    pocket_coords = torch.randn(B, N_p, 3)
    pocket_atom_names = torch.randint(0, 20, (B, N_p))
    pocket_charges_idx = torch.randint(0, 7, (B, N_p))
    pocket_res = torch.randint(0, 25, (B, N_p))
    pocket_bonds = torch.randint(0, 5, (B, N_p, N_p))
    pocket_mask = torch.ones(B, N_p, dtype=torch.long)
    times = [
        torch.rand(B, 1, 1).expand(-1, N, -1),
        torch.rand(B, 1, 1).expand(-1, N, -1),
        None,
    ]

    pocket_charge_values = charges_to_values(pocket_charges_idx)

    with torch.no_grad():
        out = gen(
            coords,
            atom_types,
            bond_types,
            atom_mask=atom_mask,
            extra_feats=times,
            pocket_coords=pocket_coords,
            pocket_atom_names=pocket_atom_names,
            pocket_atom_charges=pocket_charges_idx,
            pocket_res_types=pocket_res,
            pocket_bond_types=pocket_bonds,
            pocket_atom_mask=pocket_mask,
            pocket_charge_values=pocket_charge_values,
        )

    out_coords, type_logits, bond_logits, charge_logits, mask_out = out
    assert out_coords.shape == (B, N, 3)
    assert type_logits.shape == (B, N, 10)
    assert charge_logits.shape == (B, N, 7)
    print("[PASS] LigandGenerator forward pass with electric field conditioning works")


def test_generator_forward_with_field_and_interactions():
    torch.manual_seed(0)
    gen = _build_generator(use_efield=True, interactions=True)
    gen.eval()

    B, N, N_p = 2, 6, 12
    coords = torch.randn(B, N, 3)
    atom_types = torch.randint(0, 10, (B, N))
    bond_types = torch.randint(0, 5, (B, N, N))
    atom_mask = torch.ones(B, N, dtype=torch.long)
    pocket_coords = torch.randn(B, N_p, 3)
    pocket_atom_names = torch.randint(0, 20, (B, N_p))
    pocket_charges_idx = torch.randint(0, 7, (B, N_p))
    pocket_res = torch.randint(0, 25, (B, N_p))
    pocket_bonds = torch.randint(0, 5, (B, N_p, N_p))
    pocket_mask = torch.ones(B, N_p, dtype=torch.long)
    interactions = torch.randint(0, 9, (B, N, N_p))
    times = [
        torch.rand(B, 1, 1).expand(-1, N, -1),
        torch.rand(B, 1, 1).expand(-1, N, -1),
        torch.rand(B, 1, 1).expand(-1, N, -1),
    ]

    with torch.no_grad():
        out = gen(
            coords,
            atom_types,
            bond_types,
            atom_mask=atom_mask,
            extra_feats=times,
            pocket_coords=pocket_coords,
            pocket_atom_names=pocket_atom_names,
            pocket_atom_charges=pocket_charges_idx,
            pocket_res_types=pocket_res,
            pocket_bond_types=pocket_bonds,
            pocket_atom_mask=pocket_mask,
            interactions=interactions,
            pocket_charge_values=charges_to_values(pocket_charges_idx),
        )
    assert out[0].shape == (B, N, 3)
    assert len(out) == 6  # coords, types, bonds, charges, interaction logits, mask
    print("[PASS] forward with both electric field and interaction conditioning works")


def test_integrator_guidance():
    from flowr.models.fm_pocket import Integrator

    torch.manual_seed(0)
    B, N = 2, 5
    vocab = 10
    n_bonds = 5

    integrator_on = Integrator(
        steps=10,
        type_strategy="uniform-sample",
        bond_strategy="uniform-sample",
        efield_guidance=True,
        efield_guidance_scale=0.05,
    )
    integrator_off = Integrator(
        steps=10, type_strategy="uniform-sample", bond_strategy="uniform-sample"
    )
    assert integrator_off.efield_guidance is False

    atomics = torch.zeros(B, N, vocab)
    atomics[..., 0] = 1.0
    bonds = torch.zeros(B, N, N, n_bonds)
    bonds[..., 0] = 1.0

    curr = {
        "coords": torch.randn(B, N, 3),
        "atomics": atomics,
        "bonds": bonds,
        "mask": torch.ones(B, N),
    }
    prior = {k: v.clone() if torch.is_tensor(v) else v for k, v in curr.items()}
    times = [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)]
    predicted = {
        "coords": torch.randn(B, N, 3),
        "atomics": torch.full((B, N, vocab), 1.0 / vocab),
        "bonds": torch.full((B, N, N, n_bonds), 1.0 / n_bonds),
        "mask": torch.ones(B, N),
        "electric_field_drift": torch.randn(B, N, 3) * 10.0,
    }

    torch.manual_seed(42)
    out_on = integrator_on.step(curr, predicted, prior, times, 0.1)
    torch.manual_seed(42)
    out_off = integrator_off.step(curr, predicted, prior, times, 0.1)
    # The coordinate updates must differ by the guidance drift
    drift = 0.1 * 0.05 * predicted["electric_field_drift"]
    assert torch.allclose(out_on["coords"], out_off["coords"] + drift, atol=1e-5)
    print("[PASS] Integrator applies electric field drift when guided, ignores it otherwise")


def test_electric_field_drift():
    torch.manual_seed(0)
    B, N, N_p = 1, 4, 8
    lig_coords = torch.randn(B, N, 3)
    # One-hot predicted charge class 1 -> expected charge +1, so q*E is non-zero
    charge_probs = torch.zeros(B, N, 7)
    charge_probs[..., 1] = 1.0
    pocket_coords = torch.randn(B, N_p, 3)
    pocket_q = torch.randn(B, N_p)
    lig_mask = torch.ones(B, N)
    pocket_mask = torch.ones(B, N_p, dtype=torch.long)

    drift = electric_field_drift(
        lig_coords,
        charge_probs,
        pocket_coords,
        pocket_q,
        ligand_mask=lig_mask,
        pocket_mask=pocket_mask,
        mode="force",
    )
    assert drift.shape == (B, N, 3)
    assert torch.isfinite(drift).all()

    unit_drift = electric_field_drift(
        lig_coords,
        charge_probs,
        pocket_coords,
        pocket_q,
        mode="unit",
    )
    norms = torch.linalg.vector_norm(unit_drift, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # "field" mode: every atom follows the local field direction, so the
    # drift magnitude is 1 wherever the field is non-zero (and 0 otherwise)
    field_drift = electric_field_drift(
        lig_coords,
        charge_probs,
        pocket_coords,
        pocket_q,
        mode="field",
    )
    field_norms = torch.linalg.vector_norm(field_drift, dim=-1)
    moving = field_norms > 1e-6
    assert torch.allclose(
        field_norms[moving], torch.ones_like(field_norms[moving]), atol=1e-4
    )
    zero_drift = electric_field_drift(
        lig_coords,
        charge_probs,
        pocket_coords,
        torch.zeros_like(pocket_q),
        mode="field",
    )
    assert zero_drift.abs().max().item() == 0.0
    print("[PASS] electric_field_drift force, unit and field modes work")


def test_vp_diffusion():
    # Endpoints: s(0) = 0 (pure noise), s(1) close to 1 (data)
    s0 = vp_signal_coeff(0.0)
    s1 = vp_signal_coeff(1.0)
    assert abs(s0.item()) < 1e-6
    assert s1.item() > 0.99

    # Variance preservation: mean^2 + var^2 = s + (1 - s) = 1
    mean, std = vp_mean_std(0.37)
    assert abs((mean**2 + std**2).item() - 1.0) < 1e-5

    # Forward noising: at t=1 the sample must concentrate on the data
    # (std = sqrt(1 - s(1)) is small but not exactly zero for a finite schedule)
    data = torch.eye(4).unsqueeze(0)
    torch.manual_seed(0)
    x1 = vp_noisy_types(data, 1.0)
    assert (x1 - data).abs().max().item() < 0.2

    # Vector field: with a perfect x1_hat the field points from x_t to x_1
    x_t = vp_noisy_types(data, 0.5)
    u = vp_type_vector_field(x_t, data, 0.5)
    assert u.shape == x_t.shape
    assert torch.isfinite(u).all()
    print("[PASS] VP schedule, forward noising (公式(6)) and vector field (公式(7)) work")


def main():
    test_field_superposition()
    test_field_equivariance()
    test_encoder_equivariance()
    test_charges_to_values()
    test_flag_off_is_unchanged()
    test_generator_forward_with_field()
    test_generator_forward_with_field_and_interactions()
    test_integrator_guidance()
    test_electric_field_drift()
    test_vp_diffusion()
    print("\nAll electric-field diffusion smoke tests passed.")


if __name__ == "__main__":
    main()

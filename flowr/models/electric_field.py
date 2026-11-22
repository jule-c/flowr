"""Protein electrostatic field utilities for electric-field-guided diffusion.

This module implements the electrostatic components described on pages 15-16 of
the NSFC proposal "考虑靶蛋白活性口袋电场效应的扩散分子生成模型及其应用研究":

  * 公式(5): the electrostatic field at an arbitrary position i near the
    binding site, computed by superposition over the K partial charges of the
    nearby protein atoms j:

        E(r_i) = (1 / 4*pi*eps0) * sum_j q_j * (r_i - r_j) / |r_i - r_j|^3

    The constant 1 / (4*pi*eps0) is absorbed into the (learnable/use-controlled)
    guidance scale and feature projections, so the functions below return E in
    units of sum_j q_j * r_ij / |r_ij|^3.

  * The electric field is injected into the ligand-pocket message passing and
    attention (as an extra conditioning input next to the pocket structure C),
    see `ElectricFieldEncoder`.

  * During sampling, ligand atoms are pushed along the local electric field
    (force F_i = q_i * E_i on atom i), turning structure-based generation into
    a diffusion of atoms guided by the protein electrostatic field, see
    `electric_field_drift`.

All code in this module is additive: it is only used when the caller opts in
via the `use_electric_field` / `efield_guidance` flags of the models.

Charge conventions
------------------
FLOWR stores atom charges as one of 7 classes (see
flowr/util/rdkit.py IDX_CHARGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: -1, 5: -2, 6: -3}).
`charges_to_values` converts those class tensors to numeric charge values.
If OPLS4 partial charges are available for the target protein (as used in the
proposal), they can be passed directly as a [B, N_p] float tensor through the
`pocket_charge_values` argument of the model instead.
"""

import torch
import torch.nn as nn

# Numeric charge values for the 7 charge classes used across FLOWR.
# Must stay in sync with flowr.util.rdkit.IDX_CHARGE_MAP.
CHARGE_VALUES = (0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0)


def charges_to_values(charges: torch.Tensor) -> torch.Tensor:
    """Convert a charge tensor into numeric charge values.

    Args:
        charges (torch.Tensor): Either class indices, shape [B, N], or one-hot
            encoded charge classes, shape [B, N, 7].

    Returns:
        torch.Tensor: Numeric charge values, shape [B, N].
    """
    if charges.dim() == 2:
        values = charges.new_tensor(CHARGE_VALUES)
        return values[charges.long()]
    elif charges.dim() == 3:
        values = charges.new_tensor(CHARGE_VALUES)
        return (charges.float() * values.view(1, 1, -1)).sum(dim=-1)
    raise ValueError(f"Unsupported charge tensor shape {tuple(charges.shape)}")


def electrostatic_field(
    query_coords: torch.Tensor,
    pocket_coords: torch.Tensor,
    pocket_charge_values: torch.Tensor,
    pocket_mask: torch.Tensor = None,
    cutoff: float = None,
    softening: float = 0.5,
    eps: float = 1e-6,
):
    """Compute the electrostatic field E and potential phi at query positions (公式(5)).

    For each query position i the field is the superposition of the Coulomb
    fields of all (masked) protein atoms j with partial charge q_j:

        E_i = sum_j q_j * (r_i - r_j) / (|r_i - r_j|^2 + softening^2)^(3/2)
        phi_i = sum_j q_j / sqrt(|r_i - r_j|^2 + softening^2)

    The `softening` parameter keeps the field finite at very small distances
    (and absorbs the 1/(4*pi*eps0) unit choice, which is folded into the
    downstream learnable projections / guidance scale).

    Args:
        query_coords (torch.Tensor): Positions to evaluate the field at, shape [B, N_q, 3].
        pocket_coords (torch.Tensor): Protein atom positions, shape [B, N_p, 3].
        pocket_charge_values (torch.Tensor): Numeric protein atom charges, shape [B, N_p].
        pocket_mask (torch.Tensor, optional): Protein atom mask, shape [B, N_p], 1 for real atoms.
        cutoff (float, optional): Optional distance cutoff in the same units as the coords.
        softening (float): Softening distance added to |r_ij|^2 to avoid singularities.
        eps (float): Small constant for numerical stability.

    Returns:
        (torch.Tensor, torch.Tensor): Field E, shape [B, N_q, 3], and potential
            phi, shape [B, N_q, 1].
    """
    # Pairwise displacement vectors r_ij = r_i - r_j, shape [B, N_q, N_p, 3]
    diff = query_coords.unsqueeze(2) - pocket_coords.unsqueeze(1)
    sqrd_dists = (diff * diff).sum(dim=-1)  # [B, N_q, N_p]

    # Softened inverse distances: 1 / sqrt(d^2 + softening^2)
    inv_dists = torch.rsqrt(sqrd_dists + softening * softening + eps)

    # Signed protein charges, masked and (optionally) windowed by a cutoff
    charges = pocket_charge_values.unsqueeze(1)  # [B, 1, N_p]
    if pocket_mask is not None:
        charges = charges * pocket_mask.unsqueeze(1).float()
    if cutoff is not None:
        dists = torch.sqrt(sqrd_dists + eps)
        window = 0.5 * (torch.cos(torch.pi * dists / cutoff) + 1.0)
        window = torch.where(dists < cutoff, window, torch.zeros_like(window))
        charges = charges * window

    # Field: E_i = sum_j q_j * r_ij / |r_ij|^3 (公式(5), softened)
    field = (charges.unsqueeze(-1) * inv_dists.unsqueeze(-1) ** 3 * diff).sum(dim=2)

    # Potential: phi_i = sum_j q_j / |r_ij|
    potential = (charges * inv_dists).sum(dim=2, keepdim=True)

    return field, potential


def pairwise_electrostatic_feats(
    query_coords: torch.Tensor,
    pocket_coords: torch.Tensor,
    pocket_charge_values: torch.Tensor,
    query_mask: torch.Tensor = None,
    pocket_mask: torch.Tensor = None,
    softening: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Invariant pairwise electrostatic features for each (query, pocket) pair.

    For a pair (i, j) these are the signed Coulomb potential contribution
    q_j / |r_ij| and the magnitude |E_ij| = |q_j| * |r_ij| / |r_ij|^3 of the
    field that protein atom j generates at position i. These are the terms of
    the sum in 公式(5) before superposition, used to inject the field into the
    ligand-pocket message passing.

    Returns:
        torch.Tensor: Pairwise features, shape [B, N_q, N_p, 2].
    """
    diff = query_coords.unsqueeze(2) - pocket_coords.unsqueeze(1)
    sqrd_dists = (diff * diff).sum(dim=-1)  # [B, N_q, N_p]
    inv_dists = torch.rsqrt(sqrd_dists + softening * softening + eps)

    charges = pocket_charge_values.unsqueeze(1)  # [B, 1, N_p]
    abs_charges = charges.abs()
    if pocket_mask is not None:
        charges = charges * pocket_mask.unsqueeze(1).float()
        abs_charges = abs_charges * pocket_mask.unsqueeze(1).float()

    # Signed Coulomb potential contribution of pocket atom j at query atom i
    pair_potential = charges * inv_dists
    # Magnitude of the pairwise field |E_ij| = |q_j| * d / d^3
    pair_field_mag = abs_charges * (inv_dists**3) * torch.sqrt(sqrd_dists + eps)

    pair_feats = torch.stack((pair_potential, pair_field_mag), dim=-1)  # [B, N_q, N_p, 2]

    if query_mask is not None:
        pair_feats = pair_feats * query_mask.unsqueeze(2).unsqueeze(-1).float()

    return pair_feats


class ElectricFieldEncoder(nn.Module):
    """Encodes the protein electrostatic field into model features.

    Following the proposal, the field computed with 公式(5) is injected into the
    (equivariant) message passing and attention layers of the ligand decoder:

      * `embed_equi`: the field vector E_i is a polar (SE(3)-equivariant) 3D
        vector, so a componentwise linear map turns it into equivariant
        features which are added to the ligand equivariant features.
      * `embed_inv`: invariant summaries of the field (|E_i|, potential phi_i
        and their log-compressions) are mapped through an MLP and added to the
        ligand invariant (node) features that feed the attention layers.
      * `embed_pairwise`: the per-(ligand atom, pocket atom) electrostatic
        features are Gaussian-expanded and projected, then added to the
        ligand-pocket pairwise edge features of the cross attention.
    """

    def __init__(
        self,
        d_equi: int,
        d_inv: int,
        d_pairwise_out: int,
        cutoff: float = 10.0,
        softening: float = 0.5,
        num_rbf: int = 16,
    ):
        super().__init__()

        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_pairwise_out = d_pairwise_out
        self.cutoff = cutoff
        self.softening = softening
        self.num_rbf = num_rbf

        centers = torch.linspace(0, cutoff, num_rbf)
        width = (centers[1] - centers[0]).item() if num_rbf > 1 else 1.0
        self.register_buffer("pair_rbf_centers", centers)
        self.pair_rbf_width = width

        # Equivariant embedding: linear map applied componentwise to E (no bias,
        # so a zero field maps to zero features)
        self.field_to_equi = nn.Linear(1, d_equi, bias=False)

        # Invariant embedding of (|E|, phi, log1p|E|, log1p|phi|)
        self.field_to_inv = nn.Sequential(
            nn.Linear(4, d_inv),
            nn.SiLU(inplace=False),
            nn.Linear(d_inv, d_inv),
        )

        # Pairwise (ligand x pocket) electrostatic message features: each of the
        # two pairwise scalars (and its magnitude) is Gaussian-expanded over
        # num_rbf centers, giving 4 * num_rbf input channels
        self.pairwise_proj = nn.Linear(4 * num_rbf, d_pairwise_out)

    def embed_equi(self, field: torch.Tensor) -> torch.Tensor:
        """Embed the field vector equivariantly.

        Args:
            field (torch.Tensor): Field E at each ligand atom, shape [B, N, 3].

        Returns:
            torch.Tensor: Equivariant field features, shape [B, N, 3, d_equi].
        """
        return self.field_to_equi(field.unsqueeze(-1))

    def embed_inv(self, field: torch.Tensor, potential: torch.Tensor) -> torch.Tensor:
        """Embed invariant summaries of the field.

        Args:
            field (torch.Tensor): Field E at each ligand atom, shape [B, N, 3].
            potential (torch.Tensor): Potential phi at each ligand atom, shape [B, N, 1].

        Returns:
            torch.Tensor: Invariant field features, shape [B, N, d_inv].
        """
        field_mag = torch.linalg.vector_norm(field, dim=-1, keepdim=True)
        inv_feats = torch.cat(
            (
                field_mag,
                potential,
                torch.log1p(field_mag),
                torch.log1p(potential.abs()),
            ),
            dim=-1,
        )
        return self.field_to_inv(inv_feats)

    def embed_pairwise(
        self,
        query_coords: torch.Tensor,
        pocket_coords: torch.Tensor,
        pocket_charge_values: torch.Tensor,
        query_mask: torch.Tensor = None,
        pocket_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Pairwise electrostatic message features for the cross attention.

        Args:
            query_coords (torch.Tensor): Ligand coordinates, shape [B, N, 3].
            pocket_coords (torch.Tensor): Pocket coordinates, shape [B, N_p, 3].
            pocket_charge_values (torch.Tensor): Pocket charges, shape [B, N_p].
            query_mask (torch.Tensor, optional): Ligand atom mask, shape [B, N].
            pocket_mask (torch.Tensor, optional): Pocket atom mask, shape [B, N_p].

        Returns:
            torch.Tensor: Pairwise field features, shape [B, N, N_p, d_pairwise_out].
        """
        pair_feats = pairwise_electrostatic_feats(
            query_coords,
            pocket_coords,
            pocket_charge_values,
            query_mask=query_mask,
            pocket_mask=pocket_mask,
            softening=self.softening,
        )  # [B, N, N_p, 2]

        # Gaussian expansion of the two pairwise electrostatic scalars
        # (signed values and magnitudes, so both attraction and repulsion
        # as well as both scalars are represented over the RBF centers)
        pair_potential, pair_field_mag = pair_feats.unbind(-1)
        scalars = (
            pair_potential,
            pair_field_mag,
            pair_potential.abs(),
            pair_field_mag.abs(),
        )
        centers = self.pair_rbf_centers.view(1, 1, 1, -1)
        expansions = [
            torch.exp(
                -0.5 * ((f.unsqueeze(-1) - centers) / self.pair_rbf_width) ** 2
            )
            for f in scalars
        ]
        pair_embeds = torch.cat(expansions, dim=-1)  # [B, N, N_p, 4 * num_rbf]

        out = self.pairwise_proj(pair_embeds)
        if query_mask is not None:
            out = out * query_mask.unsqueeze(2).unsqueeze(-1).float()
        if pocket_mask is not None:
            out = out * pocket_mask.unsqueeze(1).unsqueeze(-1).float()
        return out


def electric_field_drift(
    ligand_coords: torch.Tensor,
    ligand_charge_probs: torch.Tensor,
    pocket_coords: torch.Tensor,
    pocket_charge_values: torch.Tensor,
    ligand_mask: torch.Tensor = None,
    pocket_mask: torch.Tensor = None,
    mode: str = "force",
    cutoff: float = None,
    softening: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Electrostatic drift guiding the diffusion of ligand atoms (采样期电场引导).

    The instantaneous force on ligand atom i with charge q_i in the protein
    electrostatic field E_i (公式(5)) is F_i = q_i * E_i. Adding a multiple of
    this drift to the coordinate update of each integration step makes the
    reverse diffusion a process of atoms moving under the guidance of the
    protein electrostatic field.

    Args:
        ligand_coords (torch.Tensor): Current ligand coordinates, shape [B, N, 3].
        ligand_charge_probs (torch.Tensor): Predicted charge distribution per
            ligand atom, shape [B, N, n_charge_types]. The expected charge
            q_i = sum_c p_c * value_c is used for the force.
        pocket_coords (torch.Tensor): Pocket coordinates, shape [B, N_p, 3].
        pocket_charge_values (torch.Tensor): Pocket charges, shape [B, N_p].
        ligand_mask (torch.Tensor, optional): Ligand atom mask, shape [B, N].
        pocket_mask (torch.Tensor, optional): Pocket atom mask, shape [B, N_p].
        mode (str): "force" uses the raw force q_i * E_i; "unit" normalises each
            atom's drift to unit length so only the direction is used.
        cutoff (float, optional): Optional field cutoff distance.
        softening (float): Field softening distance.
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Coordinate drift for the ligand atoms, shape [B, N, 3].
    """
    if mode not in ["force", "unit", "field"]:
        raise ValueError(f"Unknown electric field drift mode '{mode}'")

    field, _ = electrostatic_field(
        ligand_coords,
        pocket_coords,
        pocket_charge_values,
        pocket_mask=pocket_mask,
        cutoff=cutoff,
        softening=softening,
        eps=eps,
    )

    if mode == "field":
        # Direction-only field following: every atom moves along the local
        # electric field direction. This does not depend on the (formal)
        # charge prediction, so neutral molecules are guided too -- with
        # force-field partial charges the raw q*E drift is very small
        # (see scripts/demo_charge_guidance.py), whereas this mode gives a
        # cumulative displacement of ~scale Angstrom per trajectory.
        norms = torch.linalg.vector_norm(field, dim=-1, keepdim=True)
        drift = field / (norms + eps)
    else:
        # Expected (formal) charge of each ligand atom under the predicted distribution
        charge_values = ligand_charge_probs.new_tensor(CHARGE_VALUES)
        if ligand_charge_probs.size(-1) != len(CHARGE_VALUES):
            raise ValueError(
                "ligand_charge_probs must have 7 charge classes to match CHARGE_VALUES."
            )
        exp_charges = (ligand_charge_probs.float() * charge_values.view(1, 1, -1)).sum(
            dim=-1, keepdim=True
        )  # [B, N, 1]

        drift = exp_charges * field  # electrostatic force on each atom

        if mode == "unit":
            norms = torch.linalg.vector_norm(drift, dim=-1, keepdim=True)
            drift = drift / (norms + eps)

    if ligand_mask is not None:
        drift = drift * ligand_mask.unsqueeze(-1).float()

    return drift

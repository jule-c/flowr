import copy
import math
from typing import Callable, Optional

import torch
import torch.nn as nn

import flowr.util.functional as smolF
from flowr.models.semla import BondRefine, LengthsMLP, adj_to_attn_mask

# *****************************************************************************
# ******************************* Helper Modules ******************************
# *****************************************************************************


# Helper: return π as a tensor matching the input's dtype and device.
def _pi(x: torch.Tensor) -> torch.Tensor:
    return x.new_tensor(math.pi)


# --- Spherical Harmonics Definitions ---


# l = 0
def fn_Y0(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Compute the constant value.
    val = 0.5 * torch.sqrt(1.0 / (_pi(x)))
    # Return a tensor of the same shape as x (i.e. [B, N, N, 1]) filled with that value.
    return x.new_full(x.shape, val.item())


# l = 1
def fn_Y1(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Each of _Y1_1, _Y10, _Y11 will be applied elementwise.
    _Y1_1 = lambda x, y, z: torch.sqrt(3.0 / (4 * _pi(x))) * y
    _Y10 = lambda x, y, z: torch.sqrt(3.0 / (4 * _pi(x))) * z
    _Y11 = lambda x, y, z: torch.sqrt(3.0 / (4 * _pi(x))) * x
    return torch.cat(
        [_Y1_1(x, y, z), _Y10(x, y, z), _Y11(x, y, z)], dim=-1
    )  # Result shape: [B, N, N, 3]


# l = 2
def fn_Y2(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    _Y2_2 = lambda x, y, z: 0.5 * torch.sqrt(15.0 / (_pi(x))) * x * y
    _Y2_1 = lambda x, y, z: 0.5 * torch.sqrt(15.0 / (_pi(x))) * y * z
    _Y20 = lambda x, y, z: 0.25 * torch.sqrt(5.0 / (_pi(x))) * (3 * z**2 - 1)
    _Y21 = lambda x, y, z: 0.5 * torch.sqrt(15.0 / (_pi(x))) * x * z
    _Y22 = lambda x, y, z: 0.25 * torch.sqrt(15.0 / (_pi(x))) * (x**2 - y**2)
    return torch.cat(
        [_Y2_2(x, y, z), _Y2_1(x, y, z), _Y20(x, y, z), _Y21(x, y, z), _Y22(x, y, z)],
        dim=-1,
    )  # [B, N, N, 5]


# l = 3
def fn_Y3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    _Y3_3 = (
        lambda x, y, z: 0.25 * torch.sqrt(35.0 / (2 * _pi(x))) * y * (3 * x**2 - y**2)
    )
    _Y3_2 = lambda x, y, z: 0.5 * torch.sqrt(105.0 / (_pi(x))) * x * y * z
    _Y3_1 = lambda x, y, z: 0.25 * torch.sqrt(21.0 / (2 * _pi(x))) * y * (5 * z**2 - 1)
    _Y30 = lambda x, y, z: 0.25 * torch.sqrt(7.0 / (_pi(x))) * (5 * z**3 - 3 * z)
    _Y31 = lambda x, y, z: 0.25 * torch.sqrt(21.0 / (2 * _pi(x))) * x * (5 * z**2 - 1)
    _Y32 = lambda x, y, z: 0.25 * torch.sqrt(105.0 / (_pi(x))) * (x**2 - y**2) * z
    _Y33 = (
        lambda x, y, z: 0.25 * torch.sqrt(35.0 / (2 * _pi(x))) * x * (x**2 - 3 * y**2)
    )
    return torch.cat(
        [
            _Y3_3(x, y, z),
            _Y3_2(x, y, z),
            _Y3_1(x, y, z),
            _Y30(x, y, z),
            _Y31(x, y, z),
            _Y32(x, y, z),
            _Y33(x, y, z),
        ],
        dim=-1,
    )  # [B, N, N, 7]


def init_sph_fn(l: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a function that computes the spherical harmonics expansion of order l.
    Expects an input tensor of shape (..., 3) and returns a tensor of shape (..., 2l+1).
    """
    if l == 0:
        return lambda r: fn_Y0(*torch.chunk(r, chunks=3, dim=-1))
    elif l == 1:
        return lambda r: fn_Y1(*torch.chunk(r, chunks=3, dim=-1))
    elif l == 2:
        return lambda r: fn_Y2(*torch.chunk(r, chunks=3, dim=-1))
    elif l == 3:
        return lambda r: fn_Y3(*torch.chunk(r, chunks=3, dim=-1))
    else:
        raise NotImplementedError(
            "Spherical harmonics are only defined up to order l = 3."
        )


def poly_cutoff_chi(x: torch.Tensor, p: int) -> torch.Tensor:
    """
    Polynomial cutoff function as in Eq. (29):
      φ_χ_cut(x) = 1 - ((p+1)(p+2)/2) * x^p + p(p+2)*x^(p+1) - (p(p+1)/2)*x^(p+2)
    The function is applied for all x.
    Here, x = (χ̃_ij) / (χ_cut).
    """
    poly_val = (
        1
        - ((p + 1) * (p + 2) / 2) * x**p
        + p * (p + 2) * x ** (p + 1)
        - (p * (p + 1) / 2) * x ** (p + 2)
    )
    return poly_val


class EquisSphc(nn.Module):
    """
    Computes SPHC features for each point in a point cloud.

    For each point i the feature is computed as
      χ_i^(l) = (1 / C_i) ∑_{j ≠ i} φ(‖R(j)-R(i)‖) · Y^(l)(ˆr_{ij}),
    where ˆr_{ij} = (R(j)-R(i)) / ‖R(j)-R(i)‖ and
          C_i = ∑_{j ≠ i} φ(‖R(j)-R(i)‖),
          φ(r) = 0.5 * [cos(π*r/r_cut)+1]  if r < r_cut, else 0.

    The final SPHC feature for each point is the concatenation over degrees l (l_min to l_max)
    resulting in a vector of dimension ∑_{l=l_min}^{l_max}(2l+1).

    Optionally, if return_sphc_distance_matrix is True, the module also returns a SPHC distance matrix.
    This matrix is computed as follows:
      1. For each pair (i,j), compute X_ij = ||χ_i - χ_j||₂.
      2. Compute Ẋ = softmax(X) along each row.
      3. Define χ_cut = κ / n, where n is the number of valid atoms per molecule.
      4. Compute x = Ẋ / χ_cut and apply the polynomial cutoff φ_χ_cut(x) as in Eq. (29).
    """

    def __init__(
        self,
        l_min: int,
        l_max: int,
        r_cut: float,
        eps: float = 1e-8,
        return_sphc_distance_matrix: bool = False,
        p: int = 1,
        kappa: float = 1.0,
    ):
        """
        Args:
            l_min: Minimum spherical harmonic degree.
            l_max: Maximum spherical harmonic degree (must be ≤ 3).
            r_cut: Cutoff radius in Euclidean space.
            eps: Small value to avoid division by zero.
            return_sphc_distance_matrix: If True, also return the SPHC distance matrix.
            p: Polynomial order parameter for the distance cutoff function.
            kappa: Scaling factor for the SPHC cutoff, with χ_cut = κ / n.
        """
        super(EquisSphc, self).__init__()
        assert (
            0 <= l_min <= l_max <= 3
        ), "l_min and l_max must satisfy 0 ≤ l_min ≤ l_max ≤ 3."
        self.l_min = l_min
        self.l_max = l_max
        self.r_cut = r_cut
        self.eps = eps
        self.return_sphc_distance_matrix = return_sphc_distance_matrix
        self.p = p
        self.kappa = kappa

        # Create spherical harmonics functions for each degree.
        self.sph_fns = {l: init_sph_fn(l) for l in range(l_min, l_max + 1)}
        # Total output dimension: ∑_{l=l_min}^{l_max} (2l+1)
        self.out_dim = sum(2 * l + 1 for l in range(l_min, l_max + 1))

    def forward(
        self, coords: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [B, N, 3] representing the point cloud.
            mask: Optional boolean tensor of shape [B, N] indicating valid points.

        Returns:
            If return_sphc_distance_matrix is False:
                Tensor of shape [B, N, out_dim] containing SPHC features per point.
            else:
                Tuple (sphc, sphc_distance_matrix) where:
                  - sphc is [B, N, out_dim],
                  - sphc_distance_matrix is [B, N, N] with the rescaled distances.
        """
        B, N, _ = coords.shape
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=coords.device)

        # Compute pairwise difference vectors: r_ij = R(j) - R(i)
        diff = coords.unsqueeze(1) - coords.unsqueeze(2)  # [B, N, N, 3]
        dists = torch.norm(diff, dim=-1)  # [B, N, N]
        unit_diff = diff / (dists.unsqueeze(-1) + self.eps)  # [B, N, N, 3]

        # Build neighbor mask: valid if both points are valid and exclude self.
        valid_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(N, dtype=torch.bool, device=coords.device).unsqueeze(0)
        neighbor_mask = valid_mask & (~eye)

        # Compute cosine cutoff φ(r): [B, N, N]
        phi = 0.5 * (torch.cos(math.pi * dists / self.r_cut) + 1.0)
        phi = torch.where(dists < self.r_cut, phi, torch.zeros_like(phi))
        phi = phi * neighbor_mask.float()

        C = phi.sum(dim=-1, keepdim=True)  # [B, N, 1]
        C_safe = torch.where(C < self.eps, torch.ones_like(C), C)

        chi_per_l = []
        for l in range(self.l_min, self.l_max + 1):
            sph_fn = self.sph_fns[l]  # Function mapping [..., 3] -> [..., (2l+1)]
            # Evaluate spherical harmonics on each unit vector.
            Y_l = sph_fn(unit_diff)  # [B, N, N, (2l+1)]

            weighted_Y = phi.unsqueeze(-1) * Y_l  # [B, N, N, (2l+1)]
            sum_Y = weighted_Y.sum(dim=2)  # [B, N, (2l+1)]
            chi_l = sum_Y / C_safe  # [B, N, (2l+1)]
            chi_l = torch.where(C < self.eps, torch.zeros_like(chi_l), chi_l)
            chi_per_l.append(chi_l)

        # Concatenate over degrees l → [B, N, out_dim]
        sphc = torch.cat(chi_per_l, dim=-1)

        if self.return_sphc_distance_matrix:
            # Compute pairwise Euclidean distances in SPHC space.
            chi_diff = sphc.unsqueeze(2) - sphc.unsqueeze(1)  # [B, N, N, out_dim]
            X = torch.norm(chi_diff, dim=-1)  # [B, N, N]
            X_soft = torch.softmax(X, dim=-1)  # [B, N, N]
            n_valid = mask.sum(dim=1, keepdim=True).float()  # [B, 1]
            chi_cut = self.kappa / n_valid  # [B, 1]
            chi_cut = chi_cut.unsqueeze(-1)  # [B, 1, 1]
            x = X_soft / chi_cut  # [B, N, N]
            X_soft_cut = poly_cutoff_chi(x, self.p)  # [B, N, N]
            return sphc, X_soft_cut

        return sphc


class RadialBasisEmbedding(nn.Module):
    def __init__(self, d_edge: int, num_rbf: int = 16, cutoff: float = 5.0):
        """
        Args:
            d_edge (int): Output embedding dimension.
            num_rbf (int): Number of radial basis functions.
            cutoff (float): Distance cutoff in Angstroms.
        """
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        centers = torch.linspace(0, cutoff, num_rbf)
        if num_rbf > 1:
            width = centers[1] - centers[0]
        else:
            width = 1.0
        self.register_buffer("centers", centers)
        self.width = width

        self.embedding = nn.Linear(num_rbf, d_edge)

    def forward(
        self,
        ligand_coords: torch.Tensor,
        pocket_coords: torch.Tensor,
        ligand_mask: torch.Tensor,
        pocket_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ligand_coords: Tensor of shape [B, N, 3]
            pocket_coords: Tensor of shape [B, N_p, 3]
        Returns:
            Tensor of shape [B, N, N_p, d_edge]
        """
        dists = torch.cdist(ligand_coords, pocket_coords, p=2)
        dists = torch.clamp(dists, max=self.cutoff)
        centers = self.centers.view(1, 1, 1, self.num_rbf)
        dists_exp = dists.unsqueeze(-1)
        rbf_exp = torch.exp(
            -0.5 * ((dists_exp - centers) / self.width) ** 2
        )  # [B, N, N_p, num_rbf]
        mask = ligand_mask.unsqueeze(-1).unsqueeze(-1) * pocket_mask.unsqueeze(
            1
        ).unsqueeze(-1)
        out = self.embedding(rbf_exp) * mask  # [B, N, N_p, d_edge]

        return out


class RBFEmbed(torch.nn.Module):
    def __init__(self, d_edge: int, num_rbf: int, cutoff: float):
        """
        Args:
            num_rbf (int): Number of radial basis functions.
            cutoff (float): Maximum distance value to consider.
        """
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        centers = torch.linspace(0, cutoff, num_rbf)
        if num_rbf > 1:
            width = centers[1] - centers[0]
        else:
            width = 1.0
        # Register centers and width as buffers.
        self.register_buffer("centers", centers)
        self.width = width

        self.embedding = nn.Linear(num_rbf, d_edge)

    def forward(self, coords: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords (torch.Tensor): Coordinate tensor of shape [B, N, 3].
            node_mask (torch.Tensor): Boolean mask of shape [B, N] where True indicates a real node.

        Returns:
            torch.Tensor: Radial basis function features of shape [B, N, N, num_rbf].
        """
        dists = torch.cdist(coords, coords, p=2)  # [B, N, N]
        dists = torch.clamp(dists, max=self.cutoff)
        dists_exp = dists.unsqueeze(-1)  # [B, N, N, 1]
        centers = self.centers.view(1, 1, 1, self.num_rbf)  # [1, 1, 1, num_rbf]
        rbf = torch.exp(
            -0.5 * ((dists_exp - centers) / self.width) ** 2
        )  # [B, N, N, num_rbf]

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        rbf_embed = self.embedding(rbf)  # [B, N, N, d_edge]
        rbf_embed = rbf_embed * edge_mask.unsqueeze(-1)
        return rbf_embed


class _CoordNorm(nn.Module):
    def __init__(self, d_equi, zero_com=True, eps=1e-6):
        super().__init__()

        self.d_equi = d_equi
        self.zero_com = zero_com
        self.eps = eps

        self.set_weights = torch.nn.Parameter(torch.ones((1, 1, 1, d_equi)))

    def forward(self, coord_sets, node_mask):
        """Apply coordinate normlisation layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [B, N, 3, d_equi]
            node_mask (torch.Tensor): Mask for nodes, shape [B, N], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Normalised coords, shape [B, N, 3, d_equi]
        """

        if self.zero_com:
            coord_sets = smolF.zero_com(coord_sets, node_mask)

        n_atoms = node_mask.sum(dim=-1).view(-1, 1, 1, 1)
        lengths = torch.linalg.vector_norm(coord_sets, dim=2, keepdim=True)
        scaled_lengths = lengths.sum(dim=1, keepdim=True) / n_atoms
        coord_sets = (coord_sets * self.set_weights) / (scaled_lengths + self.eps)
        coord_sets = coord_sets * node_mask.unsqueeze(-1).unsqueeze(-1)

        return coord_sets

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


# *****************************************************************************
# ******************************* Model  **************************************
# *****************************************************************************


class _EquivariantMLP(nn.Module):
    def __init__(self, d_equi, d_inv):
        super().__init__()

        self.node_proj = torch.nn.Sequential(
            torch.nn.Linear(d_equi + d_inv, d_equi),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_equi, d_equi),
            torch.nn.Sigmoid(),
        )
        self.coord_proj = torch.nn.Linear(d_equi, d_equi, bias=False)
        self.attn_proj = torch.nn.Linear(d_equi, d_equi, bias=False)

    def forward(self, equis, invs):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]

        Returns:
            torch.Tensor: Updated equivariant features, shape [B, N, 3, d_equi]
        """

        lengths = torch.linalg.vector_norm(equis, dim=2)
        inv_feats = torch.cat((invs, lengths), dim=-1)

        # inv_feats shape [B, N, 1, d_equi]
        # proj_sets shape [B, N, 3, d_equi]
        inv_feats = self.node_proj(inv_feats).unsqueeze(2)
        proj_sets = self.coord_proj(equis)

        gated_equis = proj_sets * inv_feats
        equis_out = self.attn_proj(gated_equis)
        return equis_out


class _PairwiseMessages(torch.nn.Module):
    """Compute pairwise features for a set of query and a set of key nodes"""

    def __init__(
        self,
        d_equi,
        d_q_inv,
        d_kv_inv,
        d_message,
        d_out,
        d_ff,
        d_edge=None,
        include_dists=False,
        include_rbfs=False,
    ):
        super().__init__()

        in_feats = (d_message * 2) + d_equi
        in_feats = in_feats + d_edge if d_edge is not None else in_feats
        in_feats = in_feats + d_equi if include_dists else in_feats
        in_feats = in_feats + d_equi if include_rbfs else in_feats

        self.d_equi = d_equi
        self.d_edge = d_edge
        self.include_dists = include_dists
        self.include_rbfs = include_rbfs

        self.q_message_proj = torch.nn.Linear(d_q_inv, d_message)
        self.k_message_proj = torch.nn.Linear(d_kv_inv, d_message)

        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_out),
        )

    def forward(self, q_equi, q_inv, k_equi, k_inv, edge_feats=None, rbf_embeds=None):
        """Produce messages between query and key

        Args:
            q_equi (torch.Tensor): Equivariant query features, shape [B, N_q, 3, d_equi]
            q_inv (torch.Tensor): Invariant query features, shape [B, N_q, d_q_inv]
            k_equi (torch.Tensor): Equivariant key features, shape [B, N_kv, 3, d_equi]
            k_inv (torch.Tensor): Invariant key features, shape [B, N_kv, 3, d_kv_inv]
            edge_feats (torch.Tensor): Edge features, shape [B, N_q, N_kv, d_edge]

        Returns:
            torch.Tensor: Message matrix, shape [B, N_q, N_k, d_out]
        """

        if edge_feats is not None and self.d_edge is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge as None."
            )

        if edge_feats is None and self.d_edge is not None:
            raise ValueError(
                "The model was initialised with d_edge but no edge feats were provided to forward fn."
            )

        q_equi_batched = q_equi.movedim(-1, 1).flatten(0, 1)
        k_equi_batched = k_equi.movedim(-1, 1).flatten(0, 1)

        dotprods = torch.bmm(q_equi_batched, k_equi_batched.transpose(1, 2))
        dotprods = dotprods.unflatten(0, (-1, self.d_equi)).movedim(1, -1)

        q_messages = (
            self.q_message_proj(q_inv).unsqueeze(2).expand(-1, -1, k_inv.size(1), -1)
        )
        k_messages = (
            self.k_message_proj(k_inv).unsqueeze(1).expand(-1, q_inv.size(1), -1, -1)
        )

        pairwise_feats = torch.cat((q_messages, k_messages, dotprods), dim=-1)

        if self.include_dists:
            vec_dists = q_equi.unsqueeze(2) - k_equi.unsqueeze(1)
            dists = torch.linalg.vector_norm(vec_dists, dim=3)
            pairwise_feats = torch.cat((pairwise_feats, dists), dim=-1)

        if rbf_embeds is not None:
            assert (
                self.include_rbfs
            ), "RBFs were provided but include_rbfs was set to False."
            pairwise_feats = torch.cat((pairwise_feats, rbf_embeds), dim=-1)

        if edge_feats is not None:
            pairwise_feats = torch.cat((pairwise_feats, edge_feats), dim=-1)

        pairwise_messages = self.message_mlp(pairwise_feats)
        return pairwise_messages


class _EquiAttention(torch.nn.Module):
    def __init__(self, d_equi, eps=1e-6):
        super().__init__()

        self.d_equi = d_equi
        self.eps = eps

        self.coord_proj = torch.nn.Linear(d_equi, d_equi, bias=False)
        self.attn_proj = torch.nn.Linear(d_equi, d_equi, bias=False)

    def forward(self, v_equi, messages, adj_matrix):
        """Compute an attention update for equivariant features

        Args:
            v_equi (torch.Tensor): Coordinate tensor, shape [B, N_kv, 3, d_equi]
            messages (torch.Tensor): Messages tensor, shape [B, N_q, N_kv, d_equi]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N_q, N_kv]

        Returns:
            torch.Tensor: Updates for equi features, shape [B, N_q, 3, d_equi]
        """

        proj_equi = self.coord_proj(v_equi)

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        # Attentions shape now [B * d_equi, N_q, N_kv]
        # proj_equi shape now [B * d_equi, N_kv, 3]
        attentions = attentions.movedim(-1, 1).flatten(0, 1)
        proj_equi = proj_equi.movedim(-1, 1).flatten(0, 1)
        attn_out = torch.bmm(attentions, proj_equi)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.d_equi)).movedim(1, -1)
        return self.attn_proj(attn_out)


class _InvAttention(torch.nn.Module):
    def __init__(self, d_inv, n_attn_heads, d_inv_cond=None):
        super().__init__()

        d_inv_in = d_inv_cond if d_inv_cond is not None else d_inv

        d_head = d_inv_in // n_attn_heads

        if d_inv_in % n_attn_heads != 0:
            raise ValueError(
                f"n_attn_heads must divide d_inv or d_inv_cond (if provided) exactly."
            )

        self.d_inv = d_inv
        self.n_attn_heads = n_attn_heads
        self.d_head = d_head

        self.in_proj = torch.nn.Linear(d_inv_in, d_inv_in)
        self.out_proj = torch.nn.Linear(d_inv_in, d_inv)

    def forward(self, v_inv, messages, adj_matrix):
        """Accumulate edge messages to each node using attention-based message passing

        Args:
            v_inv (torch.Tensor): Node feature tensor, shape [B, N_kv, d_inv or d_inv_cond if provided]
            messages (torch.Tensor): Messages tensor, shape [B, N_q, N_kv, d_message]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N_q, N_kv]

        Returns:
            torch.Tensor: Updates to invariant features, shape [B, N_q, d_inv]
        """

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(-1)
        attentions = torch.softmax(messages, dim=2)

        proj_feats = self.in_proj(v_inv)
        head_feats = proj_feats.unflatten(-1, (self.n_attn_heads, self.d_head))

        # Put n_heads into the batch dim for both the features and the attentions
        # head_feats shape [B * n_heads, N_kv, d_head]
        # attentions shape [B * n_heads, N_q, N_kv]
        head_feats = head_feats.movedim(-2, 1).flatten(0, 1)
        attentions = attentions.movedim(-1, 1).flatten(0, 1)
        attn_out = torch.bmm(attentions, head_feats)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.n_attn_heads))
        attn_out = attn_out.movedim(1, -2).flatten(2, 3)
        return self.out_proj(attn_out)


class SemlaSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_heads,
        d_ff,
        d_edge_in=None,
        d_edge_out=None,
        fixed_equi=False,
        use_rbf=False,
        eps=1e-6,
    ):
        super().__init__()

        d_out = n_heads if fixed_equi else d_equi + n_heads
        d_out = d_out + d_edge_out if d_edge_out is not None else d_out

        messages = _PairwiseMessages(
            d_equi,
            d_inv,
            d_inv,
            d_message,
            d_out,
            d_ff,
            d_edge=d_edge_in,
            include_rbfs=use_rbf,
        )

        inv_attn = _InvAttention(d_inv, n_attn_heads=n_heads)

        self.d_equi = d_equi
        self.n_heads = n_heads
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out
        self.fixed_equi = fixed_equi

        self.messages = messages
        self.inv_attn = inv_attn

        if not fixed_equi:
            self.equi_attn = _EquiAttention(d_equi, eps=eps)

    def forward(self, equis, invs, edges, adj_matrix, rbf_embeds=None):
        """Compute output of self attention layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            edges (torch.Tensor): Edge feature matrix, shape [B, N, N, d_edge]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N, N], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updates to equi features, inv feats, edge features
            Note that equi features are None if fixed_equi is specified, and edge features are None if d_edge_out
            is None. This ordering is used to maintain consistency with the ordering in other modules and to help to
            ensure that errors will be thrown if the wrong output is taken.
        """

        messages = self.messages(
            equis, invs, equis, invs, rbf_embeds=rbf_embeds, edge_feats=edges
        )

        inv_messages = messages[..., : self.n_heads]
        inv_updates = self.inv_attn(invs, inv_messages, adj_matrix)

        equi_updates = None
        if not self.fixed_equi:
            equi_messages = messages[..., self.n_heads : self.n_heads + self.d_equi]
            equi_updates = self.equi_attn(equis, equi_messages, adj_matrix)

        edge_feats = None
        if self.d_edge_out is not None:
            edge_feats = messages[..., self.n_heads + self.d_equi :]

        return equi_updates, inv_updates, edge_feats


class SemlaCondAttention(torch.nn.Module):
    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_heads,
        d_ff,
        d_inv_cond=None,
        d_edge_in=None,
        d_edge_out=None,
        eps=1e-6,
    ):
        super().__init__()

        # Set the number of pairwise output features depending on whether edge features are generated or not
        d_out = d_equi + n_heads
        d_out = d_out if d_edge_out is None else d_out + d_edge_out

        # Use d_inv for the conditional inviariant features by default
        d_inv_cond = d_inv if d_inv_cond is None else d_inv_cond

        messages = _PairwiseMessages(
            d_equi, d_inv, d_inv_cond, d_message, d_out, d_ff, d_edge=d_edge_in
        )

        equi_attn = _EquiAttention(d_equi, eps=eps)
        inv_attn = _InvAttention(d_inv, n_attn_heads=n_heads, d_inv_cond=d_inv_cond)

        self.d_equi = d_equi
        self.n_heads = n_heads
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out

        self.messages = messages
        self.equi_attn = equi_attn
        self.inv_attn = inv_attn

    def forward(self, equis, invs, cond_equis, cond_invs, edges, adj_matrix):
        """Compute output of conditional attention layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            cond_equis (torch.Tensor): Conditional equivariant features, shape [B, N_c, 3, d_equi]
            cond_invs (torch.Tensor): Conditional invariant features, shape [B, N_c, d_inv_cond]
            edges (torch.Tensor): Edge feature matrix, shape [B, N, N_c, d_edge]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N, N_c], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updates to equi feats, inv feats, and edge feats,
            respectively. Note that the edge features will be None is d_edge_out is None.
        """

        messages = self.messages(equis, invs, cond_equis, cond_invs, edge_feats=edges)
        equi_messages = messages[..., : self.d_equi]
        inv_messages = messages[..., self.d_equi : self.d_equi + self.n_heads]

        edge_feats = None
        if self.d_edge_out is not None:
            edge_feats = messages[..., self.d_equi + self.n_heads :]

        equi_updates = self.equi_attn(cond_equis, equi_messages, adj_matrix)
        inv_updates = self.inv_attn(cond_invs, inv_messages, adj_matrix)

        return equi_updates, inv_updates, edge_feats


# *****************************************************************************
# ********************************* Semla Layer *******************************
# *****************************************************************************


class SemlaLayer(torch.nn.Module):
    """Core layer of the Semla architecture.

    The layer contains a self-attention component and a feedforward component, by default. To turn on the conditional
    -attention component in addition to the others, set d_inv_cond to the number of invariant features in the
    conditional input. Note that currently d_equi must be the same for both attention inputs.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_attn_heads,
        d_message_ff,
        d_inv_cond=None,
        d_self_edge_in=None,
        d_self_edge_out=None,
        d_cond_edge_in=None,
        d_cond_edge_out=None,
        fixed_equi=False,
        use_rbf=False,
        zero_com=False,
        eps=1e-6,
    ):
        super().__init__()

        if d_inv_cond is not None and fixed_equi:
            raise ValueError(
                "Equivariant features cannot be fixed when using conditional attention."
            )

        self.d_inv_cond = d_inv_cond
        self.d_self_edge_out = d_self_edge_out
        self.d_cond_edge_out = d_cond_edge_out
        self.fixed_equi = fixed_equi
        self.use_rbf = use_rbf

        # *** Self attention components ***
        self.self_attn_inv_norm = torch.nn.LayerNorm(d_inv)

        if not fixed_equi:
            self.self_attn_equi_norm = _CoordNorm(d_equi, zero_com=zero_com, eps=eps)

        self.self_attention = SemlaSelfAttention(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_edge_in=d_self_edge_in,
            d_edge_out=d_self_edge_out,
            fixed_equi=fixed_equi,
            use_rbf=use_rbf,
            eps=eps,
        )

        # *** Cross attention components ***
        if d_inv_cond is not None:
            self.cond_attn_self_inv_norm = torch.nn.LayerNorm(d_inv)
            self.cond_attn_cond_inv_norm = torch.nn.LayerNorm(d_inv_cond)
            self.cond_attn_equi_norm = _CoordNorm(d_equi, zero_com=zero_com, eps=eps)

            self.cond_attention = SemlaCondAttention(
                d_equi,
                d_inv,
                d_message,
                n_attn_heads,
                d_message_ff,
                d_inv_cond=d_inv_cond,
                d_edge_in=d_cond_edge_in,
                d_edge_out=d_cond_edge_out,
                eps=eps,
            )

        # *** Feedforward components ***
        self.ff_inv_norm = torch.nn.LayerNorm(d_inv)
        self.inv_ff = LengthsMLP(d_inv, d_equi)

        if not fixed_equi:
            self.ff_equi_norm = _CoordNorm(d_equi, zero_com=zero_com, eps=eps)
            self.equi_ff = _EquivariantMLP(d_equi, d_inv)

    def forward(
        self,
        equis,
        invs,
        edges,
        adj_matrix,
        node_mask,
        rbf_embeds=None,
        cond_equis=None,
        cond_invs=None,
        cond_edges=None,
        cond_adj_matrix=None,
    ):
        """Compute output of Semla layer
        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            edges (torch.Tensor): Edge features, shape [B, N, N, d_self_edge_in]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N, N], 1 is connected, 0 otherwise
            node_mask (torch.Tensor): Mask for nodes, shape [B, N], 1 for real, 0 otherwise
            cond_equis (torch.Tensor): Cond equivariant features, shape [B, N, 3, d_equi]
            cond_invs (torch.Tensor): Cond invariant features, shape [B, N, d_inv_cond]
            cond_edges (torch.Tensor): Edge features between self and cond, shape [B, N, N_c, d_cond_edge_in]
            cond_adj_matrix (torch.Tensor): Adj matrix to cond data, shape [B, N, N_c], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Updated equivariant features, updated invariant features, self pairwise features, self-conditional
            pairwise features. Note that self pairwise features will be None if d_self_edge_out is None, and self
            -conditional pairwise features will be None if d_cond_edge_out is None.
            Tensor shapes: [B, N, 3, d_equi], [B, N, d_inv], [B, N, N, d_self_edge_out], [B, N, N_c, d_cond_edge_out]
        """

        if self.d_inv_cond is not None and cond_equis is None:
            raise ValueError(
                "The layer was initialised with conditional attention but cond_equis is missing."
            )

        if self.d_inv_cond is not None and cond_invs is None:
            raise ValueError(
                "The layer was initialised with conditional attention but cond_invs is missing."
            )

        if self.d_inv_cond is not None and cond_adj_matrix is None:
            raise ValueError(
                "The layer was initialised with conditional attention but cond_adj_matrix is missing."
            )

        # *** Self attention component ***
        invs_norm = self.self_attn_inv_norm(invs)
        equis_norm = (
            equis if self.fixed_equi else self.self_attn_equi_norm(equis, node_mask)
        )
        equi_updates, inv_updates, self_edge_feats = self.self_attention(
            equis_norm, invs_norm, edges, adj_matrix, rbf_embeds=rbf_embeds
        )

        invs = invs + inv_updates
        if not self.fixed_equi:
            equis = equis + equi_updates

        # *** Conditional attention component ***
        cond_edge_feats = None
        if self.d_inv_cond is not None:
            equis, invs, cond_edge_feats = self._compute_cond_attention(
                equis,
                invs,
                cond_equis,
                cond_invs,
                cond_edges,
                node_mask,
                cond_adj_matrix,
            )

        # *** Feedforward component ***
        invs_norm = self.ff_inv_norm(invs)
        equis_norm = equis if self.fixed_equi else self.ff_equi_norm(equis, node_mask)

        inv_update = self.inv_ff(equis_norm.movedim(-1, 1), invs_norm)
        invs = invs + inv_update

        if not self.fixed_equi:
            equi_update = self.equi_ff(equis_norm, invs_norm)
            equis = equis + equi_update

        return equis, invs, self_edge_feats, cond_edge_feats

    def _compute_cond_attention(
        self, equis, invs, cond_equis, cond_invs, cond_edges, node_mask, cond_adj_matrix
    ):
        self_invs_norm = self.cond_attn_self_inv_norm(invs)
        cond_invs_norm = self.cond_attn_cond_inv_norm(cond_invs)
        equis_norm = self.cond_attn_equi_norm(equis, node_mask)

        equi_updates, inv_updates, cond_edge_feats = self.cond_attention(
            equis_norm,
            self_invs_norm,
            cond_equis,
            cond_invs_norm,
            cond_edges,
            cond_adj_matrix,
        )

        equis = equis + equi_updates
        invs = invs + inv_updates

        return equis, invs, cond_edge_feats


# *****************************************************************************
# ************************ Encoder and Decoder Stacks *************************
# *****************************************************************************


class _InvariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        d_inv,
        n_atom_types,
        n_bond_types,
        emb_size,
        n_charge_types=None,
        n_extra_feats=None,
        n_res_types=None,
        self_cond=False,
        max_size=None,
    ):
        super().__init__()

        n_embeddings = 2 if max_size is not None else 1
        n_embeddings = n_embeddings + 1 if n_charge_types is not None else n_embeddings
        n_embeddings = n_embeddings + 1 if n_res_types is not None else n_embeddings

        atom_in_feats = emb_size * n_embeddings
        atom_in_feats = atom_in_feats + n_atom_types if self_cond else atom_in_feats
        atom_in_feats = (
            atom_in_feats + n_extra_feats
            if n_extra_feats is not None
            else atom_in_feats
        )

        self.n_charge_types = n_charge_types
        self.n_extra_feats = n_extra_feats
        self.n_res_types = n_res_types
        self.self_cond = self_cond
        self.max_size = max_size

        self.atom_type_emb = torch.nn.Embedding(n_atom_types, emb_size)

        if n_charge_types is not None:
            self.atom_charge_emb = torch.nn.Embedding(n_charge_types, emb_size)

        if n_res_types is not None:
            self.res_type_emb = torch.nn.Embedding(n_res_types, emb_size)

        if max_size is not None:
            self.size_emb = torch.nn.Embedding(max_size, emb_size)

        self.atom_emb = torch.nn.Sequential(
            torch.nn.Linear(atom_in_feats, d_inv),
            torch.nn.SiLU(),
            torch.nn.Linear(d_inv, d_inv),
        )

        self.bond_emb = torch.nn.Embedding(n_bond_types, emb_size)

        if self_cond:
            self.bond_proj = torch.nn.Linear(emb_size + n_bond_types, emb_size)

    def forward(
        self,
        atom_types,
        bond_types,
        atom_mask,
        atom_charges=None,
        extra_feats=None,
        res_types=None,
        cond_types=None,
        cond_bonds=None,
    ):
        if (cond_types is not None or cond_bonds is not None) and not self.self_cond:
            raise ValueError(
                "Conditional inputs were provided but the model was initialised with self_cond as False."
            )

        if (cond_types is None or cond_bonds is None) and self.self_cond:
            raise ValueError(
                "Conditional inputs must be provided if using self conditioning."
            )

        if self.n_charge_types is not None and atom_charges is None:
            raise ValueError(
                "The invariant embedding was initialised for charge embeddings but none were provided."
            )

        if self.n_extra_feats is not None and extra_feats is None:
            raise ValueError(
                "The invariant embedding was initialised with extra feats but none were provided."
            )

        invs = self.atom_type_emb(atom_types)

        if self.n_charge_types is not None:
            charge_feats = self.atom_charge_emb(atom_charges)
            invs = torch.cat((invs, charge_feats), dim=-1)

        if self.n_extra_feats is not None:
            invs = torch.cat((invs, extra_feats), dim=-1)

        if self.n_res_types is not None:
            residue_type_feats = self.res_type_emb(res_types)
            invs = torch.cat((invs, residue_type_feats), dim=-1)

        if self.max_size is not None:
            n_atoms = atom_mask.sum(dim=-1, keepdim=True)
            size_emb = self.size_emb(n_atoms).expand(-1, atom_mask.size(1), -1)
            invs = torch.cat((invs, size_emb), dim=-1)

        if self.self_cond:
            invs = torch.cat((invs, cond_types), dim=-1)

        invs = self.atom_emb(invs)

        edges = self.bond_emb(bond_types)
        if self.self_cond:
            edges = torch.cat((edges, cond_bonds), dim=-1)
            edges = self.bond_proj(edges)

        return invs, edges


class PocketEncoder(torch.nn.Module):
    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_names,
        n_bond_types,
        n_res_types,
        n_charge_types=7,
        emb_size=64,
        fixed_equi=False,
        use_rbf=False,
        eps=1e-6,
    ):
        super().__init__()

        if fixed_equi and d_equi != 1:
            raise ValueError(f"If fixed_equi is True d_equi must be 1, got {d_equi}")

        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_message = d_message
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.d_message_ff = d_message_ff
        self.d_edge = d_edge
        self.emb_size = emb_size
        self.fixed_equi = fixed_equi
        self.use_rbf = use_rbf
        self.eps = eps

        # Embedding and encoding modules
        self.inv_emb = _InvariantEmbedding(
            d_inv,
            n_atom_names,
            n_bond_types,
            emb_size,
            n_charge_types=n_charge_types,
            n_res_types=n_res_types,
        )
        self.bond_emb = _PairwiseMessages(
            d_equi, d_inv, d_inv, d_message, d_edge, d_message_ff, emb_size
        )

        if not fixed_equi:
            self.coord_emb = torch.nn.Linear(1, d_equi, bias=False)

        if self.use_rbf:
            self.rbf_embed = RBFEmbed(d_equi, num_rbf=32, cutoff=5.0)

        # Create a stack of encoder layers
        layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_self_edge_in=d_edge,
            fixed_equi=fixed_equi,
            use_rbf=self.use_rbf,
            zero_com=False,
            eps=eps,
        )

        layers = self._get_clones(layer, n_layers)
        self.layers = torch.nn.ModuleList(layers)

    @property
    def hparams(self):
        return {
            "d_equi": self.d_equi,
            "d_inv": self.d_inv,
            "d_message": self.d_message,
            "n_layers": self.n_layers,
            "n_attn_heads": self.n_attn_heads,
            "d_message_ff": self.d_message_ff,
            "d_edge": self.d_edge,
            "emb_size": self.emb_size,
            "fixed_equi": self.fixed_equi,
            "eps": self.eps,
        }

    def forward(
        self, coords, atom_names, atom_charges, res_types, bond_types, atom_mask=None
    ):
        """Encode the protein pocket into a learnable representation

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
            atom_names (torch.Tensor): Atom name indices, shape [B, N]
            atom_charges (torch.Tensor): Atom charge indices, shape [B, N]
            residue_types (torch.Tensor): Residue type indices for each atom, shape [B, N]
            bond_types (torch.Tensor): Bond type indicies for each pair, shape [B, N, N]
            atom_mask (torch.Tensor): Mask for atoms, shape [B, N], 1 for real atom, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Equivariant and invariant features, [B, N, 3, d_equi] and [B, N, d_inv]
        """

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = smolF.adj_from_node_mask(atom_mask, self_connect=True)

        coords = coords.unsqueeze(-1)
        equis = coords if self.fixed_equi else self.coord_emb(coords)

        if self.use_rbf:
            rbf_embeds = self.rbf_embed(equis, node_mask=atom_mask)
        else:
            rbf_embeds = None

        invs, edges = self.inv_emb(
            atom_names,
            bond_types,
            atom_mask,
            atom_charges=atom_charges,
            res_types=res_types,
        )
        edges = self.bond_emb(equis, invs, equis, invs, edge_feats=edges)
        edges = edges * adj_matrix.unsqueeze(-1)

        for layer in self.layers:
            equis, invs, _, _ = layer(
                equis, invs, edges, adj_matrix, atom_mask, rbf_embeds=rbf_embeds
            )

        return equis, invs

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


class LigandDecoder(torch.nn.Module):
    """Class for generating ligands

    By default no pocket conditioning is used, to allow pocket conditioning set d_pocket_inv to the size of the pocket
    invariant feature vectors. d_equi must be the same for both pocket and ligand.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_types,
        n_bond_types,
        n_charge_types=7,
        emb_size=64,
        d_pocket_inv=None,
        n_interaction_types=None,
        predict_interactions=False,
        flow_interactions=False,
        use_lig_pocket_rbf=False,
        use_rbf=False,
        use_sphcs=False,
        n_extra_atom_feats=None,
        self_cond=False,
        coord_skip_connect=True,
        split_continuous_discrete_time=False,
        eps=1e-6,
    ):
        super().__init__()

        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_message = d_message
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.d_message_ff = d_message_ff
        self.d_edge = d_edge
        self.emb_size = emb_size
        self.d_pocket_inv = d_pocket_inv
        self.self_cond = self_cond
        self.eps = eps
        self.flow_interactions = flow_interactions
        self.predict_interactions = predict_interactions
        self.use_sphcs = use_sphcs
        self.use_lig_pocket_rbf = use_lig_pocket_rbf
        self.use_rbf = use_rbf
        self.interactions = n_interaction_types is not None
        self.coord_skip_connect = coord_skip_connect
        self.split_continuous_discrete_time = split_continuous_discrete_time

        if d_pocket_inv is None and n_interaction_types is not None:
            raise ValueError(
                "Pocket conditioning is required for interaction encoding and prediction."
            )

        coord_proj_feats = (
            3
            if self_cond and self.split_continuous_discrete_time
            else 2 if self_cond else 1
        )
        if self.use_sphcs:
            self.equis_sphcs = EquisSphc(
                l_min=0,
                l_max=3,
                r_cut=5,
                eps=1e-8,
                return_sphc_distance_matrix=True,
                p=6,
                kappa=1.0,
            )
            self.embed_sphcs = torch.nn.Linear(
                self.equis_sphcs.out_dim, d_equi, bias=False
            )
            # coord_proj_feats += self.equis_sphcs.out_dim
        d_cond_edge_in = (
            d_edge + 1
            if n_interaction_types is not None
            else d_edge if self.use_lig_pocket_rbf else None
        )
        d_cond_edge_out = d_edge if n_interaction_types is not None else None

        if self.use_rbf:
            self.rbf_embed = RBFEmbed(d_equi, num_rbf=32, cutoff=5.0)
        # *** Embedding and encoding modules ***

        self.inv_emb = _InvariantEmbedding(
            d_inv,
            n_atom_types,
            n_bond_types,
            emb_size,
            n_extra_feats=n_extra_atom_feats,
            self_cond=self_cond,
            max_size=512,
        )
        self.bond_emb = _PairwiseMessages(
            d_equi, d_inv, d_inv, d_message, d_edge, d_message_ff, emb_size
        )
        self.coord_emb = torch.nn.Linear(coord_proj_feats, d_equi, bias=False)

        # *** Layer stack ***
        enc_layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_pocket_inv,
            d_self_edge_in=d_edge,
            d_cond_edge_in=d_cond_edge_in,
            use_rbf=self.use_rbf,
            zero_com=False,
            eps=eps,
        )
        layers = self._get_clones(enc_layer, n_layers - 1)

        # Create one final layer which also produces edge feature outputs
        dec_layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_pocket_inv,
            d_self_edge_in=d_edge,
            d_self_edge_out=d_edge,
            d_cond_edge_in=d_cond_edge_in,
            d_cond_edge_out=d_cond_edge_out,
            use_rbf=self.use_rbf,
            zero_com=False,
            eps=eps,
        )
        layers.append(dec_layer)

        self.layers = torch.nn.ModuleList(layers)

        # *** Final norms and projections ***

        self.final_coord_norm = _CoordNorm(d_equi, zero_com=False, eps=eps)
        self.final_inv_norm = torch.nn.LayerNorm(d_inv)
        self.final_bond_norm = torch.nn.LayerNorm(d_edge)

        self.coord_out_proj = torch.nn.Linear(d_equi, 1, bias=False)
        self.atom_type_proj = torch.nn.Linear(d_inv, n_atom_types)
        self.atom_charge_proj = torch.nn.Linear(d_inv, n_charge_types)

        self.bond_refine = BondRefine(
            d_inv, d_message, d_edge, d_ff=d_inv, norm_feats=False
        )
        self.bond_proj = torch.nn.Linear(d_edge, n_bond_types)

        # *** Modules for interactions ***
        self.interaction_emb = (
            torch.nn.Embedding(n_interaction_types, d_edge)
            if n_interaction_types is not None and flow_interactions
            else None
        )
        self.radial_basis_embed = (
            RadialBasisEmbedding(d_edge=d_cond_edge_in, num_rbf=32, cutoff=5.0)
            if self.predict_interactions or self.use_lig_pocket_rbf
            else None
        )
        self.interaction_refine = (
            _PairwiseMessages(
                d_equi,
                d_inv,
                d_pocket_inv,
                d_message,
                d_edge,
                d_message_ff,
                d_edge=d_edge,
                include_dists=True,
            )
            if n_interaction_types is not None
            else None
        )
        self.interaction_proj = (
            torch.nn.Linear(d_edge, n_interaction_types)
            if n_interaction_types is not None
            else None
        )

    @property
    def hparams(self):
        return {
            "d_equi": self.d_equi,
            "d_inv": self.d_inv,
            "d_message": self.d_message,
            "n_layers": self.n_layers,
            "n_attn_heads": self.n_attn_heads,
            "d_message_ff": self.d_message_ff,
            "d_edge": self.d_edge,
            "emb_size": self.emb_size,
            "self_cond": self.self_cond,
            "eps": self.eps,
            "interactions": self.interactions,
            "flow_interactions": self.flow_interactions,
            "predict_interactions": self.predict_interactions,
            "use_lig_pocket_rbf": self.use_lig_pocket_rbf,
            "coord_skip_connect": self.coord_skip_connect,
            "split_continuous_discrete_time": self.split_continuous_discrete_time,
            "use_rbf": self.use_rbf,
            "use_sphcs": self.use_sphcs,
        }

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_mask=None,
        extra_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        pocket_coords=None,
        pocket_equis=None,
        pocket_invs=None,
        pocket_atom_mask=None,
        interactions=None,
    ):
        """Generate ligand atom types, coords, charges and bonds

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
            atom_types (torch.Tensor): Atom name indices, shape [B, N]
            bond_types (torch.Tensor): Bond type indicies for each pair, shape [B, N, N]
            atom_mask (torch.Tensor): Mask for atoms, shape [B, N], 1 for real atom, 0 otherwise
            extra_feats (torch.Tensor): Additional atom features, shape [B, N, n_extra_atom_feats]
            cond_coords (torch.Tensor): Self conditioning coords, shape [B, N, 3]
            cond_atomics (torch.Tensor): Self conditioning atom types, shape [B, N, n_atom_types]
            cond_bonds (torch.Tensor): Self conditioning bond types, shape [B, N, N, n_bond_types]
            pocket_coords (torch.Tensor): Original pocket coords, shape [B, N_p, 3]
            pocket_equis (torch.Tensor): Equivariant encoded pocket features, shape [B, N_p, d_equi]
            pocket_invs (torch.Tensor): Invariant encoded pocket features, shape [B, N_p, d_pocket_inv]
            pocket_atom_mask (torch.Tensor): Mask for pocket atom, shape [B, N_p], 1 for real, 0 otherwise
            interactions (torch.Tensor): Interaction types between pocket and ligand, shape [B, N, N_p]

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charge logits)
            All torch.Tensor, shapes:
                Coordinates: [B, N, 3],
                Type logits: [B, N, n_atom_types],
                Bond logits: [B, N, N, n_bond_types],
                Charge logits: [B, N, n_charge_types]
                Interaction logits: [B, N, N_p, n_interaction_types] if interactions are provided
        """

        if (cond_atomics is not None or cond_bonds is not None) and not self.self_cond:
            raise ValueError(
                "Conditional inputs were provided but the model was initialised with self_cond as False."
            )

        if (cond_atomics is None or cond_bonds is None) and self.self_cond:
            raise ValueError(
                "Conditional inputs must be provided if using self conditioning."
            )

        if (
            pocket_invs is not None or pocket_equis is not None
        ) and self.d_pocket_inv is None:
            raise ValueError(
                "Pocket cond inputs were provided but the model was not initialised for pocket cond."
            )

        if (
            pocket_invs is None or pocket_equis is None
        ) and self.d_pocket_inv is not None:
            raise ValueError(
                "Pocket cond inputs must be provided if using pocket conditioning."
            )

        if not self.interactions and interactions is not None:
            raise ValueError(
                "Interactions were provided but the model was not initialised for interactions."
            )

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = smolF.adj_from_node_mask(atom_mask, self_connect=True)

        # Work out adj matrix between pocket and ligand, if required
        cond_adj_matrix = None
        if self.d_pocket_inv is not None:
            cond_adj_matrix = atom_mask.float().unsqueeze(
                2
            ) * pocket_atom_mask.float().unsqueeze(1)
            cond_adj_matrix = cond_adj_matrix.long()

        # Embed interaction types, if required
        interaction_feats = None
        if self.interactions and interactions is not None:
            num_atoms_pocket = interactions.shape[-1]
            interaction_feats = self.interaction_emb(interactions)
            interaction_feats = torch.cat(
                (
                    interaction_feats,
                    extra_feats[-1].unsqueeze(-1).expand(-1, -1, num_atoms_pocket, 1),
                ),
                dim=-1,
            )
        elif (
            self.interactions and self.predict_interactions
        ) or self.use_lig_pocket_rbf:
            num_atoms_pocket = pocket_atom_mask.shape[-1]
            interaction_feats = self.radial_basis_embed(
                coords,
                pocket_coords,
                ligand_mask=atom_mask,
                pocket_mask=pocket_atom_mask,
            )
            if self.predict_interactions:
                interaction_feats = torch.cat(
                    (
                        interaction_feats,
                        extra_feats[-1]
                        .unsqueeze(-1)
                        .expand(-1, -1, num_atoms_pocket, 1),
                    ),
                    dim=-1,
                )

        # Project coords to d_equi
        if self.self_cond:
            _coords = torch.cat(
                (coords.unsqueeze(-1), cond_coords.unsqueeze(-1)), dim=-1
            )
        else:
            _coords = coords.unsqueeze(-1)
        if self.split_continuous_discrete_time:
            _coords = torch.cat(
                [_coords, extra_feats[0].unsqueeze(2).expand(-1, -1, 3, -1)], dim=-1
            )
        equis = self.coord_emb(_coords)
        if self.use_sphcs:
            B, N = coords.shape[:2]
            equis_sphc, sphc_dst = self.equis_sphcs(coords, mask=atom_mask)
            equis_sphc = self.embed_sphcs(equis_sphc)
            equis_sphc = equis_sphc.unsqueeze(2).expand(B, N, 3, -1)
            # _coords = torch.cat((_coords, equis_sphc), dim=-1)
            equis = equis + equis_sphc
        if self.use_rbf:
            rbf_embeds = self.rbf_embed(coords, node_mask=atom_mask)
        else:
            rbf_embeds = None

        # Embed invariant features
        invs, edges = self.inv_emb(
            atom_types,
            bond_types,
            atom_mask,
            cond_types=cond_atomics,
            cond_bonds=cond_bonds,
            extra_feats=extra_feats[1],
        )
        edges = self.bond_emb(equis, invs, equis, invs, edge_feats=edges)
        edges = edges * adj_matrix.unsqueeze(-1)

        # Iterate over Semla layers
        for layer in self.layers:
            equis, invs, edge_out, interaction_out = layer(
                equis,
                invs,
                edges,
                adj_matrix,
                atom_mask,
                rbf_embeds=rbf_embeds,
                cond_equis=pocket_equis,
                cond_invs=pocket_invs,
                cond_edges=interaction_feats,
                cond_adj_matrix=cond_adj_matrix,
            )

        if self.interactions:
            # Pass interactions through refinement layer and project to logits, if required
            interaction_out = self.interaction_refine(
                equis,
                invs,
                pocket_equis,
                pocket_invs,
                interaction_out,
            )
            interaction_logits = self.interaction_proj(interaction_out)

        # Project coords back to one equivariant feature
        equis = self.final_coord_norm(equis, atom_mask)
        out_coords = self.coord_out_proj(equis).squeeze(-1)
        if self.coord_skip_connect:
            out_coords = out_coords * atom_mask.unsqueeze(-1) + coords

        # Project invariant features to atom and charge logits
        invs_norm = self.final_inv_norm(invs)
        atom_type_logits = self.atom_type_proj(invs_norm)
        charge_logits = self.atom_charge_proj(invs_norm)

        # Pass bonds through refinement layer and project to logits
        edge_norm = self.final_bond_norm(edge_out)
        edge_out = self.bond_refine(out_coords, invs_norm, atom_mask, edge_norm)
        bond_logits = self.bond_proj(edge_out + edge_out.transpose(1, 2))

        if self.interactions:
            return (
                out_coords,
                atom_type_logits,
                bond_logits,
                charge_logits,
                interaction_logits,
                atom_mask,
            )

        return out_coords, atom_type_logits, bond_logits, charge_logits, atom_mask

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


# *****************************************************************************
# ****************************** Overall Models *******************************
# *****************************************************************************


class LigandGenerator(torch.nn.Module):
    """Main entry point class for generating ligands.

    This class allows both unconditional and pocket-conditioned models to be created. The pocket-conditioned model
    can be created by passing in a PocketEncoder object with the pocket_enc argument, this will automatically setup
    the ligand decoder to use condition attention in addition to self attention. If pocket_enc is None the ligand
    decoder is setup as an unconditional generator.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_types,
        n_bond_types,
        n_charge_types=7,
        emb_size=64,
        predict_interactions=False,
        flow_interactions=False,
        use_lig_pocket_rbf=False,
        use_rbf=False,
        use_sphcs=False,
        n_interaction_types=None,
        n_extra_atom_feats=None,
        self_cond=False,
        coord_skip_connect=True,
        split_continuous_discrete_time=False,
        pocket_enc=None,
        eps=1e-6,
    ):
        super().__init__()

        duplicate_pocket_equi = False
        if pocket_enc is not None:
            duplicate_pocket_equi = pocket_enc.d_equi == 1
            if not duplicate_pocket_equi and pocket_enc.d_equi != d_equi:
                raise ValueError(
                    "d_equi must be either the same for the pocket and ligand or 1 for the pocket."
                )

        d_pocket_inv = pocket_enc.d_inv if pocket_enc is not None else None

        self.d_equi = d_equi
        self.duplicate_pocket_equi = duplicate_pocket_equi

        ligand_dec = LigandDecoder(
            d_equi,
            d_inv,
            d_message,
            n_layers,
            n_attn_heads,
            d_message_ff,
            d_edge,
            n_atom_types,
            n_bond_types,
            n_charge_types=n_charge_types,
            emb_size=emb_size,
            d_pocket_inv=d_pocket_inv,
            predict_interactions=predict_interactions,
            flow_interactions=flow_interactions,
            use_lig_pocket_rbf=use_lig_pocket_rbf,
            use_rbf=use_rbf,
            use_sphcs=use_sphcs,
            n_interaction_types=n_interaction_types,
            n_extra_atom_feats=n_extra_atom_feats,
            self_cond=self_cond,
            coord_skip_connect=coord_skip_connect,
            split_continuous_discrete_time=split_continuous_discrete_time,
            eps=eps,
        )

        self.pocket_enc = pocket_enc
        self.ligand_dec = ligand_dec

    @property
    def hparams(self):
        dec_hparams = self.ligand_dec.hparams
        if self.pocket_enc is not None:
            pocket_hparams = {
                f"pocket-{name}": val for name, val in self.pocket_enc.hparams.items()
            }
            hparams = {**dec_hparams, **pocket_hparams}

        return hparams

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_mask=None,
        extra_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        pocket_coords=None,
        pocket_atom_names=None,
        pocket_atom_charges=None,
        pocket_res_types=None,
        pocket_bond_types=None,
        pocket_atom_mask=None,
        pocket_equis=None,
        pocket_invs=None,
        interactions=None,
    ):

        if self.pocket_enc is not None:
            if pocket_equis is None and pocket_invs is None:
                pocket_equis, pocket_invs = self.get_pocket_encoding(
                    pocket_coords,
                    pocket_atom_names,
                    pocket_atom_charges,
                    pocket_res_types,
                    pocket_bond_types,
                    pocket_atom_mask=pocket_atom_mask,
                )

        decoder_out = self.decode(
            coords,
            atom_types,
            bond_types,
            atom_mask=atom_mask,
            extra_feats=extra_feats,
            cond_coords=cond_coords,
            cond_atomics=cond_atomics,
            cond_bonds=cond_bonds,
            pocket_coords=pocket_coords,
            pocket_equis=pocket_equis,
            pocket_invs=pocket_invs,
            pocket_atom_mask=pocket_atom_mask,
            interactions=interactions,
        )

        return decoder_out

    def get_pocket_encoding(
        self,
        pocket_coords,
        pocket_atom_names,
        pocket_atom_charges,
        pocket_res_types,
        pocket_bond_types,
        pocket_atom_mask=None,
    ):
        if None in [
            pocket_coords,
            pocket_atom_names,
            pocket_atom_charges,
            pocket_res_types,
            pocket_bond_types,
        ]:
            raise ValueError(
                "All pocket inputs must be provided if the model is created with pocket cond."
            )

        if self.pocket_enc is None:
            raise ValueError(
                "Cannot call encode on a model initialised without a pocket encoder."
            )

        pocket_equis, pocket_invs = self.pocket_enc(
            pocket_coords,
            pocket_atom_names,
            pocket_atom_charges,
            pocket_res_types,
            pocket_bond_types,
            atom_mask=pocket_atom_mask,
        )

        if self.duplicate_pocket_equi:
            pocket_equis = pocket_equis.expand(-1, -1, -1, self.d_equi)

        return pocket_equis, pocket_invs

    def decode(
        self,
        coords,
        atom_types,
        bond_types,
        atom_mask=None,
        extra_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        pocket_coords=None,
        pocket_equis=None,
        pocket_invs=None,
        pocket_atom_mask=None,
        interactions=None,
    ):

        if self.pocket_enc is not None and pocket_invs is None:
            raise ValueError(
                "The model was initialised with pocket conditioning but pocket_invs was not provided."
            )

        if self.pocket_enc is not None and pocket_equis is None:
            raise ValueError(
                "The model was initialised with pocket conditioning but pocket_invs was not provided."
            )

        decoder_out = self.ligand_dec(
            coords,
            atom_types,
            bond_types,
            atom_mask=atom_mask,
            extra_feats=extra_feats,
            cond_coords=cond_coords,
            cond_atomics=cond_atomics,
            cond_bonds=cond_bonds,
            pocket_coords=pocket_coords,
            pocket_equis=pocket_equis,
            pocket_invs=pocket_invs,
            pocket_atom_mask=pocket_atom_mask,
            interactions=interactions,
        )

        return decoder_out


############### ComplexDecoder ##################


class Encoder(torch.nn.Module):
    """Class for encoding both ligand and pocket structures"""

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        d_message_ff,
        d_edge,
        n_atom_types,
        n_bond_types,
        n_charge_types=None,
        n_extra_atom_feats=None,
        n_res_types=None,
        emb_size=64,
        use_sphcs=False,
        self_cond=False,
        self_cond_inv=False,
        max_size=182,
        eps=1e-6,
    ):
        super().__init__()

        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_message = d_message
        self.d_message_ff = d_message_ff
        self.d_edge = d_edge
        self.emb_size = emb_size
        self.self_cond = self_cond
        self.self_cond_inv = self_cond_inv
        self.n_charge_types = n_charge_types
        self.n_bond_types = n_bond_types
        self.n_atom_types = n_atom_types
        self.n_res_types = n_res_types
        self.max_size = max_size

        self.eps = eps
        self.use_sphcs = use_sphcs

        coord_proj_feats = 2 if self_cond else 1
        if self.use_sphcs:
            self.equis_sphcs = EquisSphc(
                l_min=0,
                l_max=3,
                r_cut=5,
                eps=1e-8,
                return_sphc_distance_matrix=True,
                p=6,
                kappa=1.0,
            )
            coord_proj_feats += self.equis_sphcs.out_dim

        # *** Embedding and encoding modules ***
        self.inv_emb = _InvariantEmbedding(
            d_inv,
            n_atom_types,
            n_bond_types,
            emb_size,
            n_extra_feats=n_extra_atom_feats,
            n_charge_types=n_charge_types,
            n_res_types=n_res_types,
            self_cond=self_cond_inv,
            max_size=max_size,
        )
        self.bond_emb = _PairwiseMessages(
            d_equi, d_inv, d_inv, d_message, d_edge, d_message_ff, emb_size
        )
        self.coord_emb = torch.nn.Linear(coord_proj_feats, d_equi, bias=False)

    @property
    def hparams(self):
        return {
            "d_equi": self.d_equi,
            "d_inv": self.d_inv,
            "d_message": self.d_message,
            "d_message_ff": self.d_message_ff,
            "d_edge": self.d_edge,
            "emb_size": self.emb_size,
            "n_res_types": self.n_res_types,
            "n_atom_types": self.n_atom_types,
            "n_bond_types": self.n_bond_types,
            "n_charge_types": self.n_charge_types,
            "self_cond": self.self_cond,
            "self_cond_inv": self.self_cond_inv,
            "use_sphcs": self.use_sphcs,
            "max_size": self.max_size,
            "eps": self.eps,
        }

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_mask=None,
        extra_feats=None,
        atom_charges=None,
        res_types=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
    ):
        """Generate ligand atom types, coords, charges and bonds

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
            atom_types (torch.Tensor): Atom name indices, shape [B, N]
            bond_types (torch.Tensor): Bond type indicies for each pair, shape [B, N, N]
            atom_mask (torch.Tensor): Mask for atoms, shape [B, N], 1 for real atom, 0 otherwise
            extra_feats (torch.Tensor): Additional atom features, shape [B, N, n_extra_atom_feats]
            atom_charges (torch.Tensor): Atom charge indices, shape [B, N]
            res_types (torch.Tensor): Residue type indices for each atom, shape [B, N]
            cond_coords (torch.Tensor): Self conditioning coords, shape [B, N, 3]
            cond_atomics (torch.Tensor): Self conditioning atom types, shape [B, N, n_atom_types]
            cond_bonds (torch.Tensor): Self conditioning bond types, shape [B, N, N, n_bond_types]

        Returns:
            equis: [B, N, 3, d_equi]
            invs: [B, N, d_inv]
            edges: [B, N, N, d_edge]
            atom_mask: [B, N]
        """

        if self.self_cond and cond_coords is None:
            raise ValueError("self_cond is True but cond_coords is not provided.")
        if (
            cond_atomics is not None or cond_bonds is not None
        ) and not self.self_cond_inv:
            raise ValueError(
                "Conditional inputs were provided but the model was initialised with self_cond_inv as False."
            )

        if (cond_atomics is None or cond_bonds is None) and self.self_cond_inv:
            raise ValueError(
                "Conditional inputs must be provided if using self conditioning."
            )

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = smolF.adj_from_node_mask(atom_mask, self_connect=True)

        # Project coords to d_equi
        if self.self_cond:
            _coords = torch.cat(
                (coords.unsqueeze(-1), cond_coords.unsqueeze(-1)), dim=-1
            )
        else:
            _coords = coords.unsqueeze(-1)

        if self.use_sphcs:
            B, N = coords.shape[:2]
            equis_sphc, sphc_dst = self.equis_sphcs(coords, mask=atom_mask)
            # Ensure proper projection if needed. Here we simply concatenate:
            equis_sphc = equis_sphc.unsqueeze(2).expand(B, N, 3, -1)
            _coords = torch.cat((_coords, equis_sphc), dim=-1)
        if self.split_continuous_discrete_time:
            _coords = torch.cat((_coords, extra_feats[0].unsqueeze(-1)), dim=-1)
        equis = self.coord_emb(_coords)

        # Embed invariant features
        invs, edges = self.inv_emb(
            atom_types,
            bond_types,
            atom_mask,
            atom_charges=atom_charges,
            res_types=res_types,
            cond_types=cond_atomics,
            cond_bonds=cond_bonds,
            extra_feats=extra_feats,
        )
        edges = self.bond_emb(equis, invs, equis, invs, edge_feats=edges)
        edges = edges * adj_matrix.unsqueeze(-1)

        return equis, invs, edges, adj_matrix, atom_mask

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


class ComplexDecoder(nn.Module):
    """
    This decoder takes ligand latents (lig_equis, lig_invs, lig_edges, lig_mask)
    and pocket latents (pocket_equis, pocket_invs, pocket_edges, pocket_mask) as input.
    It processes the ligand branch with cross‑conditioning from the pocket and vice-versa.
    No final output projections are applied.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        d_pocket_inv,
        eps=1e-6,
    ):
        super().__init__()
        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_message = d_message
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.d_message_ff = d_message_ff
        self.d_edge = d_edge
        self.d_pocket_inv = d_pocket_inv
        self.eps = eps

        d_cond_edge_in = d_edge
        d_cond_edge_out = None

        # Ligand decoder branch: ligand is conditioned on pocket invariants.
        ligand_enc_layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_pocket_inv,
            d_self_edge_in=d_edge,
            d_cond_edge_in=d_cond_edge_in,
            zero_com=False,
            eps=eps,
        )
        ligand_dec_layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_pocket_inv,
            d_self_edge_in=d_edge,
            d_self_edge_out=d_edge,
            d_cond_edge_in=d_cond_edge_in,
            d_cond_edge_out=d_cond_edge_out,
            zero_com=False,
            eps=eps,
        )
        lig_layers = self._get_clones(ligand_enc_layer, n_layers - 1)
        lig_layers.append(ligand_dec_layer)
        self.lig_layers = nn.ModuleList(lig_layers)

        # Pocket decoder branch: pocket is conditioned on ligand latents.
        pocket_enc_layer = SemlaLayer(
            d_equi,
            d_pocket_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_inv,
            d_self_edge_in=d_edge,
            d_cond_edge_in=d_cond_edge_in,
            zero_com=False,
            eps=eps,
        )
        pocket_dec_layer = SemlaLayer(
            d_equi,
            d_pocket_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_inv,
            d_self_edge_in=d_edge,
            d_cond_edge_in=d_cond_edge_in,
            d_self_edge_out=None,
            d_cond_edge_out=None,
            zero_com=False,
            eps=eps,
        )
        pocket_layers = self._get_clones(pocket_enc_layer, n_layers - 1)
        pocket_layers.append(pocket_dec_layer)
        self.pocket_layers = nn.ModuleList(pocket_layers)
        self.pocket_equi_ff = _EquivariantMLP(d_equi, d_pocket_inv)

    @property
    def hparams(self):
        return {
            "d_equi": self.d_equi,
            "d_inv": self.d_inv,
            "d_message": self.d_message,
            "n_layers": self.n_layers,
            "n_attn_heads": self.n_attn_heads,
            "d_message_ff": self.d_message_ff,
            "d_edge": self.d_edge,
            "d_pocket_inv": self.d_pocket_inv,
            "eps": self.eps,
        }

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]

    def forward(
        self,
        lig_equis: torch.Tensor,  # [B, N, 3, d_equi]
        lig_invs: torch.Tensor,  # [B, N, d_inv]
        lig_edges: torch.Tensor,  # [B, N, N, d_edge]
        lig_adj: torch.Tensor,  # [B, N, N]
        lig_mask: torch.Tensor,  # [B, N]
        pocket_equis: torch.Tensor,  # [B, N_p, 3, d_equi]
        pocket_invs: torch.Tensor,  # [B, N_p, d_inv]
        pocket_edges: torch.Tensor,  # [B, N_p, N_p, d_edge]
        pocket_adj: torch.Tensor,  # [B, N_p, N_p]
        pocket_mask: torch.Tensor,  # [B, N_p]
        interaction_feats: torch.Tensor = None,  # optional, shape [B, N, N_p, d_edge]
    ):
        # Build cross-adjacency matrix.
        cond_adj_lig = (
            lig_mask.float().unsqueeze(2) * pocket_mask.float().unsqueeze(1)
        ).long()
        cond_adj_pock = (
            pocket_mask.float().unsqueeze(2) * lig_mask.float().unsqueeze(1)
        ).long()

        # Ligand branch: process ligand latents conditioned on pocket latents.
        for layer in self.lig_layers:
            lig_equis, lig_invs, lig_edges_out, _ = layer(
                lig_equis,
                lig_invs,
                lig_edges,
                lig_adj,
                lig_mask,
                cond_equis=pocket_equis,
                cond_invs=pocket_invs,
                cond_edges=(
                    interaction_feats if interaction_feats is not None else None
                ),
                cond_adj_matrix=cond_adj_lig,
            )

        # Pocket branch: process pocket latents conditioned on ligand.
        interaction_feats_T = (
            interaction_feats.transpose(1, 2) if interaction_feats is not None else None
        )
        for layer in self.pocket_layers:
            pocket_equis, pocket_invs, _, _ = layer(
                pocket_equis,
                pocket_invs,
                pocket_edges,
                pocket_adj,
                pocket_mask,
                cond_equis=lig_equis,
                cond_invs=lig_invs,
                cond_edges=interaction_feats_T,
                cond_adj_matrix=cond_adj_pock,
            )
            pocket_equis = self.pocket_equi_ff(pocket_equis, pocket_invs)

        # Return refined latent representations.
        return (lig_equis, lig_invs, lig_edges_out, lig_mask), (
            pocket_equis,
            pocket_mask,
        )


class ComplexGenerator(nn.Module):
    """
    ComplexGenerator takes in raw inputs for both the ligand and the pocket,
    encodes them via Encoder, refines their latent representations via ComplexDecoder,
    and then applies output modules to produce final outputs:
      - Ligand: coordinates, atom type logits, bond logits, and charge logits.
      - Pocket: 3D coordinates.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_types,
        n_bond_types,
        n_charge_types,
        emb_size,
        d_pocket_inv,
        n_pocket_atom_names,
        n_pocket_res_types,
        predict_interactions=False,
        flow_interactions=False,
        use_lig_pocket_rbf=False,
        use_rbf=False,
        use_sphcs=False,
        n_interaction_types=None,
        n_extra_atom_feats=None,
        self_cond=False,
        max_size_lig=200,
        max_size_pocket=600,
        eps=1e-6,
    ):
        super().__init__()

        self.ligand_encoder = Encoder(
            d_equi,
            d_inv,
            d_message,
            d_message_ff,
            d_edge,
            n_atom_types=n_atom_types,
            n_bond_types=n_bond_types,
            n_charge_types=None,
            emb_size=emb_size,
            n_extra_atom_feats=n_extra_atom_feats,
            self_cond=self_cond,
            self_cond_inv=self_cond,
            max_size=max_size_lig,
            eps=eps,
        )
        self.pocket_encoder = Encoder(
            d_equi,
            d_pocket_inv,
            d_message,
            d_message_ff,
            d_edge,
            n_bond_types=n_bond_types,
            n_charge_types=n_charge_types,
            n_atom_types=n_pocket_atom_names,
            n_res_types=n_pocket_res_types,
            emb_size=emb_size,
            n_extra_atom_feats=n_extra_atom_feats,
            self_cond=self_cond,
            self_cond_inv=False,
            max_size=max_size_pocket,
            eps=eps,
        )
        # Create the decoder for refined latent feature exchange.
        self.complex_decoder = ComplexDecoder(
            d_equi,
            d_inv,
            d_message,
            n_layers,
            n_attn_heads,
            d_message_ff,
            d_edge,
            d_pocket_inv,
            eps=eps,
        )
        d_cond_edge_in = (
            d_edge + 1
            if n_interaction_types is not None
            else d_edge if use_lig_pocket_rbf else None
        )
        self.radial_basis_embed = RadialBasisEmbedding(
            d_edge=d_cond_edge_in, num_rbf=32, cutoff=5.0
        )
        # Output modules for ligand.
        self.final_coord_norm = _CoordNorm(d_equi, zero_com=False, eps=eps)
        self.final_inv_norm = nn.LayerNorm(d_inv)
        self.final_bond_norm = nn.LayerNorm(d_edge)
        self.coord_out_proj = nn.Linear(d_equi, 1, bias=False)
        self.atom_type_proj = nn.Linear(d_inv, n_atom_types)
        self.atom_charge_proj = nn.Linear(d_inv, n_charge_types)
        self.bond_refine = BondRefine(
            d_inv, d_message, d_edge, d_ff=d_inv, norm_feats=False
        )
        self.bond_proj = nn.Linear(d_edge, n_bond_types)

        # Output module for pocket: only coordinate output.
        self.pocket_final_coord_norm = _CoordNorm(d_equi, zero_com=False, eps=eps)
        self.pocket_coord_out_proj = nn.Linear(d_equi, 1, bias=False)

    @property
    def hparams(self):
        dec_hparams = self.complex_decoder.hparams
        ligand_hparams = {
            f"ligand-{name}": val for name, val in self.ligand_encoder.hparams.items()
        }
        pocket_hparams = {
            f"pocket-{name}": val for name, val in self.pocket_encoder.hparams.items()
        }
        hparams = {**dec_hparams, **ligand_hparams, **pocket_hparams}

        return hparams

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_mask=None,
        pocket_coords=None,
        pocket_atom_names=None,
        pocket_atom_charges=None,
        pocket_res_types=None,
        pocket_bond_types=None,
        pocket_atom_mask=None,
        pocket_equis=None,
        pocket_invs=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        cond_pocket_coords=None,
        interactions=None,
        extra_feats=None,
    ):
        """
        Generate ligand atom types, coords, charges, bonds and pocket coords

        coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
        atom_types (torch.Tensor): Atom name indices, shape [B, N]
        bond_types (torch.Tensor): Bond type indicies for each pair, shape [B, N, N]
        atom_mask (torch.Tensor): Mask for atoms, shape [B, N], 1 for real atom, 0 otherwise
        pocket_coords (torch.Tensor): Original pocket coords, shape [B, N_p, 3]
        pocket_atom_names (torch.Tensor): Atom name indices, shape [B, N_p]
        pocket_atom_charges (torch.Tensor): Atom charge indices, shape [B, N_p]
        pocket_res_types (torch.Tensor): Residue type indices for each atom, shape [B, N_p]
        pocket_bond_types (torch.Tensor): Bond type indices for each pair, shape [B, N_p, N_p]
        pocket_atom_mask (torch.Tensor): Mask for pocket atom, shape [B, N_p], 1 for real, 0 otherwise
        pocket_equis (torch.Tensor): Equivariant encoded pocket features, shape [B, N_p, d_equi]
        pocket_invs (torch.Tensor): Invariant encoded pocket features, shape [B, N_p, d_pocket_inv]
        extra_feats (torch.Tensor): Additional atom features, shape [B, N, n_extra_atom_feats]
        cond_coords (torch.Tensor): Self conditioning coords, shape [B, N, 3]
        cond_atomics (torch.Tensor): Self conditioning atom types, shape [B, N, n_atom_types]
        cond_bonds (torch.Tensor): Self conditioning bond types, shape [B, N, N, n_bond_types]
        cond_pocket_coords (torch.Tensor): Self conditioning pocket coords, shape [B, N_p, 3]
        interactions (torch.Tensor): Interaction types between pocket and ligand, shape [B, N, N_p]

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charge logits)
            All torch.Tensor, shapes:
                Coordinates: [B, N, 3],
                Type logits: [B, N, n_atom_types],
                Bond logits: [B, N, N, n_bond_types],
                Charge logits: [B, N, n_charge_types]
                Pocket coordinates: [B, N_p, 3]
        """
        # Encode ligand
        lig_equis, lig_invs, lig_edges, lig_adj, lig_mask = self.ligand_encoder(
            coords,
            atom_types,
            bond_types,
            atom_mask=atom_mask,
            extra_feats=extra_feats[0],
            cond_coords=cond_coords,
            cond_atomics=cond_atomics,
            cond_bonds=cond_bonds,
        )
        # Encode pocket
        pocket_equis, pocket_invs, pocket_edges, pocket_adj, pocket_mask = (
            self.pocket_encoder(
                pocket_coords,
                pocket_atom_names,
                pocket_bond_types,
                atom_charges=pocket_atom_charges,
                res_types=pocket_res_types,
                atom_mask=pocket_atom_mask,
                extra_feats=extra_feats[1],
                cond_coords=cond_pocket_coords,
            )
        )
        # Ligand-pocket interaction features
        interaction_feats = self.radial_basis_embed(
            coords,
            pocket_coords,
            ligand_mask=atom_mask,
            pocket_mask=pocket_atom_mask,
        )
        # Refine latents via cross-conditioning
        (lig_equis, lig_invs, lig_edges, lig_mask), (
            pocket_equis,
            pocket_mask,
        ) = self.complex_decoder(
            lig_equis,
            lig_invs,
            lig_edges,
            lig_adj,
            lig_mask,
            pocket_equis,
            pocket_invs,
            pocket_edges,
            pocket_adj,
            pocket_mask,
            interaction_feats=interaction_feats,
        )

        # Apply ligand output modules
        ## Project coords back to one equivariant feature
        lig_equis_norm = self.final_coord_norm(lig_equis, lig_mask)
        ligand_coords_out = self.coord_out_proj(lig_equis_norm).squeeze(-1)
        ligand_coords_out = ligand_coords_out * atom_mask.unsqueeze(-1) + coords

        ## Project invariant features to atom and charge logits
        invs_norm = self.final_inv_norm(lig_invs)
        atom_type_logits = self.atom_type_proj(invs_norm)
        charge_logits = self.atom_charge_proj(invs_norm)

        ## Pass bonds through refinement layer and project to logits
        edge_norm = self.final_bond_norm(lig_edges)
        refined_edges = self.bond_refine(
            ligand_coords_out, invs_norm, lig_mask, edge_norm
        )
        bond_logits = self.bond_proj(refined_edges + refined_edges.transpose(1, 2))

        # Apply pocket output modules
        pocket_equis_norm = self.pocket_final_coord_norm(pocket_equis, pocket_mask)
        pocket_coords_out = self.pocket_coord_out_proj(pocket_equis_norm).squeeze(-1)
        pocket_coords_out = (
            pocket_coords_out * pocket_mask.unsqueeze(-1) + pocket_coords
        )

        return {
            "coords": ligand_coords_out,
            "atomics": atom_type_logits,
            "bonds": bond_logits,
            "charges": charge_logits,
            "mask": lig_mask,
            "pocket_coords": pocket_coords_out,
            "pocket_mask": pocket_mask,
        }

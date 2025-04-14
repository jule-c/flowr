import torch
import torch.nn.functional as F

import flowr.util.functional as smolF
from flowr.models.semla import CoordNorm, MolecularGenerator


def adj_to_attn_mask(adj_matrix, pos_inf=False):
    """Assumes adj_matrix is only 0s and 1s"""

    inf = float("inf") if pos_inf else float("-inf")
    attn_mask = torch.zeros_like(adj_matrix.float())
    attn_mask[adj_matrix == 0] = inf

    # Ensure nodes with no connections (fake nodes) don't have all -inf in the attn mask
    # Otherwise we would have problems when softmaxing
    n_nodes = adj_matrix.sum(dim=-1)
    attn_mask[n_nodes == 0] = 0.0

    return attn_mask


class GatedEquiUpdate(torch.nn.Module):
    def __init__(self, d_model, n_equi_feats, eps=1e-5):
        super().__init__()

        self.d_model = d_model
        self.n_equi_feats = n_equi_feats
        self.eps = eps

        self.equi_proj = torch.nn.Linear(n_equi_feats, 2 * n_equi_feats, bias=False)
        self.inv_proj = torch.nn.Linear(d_model + n_equi_feats, d_model + n_equi_feats)

    def forward(self, inv_feats, equi_feats):
        """Pass data through one layer of the model

        Args:
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            equi_feats (torch.Tensor): Equivariant atom features, shape [batch_size, n_atoms, n_equi_feats, 3]

        Returns:
            (atom feats, equi feats)
            All torch.Tensor, shapes:
                Atom feats [batch_size, n_atoms, d_model]
                Equi feats [batch_size, n_atoms, n_equi_feats, 3]
        """

        equi_feats_proj = self.equi_proj(equi_feats.transpose(2, 3)).transpose(2, 3)
        equi_feats_out = equi_feats_proj[:, :, : self.n_equi_feats, :]
        norms = (
            torch.linalg.vector_norm(
                equi_feats_proj[:, :, self.n_equi_feats :, :], dim=-1
            )
            + self.eps
        )

        inv_feats_cat = torch.cat((inv_feats, norms), dim=-1)
        inv_feats_proj = self.inv_proj(inv_feats_cat)
        inv_feats_out = inv_feats_proj[:, :, : self.d_model]
        inv_gate_feats = inv_feats_proj[:, :, self.d_model :]

        equi_feats_out = equi_feats_out * inv_gate_feats.unsqueeze(-1)

        return inv_feats_out, equi_feats_out


class EqgatLayer(torch.nn.Module):
    def __init__(self, d_model, n_equi_feats, d_edge, eps=1e-5):
        super().__init__()

        self.d_model = d_model
        self.n_equi_feats = n_equi_feats
        self.d_edge = d_edge
        self.eps = eps

        pairwise_input_feats = (2 * d_model) + d_edge + 4
        pairwise_output_feats = (2 * n_equi_feats) + d_model + d_edge + 1

        self.pairwise_mlp = torch.nn.Sequential(
            torch.nn.Linear(pairwise_input_feats, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, pairwise_output_feats),
        )

        self.node_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.equi_proj = torch.nn.Linear(n_equi_feats, n_equi_feats, bias=False)
        self.edge_in_proj = torch.nn.Linear(d_edge, d_edge, bias=False)
        self.edge_out_proj = torch.nn.Linear(d_edge, d_edge, bias=False)

        self.inv_norm = torch.nn.LayerNorm(d_model)
        self.coord_norm = CoordNorm(1)
        self.equi_norm = CoordNorm(n_equi_feats)

        self.gated_update = GatedEquiUpdate(d_model, n_equi_feats, eps=eps)

    def forward(self, coords, inv_feats, equi_feats, adj_matrix, atom_mask, edge_feats):
        """Pass data through one layer of the model

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            equi_feats (torch.Tensor): Equivariant atom features, shape [batch_size, n_atoms, n_equi_feats, 3]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_atoms, n_atoms], 1 for connected
            atom_mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms
            edge_feats (torch.Tensor): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            (coords, atom feats, equi feats, edge feats)
            All torch.Tensor, shapes:
                Coordinates [batch_size, n_atoms, 3],
                Atom feats [batch_size, n_atoms, d_model]
                Equi feats [batch_size, n_atoms, n_equi_feats, 3]
                Edge feats [batch_size, n_atoms, n_atoms, d_edge]
        """

        batch_size, n_nodes, _ = tuple(coords.shape)

        coord_norms = torch.linalg.vector_norm(coords, dim=-1).unsqueeze(-1)
        atom_feats = torch.cat((inv_feats, coord_norms), dim=-1)

        node_i = atom_feats.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        node_j = atom_feats.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)

        distances = smolF.calc_distances(coords).unsqueeze(-1)
        dotprods = torch.bmm(coords, coords.transpose(1, 2)).unsqueeze(-1)

        proj_edge_feats = self.edge_in_proj(edge_feats)
        atom_in_feats = torch.cat(
            (node_i, node_j, proj_edge_feats, distances, dotprods), dim=3
        )

        c_start = self.n_equi_feats + self.d_model
        d_start = self.d_model + (2 * self.n_equi_feats)
        d_end = self.d_model + (2 * self.n_equi_feats) + self.d_edge

        pairwise_mlp_out = self.pairwise_mlp(atom_in_feats)
        a = pairwise_mlp_out[:, :, :, : self.d_model]
        b = pairwise_mlp_out[:, :, :, self.d_model : c_start]
        c = pairwise_mlp_out[:, :, :, c_start:d_start]
        d = pairwise_mlp_out[:, :, :, d_start:d_end]
        s = pairwise_mlp_out[:, :, :, d_end : d_end + 1]

        # Compute attention weights
        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = a + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        # Apply attention to projected node features
        proj_atom_feats = self.node_proj(inv_feats)
        scaled_attentions = proj_atom_feats.unsqueeze(2) * attentions
        node_feats_out = inv_feats + scaled_attentions.sum(dim=2)

        # Apply edge update
        edge_out = self.edge_out_proj(F.silu(edge_feats + d))

        # Apply equi feat udpate
        vector_dist = coords.unsqueeze(2) - coords.unsqueeze(1)
        vector_dist_norm = torch.linalg.vector_norm(vector_dist, dim=-1)
        x_ij = vector_dist / (vector_dist_norm.unsqueeze(-1) + self.eps)

        n_atoms = atom_mask.sum(dim=-1) + self.eps

        # x_b_outer shape [B, N, N, F, 3]
        # c_outer shape [B, N, N, F, 3]
        # equi_feats shape [B, N, F, 3]
        x_b_outer = x_ij.unsqueeze(-2) * b.unsqueeze(-1)
        c_outer = torch.ones((1, 1, 1, 1, 3), device=c.device) * c.unsqueeze(-1)
        equi_feats_proj = self.equi_proj(equi_feats.transpose(2, 3)).transpose(2, 3)
        equi_feats_mult = equi_feats_proj.unsqueeze(2) * c_outer
        equi_feats_update = (x_b_outer + equi_feats_mult).sum(dim=2)
        equi_feats_out = equi_feats + (equi_feats_update / n_atoms.view(-1, 1, 1, 1))

        # Apply coord update
        coord_pairwise_updates = s * x_ij
        coords_out = coords + (
            coord_pairwise_updates.sum(dim=2) / n_atoms.view(-1, 1, 1)
        )

        node_feats_out = self.inv_norm(node_feats_out)
        coords_out = self.coord_norm(
            coords_out.unsqueeze(1), atom_mask.unsqueeze(1)
        ).squeeze(1)

        equi_atom_mask = atom_mask.unsqueeze(1).repeat(1, self.n_equi_feats, 1)
        equi_feats_out = self.equi_norm(
            equi_feats_out.transpose(1, 2), equi_atom_mask
        ).transpose(1, 2)

        inv_update, equi_update = self.gated_update(node_feats_out, equi_feats_out)
        node_feats_out = (node_feats_out + inv_update) * atom_mask.unsqueeze(-1)
        equi_feats_out = equi_feats_out + equi_update

        return coords_out, node_feats_out, equi_feats_out, edge_out


class EqgatPredictionHead(torch.nn.Module):
    def __init__(
        self, d_model, n_equi_feats, d_edge, vocab_size, n_edge_types, n_charges
    ):
        super().__init__()

        self.d_model = d_model
        self.n_equi_feats = n_equi_feats

        self.inv_proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model), torch.nn.SiLU()
        )
        self.edge_feat_proj = torch.nn.Linear(d_edge, d_edge)
        self.equi_proj = torch.nn.Linear(n_equi_feats, 1, bias=False)
        self.atom_proj = torch.nn.Linear(d_model, vocab_size)
        self.charge_proj = torch.nn.Linear(d_model, n_charges)

        edge_in_feats = (d_model * 2) + d_edge + 1
        self.bond_proj = torch.nn.Sequential(
            torch.nn.Linear(edge_in_feats, d_edge),
            torch.nn.SiLU(),
            torch.nn.Linear(d_edge, n_edge_types),
        )

    def forward(self, coords, inv_feats, equi_feats, adj_matrix, atom_mask, edge_feats):
        """Predict final atom types, charges, bonds and coordinates

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            equi_feats (torch.Tensor): Equivariant atom features, shape [batch_size, n_atoms, n_equi_feats, 3]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_atoms, n_atoms], 1 for connected
            atom_mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms
            edge_feats (torch.Tensor): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            (coords, atom logits, charge logits, bond logits)
            All torch.Tensor, shapes:
                Coordinates [batch_size, n_atoms, 3],
                Atom logits [batch_size, n_atoms, vocab_size]
                Bond logits [batch_size, n_atoms, n_atoms, n_bond_types]
                Charge logits [batch_size, n_atoms, n_charges]
        """

        batch_size, n_nodes, _ = tuple(coords.shape)

        equi_feats_proj = self.equi_proj(equi_feats.transpose(2, 3)).squeeze(-1)
        coords_out = coords + equi_feats_proj

        edge_feats = edge_feats * adj_matrix.unsqueeze(-1)
        edge_feats_in = edge_feats + edge_feats.transpose(1, 2)
        edge_feats_proj = self.edge_feat_proj(edge_feats_in)

        node_feats = self.inv_proj(inv_feats)
        node_feats_start = node_feats.unsqueeze(2).expand(
            batch_size, n_nodes, n_nodes, -1
        )
        node_feats_end = node_feats.unsqueeze(1).expand(
            batch_size, n_nodes, n_nodes, -1
        )
        node_pairs = torch.cat((node_feats_start, node_feats_end), dim=-1)

        distances = smolF.calc_distances(coords_out).unsqueeze(-1)
        pairwise_feats = torch.cat((node_pairs, edge_feats_proj, distances), dim=-1)
        bond_logits = self.bond_proj(pairwise_feats)

        atom_logits = self.atom_proj(node_feats)
        charge_logits = self.charge_proj(node_feats)

        return coords_out, atom_logits, bond_logits, charge_logits


class EqgatDynamics(torch.nn.Module):
    def __init__(self, d_model, n_layers, n_equi_feats, d_edge, eps=1e-5):
        super().__init__()

        layers = [
            EqgatLayer(d_model, n_equi_feats, d_edge, eps=eps) for _ in range(n_layers)
        ]
        self.layers = torch.nn.ParameterList(layers)

    def forward(self, coords, inv_feats, equi_feats, adj_matrix, atom_mask, edge_feats):
        """Generate molecular coordinates and atom features

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            equi_feats (torch.Tensor): Equivariant atom features, shape [batch_size, n_atoms, n_equi_feats, 3]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_atoms, n_atoms], 1 for connected
            atom_mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms
            edge_feats (torch.Tensor): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            (coords, atom feats, edge feats)
            All torch.Tensor, shapes:
                Coordinates [batch_size, n_atoms, 3],
                Atom feats [batch_size, n_atoms, d_model]
                Equi feats [batch_size, n_atoms, n_equi_feats, 3]
                Edge feats [batch_size, n_atoms, n_atoms, d_edge]
        """

        for layer in self.layers:
            coords, inv_feats, equi_feats, edge_feats = layer(
                coords, inv_feats, equi_feats, adj_matrix, atom_mask, edge_feats
            )

        return coords, inv_feats, equi_feats, edge_feats


class EqgatGenerator(MolecularGenerator):
    def __init__(
        self,
        d_model,
        n_layers,
        n_equi_feats,
        vocab_size,
        n_atom_feats,
        d_edge,
        n_edge_types,
    ):

        hparams = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_equi_feats": n_equi_feats,
            "vocab_size": vocab_size,
            "n_atom_feats": n_atom_feats,
            "d_edge": d_edge,
            "n_edge_types": n_edge_types,
        }

        super().__init__(**hparams)

        self.d_model = d_model
        self.n_equi_feats = n_equi_feats

        n_charges = 7

        self.feat_proj = torch.nn.Linear(n_atom_feats, d_model)
        self.edge_in_proj = torch.nn.Sequential(
            torch.nn.Linear(n_edge_types, d_edge),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_edge, d_edge),
        )

        self.dynamics = EqgatDynamics(d_model, n_layers, n_equi_feats, d_edge)
        self.pred_head = EqgatPredictionHead(
            d_model, n_equi_feats, d_edge, vocab_size, n_edge_types, n_charges
        )

    def forward(
        self,
        coords,
        inv_feats,
        edge_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        atom_mask=None,
    ):
        """Predict molecular coordinates and atom types

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, n_feats]
            edge_feats (torch.Tensor): In edge features, shape [batch_size, n_atoms, n_atoms, n_edge_types]
            atom_mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charges)
            All torch.Tensor, shapes:
                Coordinates: [batch_size, n_atoms, 3]
                Type logits: [batch_size, n_atoms, vocab_size],
                Bond logits: [batch_size, n_atoms, n_atoms, n_edge_types]
                Charge logits: [batch_size, n_atoms, 7]
        """

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = smolF.edges_from_nodes(coords, node_mask=atom_mask)

        inv_feats_proj = self.feat_proj(inv_feats)
        edge_feats_proj = self.edge_in_proj(edge_feats.float())

        equi_feats = torch.zeros_like(coords.unsqueeze(2)).repeat(
            1, 1, self.n_equi_feats, 1
        )

        out = self.dynamics(
            coords, inv_feats_proj, equi_feats, adj_matrix, atom_mask, edge_feats_proj
        )
        coords, atom_feats, equi_feats, edge_feats = out

        pred = self.pred_head(
            coords, atom_feats, equi_feats, adj_matrix, atom_mask, edge_feats
        )
        return pred

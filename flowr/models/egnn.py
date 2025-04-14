"""
Original EGNN implementation. Keep this mostly seperate from our implementations so that this remains consistent with
with the original version of the model.
"""

import torch

import flowr.util.functional as smolF
from flowr.models.semla import MolecularGenerator


class VanillaEgnnLayer(torch.nn.Module):
    def __init__(self, d_model, in_edge_feats, d_pred_edge=None, norm=False, eps=1e-5):
        super().__init__()

        self.d_model = d_model
        self.in_edge_feats = in_edge_feats
        self.d_pred_edge = d_pred_edge
        self.norm = norm
        self.eps = eps

        input_feats = (d_model * 2) + in_edge_feats + 1
        phi_e_out = d_model if d_pred_edge is None else d_model + d_pred_edge

        self.phi_e = torch.nn.Sequential(
            torch.nn.Linear(input_feats, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, phi_e_out),
            torch.nn.SiLU(),
        )

        self.phi_att = torch.nn.Sequential(
            torch.nn.Linear(d_model, 1), torch.nn.Sigmoid()
        )

        self.phi_h = torch.nn.Sequential(
            torch.nn.Linear(d_model * 2, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model),
        )

        self.phi_x = torch.nn.Sequential(
            torch.nn.Linear(input_feats, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, 1),
        )

        if norm:
            self.norm_layer = torch.nn.LayerNorm(d_model)

    def forward(self, coords, inv_feats, adj_matrix, atom_mask, edge_feats):
        """Pass data through the layer

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_atoms, n_atoms], 1 for connected
            atom_mask (torch.Tensor, Optional): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms
            edge_feats (torch.Tensor, Optional): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            ((torch.Tensor, torch.Tensor)): A tuple of the new node coordinates and the new node features
        """

        atom_mask = atom_mask.unsqueeze(2)

        # Add distances to edge features, then compute messages
        sqrd_dists = smolF.calc_distances(coords, sqrd=True).unsqueeze(-1)
        edge_feats = torch.cat((edge_feats, sqrd_dists), dim=-1)

        edge_messages = self._compute_edge_messages(inv_feats, edge_feats)
        if self.d_pred_edge is not None:
            edge_pred = edge_messages[:, :, :, self.d_model :]
            edge_messages = edge_messages[:, :, :, : self.d_model]

        attentions = self.phi_att(edge_messages)
        edge_messages = attentions * edge_messages
        edge_messages = edge_messages * adj_matrix.unsqueeze(-1)

        # Compute new node features
        node_messages = edge_messages.sum(dim=2)
        in_feats = torch.cat((inv_feats, node_messages), dim=-1)
        out_node_feats = self.phi_h(in_feats)

        # Compute new node coords
        coord_updates = self._compute_coord_updates(
            coords, inv_feats, edge_feats, adj_matrix, atom_mask
        )
        out_coords = coords + coord_updates

        out_node_feats = out_node_feats * atom_mask
        out_coords = out_coords * atom_mask

        if self.norm:
            out_node_feats = self.norm_layer(out_node_feats)

        if self.d_pred_edge is not None:
            return out_coords, out_node_feats, edge_pred

        return out_coords, out_node_feats

    def _compute_edge_messages(self, node_feats, edge_feats):
        """Computes messages with attention applied, for all edges

        Args:
            node_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            edge_feats (torch.Tensor, Optional): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            (torch.Tensor) Message tensor, shape [batch_size, n_nodes, n_nodes, d_model]
        """

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        node_i = node_feats.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        node_j = node_feats.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)

        in_feats = torch.cat((node_i, node_j, edge_feats), dim=3)
        messages = self.phi_e(in_feats)

        return messages

    def _compute_coord_updates(
        self, coords, node_feats, edge_feats, adj_matrix, atom_mask
    ):
        """Computes coordinate updates by summing over edges with scalar attention

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            node_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            edge_feats (torch.Tensor, Optional): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_atoms, n_atoms], 1 for connected
            atom_mask (torch.Tensor, Optional): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms

        Returns:
            (torch.Tensor) Message tensor, shape [batch_size, num_nodes, 3]
        """

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        node_i = node_feats.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        node_j = node_feats.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)

        in_feats = torch.cat((node_i, node_j, edge_feats), dim=3)
        edge_attn = self.phi_x(in_feats)

        # Compute a vector for each edge using the coord diff, edge attention score and a normaliser
        # coord_diffs = coords[batch_index, edge_is, :] - coords[batch_index, edge_js, :]
        coord_diffs = coords.unsqueeze(-2) - coords.unsqueeze(-3)
        normalisers = (
            torch.sqrt(torch.sum(coord_diffs * coord_diffs, dim=-1) + self.eps) + 1
        )
        weighted_edges = (coord_diffs * edge_attn) / normalisers.unsqueeze(-1)
        weighted_edges = weighted_edges * adj_matrix.unsqueeze(-1)

        # Sum over all of a node's edges to get a coordinate update for that node
        coord_updates = weighted_edges.sum(dim=2)

        # Take average over number of edges to reduce size of coord updates
        # *** Not part of vanilla ***
        num_nodes = atom_mask.sum(dim=1) + 1
        coord_updates = coord_updates / num_nodes.view(-1, 1, 1)

        return coord_updates


class VanillaEgnnDynamics(torch.nn.Module):
    def __init__(self, d_model, n_layers, d_edge):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.d_edge = d_edge

        in_edge_feats = 1
        layers = [
            VanillaEgnnLayer(d_model, in_edge_feats, norm=True)
            for _ in range(n_layers - 2)
        ]

        self.layers = torch.nn.ParameterList(layers)
        self.enc_layer = VanillaEgnnLayer(d_model, in_edge_feats + d_edge, norm=True)
        self.dec_layer = VanillaEgnnLayer(
            d_model, in_edge_feats, d_pred_edge=d_edge, norm=True
        )

    def forward(self, coords, inv_feats, adj_matrix, atom_mask=None, edge_feats=None):
        """Generate molecular coordinates and atom features

        Args:
            coords (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_atoms, n_atoms], 1 for connected
            atom_mask (torch.Tensor, Optional): Mask for fake atoms, shape [batch_size, n_atoms], 1 for real atoms
            edge_feats (torch.Tensor, Optional): In edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            (coords, atom feats, edge feats)
            All torch.Tensor, shapes:
                Coordinates [batch_size, n_atoms, 3],
                Atom feats [batch_size, n_atoms, d_model]
                Edge feats [batch_size, n_atoms, n_atoms, d_edge]
        """

        # Compute initial distance between edges as our only edge feature
        dist_feats = smolF.calc_distances(coords, sqrd=True).unsqueeze(-1)

        edge_feats = torch.cat((dist_feats, edge_feats), dim=-1)
        edge_feats = edge_feats * adj_matrix.unsqueeze(-1)

        coords, inv_feats = self.enc_layer(
            coords, inv_feats, adj_matrix, atom_mask, edge_feats
        )

        # Update coords and node feats using the model EGNN layers
        # Remove CoM from predicted coords before passing to next layer
        for layer in self.layers:
            coords = smolF.zero_com(coords, node_mask=atom_mask)
            coords, inv_feats = layer(
                coords, inv_feats, adj_matrix, atom_mask, dist_feats
            )

        coords, inv_feats, pred_edges = self.dec_layer(
            coords, inv_feats, adj_matrix, atom_mask, dist_feats
        )
        coords = smolF.zero_com(coords, node_mask=atom_mask)

        return coords, inv_feats, pred_edges


class VanillaEgnnGenerator(MolecularGenerator):
    def __init__(
        self,
        d_model,
        n_layers,
        vocab_size,
        n_atom_feats,
        d_edge,
        n_edge_types,
        self_cond=False,
    ):
        if self_cond:
            raise NotImplementedError("Self conditioning not implemented for EGNN")

        hparams = {
            "d_model": d_model,
            "n_layers": n_layers,
            "vocab_size": vocab_size,
            "n_atom_feats": n_atom_feats,
            "d_edge": d_edge,
            "n_edge_types": n_edge_types,
            "self_cond": self_cond,
        }

        super().__init__(**hparams)

        self.feat_proj = torch.nn.Linear(n_atom_feats, d_model)
        self.dynamics = VanillaEgnnDynamics(d_model, n_layers, d_edge)

        self.edge_in_proj = torch.nn.Sequential(
            torch.nn.Linear(n_edge_types, d_edge),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_edge, d_edge),
        )
        self.edge_out_proj = torch.nn.Sequential(
            torch.nn.Linear(d_edge, d_edge),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_edge, n_edge_types),
        )

        self.classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, vocab_size),
        )
        self.charge_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, 7),
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
            coords (torch.Tensor): Input coordinates, shape [batch_size, num_atoms, 3]
            inv_feats (torch.Tensor): Invariant atom features, shape [batch_size, num_atoms, num_feats]
            atom_mask (torch.Tensor, Optional): Mask for real and dummy atoms, shape [batch_size, num_atoms],
                    1 for real atom 0 otherwise

        Returns:
            (predicted coordinates, atom type logits)
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """

        if edge_feats is None:
            raise ValueError("edge_feats must be provided.")

        atom_mask = torch.ones_like(coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = smolF.edges_from_nodes(coords, node_mask=atom_mask)

        atom_feats = self.feat_proj(inv_feats)
        edge_feats = self.edge_in_proj(edge_feats.float())

        pred_coords, pred_feats, pred_bonds = self.dynamics(
            coords, atom_feats, adj_matrix, atom_mask=atom_mask, edge_feats=edge_feats
        )

        type_logits = self.classifier_head(pred_feats)
        charge_logits = self.charge_head(pred_feats)

        pred_edges = pred_bonds + pred_bonds.transpose(1, 2)
        edge_logits = self.edge_out_proj(pred_edges)

        return pred_coords, type_logits, edge_logits, charge_logits

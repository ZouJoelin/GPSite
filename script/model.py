### checked!
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_scatter import scatter_mean
import torch_geometric
from torch_geometric.nn import radius_graph, TransformerConv


##############  DataLoader  ##############
### checked!
class ProteinGraphDataset(data.Dataset):
    """ build ProteinGraphDataset for ID_list in outpath.

    e.g.
    Data(edge_index=[2, 9445],  # edge exists if two AAs'Ca within raidus.
         name='A0A009IHW8',     #
         X=[269, 5, 3],         # coords of AAs.
         node_feat=[269, 1033]) # feature of each AA(ProtTrans + DSSP)
    
    """
    def __init__(self, ID_list, outpath, radius=15):
        super(ProteinGraphDataset, self).__init__()
        self.IDs = ID_list
        self.path = outpath
        self.radius = radius

    def __len__(self): return len(self.IDs)

    def __getitem__(self, idx): return self._featurize_graph(idx)

    def _featurize_graph(self, idx):
        name = self.IDs[idx]
        with torch.no_grad():
            X = torch.load(self.path + "pdb/" + name + ".pt")  # coord: shape(AA_len, 5, 3)
            prottrans_feat = torch.load(self.path + "ProtTrans/" + name + ".pt")  # ProtTrans: shape(AA_len, 1024)
            dssp_feat = torch.load(self.path + 'DSSP/' + name + ".pt")  # DSSP: shape(AA_len, 9)
            pre_computed_node_feat = torch.cat([prottrans_feat, dssp_feat], dim=-1)  # ProtTrans+DSSP: shape(AA_len, 1024+9)

            X_ca = X[:, 1]  # coord of C-alpha
            edge_index = radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors = 1000, num_workers = 8)
            # link two AAs if their distance less than self.radius.
            # edge_index: shape(2, edge_num)

        graph_data = torch_geometric.data.Data(name=name, X=X, node_feat=pre_computed_node_feat, edge_index=edge_index)
        return graph_data
    

##############  Model  ##############
### checked!
class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


### checked!
class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        temp = c_V[batch_id]
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


### checked!
class GNNLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E


### checked!
class Graph_encoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))

    def forward(self, h_V, edge_index, h_E, batch_id):
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        return h_V


### checked!
class GPSite(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, augment_eps, dropout, task_list):
        super(GPSite, self).__init__()
        self.augment_eps = augment_eps
        self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, num_layers=num_layers, drop_rate=dropout)
        self.task_list = task_list
        for task in self.task_list:
            self.add_module("FC_{}1".format(task), nn.Linear(hidden_dim, hidden_dim, bias=True))
            self.add_module("FC_{}2".format(task), nn.Linear(hidden_dim, 1, bias=True))

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, h_V, edge_index, batch_id):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)

        h_V_geo, h_E = get_geo_feat(X, edge_index)
        h_V = torch.cat([h_V, h_V_geo], dim=-1)

        h_V = self.Graph_encoder(h_V, edge_index, h_E, batch_id) # [num_residue, hidden_dim]

        output = []
        for task in self.task_list:
            emb = F.elu(self._modules["FC_{}1".format(task)](h_V))
            emb = self._modules["FC_{}2".format(task)](emb)
            output.append(emb)

        output = torch.cat(output, dim=1)
        return output


##############  Geometric Featurizer  ##############
### neglected!
def get_geo_feat(X, edge_index):
    pos_embeddings = _positional_embeddings(edge_index)
    node_angles = _get_angle(X)
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

    geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

    return geo_node_feat, geo_edge_feat


def _positional_embeddings(edge_index, num_embeddings=16):
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE

def _get_angle(X, eps=1e-7):
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles # dim = 12

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _get_distance(X, edge_index):
    atom_N = X[:,0]  # [L, 3]
    atom_Ca = X[:,1]
    atom_C = X[:,2]
    atom_O = X[:,3]
    atom_R = X[:,4]

    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1) # dim = [N, 10 * 16]

    atom_list = ["N", "Ca", "C", "O", "R"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1) # dim = [E, 25 * 16]

    return node_dist, edge_dist

def _get_direction_orientation(X, edge_index): # N, CA, C, O, R
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    node_j, node_i = edge_index

    t = F.normalize(X[:, [0,2,3,4]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1) # [L, 4 * 3]

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3] # slightly improve performance
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]

    return node_direction, edge_direction, edge_orientation

def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [E,3,3]
        Q [E,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q

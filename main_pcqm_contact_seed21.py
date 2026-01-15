import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import networkx as nx
import json

from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)
from helpers import get_cayley_n, cayley_graph_size, get_cayley_graph

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

#Hyperparameters
NUM_EPOCHS = 100
LR = 0.001
BATCH_SIZE = 32
NUM_ITER = 1

#Model hyperparameters
NUM_LAYERS = 6
HIDDEN_DIM = 256
DROPOUT = 0.2

DEGREE_D = int(os.getenv('DEGREE_D', '4'))

DATASET_NAME = 'PCQM-Contact'
DATA_ROOT = os.path.join(CUR_DIR, 'datasets', 'pcqm-contact')

MAX_TRAIN_GRAPHS = None
MAX_VAL_GRAPHS = None
MAX_TEST_GRAPHS = None

INPUT_DIM =  None

HITS_K = 20
#Early stopping
ES_PATIENCE = 20
ES_MIN_DELTA = 1e-4

def _cap_list(lst, cap):
    if cap is None:
        return lst
    return lst[:max(0, int(cap))]

def get_loaders(root: str, name: str, batch_size: int = BATCH_SIZE):
    train_ds = LRGBDataset(root=root, name=name, split='train')
    val_ds = LRGBDataset(root=root, name=name, split='val')
    test_ds = LRGBDataset(root=root, name=name, split='test')

    train_list = _cap_list(list(train_ds), MAX_TRAIN_GRAPHS)
    val_list = _cap_list(list(val_ds), MAX_VAL_GRAPHS)
    test_list = _cap_list(list(test_ds), MAX_TEST_GRAPHS)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, num_workers = 4, pin_memory = True)
    val_loader = DataLoader(val_list, batch_size=batch_size, num_workers = 4, pin_memory = True)
    test_loader = DataLoader(test_list, batch_size=batch_size, num_workers = 4, pin_memory = True)
    return train_list, val_list, test_list, train_loader, val_loader, test_loader, train_ds

class PCQMExpanderTransform:
    def __init__(self, type_name: str):
        self.type = (type_name or 'base').upper()
        self.cayley_memory:dict[int, torch.Tensor] = {}
        self.cayley_node_memory:dict[int, torch.Tensor] = {}
    def _get_egp_edge_index(self, num_nodes: int) -> torch.Tensor:
        cayley_n = get_cayley_n(num_nodes)
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        cayley_graph_edge_index = self.cayley_memory[cayley_n].clone()

        if num_nodes not in self.cayley_node_memory:
            truncated_edge_index = cayley_graph_edge_index[:, (cayley_graph_edge_index[0] < num_nodes) & (cayley_graph_edge_index[1] < num_nodes)]
            self.cayley_node_memory[num_nodes] = truncated_edge_index
        return self.cayley_node_memory[num_nodes].clone()
    
    def _get_pegp_edge_index(self, num_nodes: int) -> torch.Tensor:
        cayley_n = get_cayley_n(num_nodes)
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        edge_index = self.cayley_memory[cayley_n].clone()
        num_cayley_nodes = cayley_graph_size(cayley_n)
        perm = torch.randperm(num_cayley_nodes, device=edge_index.device)
        permuted_edge_index = perm[edge_index]
        mask = (permuted_edge_index[0] < num_nodes) & (permuted_edge_index[1] < num_nodes)
        return permuted_edge_index[:, mask]

    def _get_cgp_edge_index(self, num_nodes: int) -> tuple[torch.Tensor, int]:
        cayley_n = get_cayley_n(num_nodes)
        cayley_num_nodes = cayley_graph_size(cayley_n)
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        edge_index = self.cayley_memory[cayley_n].clone()
        return edge_index, cayley_num_nodes

    def _get_p_cgp_edge_index(self, num_nodes: int) -> tuple[torch.Tensor, int]:
        cayley_n = get_cayley_n(num_nodes)
        cayley_num_nodes = cayley_graph_size(cayley_n)
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        edge_index = self.cayley_memory[cayley_n].clone()
        perm = torch.randperm(cayley_num_nodes, device=edge_index.device)
        permuted_edge_index = perm[edge_index]
        return permuted_edge_index, cayley_num_nodes

    def apply_to_data(self, data: Data):
        num_nodes = int(data.num_nodes) if getattr(data, 'num_nodes', None) is not None else int(data.x.size(0))
        t = self.type
        if t == 'EGP':
            data.expander_edge_index = self._get_egp_edge_index(num_nodes)
        elif t in ('P-EGP','PEGP','PGP'):
            # dataset provides base cayley; per-layer permutation is handled in the model
            data.expander_edge_index = self._get_egp_edge_index(num_nodes)
        elif t == 'CGP':
            if not hasattr(data, 'cgp_applied') or not bool(data.cgp_applied):
                edge_index, cayley_num_nodes = self._get_cgp_edge_index(num_nodes)
                data.expander_edge_index = edge_index
                virtual_num_nodes = cayley_num_nodes - num_nodes
                data.virtual_node_mask = torch.cat((torch.zeros(num_nodes, dtype=torch.bool), torch.ones(virtual_num_nodes, dtype=torch.bool)), dim=0)
                pad = torch.zeros((virtual_num_nodes, data.x.shape[1]), dtype=data.x.dtype)
                data.x = torch.cat((data.x, pad), dim=0)
                data.num_nodes = cayley_num_nodes
                data.cayley_num_nodes = cayley_num_nodes
                data.cgp_applied = True
        elif t in ('P-CGP','PCGP'):
            if not hasattr(data, 'cgp_applied') or not bool(data.cgp_applied):
                edge_index, cayley_num_nodes = self._get_cgp_edge_index(num_nodes)
                data.expander_edge_index = edge_index
                virtual_num_nodes = cayley_num_nodes - num_nodes
                data.virtual_node_mask = torch.cat((torch.zeros(num_nodes, dtype=torch.bool), torch.ones(virtual_num_nodes, dtype=torch.bool)), dim=0)
                pad = torch.zeros((virtual_num_nodes, data.x.shape[1]), dtype=data.x.dtype)
                data.x = torch.cat((data.x, pad), dim=0)
                data.num_nodes = cayley_num_nodes
                data.cayley_num_nodes = cayley_num_nodes
                data.cgp_applied = True
            else:
                edge_index, _ = self._get_cgp_edge_index(num_nodes)
                data.expander_edge_index = edge_index
        else:
            pass
        return data

    def apply_to_dataset(self, dataset_list: list[Data]):
        for data in dataset_list:
            self.apply_to_data(data)

class PCQMContactGNNNode(nn.Module):
    def __init__(self, transform_name: str | None, is_cgp: bool):
        super().__init__()
        self.mode = (transform_name or 'base').lower()
        self.num_layers = NUM_LAYERS
        self.drop_ratio = DROPOUT
        self.is_cgp = bool(is_cgp)
        self.degree_d = int(os.getenv('DEGREE_D', str(DEGREE_D)))
        self.pr_cache: dict[int, torch.Tensor] = {}
        self.cayley_cache: dict[int, torch.Tensor] = {}

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layers):
            input_dim = INPUT_DIM if layer == 0 else HIDDEN_DIM
            gnn_nn = nn.Sequential(
                nn.Linear(input_dim, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            )
            self.convs.append(GINConv(gnn_nn))
            self.batch_norms.append(nn.BatchNorm1d(HIDDEN_DIM))
    def begin_epoch(self):
        self.pr_cache = {}
    def _batch_counts_offsets(self, batch: torch.Tensor):
        counts = torch.bincount(batch)
        offsets = torch.empty_like(counts)
        if len(counts) > 0:
            offsets[0] = 0
            if len(counts) > 1:
                offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        else:
            offsets = torch.tensor([], device=batch.device, dtype=torch.long)
        return counts, offsets
    def _blockwise_perm(self, counts, offsets, device):
        total = int(counts.sum().item())
        perm = torch.empty(total, dtype=torch.long, device=device)
        for gid in range(len(counts)):
            n = int(counts[gid].item())
            if n == 0:
                continue
            start = int(offsets[gid].item())
            local_perm = torch.randperm(n, device=device)
            perm[start:start+n] = local_perm + start
        return perm
    def _random_regular_local_edge_index(self, n: int, d: int):
        if n <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        d = min(d, n - 1)
        if (n * d) % 2:
            if d > 1:
                d -= 1
            else:
                d = 2 if n > 2 else 1
        if d <= 0:
            return torch.empty((2, 0), dtype=torch.long)
        try:
            G = nx.random_regular_graph(d, n)
            edges = list(G.edges())
            if len(edges) == 0:
                return torch.empty((2, 0), dtype=torch.long)
            senders = []
            receivers = []
            for u, v in edges:
                senders.append(u)
                receivers.append(v)
                senders.append(v)
                receivers.append(u)
            return torch.tensor([senders, receivers], dtype=torch.long)
        except Exception:
            if n >= 2:
                senders = []
                receivers = []
                for i in range(n):
                    j = (i + 1) % n
                    senders.append(i)
                    receivers.append(j)
                    senders.append(j)
                    receivers.append(i)
                return torch.tensor([senders, receivers], dtype=torch.long)
            return torch.empty((2, 0), dtype=torch.long)
    def _random_regular_batch_edge_index(self, counts, offsets, device):
        parts = []
        for gid in range(len(counts)):
            n = int(counts[gid].item())
            if n == 0:
                continue
            base =int(offsets[gid].item())
            local_edges = self._random_regular_local_edge_index(n, self.degree_d)
            if local_edges.numel() == 0:
                continue
            parts.append((local_edges + base).to(device))
        if len(parts) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(parts, dim=1)
    def _batched_cayley_edge_index(self, batched_data: Data, counts, offsets, device):
        parts = []
        for gid in range(len(counts)):
            n = int(counts[gid].item())
            if n == 0:
                continue
            start = int(offsets[gid].item())
            if self.is_cgp and hasattr(batched_data, 'virtual_node_mask'):
                mask_block = batched_data.virtual_node_mask[start:start+n]
                n_orig = int((~mask_block).sum().item())
                cayley_n = get_cayley_n(n_orig)
                if cayley_n in self.cayley_cache:
                    local_edges = self.cayley_cache[cayley_n]
                else:
                    local_edges = get_cayley_graph(cayley_n)
                    self.cayley_cache[cayley_n] = local_edges
                if local_edges.numel() > 0:
                    max_allowed = n
                    mask = (local_edges[0] < max_allowed) & (local_edges[1] < max_allowed)
                    local_edges = local_edges[:, mask]
                parts.append((local_edges + start).to(device))
            else:
                cayley_n = get_cayley_n(n)
                if cayley_n in self.cayley_cache:
                    local_edges = self.cayley_cache[cayley_n]
                else:
                    local_edges = get_cayley_graph(cayley_n)
                    self.cayley_cache[cayley_n] = local_edges
                if local_edges.numel() > 0:
                    mask = (local_edges[0] < n) & (local_edges[1] < n)
                    local_edges = local_edges[:, mask]
                parts.append((local_edges + start).to(device))
        if len(parts) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(parts, dim=1)
    
    def _pr_base_local_graph(self, n: int):
        key = int(n)
        if key not in self.pr_cache:
            self.pr_cache[key] = self._random_regular_local_edge_index(n, self.degree_d)
        return self.pr_cache[key]
    
    def _compute_alt_edge_index(self, batched_data: Data):
        device = batched_data.edge_index.device
        batch = batched_data.batch
        counts, offsets = self._batch_counts_offsets(batch)
        mode = (self.mode or 'base').lower()
        if mode in ['egp','cgp']:
            return self._batched_cayley_edge_index(batched_data, counts, offsets, device)
        if mode in ['p-egp', 'p-cgp']:
            base_edges = self._batched_cayley_edge_index(batched_data, counts, offsets, device)
            perm = self._blockwise_perm(counts, offsets, device)
            return perm[base_edges]
        if mode == 'rand':
            return self._random_regular_batch_edge_index(counts, offsets, device)
        if mode == 'p-rand':
            parts = []
            for gid in range(len(counts)):
                n = int(counts[gid].item())
                if n == 0:
                    continue
                start = int(offsets[gid].item())
                base_local = self._pr_base_local_graph(n).to(device)
                perm_global = start + torch.randperm(n, device=device)
                parts.append(perm_global[base_local])
            if len(parts) == 0:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            return torch.cat(parts, dim=1)
        return batched_data.edge_index
    def forward(self, batched_data: Data):
        x, edge_index = batched_data.x, batched_data.edge_index
        if self.is_cgp and hasattr(batched_data, 'virtual_node_mask'):
            x0 = torch.zeros_like(x.float())
            x0[~batched_data.virtual_node_mask] = x.float()[~batched_data.virtual_node_mask]
            h_list = [x0]
        else:
            h_list = [x.float()]

        for layer in range(self.num_layers):
            base_modes = ['egp','cgp', 'p-egp','p-cgp','rand','p-rand']
            use_alt = (self.mode in base_modes and (layer % 2 == 1))
            if use_alt:
                alt_edge_index = self._compute_alt_edge_index(batched_data)
                h = self.convs[layer](h_list[layer], alt_edge_index)
            else:
                h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        return h_list[-1]
    
class PCQMContactLinkPredictor(nn.Module):
    def __init__(self, transform_name: str | None = None, is_cgp: bool = False):
        super().__init__()
        self.gnn_node = PCQMContactGNNNode(transform_name, is_cgp)
        self.link_mlp = nn.Sequential(
            nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
    def forward(self, batched_data: Data):
        h = self.gnn_node(batched_data)
        edge_label_index = getattr(batched_data, 'edge_label_index', None)
        if edge_label_index is None:
            raise RuntimeError("batched_data must have 'edge_label_index' attribute for link prediction.")
        u = edge_label_index[0]
        v = edge_label_index[1]
        pair = torch.cat([h[u], h[v]], dim=-1)
        logits = self.link_mlp(pair).squeeze(-1)
        return logits
    
def train(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn):
    model.train()
    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(DEVICE)
        y = getattr(batch, 'edge_label', None)
        if y is None:
            raise RuntimeError("Batch data must have 'edge_label' attribute for training.")
        y = y.to(DEVICE).float()
        out = model(batch)
        optimizer.zero_grad()
        loss = loss_fn(input=out, target=y)
        loss.backward()
        optimizer.step()

def eval_acc(model: nn.Module, loader: DataLoader):
    model.eval()
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            y = batch.edge_label.to(DEVICE).float()
            out = model(batch)
            pred = (torch.sigmoid(out) > 0.5).float()
            total_correct += pred.eq(y).sum().item()
            total_examples += y.numel()
    return total_correct / max(total_examples, 1)

def evaluate_link_metrics(model: nn.Module, loader: DataLoader, k: int = HITS_K):
    '''
    Compute Hits@K and MRR for link prediction task.
    Ranking is based on logits(descending).
    '''
    model.eval()
    total_pos = 0
    total_hits = 0
    total_rr = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            y = batch.edge_label.to(DEVICE).float()
            logits = model(batch)
            edge_label_index = batch.edge_label_index
            u = edge_label_index[0]
            v = edge_label_index[1]
            node_batch = batch.batch
            gid_u = node_batch[u]
            unique_gids = gid_u.unique(sorted=True)
            for gid in unique_gids.tolist():
                mask = (gid_u == gid)
                scores = logits[mask]
                labels = y[mask]
                pos_idx = (labels > 0.5).nonzero(as_tuple=False).view(-1)
                if pos_idx.numel() == 0 or scores.numel() == 0:
                    continue
                order = torch.argsort(scores, descending=True)
                inv_order = torch.empty_like(order)
                inv_order[order] = torch.arange(order.numel(), device=order.device)
                ranks = inv_order[pos_idx].float() + 1.0
                total_pos += int(pos_idx.numel())
                total_hits += int((ranks <= k).sum().item())
                total_rr += float((1.0 / ranks).sum().item())
    mrr = total_rr / max(total_pos, 1)
    hits_at_k = total_hits / max(total_pos, 1)
    return hits_at_k, mrr

def run_experiment(model: nn.Module, train_list: list[Data], val_list: list[Data], test_list: list[Data], train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                   transform_name: str | None = None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, threshold=1e-4, min_lr=1e-5, verbose=True)

    val_mrr_curve = []
    val_hits_curve = []
    test_mrr_curve = []
    test_hits_curve = []

    tname = (transform_name or 'base').lower()
    transform_obj = None
    if tname in ('cgp','p-cgp','pcgp'):
        transform_obj = PCQMExpanderTransform(tname.upper())
        transform_obj.apply_to_dataset(train_list)
        transform_obj.apply_to_dataset(val_list)
        transform_obj.apply_to_dataset(test_list)
    print('Start training...')
    best_val_mrr = -float('inf')
    best_epoch = 0
    es_counter = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch {epoch}:')
        if hasattr(model, 'gnn_node') and hasattr(model.gnn_node, 'begin_epoch'):
            model.gnn_node.begin_epoch()

        train(model, train_loader, optimizer=optimizer, loss_fn=loss_fn)
        val_hits, val_mrr = evaluate_link_metrics(model, val_loader, k=HITS_K)
        test_hits, test_mrr = evaluate_link_metrics(model, test_loader, k=HITS_K)
        scheduler.step(val_mrr)

        val_mrr_curve.append(val_mrr)
        val_hits_curve.append(val_hits)
        test_mrr_curve.append(test_mrr)
        test_hits_curve.append(test_hits)

        print(f'Val MRR: {val_mrr:.4f}, Val Hits@{HITS_K}: {val_hits:.4f} | Test MRR: {test_mrr:.4f}, Test Hits@{HITS_K}: {test_hits:.4f}\n')
        if (val_mrr - best_val_mrr) > ES_MIN_DELTA:
            best_val_mrr = val_mrr
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= ES_PATIENCE:
                print(f'Early stoppping at epoch {epoch}(no improvement for {es_counter} epochs)')
                break
    best_validation_epoch = int(np.argmax(np.array(val_mrr_curve)))

    print('Finished Training!')
    print(f'Best validation epoch: {best_validation_epoch + 1}')
    print(f'Best validation MRR: {val_mrr_curve[best_validation_epoch]:.4f}, Hits@{HITS_K}: {val_hits_curve[best_validation_epoch]:.4f}')
    print(f'Final test MRR: {test_mrr_curve[best_validation_epoch]:.4f}, Hits@{HITS_K}: {test_hits_curve[best_validation_epoch]:.4f}')

    return test_hits_curve[best_validation_epoch], test_mrr_curve[best_validation_epoch]

def main():
    global INPUT_DIM
    SEEDS = [21]
    train_list, val_list, test_list, base_train_loader, base_val_loader, base_test_loader, train_ds = get_loaders(root=DATA_ROOT, name=DATASET_NAME, batch_size=BATCH_SIZE)
    INPUT_DIM = int(train_ds.num_node_features)

    print(f'Dataset: {DATASET_NAME}')
    print(f'INPUT_DIM = {INPUT_DIM}')
    print(f'Caps: train={MAX_TRAIN_GRAPHS}, val={MAX_VAL_GRAPHS}, test={MAX_TEST_GRAPHS}')

    results = {
        'base': [],
        'egp' : [],
        'p_egp': [],
        'cgp': [],
        'p_cgp': [],
        'rand': [],
        'p_rand': []
    }

    transforms = [
        ('base', 'base', False),
        ('egp', 'EGP', False),
        ('p_egp','P-EGP', False),
        ('rand', 'rand', False),
        ('p_rand','p-rand', False),
        ('cgp','CGP', True),
        ('p_cgp','P-CGP', True)
    ]

    for key, tname, is_cgp in transforms:
        print(f'Experiments for {key} ({tname})')
        hits_list = []
        mrr_list = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_loader = DataLoader(train_list, batch_size=BATCH_SIZE, shuffle=True, generator = torch.Generator().manual_seed(seed), num_workers = 4, pin_memory = True)
            val_loader = base_val_loader
            test_loader = base_test_loader

            model = PCQMContactLinkPredictor(transform_name=tname, is_cgp=is_cgp).to(DEVICE)
            hits, mrr = run_experiment(model, train_list, val_list, test_list, train_loader, val_loader, test_loader, transform_name=tname)
            hits_list.append(hits)
            mrr_list.append(mrr)
            results_file = os.path.join(RESULTS_DIR, f'pcqm_contact_seed{seed}.jsonl')
            with open(results_file, 'a') as f:
                f.write(json.dumps({
                    'dataset': 'PCQM-Contact',
                    'mode': key,
                    'seed': int(seed),
                    'k': int(HITS_K),
                    'hits_at_k': float(hits),
                    'mrr': float(mrr)
                }) + '\n')
        results[key] = list(zip(hits_list, mrr_list))

    print(f'''\n Hyper parameters for this test\n#Training parameters\nNUM_EPOCHS = {NUM_EPOCHS}\nLR={LR}\nBATCH_SIZE = {BATCH_SIZE}\nSEEDS={SEEDS}\n\n#GNN\nNUM_LAYERS={NUM_LAYERS}\nHIDDEN_DIM = {HIDDEN_DIM}\n DROPOUT = {DROPOUT}\n#Evaluation\nHITS_K = {HITS_K} (best epoch selected by validation MRR) ''')

    def mean_std(vals):
        arr = np.array(vals, dtype = np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))
    
    for key in ['base','egp','p_egp','cgp','p_cgp','rand','p_rand']:
        pairs = results.get(key, [])
        if len(pairs) == 0:
            print(f'{key}: no results')
            continue
        hits_vals = [p[0] for p in pairs]
        mrr_vals = [p[1] for p in pairs]
        hits_mean, hits_std = mean_std(hits_vals)
        mrr_mean, mrr_std = mean_std(mrr_vals)
        print(f'{key} | Hits@{HITS_K} = {hits_mean:.4f} ± {hits_std:.4f} | MRR = {mrr_mean:.4f} ± {mrr_std:.4f}')

if __name__ == '__main__':
    main()
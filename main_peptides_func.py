import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from tqdm import tqdm
import argparse
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.datasets import LRGBDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn import global_mean_pool

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from helpers import get_cayley_n, cayley_graph_size, get_cayley_graph

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

#HYPERPARAMETERS
NUM_EPOCHS = 250
LR = 0.001
BATCH_SIZE = 128
NUM_ITER = 1

#MODEL HYPERPARAMS
NUM_LAYERS = 5
HIDDEN_DIM = 64
DROPOUT = 0.0

#Scheduler hyperparams
REDUCE_FACTOR = 0.5
PATIENCE = 20
MIN_LR = 1e-5

#Early stopping hyperparams
ES_PATIENCE = 50
ES_MIN_DELTA = 1e-4

DEGREE_D =  int(os.getenv('DEGREE_D','4'))
DATASET_NAME = 'peptides-func'
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(current_dir, 'copy-peptides-func')

INPUT_DIM = None
OUTPUT_DIM = None

def get_loaders(root: str, name: str, batch_size: int = BATCH_SIZE):
    train_ds = LRGBDataset(root=root, name=name, split='train')
    val_ds = LRGBDataset(root=root, name=name, split='val')
    test_ds = LRGBDataset(root = root, name = name, split='test')

    train_list = list(train_ds)
    val_list = list(val_ds)
    test_list = list(test_ds)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    test_loader = DataLoader(test_list, batch_size=batch_size)
    return train_list, val_list, test_list, train_loader, val_loader, test_loader, train_ds

class PeptidesExpanderTransform:
    def __init__(self, type_name: str):
        self.type = (type_name or 'base').upper()
        self.cayley_memory: dict[int, torch.Tensor] = {}
    def _get_cgp_edge_index_and_size(self, num_nodes: int) -> tuple[torch.Tensor, int]:
        cayley_n = get_cayley_n(num_nodes)
        cayley_num_nodes = cayley_graph_size(cayley_n)
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        edge_index = self.cayley_memory[cayley_n].clone()
        return edge_index, cayley_num_nodes
    def apply_to_data(self, data: Data):
        num_nodes = int(data.num_nodes) if getattr(data, 'num_nodes', None) is not None else int(data.x.size(0))
        t = self.type
        if t in ('CGP','P-CGP','PCGP'):
            if not hasattr(data, 'cgp_applied') or not bool(data.cgp_applied):
                _, cayley_num_nodes = self._get_cgp_edge_index_and_size(num_nodes)
                virtual_num_nodes = cayley_num_nodes - num_nodes
                data.virtual_node_mask = torch.cat((torch.zeros(num_nodes, dtype=torch.bool), torch.ones(virtual_num_nodes, dtype=torch.bool)), dim=0)
                pad = torch.zeros((virtual_num_nodes, data.x.shape[1]), dtype=data.x.dtype)
                data.x = torch.cat((data.x, pad), dim=0)
                data.num_nodes = cayley_num_nodes
                data.cayley_num_nodes = cayley_num_nodes
                data.cgp_applied = True
        else:
            pass
        return data
    def apply_to_dataset(self, dataset_list: list[Data]):
        for data in dataset_list:
            self.apply_to_data(data)

class PeptidesGNNNode(nn.Module):
    def __init__(self, transform_name: str | None, is_cgp: bool):
        super().__init__()
        self.mode = (transform_name or 'base').lower()
        self.is_cgp = bool(is_cgp)
        self.num_layer = NUM_LAYERS
        self.drop_ratio = DROPOUT
        self.degree_d = DEGREE_D
        self.pr_cache: dict[int, torch.Tensor] = {}
        self.cayley_cache: dict[int, torch.Tensor] = {}

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            input_dim = INPUT_DIM if layer == 0 else HIDDEN_DIM
            gnn_nn = nn.Sequential(
                nn.Linear(input_dim, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
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
            offsets = torch.tensor([], device = batch.device, dtype=torch.long)
        return counts, offsets
    def _blockwise_perm(self, counts: torch.Tensor, offsets: torch.Tensor, device):
        total = int(counts.sum().item())
        perm = torch.empty(total, dtype=torch.long, device=device)
        for gid in range(len(counts)):
            n = int(counts[gid].item())
            if n == 0:
                continue
            start = int(offsets[gid].item())
            local_perm = torch.randperm(n, device=device)
            perm[start:start+n] = start + local_perm
        return perm
    def _random_regular_local_edge_index(self, n: int, d: int):
        if n <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        d = min(d, n-1)
        if (n * d) % 2 == 1:
            if d > 1:
                d -= 1
            else:
                d = 2 if n >= 3 else 1
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
                    j = (i+1)%n
                    senders.append(i)
                    receivers.append(j)
                    senders.append(j)
                    receivers.append(i)
                return torch.tensor([senders, receivers], dtype=torch.long)
            return torch.empty((2, 0), dtype=torch.long)
    def _random_regular_batch_edge_index(self, counts: torch.Tensor, offsets: torch.Tensor, device):
        parts = []
        for gid in range(len(counts)):
            n = int(counts[gid].item())
            if n == 0:
                continue
            base = int(offsets[gid].item())
            local_edges = self._random_regular_local_edge_index(n, self.degree_d)
            if local_edges.numel() == 0:
                continue
            parts.append((base + local_edges).to(device))
        if len(parts) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat(parts, dim=1)
    
    def _batched_cayley_edge_index(self,batched_data: Data, counts: torch.Tensor, offsets: torch.Tensor, device):
        parts = []
        for gid in range(len(counts)):
            n = int(counts[gid].item())
            if n == 0:
                continue
            start = int(offsets[gid].item())
            if self.is_cgp and hasattr(batched_data, 'virtual_node_mask'):
                mask_block = batched_data.virtual_node_mask[start: start+n]
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
                parts.append((start + local_edges).to(device))
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
                parts.append((start + local_edges).to(device))
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
        if mode in ['egp', 'cgp']:
            return self._batched_cayley_edge_index(batched_data, counts, offsets, device)
        if mode in ['p-egp','p-cgp']:
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
        for layer in range(self.num_layer):
            base_modes = ['egp','cgp', 'p-egp','p-cgp','rand','p-rand']
            use_alt = (self.mode in base_modes and (layer % 2 == 1))
            if use_alt:
                alt_edge_index = self._compute_alt_edge_index(batched_data)
                h = self.convs[layer](h_list[layer], alt_edge_index)
            else:
                h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        return h_list[-1]
    
class PeptidesGNN(nn.Module):
    def __init__(self, transform_name: str | None = None, is_cgp: bool = False):
        super().__init__()
        self.is_cgp = is_cgp
        self.gnn_node = PeptidesGNNNode(transform_name, is_cgp)
        self.pool = global_mean_pool
        self.graph_pred_linear = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
    def forward(self, batched_data: Data):
        h_node = self.gnn_node(batched_data)
        if self.is_cgp and hasattr(batched_data, 'virtual_node_mask'):
            h_node = h_node[~batched_data.virtual_node_mask]
            batch_indicator = batched_data.batch[~batched_data.virtual_node_mask]
        else:
            batch_indicator = batched_data.batch
        h_graph = self.pool(h_node, batch_indicator)
        return self.graph_pred_linear(h_graph)
    
def train(model: nn.Module, loader: DataLoader, optimiser: torch.optim.Optimizer, loss_fn):
    model.train()
    for _, batch in enumerate(tqdm(loader, desc='Iteration')):
        batch = batch.to(DEVICE)
        y = batch.y.to(DEVICE).float()
        out = model(batch)
        optimiser.zero_grad()
        loss = loss_fn(input = out, target = y)
        loss.backward()
        optimiser.step()

def _average_precision_binary(scores: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    Compute average precision for a single binary label vector.
    scores: probabilities in [0,1] shape [N]
    labels: 0/1 floats shape [N]
    '''
    s = scores.detach().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(np.int32)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)
    precision = tp / k
    ap = float(precision[y_sorted == 1].sum() / max(1, n_pos))
    return ap

def eval_ap(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            y = batch.y.to(DEVICE).float()
            out = model(batch)
            all_logits.append(out)
            all_labels.append(y)
    if len(all_logits) == 0:
        return 0.0
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(logits)
    num_classes = int(labels.size(-1))
    ap_per_class = []
    for i in range(num_classes):
        ap_i = _average_precision_binary(probs[:, i], labels[:, i])
        ap_per_class.append(ap_i)
    return float(np.mean(ap_per_class))

def run_experiment(model: nn.Module,
                   train_list: list[Data], val_list: list[Data], test_list: list[Data],
                   train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                   transform_name: str | None = None):
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=REDUCE_FACTOR, patience=PATIENCE, min_lr=MIN_LR, verbose=False)
    train_curve = []
    validation_curve = []
    test_curve = []
    best_val = -float('inf')
    best_test_at_best_val = None
    epochs_no_improve = 0

    tname = (transform_name or 'base').upper()
    if tname in ('CGP', 'P-CGP', 'PCGP'):
        transform_obj = PeptidesExpanderTransform(tname)
        transform_obj.apply_to_dataset(train_list)
        transform_obj.apply_to_dataset(val_list)
        transform_obj.apply_to_dataset(test_list)
    print('Start training')
    for epoch in range(1, 1 + NUM_EPOCHS):
        print(f'Epoch: {epoch}')
        if hasattr(model, 'gnn_node') and hasattr(model.gnn_node, 'begin_epoch'):
            model.gnn_node.begin_epoch()
        train(model, train_loader, optimiser=optimiser, loss_fn=loss_fn)

        train_ap = eval_ap(model, train_loader)
        validation_ap = eval_ap(model, val_loader)
        scheduler.step(validation_ap)
        test_ap = eval_ap(model, test_loader)

        train_curve.append(train_ap)
        validation_curve.append(validation_ap)
        test_curve.append(test_ap)

        print(f'Train AP: {train_ap:.4f}, validation AP: {validation_ap:.4f}, test AP: {test_ap:.4f}\n')
        #Early stopping check(on validation AP)
        if validation_ap > best_val + ES_MIN_DELTA:
            best_val = validation_ap
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if ES_PATIENCE > 0 and epochs_no_improve >= ES_PATIENCE:
                print(f'Early stopping triggered at epoch {epoch}(no improvement for {epochs_no_improve} epochs)')
                break
    best_validation_epoch = int(np.argmax(np.array(validation_curve)))
    print('Finished training')
    print(f'Best validation score: {validation_curve[best_validation_epoch]:.4f}')
    print(f'Final test score: {test_curve[best_validation_epoch]:.4f}')
    return test_curve[best_validation_epoch]

def main():
    global INPUT_DIM, OUTPUT_DIM

    seed = 11
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_list, val_list, test_list, train_loader, val_loader, test_loader, train_ds = get_loaders(DATA_ROOT, DATASET_NAME, batch_size=BATCH_SIZE)
    INPUT_DIM = int(train_ds.num_node_features)
    first_y = train_ds[0].y
    OUTPUT_DIM = int(first_y.size(-1) if first_y.dim() > 1 else 2)

    print(f'Dataset: {DATASET_NAME}')
    print(f'INPUT_DIM = {INPUT_DIM}, OUTPUT_DIM = {OUTPUT_DIM}')

    #Multiseed evaluation and mean\pm sd reporting(LRGB practice: 10 seeds)
    SEEDS = [0,1,2,3,4]
    results = {
        'base': [], 'egp': [], 'p-egp': [], 'rand': [], 'p-rand': [], 'cgp': [], 'p-cgp': []
    }
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        base_model = PeptidesGNN(transform_name='base', is_cgp=False).to(DEVICE)
        print('Experiments for the base graph (no expanders)')
        test_ap = run_experiment(base_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, transform_name='base')
        results['base'].append(test_ap)
        results_file = os.path.join(RESULTS_DIR, f'peptides_func_seed{seed}.jsonl')
        with open(results_file, 'a') as f:
            f.write(json.dumps({
                'dataset': 'peptides-func',
                'mode': 'base',
                'seed': int(seed),
                'ap': float(test_ap)
            }) + '\n')

        egp_model = PeptidesGNN(transform_name='EGP', is_cgp=False).to(DEVICE)
        print('Experiments for egp')
        test_ap = (run_experiment(egp_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, transform_name='EGP'))
        results['egp'].append(test_ap)
        results_file = os.path.join(RESULTS_DIR, f'peptides_func_seed{seed}.jsonl')
        with open(results_file, 'a') as f:
            f.write(json.dumps({
                'dataset': 'peptides-func',
                'mode': 'egp',
                'seed': int(seed),
                'ap': float(test_ap)
            }) + '\n')

        p_egp_model = PeptidesGNN(transform_name='P-EGP', is_cgp=False).to(DEVICE)
        print('Experiments for p-egp')
        test_ap = (run_experiment(p_egp_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, transform_name='P-EGP'))
        results['p-egp'].append(test_ap)
        results_file = os.path.join(RESULTS_DIR, f'peptides_func_seed{seed}.jsonl')
        with open(results_file, 'a') as f:
            f.write(json.dumps({
                'dataset': 'peptides-func',
                'mode': 'p-egp',
                'seed': int(seed),
                'ap': float(test_ap)
            }) + '\n')

        cgp_model = PeptidesGNN(transform_name='CGP', is_cgp=True).to(DEVICE)
        print('Experiments for cgp')
        test_ap = (run_experiment(cgp_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, transform_name='CGP'))
        results['cgp'].append(test_ap)
        results_file = os.path.join(RESULTS_DIR, f'peptides_func_seed{seed}.jsonl')
        with open(results_file, 'a') as f:
            f.write(json.dumps({
                'dataset': 'peptides-func',
                'mode': 'cgp',
                'seed': int(seed),
                'ap': float(test_ap)
            }) + '\n')


    print(f'''\nHyper parameters for this test\n#Training parameters\nNUM_EPOCHS = {NUM_EPOCHS}\nLR = {LR}\nBATCH_SIZE = {BATCH_SIZE}\nSEEDS = {SEEDS}\n\n#Scheduler: ReduceLRonPlateau\nREDUCE_FACTOR:{REDUCE_FACTOR}\nPATIENCE={PATIENCE}\nMIN_LR={MIN_LR} \n
          #Early stopping
          ES_PATIENCE={ES_PATIENCE}
          ES_MIN_DELTA = {ES_MIN_DELTA}
          \n# GNN\nNUM_LAYERS = {NUM_LAYERS}\nHIDDEN_DIM={HIDDEN_DIM}\nDROPOUT = {DROPOUT}\nDEGREE_D = {DEGREE_D}''')

    print('Final Test AP (mean ± sd over seeds):')
    for key in ['base','egp','p-egp','cgp']:
        arr = np.array(results[key], dtype = float)
        print(f'{key}: {arr.mean():.4f} ± {arr.std(ddof=1):.4f}')

if __name__ == '__main__':
    main()
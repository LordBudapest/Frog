import os
import sys
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BOTTLENECK_DIR = os.path.join(CUR_DIR, 'bottleneck-main')
if BOTTLENECK_DIR not in sys.path:
    sys.path.append(BOTTLENECK_DIR)

from common import Task
from helpers import get_cayley_n, cayley_graph_size, get_cayley_graph

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 200
LR = 0.001
BATCH_SIZE = 4096
NUM_ITER = 1
WEIGHT_DECAY = 5e-4
EARLY_STOP_PATIENCE = 50
SCHEDULER_MODE = 'max'
SCHEDULER_THRESHOLD_MODE = 'abs'
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10

NUM_LAYERS = 8
HIDDEN_DIM = 64
DROPOUT = 0.0

DEPTH = 5
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1

INPUT_DIM = None
OUTPUT_DIM = None

class TreeExpanderTransform:
    def __init__(self, type_name: str):
        self.type = type_name.upper()
        self.cayley_memory: dict[int, torch.Tensor] = {}
        self.cayley_node_memory: dict[int, torch.Tensor] = {}

    def _get_egp_edge_index(self, num_nodes: int) -> torch.Tensor:
        cayley_n = get_cayley_n(num_nodes)
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        cayley_graph_edge_index = self.cayley_memory[cayley_n].clone()

        if num_nodes not in self.cayley_node_memory:
            truncated_edge_index = cayley_graph_edge_index[:, (cayley_graph_edge_index[0] < num_nodes) & (cayley_graph_edge_index[1] < num_nodes)]
            self.cayley_node_memory[num_nodes] = truncated_edge_index
        return self.cayley_node_memory[num_nodes].clone()
    
    def _get_cgp_edge_index(self, num_nodes: int) -> tuple[torch.Tensor, int]:
        cayley_n = get_cayley_n(num_nodes)
        cayley_num_nodes = cayley_graph_size(cayley_n)
        if cayley_n not in self.cayley_memory:
            self.cayley_memory[cayley_n] = get_cayley_graph(cayley_n)
        edge_index = self.cayley_memory[cayley_n].clone()
        return edge_index, cayley_num_nodes
    
    def apply_to_data(self, data: Data):
        '''
        Mutate the given Data object to add expander_edge_index and possibly virtual nodes for CGP variants.
        For P-EGP/P-CGP we now provide base EGP/CGP edges only (no permutation here): per-layer permutation are handled in the model
        '''
        num_nodes = int(data.num_nodes) if data.num_nodes is not None else int(data.x.size(0))
        t = self.type
        if t in ('EGP', 'P-EGP','PEGP','PGP'):
            data.expander_edge_index = self._get_egp_edge_index(num_nodes)
        elif t == 'CGP':
            if not hasattr(data, 'cgp_applied') or not bool(data.cgp_applied):
                edge_index, cayley_num_nodes = self._get_cgp_edge_index(num_nodes)
                data.expander_edge_index = edge_index
                virtual_num_nodes = cayley_num_nodes - num_nodes
                data.virtual_node_mask = torch.cat((torch.zeros(num_nodes, dtype=torch.bool), torch.ones(virtual_num_nodes, dtype=torch.bool)), dim=0)
                pad = torch.zeros((virtual_num_nodes, data.x.shape[1]), dtype = data.x.dtype)
                data.x = torch.cat((data.x, pad), dim=0)
                if hasattr(data, 'root_mask') and data.root_mask is not None:
                    pad_root = torch.zeros(virtual_num_nodes, dtype=torch.bool)
                    data.root_mask = torch.cat((data.root_mask, pad_root), dim=0)
                data.num_nodes = cayley_num_nodes
                data.cayley_num_nodes = cayley_num_nodes
                data.cgp_applied = True
            else:
                pass
        elif t in ('P-CGP', 'PCGP'):
            if not hasattr(data, 'cgp_applied') or not bool(data.cgp_applied):
                edge_index, cayley_num_nodes = self._get_cgp_edge_index(num_nodes)
                data.expander_edge_index = edge_index
                virtual_num_nodes = cayley_num_nodes - num_nodes
                data.virtual_node_mask = torch.cat((torch.zeros(num_nodes, dtype=torch.bool), torch.ones(virtual_num_nodes, dtype=torch.bool)), dim=0)
                pad = torch.zeros((virtual_num_nodes, data.x.shape[1]), dtype=data.x.dtype)
                data.x = torch.cat((data.x, pad), dim=0)
                if hasattr(data, 'root_mask') and data.root_mask is not None:
                    pad_root = torch.zeros(virtual_num_nodes, dtype=torch.bool)
                    data.root_mask = torch.cat((data.root_mask, pad_root), dim=0)
                data.num_nodes = cayley_num_nodes
                data.cayley_num_nodes = cayley_num_nodes
                data.cgp_applied = True
            else:
                if not hasattr(data, 'expander_edge_index') or data.expander_edge_index is None:
                    edge_index, _ = self._get_cgp_edge_index(int((~data.virtual_node_mask).sum().item()))
                    data.expander_edge_index = edge_index
        else:
            pass
        return data
    def apply_to_dataset(self, dataset_list: list[Data]):
        for data in dataset_list:
            self.apply_to_data(data)

def get_loaders(depth: int, train_fraction: float, batch_size: int = BATCH_SIZE, val_fraction: float = VAL_FRACTION, seed: int = 42):
    '''
    Generate train/val/test loaders from bottleneck-main synthetic dataset.
    Returns both lists and loaders for transform application per epoch
    '''
    X_train, X_test, dim0, out_dim, criterion = Task.NEIGHBORS_MATCH.get_dataset(depth, train_fraction)

    val_size = max(1, int(len(X_train) * val_fraction))
    train_size = max(0, len(X_train) - val_size)
    generator = torch.Generator().manual_seed(seed)
    X_train_split, X_val_split = torch.utils.data.random_split(X_train, [train_size, val_size], generator=generator)

    train_list = list(X_train_split)
    val_list = list(X_val_split)
    test_list = list(X_test)
    num_workers = max(0, min(os.cpu_count() or 0, 8))
    pin_memory = (DEVICE.type == 'cuda')
    persistent_workers = (num_workers > 0)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    val_loader = DataLoader(val_list, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_list, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    return (train_list, val_list, test_list, train_loader, val_loader, test_loader, dim0, out_dim, criterion)

class TreeGNNNode(nn.Module):
    def __init__(self,transform_name: str | None, is_cgp: bool):
        super().__init__()
        self.transform_name = (transform_name or 'base').upper()
        self.num_layer = NUM_LAYERS
        self.drop_ratio = DROPOUT
        self.is_cgp = bool(is_cgp)
        self.mode = (transform_name or 'base').lower()
        self.degree_d = int(os.getenv('DEGREE_D','4'))
        self.pr_cache: dict[int, torch.Tensor] = {}
        self.cayley_cache: dict[int, torch.Tensor] = {}

        self.layer0_keys = nn.Embedding(num_embeddings= INPUT_DIM + 1, embedding_dim=HIDDEN_DIM)
        self.layer0_values = nn.Embedding(num_embeddings=INPUT_DIM + 1, embedding_dim=HIDDEN_DIM)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layer):
            input_dim = HIDDEN_DIM
            gnn_nn = nn.Sequential(
                nn.Linear(input_dim, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM,HIDDEN_DIM)
            )
            self.convs.append(GINConv(gnn_nn))
            self.batch_norms.append(nn.BatchNorm1d(HIDDEN_DIM))
    def begin_epoch(self):
        self.pr_cache = {}

    def _batch_counts_offsets(self,batch: torch.Tensor):
        counts = torch.bincount(batch)
        offsets = torch.empty_like(counts)
        if len(counts) > 0:
            offsets[0] = 0
            if len(counts) > 1:
                offsets[1:] = torch.cumsum(counts[:-1], dim=0)
        else:
            offsets = torch.tensor([], device = batch.device, dtype=torch.long)
        return counts, offsets
    def _blockwise_perm(self, counts, offsets, device):
        total = int(counts.sum().item())
        perm = torch.empty(total, dtype=torch.long, device=device)
        for gid in range(len(counts)):
            n = int(counts[gid].item())
            if n == 0:
                continue
            start = int(offsets[gid].item())
            local_perm = torch.randperm(n, device = device)
            perm[start: start+n] = start + local_perm
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
        x_indices = batched_data.x
        x_key_embed = self.layer0_keys(x_indices[:, 0])
        x_val_embed = self.layer0_values(x_indices[:, 1])
        x0 = x_key_embed + x_val_embed

        if self.is_cgp and hasattr(batched_data, 'virtual_node_mask'):
            h_list = [torch.zeros_like(x0)]
            h_list[0][~batched_data.virtual_node_mask] = x0[~batched_data.virtual_node_mask]
        else:
            h_list = [x0]

        base_edge_index = batched_data.edge_index
        for layer in range(self.num_layer):
            base_modes = ['egp','cgp', 'p-egp','p-cgp','rand','p-rand']
            use_alt = (self.mode in base_modes and (layer % 2 == 1))
            if use_alt:
                alt_edge_index = self._compute_alt_edge_index(batched_data)
                h = self.convs[layer](h_list[layer], alt_edge_index)
            else:
                h = self.convs[layer](h_list[layer], base_edge_index)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        return h_list[-1]
    
class TreeGNN(nn.Module):
    def __init__(self, transform_name: str | None = None, is_cgp: bool = False, out_dim: int = 2):
        super().__init__()
        self.is_cgp = is_cgp
        self.gnn_node = TreeGNNNode(transform_name, is_cgp)
        self.graph_pred_linear = nn.Linear(HIDDEN_DIM, out_dim + 1)

    def forward(self, batched_data: Data):
        h_node = self.gnn_node(batched_data)
        root_nodes = h_node[batched_data.root_mask]
        return self.graph_pred_linear(root_nodes)
    
def train(model: nn.Module, loader: DataLoader, optimiser: torch.optim.Optimizer, loss_fn):
    model.train()
    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(DEVICE)
        y = batch.y.to(DEVICE)
        out = model(batch)
        optimiser.zero_grad()
        loss = loss_fn(input=out, target=y)
        loss.backward()
        optimiser.step()

def eval_acc(model: nn.Module, loader: DataLoader):
    model.eval()
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            pred = out.argmax(dim=1)
            total_correct += pred.eq(batch.y).sum().item()
            total_examples += batch.y.size(0)
    return total_correct / max(total_examples, 1)

def run_experiment(model: nn.Module,
                   train_list: list[Data], val_list: list[Data], test_list: list[Data],
                   train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                   criterion,
                   transform_name: str | None = None):
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        threshold_mode=SCHEDULER_THRESHOLD_MODE
    )

    train_curve = []
    validation_curve = []
    test_curve = []
    best_val = -float('inf')
    patience_count = 0

    transform_obj = None
    tname = (transform_name or 'base').upper()
    if tname in ('CGP', 'P-CGP', 'PCGP'):
        transform_obj = TreeExpanderTransform(tname)
        transform_obj.apply_to_dataset(train_list)
        transform_obj.apply_to_dataset(val_list)
        transform_obj.apply_to_dataset(test_list)
    print('Start training')
    for epoch in range(1, 1 + NUM_EPOCHS):
        print(f'Epoch: {epoch}')
        if hasattr(model, 'gnn_node') and hasattr(model.gnn_node, 'begin_epoch'):
            model.gnn_node.begin_epoch()
        train(model, train_loader, optimiser=optimiser, loss_fn=criterion)

        train_acc = eval_acc(model, train_loader)
        validation_acc = eval_acc(model, val_loader)
        test_acc = eval_acc(model, test_loader)

        train_curve.append(train_acc)
        validation_curve.append(validation_acc)
        test_curve.append(test_acc)

        print(f'Train acc: {train_acc:.4f}, validation Acc: {validation_acc:.4f}, test Acc: {test_acc:.4f}\n')

        scheduler.step(validation_acc)

        #Early stopping on validation accuracy
        if validation_acc > best_val:
            best_val = validation_acc
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= EARLY_STOP_PATIENCE:
            print(f'Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)')
            break
    best_validation_epoch = int(np.argmax(np.array(validation_curve)))
    print('Finished training')
    print(f'Best validation score: {validation_curve[best_validation_epoch]:.4f}')
    print(f'Final test score: {test_curve[best_validation_epoch]:.4f}')
    return test_curve[best_validation_epoch]

def parse_args():
    parser = argparse.ArgumentParser(description="Tree-NeighborsMatch synthetic benchmark")
    parser.add_argument('--seeds', type=str, default='1,11,21,31,41', help='Comma-seprated list of RNG seeds')
    parser.add_argument('--depth', type=int, default=DEPTH, help='Tree depth for dataset generation')
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS, help='Number of GNN layers')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size for the dataloader')
    return parser.parse_args()

def main():
    global INPUT_DIM, OUTPUT_DIM, NUM_LAYERS, HIDDEN_DIM, BATCH_SIZE
    args = parse_args()
    depth = int(args.depth)
    NUM_LAYERS = int(args.num_layers)
    HIDDEN_DIM = int(args.hidden_dim)
    BATCH_SIZE = int(args.batch_size)
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    print('Tree-NeighborsMatch (synthetic benchmark): Accuracy( higher is better)')
    print(f'Reporting: mean ± standard deviation over {len(seeds)} seeds on the test split, evaluated at the best validation epoch; depth={depth}')

    base_scores = []
    egp_scores = []
    p_egp_scores = []
    cgp_scores = []
    p_cgp_scores = []
    rand_scores = []
    p_rand_scores = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        train_list, val_list, test_list, train_loader, val_loader, test_loader, dim0, out_dim, criterion = get_loaders(
            depth, TRAIN_FRACTION, batch_size=BATCH_SIZE, val_fraction=VAL_FRACTION, seed=seed
        )
        INPUT_DIM = int(dim0)
        OUTPUT_DIM = int(out_dim)
        print(f'\nDataset: Tree-NeighborsMatch (depth={depth}, seed={seed})')
        print(f'INPUT_DIM = {INPUT_DIM}, OUTPUT_DIM = {OUTPUT_DIM}( +1 LOGITS)')

        base_model = TreeGNN(transform_name='base', is_cgp=False, out_dim=OUTPUT_DIM).to(DEVICE)
        print('Experiments for the base graph (no expander)')
        base_score = 0.0
        for _ in range(NUM_ITER):
            base_score += run_experiment(base_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, criterion, transform_name='base')
        base_scores.append(base_score/ max(1, NUM_ITER))

        egp_model = TreeGNN(transform_name='EGP', is_cgp=False, out_dim=OUTPUT_DIM).to(DEVICE)
        print('Experiments for egp')
        egp_score = 0.0
        for _ in range(NUM_ITER):
            egp_score += run_experiment(egp_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, criterion, transform_name='EGP')
        egp_scores.append(egp_score/ max(1, NUM_ITER))

        p_egp_model = TreeGNN(transform_name='P-EGP', is_cgp=False, out_dim=OUTPUT_DIM).to(DEVICE)
        print('Experiments for p-egp')
        p_egp_score = 0.0
        for _ in range(NUM_ITER):
            p_egp_score += run_experiment(p_egp_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, criterion, transform_name='P-EGP')
        p_egp_scores.append(p_egp_score/ max(1, NUM_ITER))

        rand_model = TreeGNN(transform_name='rand', is_cgp=False, out_dim=OUTPUT_DIM).to(DEVICE)
        print('Experiments for rand(random regular per odd layer)')
        rand_score = 0.0
        for _ in range(NUM_ITER):
            rand_score += run_experiment(rand_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, criterion, transform_name='base')
        rand_scores.append(rand_score/ max(1, NUM_ITER))

        p_rand_model = TreeGNN(transform_name='p-rand', is_cgp=False, out_dim=OUTPUT_DIM).to(DEVICE)
        print('Experiments for p-rand( per-epoch random base, permuted per odd layer)')
        p_rand_score = 0.0
        for _ in range(NUM_ITER):
            p_rand_score += run_experiment(p_rand_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, criterion, transform_name='base')
        p_rand_scores.append(p_rand_score/ max(1, NUM_ITER))

        cgp_model = TreeGNN(transform_name='CGP', is_cgp=True, out_dim=OUTPUT_DIM).to(DEVICE)
        print('Experiments for cgp')
        cgp_score = 0.0
        for _ in range(NUM_ITER):
            cgp_score += run_experiment(cgp_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, criterion, transform_name='CGP')
        cgp_scores.append(cgp_score/ max(1, NUM_ITER))

        p_cgp_model = TreeGNN(transform_name='P-CGP', is_cgp=True, out_dim=OUTPUT_DIM).to(DEVICE)
        print('Experiments for p-cgp')
        p_cgp_score = 0.0
        for _ in range(NUM_ITER):
            p_cgp_score += run_experiment(p_cgp_model, train_list, val_list, test_list, train_loader, val_loader, test_loader, criterion, transform_name='P-CGP')
        p_cgp_scores.append(p_cgp_score/ max(1, NUM_ITER))

    def mean_std(x):
        arr = np.array(x, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))
    
    base_mean, base_std = mean_std(base_scores)
    egp_mean, egp_std = mean_std(egp_scores)
    p_egp_mean, p_egp_std = mean_std(p_egp_scores)
    cgp_mean, cgp_std = mean_std(cgp_scores)
    p_cgp_mean, p_cgp_std = mean_std(p_cgp_scores)
    rand_mean, rand_std = mean_std(rand_scores)
    p_rand_mean, p_rand_std = mean_std(p_rand_scores)

    print("\nFinal aggregated scores:")
    print(f'base graph (no expander): {base_mean:.4f} ± {base_std:.4f}')
    print(f'egp graph (no expander): {egp_mean:.4f} ± {egp_std:.4f}')
    print(f'p-egp graph (no expander): {p_egp_mean:.4f} ± {p_egp_std:.4f}')
    print(f'cgp graph (no expander): {cgp_mean:.4f} ± {cgp_std:.4f}')
    print(f'p-cgp graph (no expander): {p_cgp_mean:.4f} ± {p_cgp_std:.4f}')
    print(f'rand graph (no expander): {rand_mean:.4f} ± {rand_std:.4f}')
    print(f'p-rand graph (no expander): {p_rand_mean:.4f} ± {p_rand_std:.4f}')

if __name__ == '__main__':
    main()

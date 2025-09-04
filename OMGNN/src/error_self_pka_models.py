import os
import math
import logging
from typing import Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import set_detect_anomaly
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# =====================
# Logging utilities
# =====================

def setup_logger(name: str = "pka_model", log_dir: str = "logs") -> logging.Logger:
    """Configure and return a logger that writes DEBUG to file and ERROR to console."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger already configured in this session
        return logger

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger()

# Enable autograd anomaly detection once at import time (optional)
set_detect_anomaly(True)
logger.info("Autograd anomaly detection enabled – may slow down training.")

# =====================
#  Helper layers
# =====================

class Squeeze1D(nn.Module):
    """Remove the last dimension if it equals 1."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401, N802
        return x.squeeze(-1) if x.size(-1) == 1 else x


# =====================
# Message‑passing layers
# =====================

class BondMessagePassing(nn.Module):
    """A simple edge‑aware residual MPNN (used in the original pka_GNN)."""

    def __init__(self, node_dim: int, bond_dim: int, hidden_dim: int,
                 *, depth: int = 5, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        self.depth = depth
        self.dropout_p = dropout

        self.W_i = nn.Linear(node_dim + bond_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(node_dim + hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    # ------------------------------------------------------------------
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize(t: torch.Tensor) -> torch.Tensor:
        """Replace NaN/Inf with finite values to prevent gradient explosions."""
        return torch.nan_to_num(t, nan=0.0, posinf=1e3, neginf=-1e3)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,          # [N, node_dim]
        edge_index: torch.Tensor, # [2, E]
        edge_attr: torch.Tensor,  # [E, bond_dim]
        rev_edge_index: torch.Tensor = None,  # Needed for backwards message; optional
    ) -> torch.Tensor:
        device = x.device
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        # (1) initial messages
        h0 = self.relu(self.W_i(torch.cat([x[edge_index[0]], edge_attr], dim=1)))
        h = h0.clone()

        # (2) iterative message passing
        for _ in range(1, self.depth):
            # Aggregate incoming messages
            m = torch.zeros_like(h)
            m.index_add_(0, edge_index[1], h)  # sum over neighbours -> dst

            # subtract reverse edge message if provided (optional)
            if rev_edge_index is not None and (rev_edge_index >= 0).any():
                valid = rev_edge_index >= 0
                m.index_add_(0, edge_index[0, valid], -h[rev_edge_index[valid]])

            m = self.W_h(self._sanitize(m))
            h = self.dropout(self.relu(h0 + m))  # residual

        # (3) final update
        m_final = torch.zeros_like(h)
        m_final.index_add_(0, edge_index[1], h)
        h_cat = torch.cat([x, self._sanitize(m_final)], dim=1)
        out = self.dropout(self.relu(self.W_o(h_cat)))
        return self._sanitize(out)


class BondGATMessagePassing(nn.Module):
    """A multi‑head edge‑aware GAT block, wrapped to match the API above."""

    def __init__(self, node_dim: int, bond_dim: int, hidden_dim: int,
                 *, depth: int = 5, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.depth = depth
        self.heads = heads
        self.dropout_p = dropout

        self.layers = nn.ModuleList()
        dim_in = node_dim
        for _ in range(depth):
            self.layers.append(
                GATConv(
                    in_channels=dim_in,
                    out_channels=hidden_dim,
                    heads=heads,
                    edge_dim=bond_dim,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            dim_in = hidden_dim * heads  # concat heads

        self.fc_out = nn.Linear(node_dim + hidden_dim * heads, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        nn.init.kaiming_normal_(self.fc_out.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.fc_out.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize(t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=1e3, neginf=-1e3)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        rev_edge_index: torch.Tensor = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for gat in self.layers:
            h = F.relu(gat(h, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout_p, training=self.training)
            h = self._sanitize(h)
        h_cat = torch.cat([x, h], dim=1)
        out = self.dropout_p if isinstance(self.dropout_p, float) else self.dropout_p
        out = self._sanitize(F.dropout(F.relu(self.fc_out(h_cat)), p=self.dropout_p, training=self.training))
        return out


class BondGraphTransformer(nn.Module):
    def __init__(self, node_features, bond_features, hidden_size, depth=5, n_heads=4, dropout=0.3):
        super().__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.depth = depth

        # 初始節點特徵投影
        self.node_proj = nn.Linear(node_features, hidden_size)
        # 邊特徵作為注意力偏置
        self.edge_proj = nn.Linear(bond_features, n_heads)

        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        # 輸出投影
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # 前饋網絡
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # 層歸一化
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 權重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr, rev_edge_index=None):
        """
        x: Tensor [N, node_features]
        edge_index: LongTensor [2, E] (source, target)
        edge_attr: Tensor [E, bond_features]
        returns: Tensor [N, hidden_size]
        """
        device = x.device
        N = x.size(0)

        # 投影節點特徵
        h = self.node_proj(x)  # [N, hidden_size]
        # 處理邊特徵為每 head 的偏置
        bias = torch.zeros(self.n_heads, N, N, device=device)
        e_bias = self.edge_proj(edge_attr)  # [E, n_heads]
        src, dst = edge_index
        for head in range(self.n_heads):
            bias[head, src, dst] = e_bias[:, head]

        # 多層 Transformer
        for _ in range(self.depth):
            # 預計算 Q, K, V
            Q = self.q_proj(h).view(N, self.n_heads, self.head_dim).permute(1, 0, 2)  # [heads, N, d]
            K = self.k_proj(h).view(N, self.n_heads, self.head_dim).permute(1, 0, 2)
            V = self.v_proj(h).view(N, self.n_heads, self.head_dim).permute(1, 0, 2)

            # 注意力計分
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [heads, N, N]
            scores = scores + bias  # 加上邊偏置
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # 聚合
            out = torch.matmul(attn, V)  # [heads, N, d]
            out = out.permute(1, 0, 2).contiguous().view(N, self.hidden_size)  # [N, hidden]
            out = self.out_proj(out)

            # 殘差 + LayerNorm
            h = self.norm1(h + self.dropout(out))
            # 前饋 + 殘差 + LayerNorm
            h_ff = self.ff(h)
            h = self.norm2(h + self.dropout(h_ff))

            # 處理 NaN/Inf
            if torch.isnan(h).any() or torch.isinf(h).any():
                h = torch.nan_to_num(h, nan=0.0, posinf=1e3, neginf=-1e3)

        return h
# =====================
# Base GNN for pKa prediction
# =====================

class BasePkaGNN(nn.Module):
    """Shared implementation; pass a message‑passing layer class to change backbone."""

    def __init__(
        self,
        node_dim: int,
        bond_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        mp_layer_cls: Type[nn.Module],
        dropout: float = 0.0,
        depth: int = 4,
        **mp_kwargs,
    ):
        super().__init__()
        self.dropout_p = dropout
        self.depth = depth

        # 1) Fixed backbone (gcn1 frozen, gcn3 updated every loop)
        self.gcn1 = mp_layer_cls(node_dim, bond_dim, hidden_dim, depth=depth, dropout=dropout, **mp_kwargs)
        self.gcn3 = mp_layer_cls(node_dim, bond_dim, hidden_dim, depth=depth, dropout=dropout, **mp_kwargs)

        # 2) Gate to combine features
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

        # 3) Classifier & regressor
        self.atom_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2)
        )
        self.atom_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, output_dim), Squeeze1D()
        )

        # 4) Losses & normalisation params
        self.criterion_reg = nn.MSELoss(reduction="mean")
        self.register_buffer("pka_mean", torch.tensor([7.0]))
        self.register_buffer("pka_std", torch.tensor([3.0]))

    # ------------------------------------------------------------------
    @staticmethod
    def _rev_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
        device = edge_index.device
        mapping = {(edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(edge_index.size(1))}
        rev = torch.full((edge_index.size(1),), -1, dtype=torch.long, device=device)
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in mapping:
                rev[i] = mapping[(v, u)]
        return rev

    # ------------------------------------------------------------------
    def forward(self, batch: Data, *, return_latent: bool = False):  # noqa: C901
        device = batch.x.device
        x, ei, ea = batch.x, batch.edge_index, batch.edge_attr
        rev_ei = self._rev_edge_index(ei)

        # (1) one‑time backbone pass
        h_static = self.gcn1(x, ei, ea, rev_ei)  # [N, H]
        h_cur = self.gcn3(h_static, ei, ea, rev_ei)

        # (2) teacher forcing loop over ground‑truth sites
        gt_mask = batch.pka_labels > 0
        idx_gt = torch.nonzero(gt_mask).squeeze(1)
        if idx_gt.numel() == 0:
            logits = self.atom_classifier(h_cur)
            pka_raw = self.atom_regressor(h_cur).view(-1)
            loss_cla = F.cross_entropy(logits, torch.zeros_like(gt_mask, dtype=torch.long))
            return logits, pka_raw, (0.5 * loss_cla, loss_cla, torch.tensor(0.0, device=device))

        idx_sorted = idx_gt[torch.argsort(batch.pka_labels[idx_gt])]

        loss_cla_steps, loss_reg_steps = [], []
        latent_steps, pka_steps = [], []

        for idx in idx_sorted:
            if return_latent:
                latent_steps.append(h_cur[idx].detach().cpu())
                pka_steps.append(batch.pka_labels[idx].detach().cpu())

            logits = self.atom_classifier(h_cur)
            target = torch.zeros_like(gt_mask, dtype=torch.long)
            target[idx] = 1
            ratio = float((target == 0).sum()) / (target.sum() + 1e-6)
            loss_c = F.cross_entropy(logits, target, weight=torch.tensor([1.0, ratio], device=device), reduction="none")
            loss_cla_steps.extend(loss_c)

            pred_pka_norm = (self.atom_regressor(h_cur)[idx] - self.pka_mean) / self.pka_std
            true_pka_norm = (batch.pka_labels[idx] - self.pka_mean) / self.pka_std
            loss_r = self.criterion_reg(pred_pka_norm, true_pka_norm)
            loss_reg_steps.append(loss_r)

            # (gate) update one atom, reset others
            h_upd = h_static.clone()
            h_upd[idx] = h_cur[idx] * self.gate(h_cur[idx]) + h_cur[idx]
            h_cur = self.gcn3(h_upd, ei, ea, rev_ei)

        loss_cla = torch.stack(loss_cla_steps).mean()
        loss_reg = torch.stack(loss_reg_steps).mean()
        total = loss_cla + loss_reg

        final_logits = logits  # from last loop
        final_pka = self.atom_regressor(h_cur).view(-1)

        outputs = (final_logits, final_pka, (total, loss_cla, loss_reg))
        if return_latent:
            outputs += (latent_steps, pka_steps)
        return outputs

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def set_pka_normalization(self, mean: float, std: float):
        self.pka_mean[0] = float(mean)
        self.pka_std[0] = max(float(std), 1e-6)
        logger.info(f"Set pKa normalisation: mean={mean:.4f}, std={std:.4f}")

    def sample(self, smiles: str, *, device: Union[torch.device, None] = None,
               eval_mode: str = "predicted", return_latent: bool = False):
        from rdkit import Chem  # local import to avoid RDKit dependency when unused
        from self_pka_chemutils import tensorize_for_pka
        if device is None:
            device = next(self.parameters()).device
        self.to(device).eval()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles}")

        fatoms, ei, ea = tensorize_for_pka(smiles)
        n = fatoms.size(0)
        data = Data(x=fatoms, edge_index=ei, edge_attr=ea,
                    pka_labels=torch.zeros(n), batch=torch.zeros(n, dtype=torch.long), smiles=smiles).to(device)

        with torch.no_grad():
            if return_latent:
                logits, pka_pred, _, latent_steps, pka_steps = self(data, return_latent=True)
            else:
                logits, pka_pred, _ = self(data)

        has_pka = (logits.argmax(1) == 1).cpu().numpy()
        pka_pred = pka_pred.cpu().numpy()
        idx = np.where(has_pka == 1)[0] if eval_mode == "predicted" else np.arange(n)

        result = {
            "smiles": smiles,
            "mol": mol,
            "atom_has_pka": has_pka,
            "atom_pka_values": pka_pred,
            "pka_positions": idx.tolist(),
            "pka_values": pka_pred[idx].tolist(),
            "eval_mode": eval_mode,
        }
        if return_latent:
            result["latent_steps"] = latent_steps
            result["pka_steps"] = pka_steps
        return result


# =====================
# Concrete models
# =====================

class pka_GNN(BasePkaGNN):
    """Original backbone based on custom BondMessagePassing."""

    def __init__(self, node_dim: int, bond_dim: int, hidden_dim: int, output_dim: int,
                 *, dropout: float = 0.0, depth: int = 4):
        super().__init__(
            node_dim, bond_dim, hidden_dim, output_dim,
            mp_layer_cls=BondMessagePassing,
            dropout=dropout,
            depth=depth,
        )


class pka_GNN_ver2(BasePkaGNN):
    """Alternate backbone using multi‑head edge‑aware GAT."""

    def __init__(self, node_dim: int, bond_dim: int, hidden_dim: int, output_dim: int,
                 *, dropout: float = 0.0, depth: int = 4, heads: int = 4):
        super().__init__(
            node_dim, bond_dim, hidden_dim, output_dim,
            mp_layer_cls=BondGATMessagePassing,
            dropout=dropout,
            depth=depth,
            heads=heads,
        )

# 使用Transformer
class pka_GNN_ver3(BasePkaGNN):
    
    def __init__(self, node_dim: int, bond_dim: int, hidden_dim: int, output_dim: int,
                 *, dropout: float = 0.0, depth: int = 4, heads: int = 4):
        super().__init__(
            node_dim, bond_dim, hidden_dim, output_dim,
            mp_layer_cls=BondGraphTransformer,
            dropout=dropout,
            depth=depth,
            n_heads=heads,
        )

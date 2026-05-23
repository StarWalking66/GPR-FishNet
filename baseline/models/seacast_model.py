from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from baseline.models.task_adapter import validate_sequence_grid_input


def _build_grid_node_index(mask_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ocean_mask = np.asarray(mask_2d, dtype=bool)
    coords = np.argwhere(ocean_mask)
    node_id = np.full(ocean_mask.shape, fill_value=-1, dtype=np.int64)
    for idx, (row, col) in enumerate(coords):
        node_id[row, col] = idx
    return coords, node_id


def _build_normalized_grid_edges(occupied_cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coords, node_id = _build_grid_node_index(occupied_cells)
    num_nodes = int(coords.shape[0])

    if num_nodes == 0:
        empty_index = np.zeros((2, 0), dtype=np.int64)
        empty_weight = np.zeros((0,), dtype=np.float32)
        return empty_index, empty_weight

    offsets = [
        (-1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
    ]

    src_nodes = []
    dst_nodes = []
    raw_weights = []
    rows, cols = occupied_cells.shape

    for src_idx, (row, col) in enumerate(coords):
        src_nodes.append(src_idx)
        dst_nodes.append(src_idx)
        raw_weights.append(1.0)

        for d_row, d_col, distance in offsets:
            n_row = row + d_row
            n_col = col + d_col
            if not (0 <= n_row < rows and 0 <= n_col < cols):
                continue
            dst_idx = node_id[n_row, n_col]
            if dst_idx < 0:
                continue
            src_nodes.append(src_idx)
            dst_nodes.append(int(dst_idx))
            raw_weights.append(float(1.0 / distance))

    src_array = np.asarray(src_nodes, dtype=np.int64)
    dst_array = np.asarray(dst_nodes, dtype=np.int64)
    weight_array = np.asarray(raw_weights, dtype=np.float32)

    incoming_sum = np.zeros((num_nodes,), dtype=np.float32)
    np.add.at(incoming_sum, dst_array, weight_array)
    norm_weights = weight_array / np.maximum(incoming_sum[dst_array], 1e-6)

    edge_index = np.stack([src_array, dst_array], axis=0)
    return edge_index, norm_weights.astype(np.float32)


def build_seacast_graph(mask_2d: np.ndarray, coarse_factor: int = 4) -> Dict[str, torch.Tensor]:
    ocean_mask = np.asarray(mask_2d, dtype=bool)
    coords, _ = _build_grid_node_index(ocean_mask)

    if coords.size == 0:
        raise ValueError("SeaCast graph construction requires at least one valid ocean cell.")

    height, width = ocean_mask.shape
    node_positions = (coords[:, 0] * width + coords[:, 1]).astype(np.int64)

    fine_edge_index, fine_edge_weight = _build_normalized_grid_edges(ocean_mask)

    coarse_rows = (height + coarse_factor - 1) // coarse_factor
    coarse_cols = (width + coarse_factor - 1) // coarse_factor
    coarse_occupied = np.zeros((coarse_rows, coarse_cols), dtype=bool)
    for row, col in coords:
        coarse_occupied[row // coarse_factor, col // coarse_factor] = True

    coarse_coords, coarse_node_id = _build_grid_node_index(coarse_occupied)
    fine_to_coarse = np.asarray(
        [coarse_node_id[row // coarse_factor, col // coarse_factor] for row, col in coords],
        dtype=np.int64,
    )
    coarse_counts = np.bincount(fine_to_coarse, minlength=int(coarse_coords.shape[0])).astype(np.float32)
    coarse_counts = np.maximum(coarse_counts, 1.0)

    coarse_edge_index, coarse_edge_weight = _build_normalized_grid_edges(coarse_occupied)

    return {
        "height": torch.tensor(height, dtype=torch.long),
        "width": torch.tensor(width, dtype=torch.long),
        "node_positions": torch.from_numpy(node_positions),
        "fine_edge_index": torch.from_numpy(fine_edge_index),
        "fine_edge_weight": torch.from_numpy(fine_edge_weight),
        "fine_to_coarse": torch.from_numpy(fine_to_coarse),
        "coarse_edge_index": torch.from_numpy(coarse_edge_index),
        "coarse_edge_weight": torch.from_numpy(coarse_edge_weight),
        "coarse_counts": torch.from_numpy(coarse_counts),
    }


def _aggregate_neighbors(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    edge_weight = edge_weight.to(dtype=x.dtype)
    messages = x[:, src_idx, :] * edge_weight.view(1, -1, 1)
    aggregated = x.new_zeros(x.shape)
    aggregated.index_add_(1, dst_idx, messages)
    return aggregated


def _pool_to_coarse(
    x: torch.Tensor,
    fine_to_coarse: torch.Tensor,
    coarse_counts: torch.Tensor,
) -> torch.Tensor:
    coarse = x.new_zeros(x.size(0), coarse_counts.numel(), x.size(-1))
    coarse.index_add_(1, fine_to_coarse, x)
    coarse_counts = coarse_counts.to(dtype=x.dtype)
    coarse = coarse / coarse_counts.view(1, -1, 1).clamp_min(1.0)
    return coarse


class GraphMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_state = _aggregate_neighbors(self.message_proj(x), edge_index, edge_weight)
        updated = self.update(torch.cat([x, neighbor_state], dim=-1))
        return self.norm(x + updated)


class HierarchicalGraphProcessor(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int = 2):
        super().__init__()
        self.fine_blocks = nn.ModuleList(GraphMessageBlock(hidden_dim) for _ in range(num_blocks))
        self.coarse_blocks = nn.ModuleList(GraphMessageBlock(hidden_dim) for _ in range(num_blocks))
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        fine_edge_index: torch.Tensor,
        fine_edge_weight: torch.Tensor,
        fine_to_coarse: torch.Tensor,
        coarse_counts: torch.Tensor,
        coarse_edge_index: torch.Tensor,
        coarse_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        fine_state = x
        for block in self.fine_blocks:
            fine_state = block(fine_state, fine_edge_index, fine_edge_weight)

        coarse_state = _pool_to_coarse(fine_state, fine_to_coarse, coarse_counts)
        for block in self.coarse_blocks:
            coarse_state = block(coarse_state, coarse_edge_index, coarse_edge_weight)

        coarse_up = coarse_state[:, fine_to_coarse, :]
        fused = self.fusion(torch.cat([fine_state, coarse_up], dim=-1))
        return self.output_norm(fine_state + fused)


class SeaCast(nn.Module):
    """
    SeaCast-style adapted baseline for this project.

    The model keeps the core idea of ocean-graph forecasting:
    1. encode per-ocean-cell temporal histories,
    2. run hierarchical message passing on an ocean graph,
    3. decode the graph state back to a regular heatmap.

    This adaptation intentionally uses the same historical inputs and
    training/evaluation protocol as the other baselines for fair comparison.
    """

    def __init__(
        self,
        graph_data: Dict[str, torch.Tensor],
        in_chans: int = 8,
        hidden_dim: int = 64,
        seq_len: int = 12,
        pred_len: int = 1,
        temporal_layers: int = 1,
        graph_blocks: int = 2,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.height = int(graph_data["height"].item())
        self.width = int(graph_data["width"].item())

        self.register_buffer("node_positions", graph_data["node_positions"].long(), persistent=False)
        self.register_buffer("fine_edge_index", graph_data["fine_edge_index"].long(), persistent=False)
        self.register_buffer("fine_edge_weight", graph_data["fine_edge_weight"].float(), persistent=False)
        self.register_buffer("fine_to_coarse", graph_data["fine_to_coarse"].long(), persistent=False)
        self.register_buffer("coarse_edge_index", graph_data["coarse_edge_index"].long(), persistent=False)
        self.register_buffer("coarse_edge_weight", graph_data["coarse_edge_weight"].float(), persistent=False)
        self.register_buffer("coarse_counts", graph_data["coarse_counts"].float(), persistent=False)

        self.input_proj = nn.Sequential(
            nn.Linear(in_chans, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
        )
        self.processor = HierarchicalGraphProcessor(hidden_dim=hidden_dim, num_blocks=graph_blocks)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len),
        )

    @classmethod
    def from_mask(
        cls,
        mask_2d: np.ndarray,
        *,
        coarse_factor: int = 4,
        **kwargs,
    ) -> "SeaCast":
        graph_data = build_seacast_graph(np.asarray(mask_2d, dtype=bool), coarse_factor=coarse_factor)
        return cls(graph_data=graph_data, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, in_chans, height, width = validate_sequence_grid_input(
            x,
            model_name=self.__class__.__name__,
            seq_len=self.seq_len,
            in_chans=self.in_chans,
            img_size=(self.height, self.width),
        )

        flat = x.view(batch_size, seq_len, in_chans, height * width).permute(0, 1, 3, 2).contiguous()
        node_series = flat[:, :, self.node_positions, :]

        node_features = self.input_proj(node_series)
        node_features = node_features.permute(0, 2, 1, 3).contiguous()
        node_features = node_features.view(batch_size * self.node_positions.numel(), seq_len, self.hidden_dim)

        _, hidden = self.temporal_encoder(node_features)
        node_state = hidden[-1].view(batch_size, self.node_positions.numel(), self.hidden_dim)

        node_state = self.processor(
            node_state,
            self.fine_edge_index,
            self.fine_edge_weight,
            self.fine_to_coarse,
            self.coarse_counts,
            self.coarse_edge_index,
            self.coarse_edge_weight,
        )

        node_pred = self.decoder(node_state).permute(0, 2, 1).contiguous()
        full_grid = x.new_zeros(batch_size, self.pred_len, self.height * self.width)
        node_pred = node_pred.to(dtype=full_grid.dtype)
        full_grid[:, :, self.node_positions] = node_pred
        full_grid = full_grid.view(batch_size, self.pred_len, self.height, self.width)
        return full_grid.unsqueeze(2).contiguous()

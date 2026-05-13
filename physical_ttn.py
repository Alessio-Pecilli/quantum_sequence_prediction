from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def infer_num_qubits_from_dim(dim_2n: int) -> int:
    dim_2n = int(dim_2n)
    if dim_2n < 1 or dim_2n & (dim_2n - 1):
        raise ValueError(f"Expected last dimension to be a power of 2, got {dim_2n}")
    return dim_2n.bit_length() - 1


@dataclass(frozen=True)
class TTNNodeSpec:
    node_id: int
    level: int
    dim: int
    qubits: Tuple[int, ...]


@dataclass(frozen=True)
class TTNMergeSpec:
    level: int
    left_node_id: int
    right_node_id: int
    parent_node_id: int
    left_dim: int
    right_dim: int
    parent_dim: int
    qubits: Tuple[int, ...]


@dataclass(frozen=True)
class TTNTreeSpec:
    num_qubits: int
    pairing: str
    root_node_id: int
    root_dim: int
    node_specs: Dict[int, TTNNodeSpec]
    levels: Tuple[Tuple[int, ...], ...]
    merges_by_level: Tuple[Tuple[TTNMergeSpec, ...], ...]
    carried_by_level: Tuple[Tuple[int, ...], ...]

    @classmethod
    def build(
        cls,
        *,
        num_qubits: int,
        pairing: str = config.TTN_TREE_PAIRING,
        bond_dim: int = config.TTN_BOND_DIM,
        root_dim: int = config.TTN_ROOT_DIM,
        use_bond_cap: bool = config.TTN_USE_BOND_CAP,
    ) -> "TTNTreeSpec":
        num_qubits = int(num_qubits)
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        pairing = str(pairing).lower()
        if pairing != "adjacent":
            raise ValueError(f"Unsupported TTN pairing strategy: {pairing}")

        node_specs: Dict[int, TTNNodeSpec] = {}
        current_nodes: list[int] = []
        next_node_id = 0
        for qubit in range(num_qubits):
            dim = config.bond_dim_for_block(1, base_dim=2, use_bond_cap=use_bond_cap)
            node_specs[next_node_id] = TTNNodeSpec(
                node_id=next_node_id,
                level=0,
                dim=dim,
                qubits=(qubit,),
            )
            current_nodes.append(next_node_id)
            next_node_id += 1

        levels: list[Tuple[int, ...]] = [tuple(current_nodes)]
        merges_by_level: list[Tuple[TTNMergeSpec, ...]] = []
        carried_by_level: list[Tuple[int, ...]] = []
        level = 0

        while len(current_nodes) > 1:
            next_nodes: list[int] = []
            level_merges: list[TTNMergeSpec] = []
            level_carries: list[int] = []
            index = 0
            while index < len(current_nodes):
                if index + 1 < len(current_nodes):
                    left_id = current_nodes[index]
                    right_id = current_nodes[index + 1]
                    left_node = node_specs[left_id]
                    right_node = node_specs[right_id]
                    parent_id = next_node_id
                    next_node_id += 1
                    parent_qubits = tuple(left_node.qubits + right_node.qubits)
                    is_root = len(current_nodes) == 2
                    if is_root:
                        parent_dim = config.bond_dim_for_block(
                            len(parent_qubits),
                            base_dim=root_dim,
                            use_bond_cap=use_bond_cap,
                        )
                    else:
                        parent_dim = config.bond_dim_for_block(
                            len(parent_qubits),
                            base_dim=bond_dim,
                            use_bond_cap=use_bond_cap,
                        )
                    node_specs[parent_id] = TTNNodeSpec(
                        node_id=parent_id,
                        level=level + 1,
                        dim=parent_dim,
                        qubits=parent_qubits,
                    )
                    level_merges.append(
                        TTNMergeSpec(
                            level=level,
                            left_node_id=left_id,
                            right_node_id=right_id,
                            parent_node_id=parent_id,
                            left_dim=left_node.dim,
                            right_dim=right_node.dim,
                            parent_dim=parent_dim,
                            qubits=parent_qubits,
                        )
                    )
                    next_nodes.append(parent_id)
                    index += 2
                else:
                    carry_id = current_nodes[index]
                    next_nodes.append(carry_id)
                    level_carries.append(carry_id)
                    index += 1
            merges_by_level.append(tuple(level_merges))
            carried_by_level.append(tuple(level_carries))
            current_nodes = next_nodes
            levels.append(tuple(current_nodes))
            level += 1

        root_node_id = current_nodes[0]
        root_dim_resolved = int(node_specs[root_node_id].dim)
        return cls(
            num_qubits=num_qubits,
            pairing=pairing,
            root_node_id=root_node_id,
            root_dim=root_dim_resolved,
            node_specs=node_specs,
            levels=tuple(levels),
            merges_by_level=tuple(merges_by_level),
            carried_by_level=tuple(carried_by_level),
        )


class _PhysicalTTNBase(nn.Module):
    def __init__(self, *, num_qubits: int, tree_spec: TTNTreeSpec | None = None):
        super().__init__()
        self.num_qubits = int(num_qubits)
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        self.dim_2n = 2 ** self.num_qubits
        self.tree_spec = tree_spec or TTNTreeSpec.build(num_qubits=self.num_qubits)
        self.root_dim = int(self.tree_spec.root_dim)

    @staticmethod
    def _dynamic_level_norm(tensor: torch.Tensor) -> torch.Tensor:
        normalized_shape = tuple(int(dim) for dim in tensor.shape[1:])
        if not normalized_shape:
            return tensor
        return F.layer_norm(tensor, normalized_shape)

    @staticmethod
    def _contract_adjacent_axes(tensor: torch.Tensor, *, left_axis: int, weight: torch.Tensor) -> torch.Tensor:
        merged = torch.tensordot(tensor, weight, dims=([left_axis, left_axis + 1], [0, 1]))
        return merged.movedim(-1, left_axis)

    @staticmethod
    def _expand_axis(tensor: torch.Tensor, *, parent_axis: int, weight: torch.Tensor) -> torch.Tensor:
        expanded = torch.tensordot(tensor, weight, dims=([parent_axis], [0]))
        return expanded.movedim((-2, -1), (parent_axis, parent_axis + 1))


class PhysicalQubitTTNEncoder(_PhysicalTTNBase):
    def __init__(
        self,
        *,
        num_qubits: int = config.N_QUBITS,
        d_model: int = config.D_MODEL,
        bond_dim: int = config.TTN_BOND_DIM,
        root_dim: int = config.TTN_ROOT_DIM,
        use_bond_cap: bool = config.TTN_USE_BOND_CAP,
        pairing: str = config.TTN_TREE_PAIRING,
    ):
        tree_spec = TTNTreeSpec.build(
            num_qubits=int(num_qubits),
            pairing=pairing,
            bond_dim=bond_dim,
            root_dim=root_dim,
            use_bond_cap=use_bond_cap,
        )
        super().__init__(num_qubits=num_qubits, tree_spec=tree_spec)
        self.merge_weights = nn.ParameterDict()
        for level_merges in self.tree_spec.merges_by_level:
            for merge_spec in level_merges:
                scale = 1.0 / math.sqrt(max(1, merge_spec.left_dim * merge_spec.right_dim))
                self.merge_weights[str(merge_spec.parent_node_id)] = nn.Parameter(
                    torch.randn(
                        merge_spec.left_dim,
                        merge_spec.right_dim,
                        merge_spec.parent_dim,
                        dtype=torch.float32,
                    )
                    * scale
                )
        self.output_norm = nn.LayerNorm(self.root_dim * 2)
        self.output_projection = nn.Linear(self.root_dim * 2, int(d_model))

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x_complex):
            raise ValueError(
                f"PhysicalQubitTTNEncoder expected complex input, got dtype={x_complex.dtype}"
            )
        if x_complex.shape[-1] != self.dim_2n:
            raise ValueError(
                f"Expected last dimension 2**num_qubits={self.dim_2n}, got {x_complex.shape[-1]}"
            )

        leading_shape = tuple(int(dim) for dim in x_complex.shape[:-1])
        flat_batch = math.prod(leading_shape) if leading_shape else 1
        x = x_complex.reshape(flat_batch, self.dim_2n).to(torch.complex64)
        x_ri = torch.view_as_real(x).to(torch.float32)
        tensor = x_ri.reshape(flat_batch, *([2] * self.num_qubits), 2)

        for level_merges in self.tree_spec.merges_by_level:
            new_tensor = tensor
            new_nodes_so_far = 0
            for merge_spec in level_merges:
                new_tensor = self._contract_adjacent_axes(
                    new_tensor,
                    left_axis=1 + new_nodes_so_far,
                    weight=self.merge_weights[str(merge_spec.parent_node_id)],
                )
                new_nodes_so_far += 1
            tensor = F.gelu(self._dynamic_level_norm(new_tensor))

        if tensor.shape != (flat_batch, self.root_dim, 2):
            raise RuntimeError(
                f"PhysicalQubitTTNEncoder produced invalid root shape {tuple(tensor.shape)}, "
                f"expected ({flat_batch}, {self.root_dim}, 2)"
            )

        root_flat = tensor.reshape(flat_batch, self.root_dim * 2)
        hidden = self.output_projection(self.output_norm(root_flat))
        return hidden.reshape(*leading_shape, hidden.shape[-1])


class PhysicalQubitTTNDecoder(_PhysicalTTNBase):
    def __init__(
        self,
        *,
        num_qubits: int = config.N_QUBITS,
        d_model: int = config.D_MODEL,
        bond_dim: int = config.TTN_BOND_DIM,
        root_dim: int = config.TTN_ROOT_DIM,
        use_bond_cap: bool = config.TTN_USE_BOND_CAP,
        pairing: str = config.TTN_TREE_PAIRING,
    ):
        tree_spec = TTNTreeSpec.build(
            num_qubits=int(num_qubits),
            pairing=pairing,
            bond_dim=bond_dim,
            root_dim=root_dim,
            use_bond_cap=use_bond_cap,
        )
        super().__init__(num_qubits=num_qubits, tree_spec=tree_spec)
        self.initial_projection = nn.Linear(int(d_model), self.root_dim * 2)
        self.root_norm = nn.LayerNorm(self.root_dim * 2)
        self.split_weights = nn.ParameterDict()
        for level_merges in self.tree_spec.merges_by_level:
            for merge_spec in level_merges:
                scale = 1.0 / math.sqrt(max(1, merge_spec.parent_dim))
                self.split_weights[str(merge_spec.parent_node_id)] = nn.Parameter(
                    torch.randn(
                        merge_spec.parent_dim,
                        merge_spec.left_dim,
                        merge_spec.right_dim,
                        dtype=torch.float32,
                    )
                    * scale
                )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.initial_projection.in_features:
            raise ValueError(
                f"Expected last dimension d_model={self.initial_projection.in_features}, got {hidden.shape[-1]}"
            )

        leading_shape = tuple(int(dim) for dim in hidden.shape[:-1])
        flat_batch = math.prod(leading_shape) if leading_shape else 1
        root_flat = self.root_norm(self.initial_projection(hidden.reshape(flat_batch, hidden.shape[-1]).to(torch.float32)))
        tensor = F.gelu(root_flat.reshape(flat_batch, self.root_dim, 2))

        for level_index in range(len(self.tree_spec.merges_by_level) - 1, -1, -1):
            current_nodes = self.tree_spec.levels[level_index + 1]
            restored_nodes: list[int] = []
            new_tensor = tensor
            for node_id in current_nodes:
                merge_spec = next(
                    (spec for spec in self.tree_spec.merges_by_level[level_index] if spec.parent_node_id == node_id),
                    None,
                )
                if merge_spec is None:
                    restored_nodes.append(node_id)
                    continue
                new_tensor = self._expand_axis(
                    new_tensor,
                    parent_axis=1 + len(restored_nodes),
                    weight=self.split_weights[str(merge_spec.parent_node_id)],
                )
                restored_nodes.extend([merge_spec.left_node_id, merge_spec.right_node_id])

            tensor = new_tensor
            if level_index > 0:
                tensor = F.gelu(self._dynamic_level_norm(tensor))

        expected_shape = (flat_batch, *([2] * self.num_qubits), 2)
        if tensor.shape != expected_shape:
            raise RuntimeError(
                f"PhysicalQubitTTNDecoder produced invalid tensor shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}"
            )

        x_ri_flat = tensor.reshape(flat_batch, self.dim_2n, 2)
        out_complex = torch.complex(x_ri_flat[..., 0], x_ri_flat[..., 1]).to(torch.complex64)
        return out_complex.reshape(*leading_shape, self.dim_2n)

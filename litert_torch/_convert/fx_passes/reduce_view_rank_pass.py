# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pass to keep view-chain intermediates within the GPU delegate's rank cap.

Concretely, this rewrites

    (1,56,56,96) -> view(1,8,7,8,7,96) -> permute -> clone -> view(3136,96)

into

    reshape(8,7,8,672) -> permute(0,2,1,3) -> reshape(3136,96)

which computes the same thing without ever exceeding rank 4.

Why
---
The ML Drift GPU delegate caps `RESHAPE` and `TRANSPOSE` at rank 5 and
`FULLY_CONNECTED` at rank 4. Two common PyTorch idioms blow past that with
tensors that are *transient*: they exist only inside a run of pure view ops and
are flattened away again immediately.

  * Packed QKV projection. `torch.nn.functional._in_projection_packed` (used by
    `nn.MultiheadAttention` and `F.scaled_dot_product_attention`) unpacks the
    fused projection with `unsqueeze -> permute -> squeeze`, briefly going to
    rank 5 even though the inserted axis has size 1.

  * Block transpose / window partition (Swin, MaxViT, Twins, XCiT). An axis is
    split into several, the pieces are permuted, then everything is flattened
    back:

        x.view(B, H//M, M, W//M, M, C).permute(0, 1, 3, 2, 4, 5).reshape(...)

    which briefly goes to rank 6.

Both are the same thing: a maximal chain of pure view ops whose *net* effect is
a single permutation of a regrouping of the source tensor's axes. Rather than
pattern-matching each idiom, this pass builds one model of the whole chain and
re-emits it at a rank the delegate accepts.

Modeling
--------
Split the source tensor into `_Atom`s, one per non-unit source axis. An atom is
an indivisible run of the source's flattened element order. Track
`groups: list[list[_Atom]]`, one group per current axis, whose concatenation is
the current element order. Each view op is a cheap edit of that state:

  ============== ==========================================================
  `permute`      reorder groups
  `transpose`    swap two groups
  `unsqueeze`    insert an empty group (an empty product is 1)
  `squeeze`      drop an empty group
  `reshape`      re-partition the flattened atom list, splitting an atom
                 when a target axis straddles one
  `clone`        no-op; a copy does not change logical element order
  ============== ==========================================================

Nothing here inspects tensor data; only shapes are read. The correctness
criterion throughout is *element order*: a view chain computes nothing, it only
decides what sequence you get when you flatten the result, so any rewrite that
preserves that sequence is equivalent.

Re-emission
-----------
Two axes may be merged into one exactly when they are adjacent in the source
*and* adjacent, in the same order, in the result. Scanning the source atom
sequence for that condition yields the coarsest legal partition, and every
legal partition is a refinement of it. So the pass starts from the source's own
axes (refined wherever they straddle a coarse boundary) and merges only as far
as the rank cap demands -- it does not hunt for the theoretical minimum rank,
because leaving axes that already fit undisturbed keeps the emitted shape close
to the original. The chain is then rewritten as

    reshape(pre_shape) -> permute(perm) -> reshape(out_shape)

dropping whichever of the three is the identity. The packed-QKV chain needs no
merging at all, so it collapses to a bare `permute`; the block-transpose chain
merges down to rank 4.

A copy at the *end* of a chain is excluded from the rewrite. Unlike one in the
middle, it is not incidental: it is there to materialize a layout that later
ops depend on, so it is left in place to re-read the new, lower-rank result.

Scope
-----
Static shapes only. A chain with a symbolic extent anywhere is declined and
left exactly as it was, because deciding whether two symbolic extents describe
the same number of elements needs machinery no model has yet demanded. Handling
it is a follow-up.

This is a numerical no-op. It only fires when the chain actually exceeds
`max_rank` and the rewrite strictly lowers the peak rank, so graphs that are
already GPU-clean are left untouched.
"""

import dataclasses
import math

from litert_torch import fx_infra
import torch

aten = torch.ops.aten

# Debug-info meta fields that map a node back to its source model code.
_PROVENANCE_META_KEYS = (
    "stack_trace",
    "nn_module_stack",
    "source_fn_stack",
    "from_node",
)

# Ops that reinterpret a tensor's shape without moving data.
_RESHAPE_TARGETS = (
    aten.reshape.default,
    aten.view.default,
    aten._unsafe_view.default,
)
_PERMUTE_TARGETS = (aten.permute.default,)
_TRANSPOSE_TARGETS = (aten.transpose.int,)
_UNSQUEEZE_TARGETS = (aten.unsqueeze.default,)
_SQUEEZE_TARGETS = (
    aten.squeeze.default,
    aten.squeeze.dim,
    aten.squeeze.dims,
)
# Decomposing a reshape of a permuted (non-contiguous) tensor splits it into a
# contiguous copy followed by a true view, so a copy can sit in the middle of a
# chain. One there is incidental and is absorbed; one at the end of a chain is
# load-bearing and is left alone. See `_rewrite_chain`.
_COPY_TARGETS = (
    aten.clone.default,
    aten.contiguous.default,
)

_VIEW_TARGETS = frozenset(
    _RESHAPE_TARGETS
    + _PERMUTE_TARGETS
    + _TRANSPOSE_TARGETS
    + _UNSQUEEZE_TARGETS
    + _SQUEEZE_TARGETS
    + _COPY_TARGETS
)


def _static_shape(val) -> tuple[int, ...] | None:
  """Returns `val`'s shape, or None if it is missing or not fully static.

  `torch.SymInt` is not an `int` subclass and concrete fake-tensor dims are
  plain `int`s, so the isinstance check is exact. It also has to come before
  any arithmetic or comparison on the dim: `==` on a `SymInt` installs a shape
  guard as a side effect, which would silently specialize the model.
  """
  if val is None or not hasattr(val, "shape"):
    return None
  shape = tuple(val.shape)
  if any(not isinstance(dim, int) for dim in shape):
    return None
  return shape


class _Atom:
  """An indivisible run of the source tensor's flattened element order.

  Compared by identity, never by value: two axes that both happen to have
  extent 7 are different atoms, and `list.index`, `in` and dict keys all rely
  on telling them apart. Hence no `__eq__`.

  Attributes:
    size: the number of elements. Always >= 2; unit axes carry no elements and
      are modeled as empty groups rather than atoms.
  """

  __slots__ = ("size",)

  def __init__(self, size: int):
    self.size = size


def _extent(group: list[_Atom]) -> int:
  """Returns the extent of an axis, i.e. how many elements lie along it."""
  return math.prod(atom.size for atom in group)


@dataclasses.dataclass
class _ChainModel:
  """State of a view chain, as a partition of the source's element order.

  Attributes:
    atoms: every atom, in *source* element order.
    source_axes: the source tensor's own axes, as groups of atoms. Empty groups
      are unit axes. Flattening this reproduces `atoms`.
    groups: the *current* axes, as groups of atoms. Flattening this gives the
      current element order, which is a permutation of `atoms`.
  """

  atoms: list[_Atom]
  source_axes: list[list[_Atom]]
  groups: list[list[_Atom]]

  def flatten(self) -> list[_Atom]:
    return [atom for group in self.groups for atom in group]

  def split(self, atom: _Atom, head_size: int, tail_size: int):
    """Replaces `atom` with two smaller atoms everywhere it is recorded."""
    head = _Atom(head_size)
    tail = _Atom(tail_size)
    index = self.atoms.index(atom)
    self.atoms[index : index + 1] = [head, tail]
    for axis in self.source_axes:
      if atom in axis:
        at = axis.index(atom)
        axis[at : at + 1] = [head, tail]
        break
    return head, tail


def _init_model(shape: tuple[int, ...]) -> _ChainModel:
  """Builds the initial chain state for a tensor of the given shape."""
  atoms = []
  source_axes = []
  for dim in shape:
    if dim == 1:
      # A unit axis holds no elements, so it constrains nothing.
      source_axes.append([])
      continue
    atom = _Atom(dim)
    atoms.append(atom)
    source_axes.append([atom])
  return _ChainModel(
      atoms=atoms,
      source_axes=source_axes,
      groups=[list(axis) for axis in source_axes],
  )


def _apply_reshape(model: _ChainModel, target_shape: tuple[int, ...]) -> bool:
  """Re-partitions the flattened atom list to match `target_shape`."""
  flat = model.flatten()
  index = 0
  new_groups = []
  for dim in target_shape:
    if dim == 1:
      new_groups.append([])
      continue
    group = []
    remaining = dim
    while remaining > 1:
      if index >= len(flat):
        return False
      atom = flat[index]
      if atom.size <= remaining:
        if remaining % atom.size:
          return False
        group.append(atom)
        remaining //= atom.size
        index += 1
      else:
        # The target axis ends partway through this atom, so cut the atom.
        if atom.size % remaining:
          return False
        head, tail = model.split(atom, remaining, atom.size // remaining)
        flat[index : index + 1] = [head, tail]
        group.append(head)
        remaining = 1
        index += 1
    new_groups.append(group)

  if index != len(flat):
    return False
  model.groups = new_groups
  return True


def _apply_squeeze(model: _ChainModel, node: torch.fx.Node) -> bool:
  """Drops unit axes. Squeezing a non-unit axis is a no-op in PyTorch."""
  rank = len(model.groups)
  if node.target == aten.squeeze.default:
    dims = range(rank)
  elif node.target == aten.squeeze.dim:
    dims = [node.args[1] % rank]
  else:  # aten.squeeze.dims
    dims = [dim % rank for dim in node.args[1]]
  drop = {dim for dim in dims if not model.groups[dim]}
  model.groups = [g for i, g in enumerate(model.groups) if i not in drop]
  return True


def _apply(model: _ChainModel, node: torch.fx.Node) -> bool:
  """Advances `model` through `node`. Returns False if it cannot be modeled."""
  target = node.target
  if target in _COPY_TARGETS:
    return True
  if target in _PERMUTE_TARGETS:
    perm = list(node.args[1])
    if sorted(perm) != list(range(len(model.groups))):
      return False
    model.groups = [model.groups[p] for p in perm]
    return True
  if target in _TRANSPOSE_TARGETS:
    rank = len(model.groups)
    i, j = node.args[1] % rank, node.args[2] % rank
    model.groups[i], model.groups[j] = model.groups[j], model.groups[i]
    return True
  if target in _UNSQUEEZE_TARGETS:
    model.groups.insert(node.args[1] % (len(model.groups) + 1), [])
    return True
  if target in _SQUEEZE_TARGETS:
    return _apply_squeeze(model, node)
  if target in _RESHAPE_TARGETS:
    # Read the resolved shape off the node rather than its argument, which may
    # contain a -1 placeholder.
    shape = _static_shape(node.meta.get("val"))
    return shape is not None and _apply_reshape(model, shape)
  return False


def _consistent_with(model: _ChainModel, val) -> bool:
  """Cross-checks the modeled axes against what PyTorch actually computed."""
  shape = _static_shape(val)
  if shape is None or len(model.groups) != len(shape):
    return False
  return all(_extent(group) == dim for group, dim in zip(model.groups, shape))


def _coarse_ids(model: _ChainModel, positions: dict) -> dict:
  """Labels each atom with its block in the coarsest legal partition.

  Two adjacent source atoms may share an axis exactly when they are still
  adjacent, and in the same order, in the result.
  """
  ids = {}
  block = 0
  for i, atom in enumerate(model.atoms):
    if i and positions[atom] != positions[model.atoms[i - 1]] + 1:
      block += 1
    ids[atom] = block
  return ids


def _initial_blocks(model: _ChainModel, coarse: dict) -> list[list[_Atom]]:
  """Returns the source axes, cut wherever they straddle a coarse block."""
  blocks = []
  for axis in model.source_axes:
    if not axis:
      blocks.append([])
      continue
    current = [axis[0]]
    for atom in axis[1:]:
      if coarse[atom] == coarse[current[-1]]:
        current.append(atom)
      else:
        blocks.append(current)
        current = [atom]
    blocks.append(current)
  return blocks


def _mergeable(blocks: list[list[_Atom]], coarse: dict, i: int) -> bool:
  """Returns whether axes `i` and `i + 1` may be fused into one."""
  head, tail = blocks[i], blocks[i + 1]
  # A unit axis holds no elements, so it can always be absorbed.
  if not head or not tail:
    return True
  return coarse[head[-1]] == coarse[tail[0]]


def _merge_blocks(
    blocks: list[list[_Atom]], coarse: dict, max_rank: int
) -> list[list[_Atom]] | None:
  """Fuses adjacent axes until the rank fits, or gives up."""
  blocks = [list(block) for block in blocks]
  while len(blocks) > max_rank:
    for i in range(len(blocks) - 1):
      if _mergeable(blocks, coarse, i):
        blocks[i : i + 2] = [blocks[i] + blocks[i + 1]]
        break
    else:
      return None
  return blocks


def _permutation(blocks: list[list[_Atom]], positions: dict) -> list[int]:
  """Returns the permutation that puts `blocks` into result order.

  Unit axes carry no elements, so they are unconstrained; parking them at the
  end keeps the permutation a bijection and the trailing reshape fixes up the
  final shape.
  """
  ordered = sorted(
      (positions[block[0]], i) for i, block in enumerate(blocks) if block
  )
  perm = [i for _, i in ordered]
  perm += [i for i, block in enumerate(blocks) if not block]
  return perm


def _copy_provenance_meta(src: torch.fx.Node, dst: torch.fx.Node):
  for key in _PROVENANCE_META_KEYS:
    if key in src.meta:
      dst.meta[key] = src.meta[key]


def _is_view_node(node) -> bool:
  return (
      isinstance(node, torch.fx.Node)
      and node.op == "call_function"
      and node.target in _VIEW_TARGETS
  )


def _collect_chain(node: torch.fx.Node) -> list[torch.fx.Node]:
  """Extends a chain forward while each node feeds exactly one view op."""
  chain = [node]
  while len(chain[-1].users) == 1:
    user = next(iter(chain[-1].users))
    if not _is_view_node(user):
      break
    chain.append(user)
  return chain


def _plan(
    chain: list[torch.fx.Node], src_shape: tuple[int, ...], max_rank: int
) -> tuple[list[list[_Atom]], list[int]] | None:
  """Plans the rewrite of `chain`, or returns None if it cannot be rewritten.

  Returns:
    `(blocks, perm)`, where `blocks` is the axis partition to reshape the
    source into and `perm` reorders those axes into result order.
  """
  model = _init_model(src_shape)
  for node in chain:
    if not _apply(model, node):
      return None
    if not _consistent_with(model, node.meta["val"]):
      return None

  flat_out = model.flatten()
  if len(flat_out) != len(model.atoms):
    return None
  positions = {atom: i for i, atom in enumerate(flat_out)}

  coarse = _coarse_ids(model, positions)
  blocks = _merge_blocks(_initial_blocks(model, coarse), coarse, max_rank)
  if blocks is None:
    return None
  return blocks, _permutation(blocks, positions)


def _rewrite_chain(
    graph_module: torch.fx.GraphModule,
    chain: list[torch.fx.Node],
    max_rank: int,
) -> bool:
  """Re-emits `chain` at lower rank. Returns True if the graph was rewritten."""
  # A copy at the end of a chain is there to materialize a layout, not to
  # reshape anything, and downstream `view` ops rely on the result being
  # contiguous. Leave it in place and rewrite only the chain ahead of it; it
  # will simply re-read the new, lower-rank result.
  while chain and chain[-1].target in _COPY_TARGETS:
    chain = chain[:-1]
  if not chain:
    return False

  src = chain[0].args[0]
  if not isinstance(src, torch.fx.Node):
    return False
  src_shape = _static_shape(src.meta.get("val"))
  last_shape = _static_shape(chain[-1].meta.get("val"))
  if src_shape is None or last_shape is None:
    return False
  # Every intermediate must be static too, so that the model can be checked
  # against it and so that no dim comparison installs a shape guard.
  shapes = [_static_shape(node.meta.get("val")) for node in chain]
  if any(shape is None for shape in shapes):
    return False

  old_rank = max([len(src_shape)] + [len(shape) for shape in shapes])
  if old_rank <= max_rank:
    # Already GPU-clean; leave it alone.
    return False
  if len(src_shape) > max_rank or len(last_shape) > max_rank:
    # The chain's endpoints are over the cap, so no rewrite of its interior
    # can bring the peak rank down.
    return False

  plan = _plan(chain, src_shape, max_rank)
  if plan is None:
    return False
  blocks, perm = plan
  if max(len(blocks), len(last_shape), len(src_shape)) >= old_rank:
    return False

  pre_shape = [_extent(block) for block in blocks]
  if tuple(pre_shape) == src_shape:
    # The partition is just the source's own axes; no reshape needed.
    pre_shape = None
  need_permute = perm != list(range(len(perm)))

  # Decide all three steps up front, against fake tensors, so that a late
  # bail-out never leaves a half-built chain behind in the graph.
  src_val = src.meta["val"]
  with src_val.fake_mode:
    pre_val = src_val
    if pre_shape is not None:
      pre_val = aten.reshape.default(src_val, pre_shape)
    mid_val = pre_val
    if need_permute:
      mid_val = aten.permute.default(pre_val, perm)
    out_shape = None
    if _static_shape(mid_val) != last_shape:
      out_shape = list(last_shape)
      out_val = aten.reshape.default(mid_val, out_shape)

  if pre_shape is None and not need_permute and out_shape is None:
    # The chain was already a no-op; there is nothing to emit.
    return False

  graph = graph_module.graph
  last = chain[-1]
  with graph.inserting_before(last):
    current = src
    if pre_shape is not None:
      current = graph.call_function(aten.reshape.default, (current, pre_shape))
      current.meta["val"] = pre_val
      _copy_provenance_meta(last, current)
    if need_permute:
      current = graph.call_function(aten.permute.default, (current, perm))
      current.meta["val"] = mid_val
      _copy_provenance_meta(last, current)
    if out_shape is not None:
      current = graph.call_function(aten.reshape.default, (current, out_shape))
      current.meta["val"] = out_val
      _copy_provenance_meta(last, current)

  # The rewritten chain produces exactly the tensor the old one did, so adopt
  # its metadata rather than keeping the recomputed approximation.
  current.meta["val"] = last.meta["val"]
  if "tensor_meta" in last.meta:
    current.meta["tensor_meta"] = last.meta["tensor_meta"]

  last.replace_all_uses_with(current)
  return True


class ReduceViewRankPass(fx_infra.PassBase):
  """Re-emits pure view chains at a rank the GPU delegate accepts."""

  def __init__(self, max_rank: int = 4):
    super().__init__()
    self._max_rank = max_rank

  def call(self, graph_module: torch.fx.GraphModule):
    modified = False
    visited = set()
    # Snapshot the nodes since the graph is mutated during iteration.
    for node in list(graph_module.graph.nodes):
      if node in visited or not _is_view_node(node):
        continue
      chain = _collect_chain(node)
      visited.update(chain)
      if _rewrite_chain(graph_module, chain, self._max_rank):
        modified = True

    if modified:
      graph_module.graph.eliminate_dead_code()
      graph_module.graph.lint()
      graph_module.recompile()
    return fx_infra.PassResult(graph_module, modified)

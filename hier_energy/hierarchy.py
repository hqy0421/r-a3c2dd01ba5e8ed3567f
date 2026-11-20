"""Declared meter hierarchy: validation, leaf sets and the summing matrix."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class HierarchyError(ValueError):
    """The declared tree is malformed (not a single rooted tree)."""


@dataclass(frozen=True)
class Hierarchy:
    """A declared meter tree mapping each parent to its ordered children.

    Terminal nodes are the leaf meters that carry raw signals.  Every
    non-leaf node's value is *defined* as the sum of the leaves beneath it;
    this class is the single source of truth for that structure.
    """

    tree: dict  # {parent: [child, ...]}

    def __post_init__(self):
        if not isinstance(self.tree, dict) or not self.tree:
            raise HierarchyError("tree must be a non-empty dict of parent -> children")
        tree = {}
        for parent, children in self.tree.items():
            children = [str(c) for c in children]
            if not children:
                raise HierarchyError(f"parent {parent!r} declares no children")
            if len(set(children)) != len(children):
                raise HierarchyError(f"parent {parent!r} lists a child twice")
            tree[str(parent)] = children
        object.__setattr__(self, "tree", tree)

        parents = list(tree)
        children = [c for cs in tree.values() for c in cs]
        seen_children = set()
        for c in children:
            if c in seen_children:
                raise HierarchyError(f"node {c!r} has more than one parent")
            seen_children.add(c)
        roots = [p for p in parents if p not in seen_children]
        if len(roots) != 1:
            raise HierarchyError(f"expected exactly one root, found {sorted(roots)}")
        root = roots[0]

        # Reachability from the single root must cover every declared node;
        # together with in-degree <= 1 this rules out detached cycles.
        order, visited, stack = [], set(), [root]
        while stack:
            node = stack.pop()
            if node in visited:
                raise HierarchyError(f"cycle detected at node {node!r}")
            visited.add(node)
            order.append(node)
            stack.extend(reversed(tree.get(node, [])))
        all_nodes = set(parents) | set(children)
        if visited != all_nodes:
            raise HierarchyError(
                f"nodes unreachable from root {root!r}: {sorted(all_nodes - visited)}"
            )

        leaves = [n for n in order if n not in tree]
        if not leaves:
            raise HierarchyError("tree declares no leaf meters")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "nodes", order)  # pre-order, root first
        object.__setattr__(self, "leaves", leaves)
        object.__setattr__(
            self, "_leaf_sets", {n: tuple(self._collect_leaves(n)) for n in order}
        )

    # ------------------------------------------------------------------ #
    def _collect_leaves(self, node):
        if node not in self.tree:
            return [node]
        out = []
        for child in self.tree[node]:
            out.extend(self._collect_leaves(child))
        return out

    def children(self, node):
        return list(self.tree.get(node, []))

    def leaves_under(self, node):
        """Ordered leaf meters below ``node`` (the node itself if it is a leaf)."""
        if node not in self._leaf_sets:
            raise KeyError(f"unknown node {node!r}")
        return list(self._leaf_sets[node])

    def summing_matrix(self) -> pd.DataFrame:
        """S with rows = nodes (pre-order), columns = leaves; S[n, l] = 1 iff
        leaf ``l`` lies under node ``n``.  Node values are ``S @ x`` for leaf
        values ``x``."""
        leaves = self.leaves
        data = {
            n: [1.0 if l in self._leaf_sets[n] else 0.0 for l in leaves]
            for n in self.nodes
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=leaves)

    def render(self) -> str:
        """ASCII rendering of the declared tree (leaf meters marked with *)."""
        lines = [self.root]

        def walk(node, prefix):
            kids = self.tree.get(node, [])
            for i, child in enumerate(kids):
                last = i == len(kids) - 1
                marker = "└── " if last else "├── "
                leaf = " *" if child not in self.tree else ""
                lines.append(f"{prefix}{marker}{child}{leaf}")
                walk(child, prefix + ("    " if last else "│   "))

        walk(self.root, "")
        return "\n".join(lines)

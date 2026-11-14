"""Graph Executor — dependency resolution and level-by-level execution.

Nodes in the same topological level are independent and are run
concurrently (asyncio.gather); levels run in dependency order.
Phase 1 node work is in-memory, so concurrency is structural — the
same executor shape carries over to real network-bound models.
"""
from __future__ import annotations

import asyncio

from backend.execution.node_executor import NodeExecutor
from backend.schemas import ExecutionPlan, NodeResult


def _levels(plan: ExecutionPlan) -> list[list]:
    done: set[str] = set()
    remaining = list(plan.nodes)
    levels = []
    while remaining:
        level = [n for n in remaining if set(n.depends_on) <= done]
        if not level:  # validator prevents cycles; defensive only
            raise ValueError("Unresolvable node dependencies")
        levels.append(level)
        done |= {n.node_id for n in level}
        remaining = [n for n in remaining if n.node_id not in done]
    return levels


class GraphExecutor:
    def __init__(self, node_executor: NodeExecutor | None = None):
        self.node_executor = node_executor or NodeExecutor()

    async def execute(self, plan: ExecutionPlan, query: str) -> dict[str, NodeResult]:
        results: dict[str, NodeResult] = {}
        for level in _levels(plan):
            async def run(node):
                return node.node_id, self.node_executor.execute(
                    node, query, dict(results))
            for node_id, result in await asyncio.gather(*(run(n) for n in level)):
                results[node_id] = result
        return results

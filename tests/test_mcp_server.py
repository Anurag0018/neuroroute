"""MCP server: a real protocol round-trip over stdio — spawn the
server as a subprocess, connect an MCP client, list its tools and
call them. Skipped when the optional mcp package is absent."""
import asyncio
import json
import os
import sys

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

EXPECTED_TOOLS = {"neuroroute_query", "neuroroute_route_preview",
                  "neuroroute_calculator", "neuroroute_datetime",
                  "neuroroute_web_search", "neuroroute_metrics",
                  "neuroroute_health"}


async def _roundtrip() -> dict:
    env = {**os.environ, "NR_MODEL_MODE": "mock",
           "NR_DECISION_MODEL": "rules", "NR_PLAN_MODEL": "strategy",
           "NR_AUDIO_MODE": "mock", "NR_VECTOR_STORE": "memory"}
    env.pop("NR_API_KEY", None)
    params = StdioServerParameters(command=sys.executable,
                                   args=["mcp_server.py"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            calc = await session.call_tool(
                "neuroroute_calculator",
                {"query": "What is 1,247 * 38 + 5,600?"})
            health = await session.call_tool("neuroroute_health", {})
            query = await session.call_tool(
                "neuroroute_query", {"query": "What is Python?"})
            preview = await session.call_tool(
                "neuroroute_route_preview",
                {"query": "What is Python?"})
            return {
                "tools": {t.name for t in tools.tools},
                "calc": calc.content[0].text,
                "health": json.loads(health.content[0].text),
                "query": json.loads(query.content[0].text),
                "preview": json.loads(preview.content[0].text),
            }


def test_mcp_protocol_roundtrip():
    out = asyncio.run(_roundtrip())
    assert out["tools"] == EXPECTED_TOOLS
    assert "52,986" in out["calc"]
    assert out["health"]["interface"] == "mcp"
    assert out["health"]["model_mode"] == "mock"
    assert out["query"]["strategy"] == "direct_small"
    assert out["query"]["answer"]
    assert out["preview"]["strategy"] == "direct_small"
    assert out["preview"]["nodes"]

"""NeuroRoute pipeline — wires the subsystems together:

analyse -> estimate complexity -> decision model proposes a plan ->
validator authorises it -> graph executor runs it -> confidence ->
(optional ad-hoc verification) -> escalation if required -> trace.

This is the single orchestration entry point used by the API.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from backend.execution.graph_executor import GraphExecutor
from backend.execution.node_executor import _answer_from, _docs_from
from backend.intelligence import complexity_estimator, input_analyzer
from backend.intelligence.decision_model import default_decision_model
from backend.planning.graph_generator import build_plan
from backend.planning.graph_validator import PlanRejected, validate_plan
from backend.schemas import Budget, BudgetUsage, PlanNode
from backend.observability.execution_tracer import ExecutionTracer
from backend.validation import confidence_estimator
from backend.validation.escalation_controller import escalation_target


class NeuroRoutePipeline:
    def __init__(self, tracer: ExecutionTracer | None = None,
                 decision_model=None):
        # Default honours NR_DECISION_MODEL (rules | llm); tests and the
        # API may inject a specific model.
        self.decision_model = decision_model or default_decision_model()
        self.executor = GraphExecutor()
        self.tracer = tracer or ExecutionTracer()

    async def process(self, query: str, attachments: list[str] | None = None,
                      budget: Budget | None = None,
                      request_id: str | None = None) -> dict:
        budget = budget or Budget()
        analysis = input_analyzer.analyze(query, attachments, request_id)
        complexity = complexity_estimator.estimate(analysis)
        plan = validate_plan(
            self.decision_model.propose_plan(analysis, complexity, budget), budget)

        usage = BudgetUsage()
        # Charge the decision step itself: an LLM router's tokens and
        # latency are part of NeuroRoute's cost (zero for rule-based).
        decision_usage = getattr(self.decision_model, "last_usage", None)
        if decision_usage is not None:
            usage.add(tokens=decision_usage.tokens,
                      cost_usd=decision_usage.cost_usd,
                      latency_ms=decision_usage.latency_ms, model_calls=1)
        decision_info = {
            "source": getattr(self.decision_model, "last_source", "rules"),
            "tier": getattr(self.decision_model, "tier", None),
            "tokens": decision_usage.tokens if decision_usage else 0,
            "latency_ms": decision_usage.latency_ms if decision_usage else 0,
        }
        escalation_history: list[dict] = []
        answer, confidence, node_results, verified_adhoc = "", None, {}, False

        while True:
            # Ad-hoc verification belongs to THIS iteration's results;
            # reset it so an escalation re-run never references the
            # previous plan's node_results (KeyError regression).
            verified_adhoc = False
            node_results = await self.executor.execute(plan, query)
            for r in node_results.values():
                usage.add(tokens=r.tokens, cost_usd=r.cost_usd,
                          latency_ms=r.latency_ms,
                          model_calls=1 if r.kind in ("generate", "verify") else 0)
            answer = _answer_from(node_results)
            docs = _docs_from(node_results)
            verification = next(
                (r.output for r in node_results.values() if r.kind == "verify"), None)
            relevance = max((d["relevance"] for d in docs), default=0.0)
            confidence = confidence_estimator.estimate(
                has_retrieval=any(n.kind == "retrieve" for n in plan.nodes),
                retrieval_relevance=relevance, verification=verification)

            # Middle band: if no verifier node ran, run one ad-hoc check
            # and re-score. The ad-hoc check is evidence-only (local
            # coverage measurement, no extra model call): per the
            # architecture, verification is evidence-based, not a second
            # model's opinion. The verified_large path keeps its in-graph
            # verifier model call for high-risk / high-complexity work.
            if confidence.decision == "verify" and verification is None:
                vnode = PlanNode("verify_adhoc", "verify", "verifier")
                vres = self.executor.node_executor.execute(
                    vnode, query, dict(node_results), use_llm_verifier=False)
                usage.add(tokens=vres.tokens, cost_usd=vres.cost_usd,
                          latency_ms=vres.latency_ms, model_calls=0)
                node_results["verify_adhoc"] = vres
                verified_adhoc = True
                confidence = confidence_estimator.estimate(
                    has_retrieval=any(n.kind == "retrieve" for n in plan.nodes),
                    retrieval_relevance=relevance, verification=vres.output)

            target = escalation_target(confidence, plan.strategy, usage, budget)
            if target is None:
                break
            escalation_history.append({
                "from": plan.strategy, "to": target,
                "confidence_before": confidence.confidence_score,
                "reason": "confidence below escalation threshold"})
            usage.escalations += 1
            try:
                plan = validate_plan(
                    build_plan(target, analysis, complexity, budget), budget)
            except PlanRejected:
                break

        trace = {
            "request_id": analysis.request_id,
            "query": query,
            "analysis": asdict(analysis),
            "complexity": asdict(complexity),
            "strategy": plan.strategy,
            "rationale": plan.rationale,
            "decision": decision_info,
            "execution_graph": [
                {"node_id": n.node_id, "kind": n.kind, "model_tier": n.model_tier,
                 "depends_on": n.depends_on,
                 "status": node_results[n.node_id].status,
                 "latency_ms": node_results[n.node_id].latency_ms}
                for n in plan.nodes] + (
                [{"node_id": "verify_adhoc", "kind": "verify",
                  "model_tier": "verifier", "depends_on": ["generate"],
                  "status": node_results["verify_adhoc"].status,
                  "latency_ms": node_results["verify_adhoc"].latency_ms}]
                if verified_adhoc else []),
            "answer": answer,
            "confidence": asdict(confidence),
            "uncertain": confidence.decision == "escalate",
            "escalation_history": escalation_history,
            "metrics": {
                "tokens": usage.tokens,
                "cost_usd": round(usage.cost_usd, 6),
                "latency_ms": usage.latency_ms,
                "model_calls": usage.model_calls,
                "escalations": usage.escalations,
                "estimated_cost_usd": plan.estimated_cost_usd,
            },
        }
        self.tracer.record(trace)
        return trace

    def process_sync(self, *args, **kwargs) -> dict:
        return asyncio.run(self.process(*args, **kwargs))

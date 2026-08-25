from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from corpus.agent.graph import build_graph
from corpus.agent.llm import LLMProvider, default_provider
from corpus.agent.tracing import SQLiteTraceHandler, new_trace_id
from corpus.eval.dataset import Case, load_golden
from corpus.eval.judge import JudgeVerdict, judge_answer, judge_decline
from corpus.store.eval import finish_run, insert_case, start_run


@dataclass
class CaseResult:
    case: Case
    answer: str
    retrieval_hit: bool | None  # None = not applicable to this category
    verdict: JudgeVerdict | None  # None = no LLM judge call for this category (conversational)
    passed: bool
    trace_id: str


@dataclass
class EvalRunResult:
    run_id: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _bool_str(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "true" if value else "false"


def _run_case(llm: LLMProvider, graph, case: Case) -> CaseResult:
    trace_id = new_trace_id()
    tracer = SQLiteTraceHandler(trace_id, case.query)
    try:
        result = graph.invoke(
            {"query": case.query, "loop_count": 0, "messages": []},
            config={
                "callbacks": [tracer],
                "run_id": uuid.UUID(trace_id),
                "metadata": {"trace_id": trace_id, "eval_case_id": case.id},
            },
        )
    except Exception:
        tracer.finish("error")
        raise
    tracer.finish("ok")

    answer = result.get("answer", "")

    if case.category in ("factual", "multi_hop"):
        docs = result.get("docs") or []
        retrieved_sources = {d.metadata.get("source") for d in docs}
        retrieval_hit = any(src in retrieved_sources for src in case.expected_sources)
        verdict = judge_answer(llm, case.query, case.expected_keyphrases, case.notes, answer)
        passed = retrieval_hit and verdict.correct
        return CaseResult(case, answer, retrieval_hit, verdict, passed, trace_id)

    if case.category == "negative":
        verdict = judge_decline(llm, case.query, answer)
        return CaseResult(case, answer, None, verdict, verdict.correct, trace_id)

    # conversational — cheap deterministic check, no LLM judge call
    passed = result.get("route_type") == "direct"
    return CaseResult(case, answer, None, None, passed, trace_id)


def run_eval(golden_path: str | Path = "eval/golden.json") -> EvalRunResult:
    cases = load_golden(golden_path)
    llm = default_provider()
    graph = build_graph(llm=llm)

    run_id = new_trace_id()
    start_run(run_id, _git_sha())

    outcome = EvalRunResult(run_id=run_id)
    for case in cases:
        case_result = _run_case(llm, graph, case)
        outcome.cases.append(case_result)
        insert_case(
            {
                "case_run_id": f"{run_id}:{case.id}",
                "run_id": run_id,
                "case_id": case.id,
                "category": case.category,
                "retrieval_hit": _bool_str(case_result.retrieval_hit),
                "judge_correct": _bool_str(
                    case_result.verdict.correct if case_result.verdict else None
                ),
                "passed": _bool_str(case_result.passed),
                "trace_id": case_result.trace_id,
                "reasoning": case_result.verdict.reasoning if case_result.verdict else "",
            }
        )

    finish_run(run_id, outcome.total, outcome.passed)
    return outcome

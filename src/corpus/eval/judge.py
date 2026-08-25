from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from corpus.agent.llm import LLMProvider


class JudgeVerdict(BaseModel):
    correct: bool = Field(description="Whether the answer meets the stated criterion.")
    reasoning: str = Field(description="One concise sentence explaining the verdict.")


def judge_answer(
    llm: LLMProvider, query: str, expected_keyphrases: list[str], notes: str, answer: str
) -> JudgeVerdict:
    """Reference-based check: does the answer actually cover the expected content?"""
    structured = llm.fast.with_structured_output(JudgeVerdict)
    keyphrases = ", ".join(expected_keyphrases) if expected_keyphrases else "(none listed)"
    return structured.invoke(
        [
            HumanMessage(
                content=(
                    "You are grading an answer from a RAG assistant against a reference. "
                    "Mark correct=true only if the answer substantively covers the expected "
                    "content below — near-exact wording is not required, but the key facts "
                    "must be present and not contradicted. Output nothing except the JSON.\n\n"
                    f"Question: {query}\n\n"
                    f"Expected key phrases/concepts: {keyphrases}\n"
                    f"Additional notes: {notes or '(none)'}\n\n"
                    f"Answer to grade:\n{answer}"
                )
            )
        ]
    )


def judge_decline(llm: LLMProvider, query: str, answer: str) -> JudgeVerdict:
    """Checks the assistant appropriately declined rather than fabricating an answer."""
    structured = llm.fast.with_structured_output(JudgeVerdict)
    return structured.invoke(
        [
            HumanMessage(
                content=(
                    "You are grading whether a RAG assistant correctly declined to answer a "
                    "question its knowledge base has no information on. Mark correct=true only "
                    "if the answer clearly states it lacks relevant information and does NOT "
                    "fabricate a substantive answer or cite sources. Output nothing except the "
                    "JSON.\n\n"
                    f"Question: {query}\n\n"
                    f"Answer to grade:\n{answer}"
                )
            )
        ]
    )

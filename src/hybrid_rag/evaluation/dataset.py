"""Load and validate manually curated RAG evaluation cases."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class EvidenceLocation:
    source: str
    page: int


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_answer: str
    relevant_chunk_ids: frozenset[str]
    expected_evidence: tuple[EvidenceLocation, ...]


def load_evaluation_dataset(path: str | Path) -> list[EvaluationCase]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Evaluation dataset must be a nonempty JSON list")
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for record in records:
        try:
            case_id = record["id"]
            question = record["question"]
            expected_answer = record["expected_answer"]
            chunk_ids = record["relevant_chunk_ids"]
            evidence = record["expected_evidence"]
        except (KeyError, TypeError) as error:
            raise ValueError("Evaluation case is missing required fields") from error
        if (not isinstance(case_id, str) or not case_id.strip()
                or not isinstance(question, str) or not question.strip()
                or not isinstance(expected_answer, str) or not expected_answer.strip()
                or not isinstance(chunk_ids, list) or not chunk_ids
                or any(not isinstance(value, str) or not value for value in chunk_ids)
                or len(set(chunk_ids)) != len(chunk_ids)
                or not isinstance(evidence, list) or not evidence):
            raise ValueError("Evaluation case contains invalid fields")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case id: {case_id}")
        seen_ids.add(case_id)
        locations = []
        for item in evidence:
            if (not isinstance(item, dict) or not isinstance(item.get("source"), str)
                    or type(item.get("page")) is not int):
                raise ValueError("Expected evidence needs source and integer page")
            locations.append(EvidenceLocation(item["source"], item["page"]))
        cases.append(EvaluationCase(
            case_id=case_id,
            question=question,
            expected_answer=expected_answer,
            relevant_chunk_ids=frozenset(chunk_ids),
            expected_evidence=tuple(locations),
        ))
    return cases

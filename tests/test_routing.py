from backend.intelligence import complexity_estimator, input_analyzer
from backend.intelligence.routing_policy import choose_strategy


def route(query, attachments=None):
    a = input_analyzer.analyze(query, attachments)
    c = complexity_estimator.estimate(a)
    return choose_strategy(a, c), a, c


def test_simple_question_takes_small_path():
    strategy, a, c = route("What is Python?")
    assert strategy == "direct_small"
    assert c.complexity_score < 0.35


def test_research_question_takes_rag_path():
    strategy, a, c = route("Research how retrieval augmented generation improves factual accuracy")
    assert strategy in ("rag_medium", "verified_large")
    assert c.retrieval_required


def test_document_research_is_highest_path():
    strategy, a, c = route(
        "Analyse this research paper and compare its findings with the literature",
        ["paper.pdf"])
    assert a.modality == "document"
    assert strategy == "verified_large"


def test_high_risk_forces_verified_path():
    strategy, a, c = route("What dosage is a safe medical prescription for this?")
    assert a.risk_level == "high"
    assert strategy == "verified_large"

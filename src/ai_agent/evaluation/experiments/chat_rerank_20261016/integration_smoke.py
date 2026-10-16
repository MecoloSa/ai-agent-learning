"""用冻结教材候选验证正式hybrid与PDF评测接入；2次套餐调用，不重建索引。"""
import json
import time

from langchain_core.documents import Document

from ai_agent.config import get_settings
from ai_agent.rag import LearningRAG
from ai_agent.evaluation.pdf_rag_eval import evaluate_case
from ai_agent.evaluation.rerank_probe import configured_rerank_options, rerank_candidates
from run import HERE, write_json


def main():
    case = json.loads((HERE / "inputs.local.json").read_text(encoding="utf-8"))[0]
    docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in case["documents"]]
    settings = get_settings()
    assert settings.rag_rerank_enabled and settings.rag_rerank_backend == "chat"
    assert settings.rag_rerank_chat_model == "deepseek-v4.1-flash"
    rag = object.__new__(LearningRAG)
    rag.retrieve_hybrid_candidates = lambda *args, **kwargs: docs
    # 向量回退不应被悄悄当成聊天接入成功；烟测中出现回退就失败。
    rag.retrieve = lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected fallback"))
    started = time.perf_counter()
    hits = rag.retrieve_with_strategy(case["question"], strategy="hybrid", k=8, rerank=True)
    assert hits and all(d.metadata.get("rerank_backend") == "chat" for d in hits)
    assert all("rerank_score" not in d.metadata for d in hits)
    runtime = {"elapsed_ms": round((time.perf_counter() - started) * 1000),
               "ranked_page_ids": [f"{d.metadata['source']}#page={d.metadata['page']}" for d in hits],
               "backend": "chat", "model": settings.rag_rerank_chat_model, "status": "succeeded"}

    class ReplayRAG:
        def retrieve(self, **kwargs):
            kwargs["candidate_audit"].extend([
                {**d.metadata, "raw_rank": i, "score": .8, "decision": "selected"}
                for i, d in enumerate(docs, 1)])
            return docs[:8]

    class ReplayLexical:
        documents = docs

        def search(self, *args, **kwargs):
            return docs

    gold = []
    for item in case["gold_ids"]:
        source, page = item.rsplit("#page=", 1)
        gold.append({"source": source, "pages": [int(page)]})
    record = evaluate_case(ReplayRAG(), {"case_id": case["case_id"], "question": case["question"],
        "gold_evidence": gold}, 8, lexical=ReplayLexical(), rerank_candidate_limit=16,
        reranker=lambda q, candidates: rerank_candidates(q, candidates, **configured_rerank_options(settings)))
    assert record["rerank_status"] == "succeeded" and record["rerank_backend"] == "chat"
    summary = {"case_id": case["case_id"], "runtime": runtime, "evaluation": {key: record[key] for key in [
        "rerank_status", "rerank_backend", "rerank_completed_missing_count", "rerank_recall_at_k",
        "rerank_rr_at_k", "rerank_ms", "rerank_total_tokens", "rerank_request_id"]}}
    write_json(HERE / "integration_smoke.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

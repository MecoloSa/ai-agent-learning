"""Plan and execute delegated reading tasks."""

import asyncio
import json
import re
from functools import lru_cache
from hashlib import sha256
from itertools import count
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal, TypedDict

from chromadb.errors import ChromaError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from openai import APIError
from pydantic import BaseModel, Field, StringConstraints, model_validator
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ai_agent.config import create_embeddings, create_llm, get_settings
from ai_agent.learning.learning_context import build_learning_state_message
from ai_agent.persistence.history_readback import HistoryReadAccess
from ai_agent.persistence.memory import ConversationMemory
from ai_agent.persistence.memory_state import HistoryArchiveState
from ai_agent.resources.document_structure import build_inspect_document_structure_tool
from ai_agent.resources.pdf_materials import MAX_READ_PDF_PAGES, PdfPageOutOfBounds, read_pdf_pages
from ai_agent.resources.tools import build_read_material_lines_tool, build_retrieve_learning_context_tool
from ai_agent.retrieval.rag import RAGError
from ai_agent.runtime.source_delivery import TurnSourceLedger
from ai_agent.runtime.reading_dependency import execute_ready
from ai_agent.runtime.report_delivery import DeliveryFailure, build_report_input, deliver_reports
from ai_agent.runtime.shared_reading_evidence import CitationArchive, SharedPdfReads
from graphlib import TopologicalSorter

DEEP_READING_MODEL = "qwen4-max"
DEEP_READING_MODEL_REQUEST_TIMEOUT = 60
DEFAULT_TASK_EVIDENCE_CHARS = 48000

ACQUISITION_NOTICE = (
    "最近几次取证没有增加正文。请根据当前子任务判断是否已有足够依据。"
    "若还缺少会影响结论的信息，可以继续查找。"
)


class ModelRequestTimeout(TimeoutError):
    """A single planning, worker, or delivery model request expired."""

PLANNER = """需要派发时，为每项写明独立交付目标、所需素材和完成条件，并传递适用的用户约束。
共享证据的工作可合并；后续工作确实需要上游产出时，填写依赖。已有报告足以完成的比较、解释或出题交给最终答复。
返回 JSON：tasks 含 id、instruction、sources（优先查阅的授权文件名数组）。有前置任务时填 depends_on（任务 ID 数组）；必须取得指定产物才能继续时，再填 required_artifacts（前置任务 ID 到产物名数组的映射）。"""

FOCUSED_PLANNER = """需要派发时，在 instruction 写清子任务的交付目标、所需素材和适用的用户约束。
可独立完成的任务并行，依赖已有报告的综合结论与教学练习交给最终答复。
返回 JSON：tasks 含 id、instruction、sources（使用 authorized_scope 的文件名，页码写入 instruction）。有前置任务时填 depends_on（任务 ID 数组）；必须取得指定产物才能继续时，再填 required_artifacts（前置任务 ID 到产物名数组的映射）。"""

ROLES = {
    "reader": "定位或核对资料，交付有出处的发现。",
    "analyst": "依据相关材料或已有发现进行解释、比较或推演。",
    "practice_designer": "交付用户确需单独设计的练习或测验。",
    "generalist": "处理本轮不适合交给其他专长角色的子任务。",
}
ROLE_OPENERS = {
    "reader": "你是资料检索智能体。定位或核对资料，交付能追溯到原文的发现。",
    "analyst": "你是证据分析智能体。根据相关证据解释、比较或推演，区分原文与推断。",
    "practice_designer": "你是学习练习设计智能体。围绕学习目标和已有材料，设计适合学习者的练习。",
    "generalist": "你是通用子任务智能体。完成当前指派，交付可供后续使用的结果。",
}
PLANNER_ROLES = "可按交付职责为子任务填写 role。" + " ".join(
    f"{key}：{description}" for key, description in ROLES.items()
)

WORKER = """把委派当作一个独立问题，完成取证、核对和解释，交付一份可单独阅读的回答。
按需要使用资料工具和上游发现。核心问题已有依据时即可提交，证据不足时说明具体缺口。
在 submit_report 中提交交付内容。用实际回执的证据 ID 标记有依据的发现。"""
WORKER_CONTRACT = WORKER.partition("\n")[2]  # 角色负责说明工作内容，共用段只保留工具与交付原则。
WORKER_ASSIGNMENT_RULE = "按当前指派完成本次交付。"

REPORT_FORMAT = """完成当前指派并返回 JSON。findings 记录有依据的发现，每项含 statement 和 evidence_ids。gaps 记录影响结论的缺口，无则为空数组。"""
GENERATED_REPORT_FORMAT = "生成成果放在 deliverables，每项含 content。"
ARTIFACT_RULE = "交付 requested_artifacts 中的产物时，在对应发现或生成成果的 artifact 填写产物名。"

DELIVERY = """根据用户请求、子任务报告与可用证据完成答复。
区分资料事实、转述、推断和生成内容；证据缺口影响结论时说明。"""
DIRECT_DELIVERY = "直接完成用户请求，按指定范围和形式交付。改写时保留原意，不补充原材料未表达的关系或结论。"
PLAN_ROUTE_RULE = "素材已在用户消息或已有对话中、无需派发时，返回 mode=direct 和空 tasks。需要查阅资料或派发独立工作时，返回 mode=delegate 和具体任务。"
PLAN_GENERATION_RULE = "子任务需要交付新编练习、改写稿等生成内容时，可填写 generates_content=true。"
GENERATED_CONTENT_RULE = "将练习、改写稿等生成内容作为交付物使用；资料结论仍需依据相应证据。用户只要成品时，直接交付成品。"

UNANSWERED_EXERCISE_RULE = "用户要求练习不给答案时，围绕用户指定的考点出题。检查题干是否已经说出或排除了所问的答案；如有，改成可供推断的线索。"

REPORT_DELIVERY = """依据子任务报告回答用户问题，综合各路发现。若证据有影响结论的缺口，再说明其影响。
有证据支持且无需进一步核对的解释可直接交付。需要逐字引用、核对分歧或补足结论所需证据时，回读相关原文。
区分原文、转述和推断，按用户要求组织回答。""" + "\n" + UNANSWERED_EXERCISE_RULE

ArtifactName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ArtifactNames = Annotated[list[ArtifactName], Field(min_length=1)]


class Assignment(BaseModel):
    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    sources: list[str]
    depends_on: list[str] = Field(default_factory=list)
    required_artifacts: dict[str, ArtifactNames] = Field(default_factory=dict)
    role: str = "generalist"
    generates_content: bool = False


class RoomyPlan(BaseModel):
    mode: Literal["direct", "delegate"] = "delegate"
    tasks: list[Assignment]

    @model_validator(mode="after")
    def validate_route(self):
        # 兼容历史任务计划。只有显式选择直答时才允许空 tasks。
        if (self.mode == "direct") == bool(self.tasks):
            raise ValueError("direct requires no tasks; delegate requires tasks")
        return self


class Plan(RoomyPlan):
    tasks: list[Assignment] = Field(max_length=6)


class Finding(BaseModel):
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    artifact: ArtifactName | None = None


class Deliverable(BaseModel):
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    artifact: ArtifactName | None = None


class Report(BaseModel):
    findings: list[Finding]
    gaps: list[str]
    deliverables: list[Deliverable] = Field(default_factory=list)


class State(TypedDict, total=False):
    question: str
    tasks: list[dict]
    results: dict
    events: list[dict]
    answer: str
    delivery_status: str
    delivery_error: dict[str, str]
    worker_seconds: dict[str, float]
    delegation_goal: str
    delegated_report: dict


class ExecutionState(State, total=False):
    evidence_store: dict
    turn_context: list[BaseMessage]
    archive_scope: HistoryArchiveState | None
    delivery_deadline: float
    research_deadline: float
    planning_timed_out: bool
    mode: Literal["direct", "delegate"]


def parse(text, schema):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.partition("\n")[2].removesuffix("```").strip()
    return schema.model_validate_json(text)


def validate(tasks, scope, *, history_available=False):
    ids = {task.id for task in tasks}
    if len(ids) != len(tasks):
        raise ValueError("duplicate id")
    for task in tasks:
        # 模型偶尔在文件名后附上已授权的页码。拆掉表示差异，读取权限仍以 scope 为准。
        normalized = []
        for source in task.sources:
            if source not in scope:
                match = re.fullmatch(r"(.+)#page=([1-9][0-9]*)", source)
                if match:
                    path, page = match.groups()
                    if path in scope and scope[path] is not None and int(page) in scope[path]:
                        source = path
            normalized.append(source)
        task.sources = list(dict.fromkeys(normalized))
        if set(task.sources) - scope.keys() or set(task.depends_on) - ids:
            raise ValueError("invalid source or dependency")
        if not task.sources and not task.depends_on and not history_available:
            raise ValueError("task has neither source nor upstream input")
        if set(task.required_artifacts) - set(task.depends_on):
            raise ValueError("required artifacts must come from direct dependencies")
    TopologicalSorter({task.id: task.depends_on for task in tasks}).prepare()


@lru_cache(maxsize=4)
def _index_pdf_snapshot(path, signature):
    with path.open("rb") as stream:
        return tuple((page.extract_text() or "").casefold() for page in PdfReader(stream).pages)


def index_pdf(path):
    stat = path.stat()
    # 同路径替换教材后，不复用上一个文件的定位索引。
    return _index_pdf_snapshot(path, (stat.st_size, stat.st_mtime_ns))


def build_tools(root, allowed, store, acquired, *, roomy=False, shared_reads=None):
    @tool
    def locate_pdf(relative_path: str, terms: list[str]) -> str:
        """搜索 PDF 的候选物理页及命中片段，用于定位。"""
        if relative_path not in allowed or allowed[relative_path] is not None:
            return json.dumps({"error": "source_scope"})
        terms = [term.casefold().strip() for term in (terms if roomy else terms[:8]) if term.strip()]
        pages = index_pdf(root / relative_path)
        ranked = sorted(((sum(term in text for term in terms), i)
                         for i, text in enumerate(pages, 1)),
                        key=lambda item: (-item[0], item[1]))
        hits = []
        for score, page in (ranked if roomy else ranked[:5]):
            if not score:
                break
            text = pages[page-1]
            offset = next(text.index(term) for term in terms if term in text)
            hits.append({"page": page, "match": text[max(0, offset-60):offset+130]})
        return json.dumps({"candidates": hits}, ensure_ascii=False)

    @tool
    def read_pdf(relative_path: str, start_page: int, end_page: int) -> str:
        """读取 PDF 正文并返回证据 ID。物理页从 1 开始，包含起止页。"""
        if relative_path not in allowed:
            return json.dumps({"error": "source_scope"})
        pages = allowed[relative_path]
        if pages is not None and (end_page < start_page or not {start_page, end_page} <= pages
                                  or end_page - start_page + 1 > len(pages)):
            return json.dumps({"error": "page_scope"})
        if pages is not None and any(i not in pages for i in range(start_page, end_page + 1)):
            return json.dumps({"error": "page_scope"})
        path = (root / relative_path).resolve()
        if shared_reads is None or not path.is_relative_to(root.resolve()):
            return read_uncached(relative_path, start_page, end_page)
        raw, reused = shared_reads.read(path, relative_path, start_page, end_page,
                                        lambda: (read_uncached(relative_path, start_page, end_page), False))
        if reused:
            payload = json.loads(raw)
            acquired.update(row["evidence_id"] for row in payload["evidence"])
            raw = json.dumps({**payload, "reused_shared_read": True}, ensure_ascii=False)
        return raw

    def read_uncached(relative_path, start_page, end_page):
        try:
            receipt = read_pdf_pages(str(root), relative_path, start_page, end_page)
            receipts = [receipt]
        except PdfPageOutOfBounds as error:
            return json.dumps({"error": error.to_error()}, ensure_ascii=False)
        except ValueError as error:
            if not roomy or end_page - start_page + 1 <= MAX_READ_PDF_PAGES or start_page < 1:
                return json.dumps({"error": "ValueError", "detail": str(error)}, ensure_ascii=False)
            try:
                # 宽松实验接受长页段，内部复用现有的有界 PDF 读页函数。
                first = read_pdf_pages(str(root), relative_path, start_page, start_page + MAX_READ_PDF_PAGES - 1)
                receipts = [first]
                final_page = min(end_page, first["total_pages"])
                for start in range(start_page + MAX_READ_PDF_PAGES, final_page + 1, MAX_READ_PDF_PAGES):
                    receipts.append(read_pdf_pages(str(root), relative_path, start,
                                                   min(final_page, start + MAX_READ_PDF_PAGES - 1)))
            except PdfPageOutOfBounds as error:
                return json.dumps({"error": error.to_error()}, ensure_ascii=False)
            except (ValueError, OSError) as error:
                return json.dumps({"error": type(error).__name__, "detail": str(error)}, ensure_ascii=False)
        except OSError as error:
            return json.dumps({"error": type(error).__name__, "detail": str(error)}, ensure_ascii=False)
        evidence = []
        for page in (item for receipt in receipts for item in receipt["pages"]):
            if not page["text"].strip():
                continue
            # 同页同长度的不同正文也必须能分别回查。
            digest = sha256(page["text"].encode("utf-8")).hexdigest()
            evidence_id = f"{relative_path}#p{page['page']}:{digest}"
            item = {"evidence_id": evidence_id, "source": relative_path, **page}
            store[evidence_id] = item
            acquired.add(evidence_id)
            evidence.append(item)
        return json.dumps({"evidence": evidence, "truncated": any(r["truncated"] for r in receipts)}, ensure_ascii=False)

    return [read_pdf, locate_pdf] if any(pages is None for pages in allowed.values()) else [read_pdf]


def project_history(messages):
    # 旧工具回执保留调用配对，正文由任务的证据仓保存。最近一批仍保留原回执。
    last_batch = max((i for i, message in enumerate(messages)
                      if getattr(message, "tool_calls", [])), default=0)
    visible = []
    for i, message in enumerate(messages):
        if isinstance(message, ToolMessage) and i < last_batch:
            payload = json.loads(message.content)
            if "evidence" in payload:
                payload["evidence"] = [{k: v for k, v in item.items() if k != "text"}
                                       for item in payload["evidence"]]
                message = message.model_copy(update={"content": json.dumps(payload, ensure_ascii=False)})
            elif "locations" in payload:
                # 检索返回定位线索，原文已经由读取工具单独保存。
                message = message.model_copy(update={"content": json.dumps(
                    {key: value for key, value in payload.items() if key != "snippets"}, ensure_ascii=False,
                )})
        visible.append(message)
    return visible


def project_task_history(messages, evidence_chars=DEFAULT_TASK_EVIDENCE_CHARS, *, keep_snippets=False):
    """保留本任务的调用过程，正文按证据 ID 去重，超出预算的正文可回读。"""
    visible = list(messages)
    payloads, positions = {}, {}
    recall_ids = set()
    last_batch = max((index for index, message in enumerate(messages)
                      if getattr(message, "tool_calls", [])), default=-1)
    for index, message in enumerate(messages):
        calls = getattr(message, "tool_calls", [])
        if calls:
            recall_ids.update(call["id"] for call in calls if call["name"] == "recall_evidence")
        if not isinstance(message, ToolMessage):
            continue
        payload = json.loads(message.content)
        if "evidence" not in payload:
            if not keep_snippets and index < last_batch and "locations" in payload and "snippets" in payload:
                payload.pop("snippets")
                visible[index] = message.model_copy(update={"content": json.dumps(payload, ensure_ascii=False)})
            continue
        locations = payload.get("locations")
        if locations and locations == [
            {key: value for key, value in row.items() if key not in {"evidence_id", "text"}}
            for row in payload["evidence"]
        ]:
            # 两份定位字段完全相同时只保留证据行。原始回执仍留在审计记录中。
            del payload["locations"]
        payloads[index] = payload
        recalled = message.tool_call_id in recall_ids
        for offset, row in enumerate(payload["evidence"]):
            if row.get("text") and (recalled or row["evidence_id"] not in positions):
                # 普通重复读取不提升正文优先级，主动回读可让原文重新进入当前视图。
                positions[row["evidence_id"]] = (index, offset, recalled, len(row["text"]))

    retained = set()
    remaining = evidence_chars
    for index, offset, recalled, chars in sorted(positions.values(), reverse=True):
        # 当前批次的主动回读保留全文，否则模型可能反复回读同一个过大结果。
        if remaining is None or chars <= remaining or (recalled and index > last_batch):
            retained.add((index, offset))
            if remaining is not None:
                remaining -= chars

    resident = {key for key, (index, offset, _, _) in positions.items() if (index, offset) in retained}
    for index, payload in payloads.items():
        rows = []
        for offset, row in enumerate(payload["evidence"]):
            if not row.get("text") or (index, offset) in retained:
                rows.append(row)
            else:
                rows.append({
                    **{key: value for key, value in row.items() if key != "text"},
                    "text_status": (
                        "shown_in_another_receipt" if row["evidence_id"] in resident else "available_via_recall"
                    ),
                })
        payload["evidence"] = rows
        visible[index] = messages[index].model_copy(
            update={"content": json.dumps(payload, ensure_ascii=False)},
        )
    return visible


def build_scoped_search(root, scope, lock, store=None, acquired=None):
    sources = frozenset(source for source, pages in scope.items() if pages is None)
    if not sources:
        return None
    settings = get_settings()
    retrievals = {}

    @tool
    def search_material(queries: list[str], relative_path: str | None = None) -> str:
        """查找有出处的原文片段。可指定文件；需要完整页面或上下文时按物理页回读。"""
        if relative_path is not None and relative_path not in sources:
            return json.dumps({"error": "source_not_open_for_search"})
        selected = sources if relative_path is None else frozenset({relative_path})
        # 多路共用一个持久索引。实验中保护同一轮内的增量同步和索引更新。
        with lock:
            if selected not in retrievals:
                retrievals[selected] = build_retrieve_learning_context_tool(
                    str(root), source_paths=selected, structured_result=True,
                    strategy="dense", rerank=True,
                    max_queries_per_call=settings.interactive_max_queries_per_tool_call,
                    max_context_chars=settings.interactive_retrieval_max_context_chars,
                )
            raw = retrievals[selected].invoke({"queries": queries, "neighbor_window": 0})
        try:
            data = json.loads(raw)["data"]
        except (ValueError, KeyError):
            return json.dumps({"locations": [], "notice": raw}, ensure_ascii=False)
        locations = [{key: value for key, value in row.items() if key != "text"}
                     for row in data["locations"]]
        evidence = []
        for row, location in zip(data["locations"], locations):
            source, chunk, text = row.get("source"), row.get("chunk_index"), row.get("text")
            if store is None or acquired is None or source not in selected or chunk is None or not text:
                continue
            digest = sha256(text.encode("utf-8")).hexdigest()
            key = f"{source}#chunk{chunk}:{digest}"
            item = {"evidence_id": key, **location, "text": text}
            store[key] = item
            acquired.add(key)
            evidence.append(item)
        return json.dumps({
            "locations": locations,
            **({"evidence": evidence} if evidence else {"snippets": data["text"]}),
            **({"omitted_blocks": data["omitted_blocks"]} if "omitted_blocks" in data else {}),
        }, ensure_ascii=False)

    return search_material


def build_scoped_line_reader(root, scope, store, acquired):
    sources = {source for source, pages in scope.items()
               if pages is None and Path(source).suffix.lower() in {".md", ".txt", ".py"}}
    if not sources:
        return None
    reader = build_read_material_lines_tool(str(root))

    @tool
    def read_material_lines(relative_path: str, start_line: int, end_line: int) -> str:
        """读取当前资料中 Markdown、TXT 或 Python 文件的实际行号和正文。"""
        if relative_path not in sources:
            return json.dumps({"error": "source_scope"})
        result = json.loads(reader.invoke({
            "relative_path": relative_path, "start_line": start_line, "end_line": end_line,
        }))
        if not result["ok"]:
            return json.dumps({"error": result["error"]}, ensure_ascii=False)
        data = result["data"]
        text = "\n".join(row["text"] for row in data["lines"])
        if not text.strip():
            return json.dumps({"evidence": [], "notice": "未返回非空正文"}, ensure_ascii=False)
        start, end = data["returned_start_line"], data["returned_end_line"]
        digest = sha256(text.encode("utf-8")).hexdigest()
        key = f"{relative_path}#l{start}-{end}:{digest}"
        item = {
            "evidence_id": key, "source": data["source"], "text": text,
            "lines": [start, end], "citation": data["citation"],
            "truncated": data["truncated"] or any(row["partial"] for row in data["lines"]),
        }
        store[key] = item
        acquired.add(key)
        return json.dumps({"evidence": [item]}, ensure_ascii=False)

    return read_material_lines


def build_scoped_structure_reader(root, scope, store=None, acquired=None):
    sources = {source for source, pages in scope.items()
               if pages is None and Path(source).suffix.lower() in {".pdf", ".md", ".txt"}}
    if not sources:
        return None
    structure = build_inspect_document_structure_tool(str(root))

    @tool
    def inspect_document_structure(relative_path: str) -> str:
        """查看章节组织和原文位置。短文可返回全文，目录证据仅支持标题和位置。"""
        if relative_path not in sources:
            return json.dumps({"error": "source_scope"})
        raw = structure.invoke({"relative_path": relative_path})
        if store is None or acquired is None:
            return raw
        receipt = json.loads(raw)
        data = receipt["data"]
        mode = data["mode"]
        if mode == "whole_document":
            text = data.pop("text", "")
        else:
            # 模型选中的书签未必核对过正文。只登记经过程序定位的标题与起点。
            located = [
                {field: section[field] for field in ("title", "start", "boundary_status")}
                for section in data["sections"]
                if section["boundary_status"] in {"verified_opening_text", "located_in_extracted_text"}
            ]
            text = json.dumps(located, ensure_ascii=False) if located else ""
        if text.strip() and data["sources"]:
            digest = sha256(text.encode("utf-8")).hexdigest()
            key = f"{relative_path}#structure:{digest}"
            item = {"evidence_id": key, "source": data["source"], "text": text,
                    "kind": "document_structure", "mode": mode, "sources": data["sources"],
                    "limitations": data.get("limitations", [])}
            store[key] = item
            acquired.add(key)
            receipt["evidence"] = [item]
        return json.dumps(receipt, ensure_ascii=False)

    return inspect_document_structure


def build_history_access(root, archive_scope):
    settings = get_settings()
    access = HistoryReadAccess(root, strategy=settings.history_retrieval_strategy,
                              min_similarity=settings.history_semantic_min_similarity,
                              embedding_factory=create_embeddings)
    access.start_turn(archive_scope)
    return access


def wrap_history_read(original, store, acquired):
    def read_turn(**args):
        receipt = json.loads(original.invoke(args))
        if receipt["ok"]:
            data = receipt["data"]
            # 对话原文只有成功回读才成为证据，定位命中和摘要均不能进入正文仓。
            key = data["citation"]
            item = {"evidence_id": key, "source": "历史对话", "kind": "archived_conversation",
                    "citation": key, "citation_ref": data["citation_ref"],
                    "text": json.dumps({field: data.pop(field) for field in ("user_text", "assistant_text")},
                                       ensure_ascii=False)}
            store[key] = item
            acquired.add(key)
            receipt["evidence"] = [item]
        return json.dumps(receipt, ensure_ascii=False)

    return original.model_copy(update={"func": read_turn})


def cited_items(findings, store):
    ids = dict.fromkeys(evidence_id for finding in findings for evidence_id in finding["evidence_ids"])
    return [store[evidence_id] for evidence_id in ids]


def footer_evidence(results, store, delivered_ids):
    # 优先展示进入汇总请求的正文。没有正文时，回退到报告引用而非搜索命中。
    visible = [item for key, item in store.items() if key in delivered_ids
               and item.get("kind") != "archived_conversation"]
    if visible:
        return visible
    cited = {key for result in results.values() for finding in result.get("findings", [])
             for key in finding["evidence_ids"]}
    return [item for key, item in store.items() if key in cited and item.get("kind") != "archived_conversation"]


def missing_dependencies(task, upstream):
    missing = {}
    for task_id, names in task.get("required_artifacts", {}).items():
        report = upstream[task_id]
        available = {item.get("artifact") for item in [*report.get("findings", []), *report.get("deliverables", [])]}
        absent = list(dict.fromkeys(name for name in names if name not in available))
        if absent:
            missing[task_id] = absent
    if missing:
        return {"status": "blocked", "reason": "前置任务缺少必需产物", "missing_artifacts": missing}
    return None


def rank_fallback_evidence(question, evidence):
    # 用用户原问中的英文术语给未被引用的正文排序，优先保留直接命中项。
    # 候选清单仍然是实际回执，关键词本身不参与生成证据。
    terms = set(re.findall(r"\b[A-Za-z][A-Za-z_0-9]{2,}\b", question.casefold()))

    def score(item):
        text = item["text"].casefold()
        return sum((4 if "_" in term else 1) * min(3, text.count(term)) for term in terms)

    return sorted(evidence, key=score, reverse=True)


def attach_partial_excerpts(catalog, tasks, results, store):
    """Attach verbatim receipts for incomplete branches within a shared character budget."""
    remaining = 6000
    for task in tasks:
        result = results.get(task["id"])
        if not result or result["status"] == "done":
            continue
        ranked = rank_fallback_evidence(task["instruction"], [
            store[key] for key in result.get("evidence_ids", ())
            if key in store and key in catalog and store[key].get("text")
        ])
        locations = set()
        for item in ranked:
            if len(locations) == 2 or remaining <= 0:
                break
            # 同一页的切片与整页只附一份。按行读取和未知位置保留各自身份。
            key = item["evidence_id"]
            position = (item.get("source"), item["page"]) if "page" in item else (
                item.get("source"), tuple(item.get("lines") or ()) or key,
            )
            if position in locations:
                continue
            locations.add(position)
            if "excerpt" in catalog[key]:
                continue
            original_text = item["text"]
            excerpt = original_text[:min(1200, remaining)]
            catalog[key]["excerpt"] = excerpt
            # 这里只记录交接时的裁剪，原始回执本身也可能只是检索片段。
            catalog[key]["excerpt_truncated"] = len(excerpt) < len(original_text)
            remaining -= len(excerpt)


def prepare_partial_evidence_handoff(report, tasks, results, store):
    catalog = report["evidence_catalog"]
    attach_partial_excerpts(catalog, tasks, results, store)
    if any("excerpt" in location for location in catalog.values()):
        report["excerpt_note"] = (
            "excerpt 是已读原文的节选。需要核对原句边界、"
            "确认某范围已完整覆盖或判断其中是否未提及某内容时，请回读对应范围。"
        )


def build_graph(root, scope, ask, trace, *, project=True, planner_role="auxiliary", bounded=False,
                roomy=False, worker_timeout=None, turn_timeout=None, delivery_reserve=20.0,
                progressive_reports=False, reuse_retrieval=False, source_footer=False, structure_lookup=False,
                focused_planning=False, model_request_timeout=None,
                learning_state=None, conversation_memory=None, history_archive_root=None,
                acquisition_notice=False, notice_after=4, task_evidence_chars=DEFAULT_TASK_EVIDENCE_CHARS,
                latest_only=False, shared_pdf_reads=False, citation_archive=None, worker_roles=True,
                worker_full_question=False, collect_only=False, branch_report_reserve=0, fast_report=None,
                submitted_tasks=None):
    if submitted_tasks is not None:
        if not collect_only:
            raise ValueError("submitted tasks require collect-only delegation")
        assignment_plan = RoomyPlan(tasks=submitted_tasks)
        validate(assignment_plan.tasks, scope)
        task_rows = [task.model_dump(exclude_defaults=True) for task in assignment_plan.tasks]
    if turn_timeout is not None and not 0 < delivery_reserve < turn_timeout:
        raise ValueError("delivery reserve must be between zero and the turn timeout")
    if acquisition_notice and notice_after < 1:
        raise ValueError("notice_after must be positive")
    if branch_report_reserve < 0 or (
        branch_report_reserve and (
            (worker_timeout is None and
             (turn_timeout is None or branch_report_reserve >= turn_timeout - delivery_reserve))
            or (worker_timeout is not None and branch_report_reserve >= worker_timeout)
        )
    ):
        raise ValueError("branch_report_reserve must fit within the research deadline")
    search_lock = Lock()

    async def ask_model(role, stage, messages, tools=()):
        if model_request_timeout is None:
            return await ask(role, stage, messages, tools)
        try:
            # 每次请求重新计时，串行子任务不会继承上游已经花掉的时间。
            async with asyncio.timeout(model_request_timeout) as deadline:
                return await ask(role, stage, messages, tools)
        except TimeoutError as error:
            if not deadline.expired():
                raise
            trace.append({"task": stage, "tool": "model_request_timeout", "seconds": model_request_timeout})
            raise ModelRequestTimeout(f"单次模型请求超过 {model_request_timeout:g} 秒") from error

    def conversation_context():
        # 给统筹和汇总相同的会话视图；执行者只收当前指派及所需的上游交付。
        memory = conversation_memory
        if memory is None and learning_state is not None and learning_state.conversation_memory is not None:
            memory = ConversationMemory.from_snapshot("", learning_state.conversation_memory)
        history = memory.get_messages_for_llm()[1:] if memory is not None else []
        # 摘要是背景数据。复用原记忆策略，但不把旧会话的系统提示复制到新角色。
        history = [HumanMessage(content=message.content) if isinstance(message, SystemMessage) else message
                   for message in history]
        state_view = [build_learning_state_message(learning_state)] if learning_state else []
        archive = memory.history_archive if memory is not None else None
        return history + state_view, archive if history_archive_root is not None else None

    async def plan(state):
        turn_context, archive_scope = conversation_context()
        if archive_scope is not None:
            turn_context.append(build_history_access(history_archive_root, archive_scope).source_message())
        deadlines = {}
        if turn_timeout is not None:
            end = asyncio.get_running_loop().time() + turn_timeout
            deadlines = {"delivery_deadline": end, "research_deadline": end - delivery_reserve}
        if submitted_tasks is not None:
            # 主模型已提交任务。该节点只初始化执行时间和本轮状态。
            return {**deadlines, "turn_context": turn_context, "archive_scope": archive_scope,
                    "mode": "delegate", "tasks": task_rows}
        planner_prompt = FOCUSED_PLANNER if focused_planning else PLANNER
        planner_prompt = f"{planner_prompt}\n{PLAN_ROUTE_RULE}\n{PLAN_GENERATION_RULE}"
        if worker_roles:
            planner_prompt = f"{planner_prompt}\n{PLANNER_ROLES}"
        payload = {"question": state["question"], "authorized_scope": {
            book: sorted(pages) if pages is not None else None for book, pages in scope.items()
        }}
        if collect_only:
            payload["delegation_goal"] = state["delegation_goal"]
            planner_prompt += "\n围绕委派目标规划，保留原问题中适用的要求。"
        try:
            async with asyncio.timeout_at(deadlines.get("research_deadline")) as deadline:
                answer = await ask_model(planner_role, "planner", [
                    SystemMessage(content=planner_prompt),
                    *turn_context,
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ])
        except ModelRequestTimeout:
            return {**deadlines, "tasks": [], "planning_timed_out": True}
        except TimeoutError:
            if not deadline.expired():
                raise
            return {**deadlines, "tasks": [], "planning_timed_out": True}
        outcome = parse(answer.content, RoomyPlan if roomy else Plan)
        validate(outcome.tasks, scope, history_available=bool(archive_scope and archive_scope.sources))
        return {**deadlines, "turn_context": turn_context,
                "archive_scope": archive_scope,
                "mode": outcome.mode,
                "tasks": [task.model_dump(exclude_defaults=True) for task in outcome.tasks]}

    async def execute(state):
        # 同一编译图可以处理多个请求，证据只属于本次执行。
        store = {}
        shared_reads = SharedPdfReads() if shared_pdf_reads else None
        artifact_requests = {}
        for task in state["tasks"]:
            for upstream_id, names in task.get("required_artifacts", {}).items():
                artifact_requests.setdefault(upstream_id, set()).update(names)

        async def worker(task, inputs, completed_receipts, progress, end):
            # 角色只决定系统消息首段。任务 ID、工具范围和依赖沿用各自的字段。
            opener = ROLE_OPENERS.get(task.get("role"), ROLE_OPENERS["generalist"]) if worker_roles else ""
            # 执行和独立收尾都使用当前指派，完整原问题留在规划与最终汇总。
            worker_context = {"user_request": state["question"]} if worker_full_question else {}
            assignment_rule = "" if worker_full_question else WORKER_ASSIGNMENT_RULE
            # 保留来源任务及其完成情况；正文在单独的去重列表中传递。
            upstream_reports = {
                task_id: {key: value for key, value in result.items() if key != "evidence_ids"}
                for task_id, result in inputs.items()
            }
            upstream_evidence = cited_items(
                [finding for result in inputs.values() for finding in result.get("findings", [])], store,
            )
            acquired = {item["evidence_id"] for item in upstream_evidence}
            requested = artifact_requests.get(task["id"], set())
            artifact_context = {"requested_artifacts": sorted(requested)} if requested else {}
            local_evidence = set()
            read_failures = {}
            terminal_reads = {}
            repeated_receipts = 0
            noticed = False
            generates_content = task.get("generates_content", False)

            @tool
            def submit_report(findings: list[Finding], gaps: list[str]) -> str:
                """提交有依据的发现，gaps 记录影响交付的缺口。"""
                return Report(findings=findings, gaps=gaps).model_dump_json()

            @tool("submit_report")
            def submit_generated_report(findings: list[Finding], gaps: list[str],
                                        deliverables: list[Deliverable] | None = None) -> str:
                """提交当前指派：findings 为带证据 ID 的发现，deliverables 为生成的练习、改写等成果，gaps 记录影响交付的缺口。"""
                return Report(findings=findings, gaps=gaps, deliverables=deliverables or []).model_dump_json()

            def validated_findings(report):
                findings = []
                for item in report.findings:
                    statement = item.statement.strip()
                    if not statement or not set(item.evidence_ids) <= acquired | completed_receipts:
                        continue
                    if re.fullmatch(r"(?:placeholder|占位)(?:提交)?[。.!！]?", statement, re.IGNORECASE):
                        continue
                    finding = item.model_dump(exclude_none=True)
                    if item.artifact not in requested:
                        finding.pop("artifact", None)
                    findings.append(finding)
                return findings

            def finish(report=None, reason=None):
                # 正文回执和已保存发现都可以在最终报告失败后继续使用。
                ids = sorted(completed_receipts)
                facts = {"read_failures": list(read_failures.values())} if read_failures else {}
                if report is None:
                    return {"status": "unavailable", "reason": reason, "evidence_ids": ids,
                            **({"findings": list(progress.values())} if progress else {}), **facts}
                findings = validated_findings(report)
                deliverables = [item.model_dump(exclude_none=True) if item.artifact in requested
                                else {"content": item.content} for item in report.deliverables] if generates_content else []
                if not findings and not deliverables:
                    gaps = report.gaps or ["缺少可追溯的发现"]
                    return {"status": "unavailable", "reason": "；".join(gaps), "gaps": gaps,
                            "evidence_ids": ids, **({"findings": list(progress.values())} if progress else {}), **facts}
                return {"status": "done", "findings": findings, "gaps": report.gaps, "evidence_ids": ids,
                        **({"deliverables": deliverables} if deliverables else {}), **facts}

            # 分派的 sources 指引取证重点，工具继承本轮允许的文件和页码。
            pdf_scope = ({source: pages for source, pages in scope.items()
                          if Path(source).suffix.lower() == ".pdf"} if reuse_retrieval else scope)
            tools = build_tools(root, pdf_scope, store, local_evidence, roomy=roomy,
                                shared_reads=shared_reads) if pdf_scope else []
            if reuse_retrieval:
                search_material = build_scoped_search(root, scope, search_lock, store, local_evidence)
                if search_material is not None:
                    # 候选对照只保留一个检索入口，基线仍用 locate_pdf。
                    tools = [item for item in tools if item.name != "locate_pdf"]
                    tools.insert(1, search_material)
                line_reader = build_scoped_line_reader(root, scope, store, local_evidence)
                if line_reader is not None:
                    tools.append(line_reader)
            structure_reader = (build_scoped_structure_reader(root, scope, store, local_evidence)
                                if structure_lookup else None)
            if structure_reader is not None:
                tools.append(structure_reader)
            worker_history = None
            if state["archive_scope"] is not None and state["archive_scope"].sources:
                worker_history = build_history_access(history_archive_root, state["archive_scope"])
                tools.extend(
                    wrap_history_read(item, store, local_evidence)
                    if item.name == "read_conversation_turn" else item for item in worker_history.tools
                )
            if progressive_reports:
                @tool
                def save_findings(findings: list[Finding]) -> str:
                    """保存已经核实的发现及其证据 ID。保存后可以继续取证，也可以提交完整报告。"""
                    for finding in validated_findings(Report(findings=findings, gaps=[])):
                        key = (finding["statement"], tuple(sorted(set(finding["evidence_ids"]))),
                               finding.get("artifact"))
                        progress[key] = finding
                    return json.dumps({"saved_count": len(progress)})

                tools.append(save_findings)
            if roomy:
                @tool
                def recall_evidence(evidence_id: str) -> str:
                    """重新查看本任务已经取得或由上游交付的证据正文。"""
                    if evidence_id not in local_evidence | acquired:
                        return json.dumps({"error": "evidence_scope"})
                    return json.dumps({"evidence": [store[evidence_id]]}, ensure_ascii=False)

                tools.append(recall_evidence)
            allowed_scope = {source: sorted(pages) if pages is not None else None
                             for source, pages in scope.items()}
            assignment_payload = {
                **worker_context, "assignment": task, "authorized_scope": allowed_scope,
                **artifact_context, "upstream_reports": upstream_reports, "upstream_evidence": upstream_evidence,
            }
            messages = [
                SystemMessage(content="\n".join(filter(None, (opener, WORKER_CONTRACT if opener else WORKER,
                    ARTIFACT_RULE if requested else "",
                    assignment_rule,
                    "还需要继续取证时，可先用 save_findings 保存已有依据的发现。" if progressive_reports else "")))),
                HumanMessage(content=json.dumps(assignment_payload, ensure_ascii=False)),
            ]
            if worker_history is not None:
                messages.insert(1, worker_history.source_message())
            async def finalize(*, deadline_closeout=False):
                # 提交单独起请求，给出选中的原文以及定位信息。没有继续取证的工具。
                use_fast_report = deadline_closeout and fast_report is not None
                ranked = [] if use_fast_report else rank_fallback_evidence(
                    f"{state['question']} {task['instruction']}",
                    [item for key, item in store.items()
                     if key in (completed_receipts if deadline_closeout else local_evidence) | acquired],
                )
                evidence = ranked
                if deadline_closeout and not use_fast_report:
                    evidence = []
                    remaining = 8000
                    positions = set()
                    for item in ranked:
                        position = (item["source"], item.get("page"), tuple(item.get("lines") or ()))
                        if item.get("page") is None and not position[-1]:
                            position = (*position, item["evidence_id"])
                        if position in positions or len(item["text"]) > remaining:
                            continue
                        evidence.append(item)
                        positions.add(position)
                        remaining -= len(item["text"])
                    trace.append({"task": task["id"], "tool": "branch_closeout_evidence",
                                  "evidence_ids": [item["evidence_id"] for item in evidence],
                                  "text_chars": 8000 - remaining})
                report_data = {
                    "assignment": task, **worker_context, **artifact_context,
                    "upstream_reports": upstream_reports, "evidence": evidence,
                }
                if use_fast_report:
                    # 复用请求投影去掉同一证据的重复正文，按历史保留每条取证及回执。
                    # 工具历史作为数据传入。截止时未返回的调用没有回执，不能被当作证据。
                    history = project_task_history(messages, None, keep_snippets=True)
                    returned = {item.tool_call_id for item in history if isinstance(item, ToolMessage)}
                    report_data = {
                        **assignment_payload,
                        "tool_history": [
                            {"tool_call_id": item.tool_call_id, "result": json.loads(item.content)}
                            if isinstance(item, ToolMessage) else
                            {"tool_calls": item.tool_calls}
                            for item in history if isinstance(item, ToolMessage) or getattr(item, "tool_calls", ())
                        ],
                        "pending_tool_call_ids": [
                            call["id"] for item in history for call in getattr(item, "tool_calls", ())
                            if call["id"] not in returned
                        ],
                    }
                try:
                    report_messages = [
                        SystemMessage(content="\n".join(filter(None, (
                            opener, REPORT_FORMAT, GENERATED_REPORT_FORMAT if generates_content else "",
                            ARTIFACT_RULE if requested else "", assignment_rule,
                                    ("先交付现有证据能支持的核心发现。影响结论的未核实要求写入 gaps。"
                                     if deadline_closeout else ""),
                        )))),
                        HumanMessage(content=json.dumps(report_data, ensure_ascii=False)),
                    ]
                    answer = (await fast_report(report_messages) if use_fast_report
                              else await ask_model("auxiliary", task["id"], report_messages))
                    if not answer.content:
                        reason = answer.response_metadata.get("finish_reason")
                        return finish(reason=f"独立交付正文为空（结束原因：{reason or '未知'}）")
                    return finish(parse(answer.content, Report))
                except ModelRequestTimeout as error:
                    return finish(reason=str(error))
                except (APIError, ValueError) as error:
                    return finish(reason=f"独立交付失败：{type(error).__name__}")

            async def explore():
                nonlocal noticed, repeated_receipts
                # 宽松实验由执行者自行决定何时停止，历史对照保留原预算。
                tool_steps = 3 if bounded else 6
                report_tool = submit_generated_report if generates_content else submit_report
                for step in count() if roomy else range(tool_steps + 1):
                    allowed_tools = ([*tools, report_tool] if roomy or
                                     tools and step < tool_steps else [report_tool])
                    try:
                        view = messages
                        if project:
                            view = (project_task_history(messages, task_evidence_chars)
                                    if roomy and not latest_only else project_history(messages))
                        if acquisition_notice and repeated_receipts >= notice_after and not noticed:
                            # 只改变这次请求。工具和长期消息保持原状，模型可以继续取证。
                            view = [*view, HumanMessage(content=ACQUISITION_NOTICE)]
                            noticed = True
                            trace.append({"task": task["id"], "tool": "acquisition_notice"})
                        answer = await ask_model(
                            "auxiliary", task["id"], view, allowed_tools
                        )
                    except ModelRequestTimeout as error:
                        return finish(reason=str(error))
                    except APIError as error:
                        # 单路请求失败仍保留已实际取得的证据，让其他分支继续。
                        return finish(reason=type(error).__name__)
                    if len(answer.tool_calls) == 1 and answer.tool_calls[0]["name"] == "submit_report":
                        try:
                            result = finish(Report.model_validate(answer.tool_calls[0]["args"]))
                            return await finalize() if roomy and result["status"] != "done" and (local_evidence or acquired) else result
                        except ValueError:
                            return await finalize() if roomy else finish(reason="提交参数无效")
                    if not answer.tool_calls:
                        try:
                            result = finish(parse(answer.content, Report))
                            return await finalize() if roomy and result["status"] != "done" and (local_evidence or acquired) else result
                        except ValueError:
                            trace.append({"task": task["id"], "tool": "invalid_report", "text": answer.content})
                            return await finalize() if roomy else finish(reason="交付格式无效")
                    messages.append(answer)
                    tool_by_name = {item.name: item for item in allowed_tools if item.name != "submit_report"}
                    all_reused = True
                    for call in answer.tool_calls:
                        name, args = call["name"], call["args"]
                        tool_started = perf_counter()
                        # 仅为本子任务缓存明确不可重试的 PDF 读页回执。
                        key = json.dumps(args, sort_keys=True) if name == "read_pdf" else None
                        reused = name in tool_by_name and key in terminal_reads
                        all_reused &= reused
                        try:
                            if reused:
                                result = terminal_reads[key]
                            elif name == submit_report.name and submit_report in allowed_tools:
                                # 先完成取证批次，报告须能根据这批回执更新。
                                result = json.dumps({"error": "submit_after_tool_results"})
                            else:
                                # 阶段保存留在事件循环，保证超时后后台线程不能更新已交付的发现。
                                if name == "save_findings" and name in tool_by_name:
                                    result = tool_by_name[name].invoke(args)
                                else:
                                    result = await asyncio.to_thread(tool_by_name[name].invoke, args)
                        except KeyError:
                            result = json.dumps({"error": "tool_unavailable_in_current_stage"})
                        except (ValueError, OSError, PdfReadError, RAGError, ChromaError, APIError) as exc:
                            result = json.dumps({"error": type(exc).__name__})
                        receipt = json.loads(result)
                        # 只登记已经由本轮工具任务实际返回的正文。超时后仍在后台结束的
                        # 同步读取，无法再往这份已确认回执集合添加内容。
                        receipt_ids = {item["evidence_id"] for item in receipt.get("evidence", ())}
                        if acquisition_notice and name in {
                            "search_material", "read_pdf", "read_material_lines", "inspect_document_structure",
                        }:
                            if receipt_ids - completed_receipts:
                                repeated_receipts = 0
                                noticed = False
                            elif receipt_ids:
                                repeated_receipts += 1
                            else:
                                repeated_receipts = 0
                        completed_receipts.update(receipt_ids)
                        error = receipt.get("error")
                        if name == "read_pdf" and isinstance(error, dict) and error.get("code") == "pdf_page_out_of_bounds":
                            source = args["relative_path"]
                            read_failures[source, error["requested_start_page"]] = {"source": source, **error}
                            if error.get("retryable") is False:
                                terminal_reads[key] = result
                        trace.append({"task": task["id"], "tool": name, "args": args, "result": receipt,
                                      "seconds": perf_counter() - tool_started})
                        if reused:
                            trace[-1]["reused_terminal_receipt"] = True
                        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
                    if all_reused:
                        return (await finalize() if local_evidence or acquired
                                else finish(reason="重复请求的 PDF 页面已确定不可读取"))
                return await finalize() if roomy else finish(reason="工具循环未形成交付")

            if not branch_report_reserve:
                return await explore()
            now = asyncio.get_running_loop().time()
            reserve = branch_report_reserve
            if worker_timeout is not None:
                # 若整轮只剩更短的时间，按比例缩短两阶段，仍服从整轮截止。
                reserve = min(reserve, max(0, end - now) * branch_report_reserve / worker_timeout)
            cutoff = end - reserve
            try:
                async with asyncio.timeout_at(cutoff) as exploration:
                    return await explore()
            except TimeoutError:
                if not exploration.expired():
                    raise
            if not completed_receipts and not acquired:
                return finish(reason="收尾前未取得可用证据")
            trace.append({"task": task["id"], "tool": "branch_deadline_closeout",
                          "confirmed_evidence": len(completed_receipts | acquired)})
            return await finalize(deadline_closeout=True)

        async def run_worker(task, inputs):
            completed_receipts = set()
            progress = {}
            loop = asyncio.get_running_loop()
            ends = [state["research_deadline"]] if "research_deadline" in state else []
            if worker_timeout is not None:
                ends.append(loop.time() + worker_timeout)
            end = min(ends) if ends else None
            try:
                async with asyncio.timeout_at(end) as deadline:
                    # 依赖刚放行也要检查时间，避免在取证截止后启动新模型调用。
                    if end is not None and loop.time() >= end:
                        return {"status": "unavailable", "reason": "取证时间已到", "evidence_ids": []}
                    return await worker(task, inputs, completed_receipts, progress, end)
            except TimeoutError:
                if not deadline.expired():
                    raise
                return {"status": "unavailable", "reason": "取证分支超过等待边界",
                        "evidence_ids": sorted(completed_receipts),
                        **({"findings": list(progress.values())} if progress else {})}

        results, events = await execute_ready(
            state["tasks"], run_worker, concurrency=len(state["tasks"]) if roomy else 2,
            dependency_blocker=missing_dependencies,
        )
        starts = {event["id"]: event["seconds"] for event in events if event["status"] == "started"}
        worker_seconds = {event["id"]: event["seconds"] - starts[event["id"]] for event in events
                          if event["status"] != "started" and event["id"] in starts}
        return {"results": results, "events": events, "evidence_store": store, "worker_seconds": worker_seconds}

    async def synthesize(state):
        store = state["evidence_store"]
        if roomy:
            delivered_ids = set()
            instructions = REPORT_DELIVERY
            if any(result.get("deliverables") for result in state["results"].values()):
                instructions = f"{instructions}\n{GENERATED_CONTENT_RULE}"
            reply = await deliver_reports(
                state["question"], state["results"], store, ask_model, trace, instructions,
                frontload_findings=progressive_reports,
                conversation_context=state["turn_context"],
                citation_archive=citation_archive, citation_scope=scope,
                delivered_evidence_ids=delivered_ids,
            )
            return reply.text, delivered_ids
        findings = [finding for result in state["results"].values() for finding in result.get("findings", [])]
        selected = {item["evidence_id"]: item for item in cited_items(findings, store)}
        for result in state["results"].values():
            if result["status"] != "done":
                for evidence_id in result.get("evidence_ids", []):
                    selected[evidence_id] = store[evidence_id]
        if bounded and not roomy:
            # 优先保留被发现引用的证据。补救材料按原始取得顺序填充，
            # 每本最多两页。同页的多种回执只保留先选中的版本。
            candidates = list(cited_items(findings, store))
            candidates.extend(rank_fallback_evidence(
                state["question"], [item for key, item in store.items() if key in selected]
            ))
            by_page = {}
            counts = {}
            for item in candidates:
                source = item["source"]
                page_key = (source, item.get("page"), tuple(item.get("lines", ())))
                if page_key in by_page or counts.get(source, 0) >= 2:
                    continue
                counts[source] = counts.get(source, 0) + 1
                by_page[page_key] = item
            selected = {item["evidence_id"]: item for item in by_page.values()}
            # 保留交付范围内可复核的声明，避免发送正文已被排除的引用。
            results = {
                task_id: {
                    **result,
                    "findings": [f for f in result.get("findings", [])
                                 if set(f["evidence_ids"]) <= selected.keys()],
                    "evidence_ids": [key for key in result.get("evidence_ids", []) if key in selected],
                }
                for task_id, result in state["results"].items()
            }
        else:
            results = state["results"]
        payload = {"question": state["question"], "results": results,
                   "evidence": list(selected.values())}
        reply = await ask_model("standard", "deliver", [
            SystemMessage(content=DELIVERY), *state["turn_context"],
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        return reply.text, set(selected)

    async def deliver(state):
        if state.get("planning_timed_out"):
            return {"delivery_status": "failed",
                    "delivery_error": {"type": "TimeoutError", "message": "规划超时，未取得资料"}}
        try:
            async with asyncio.timeout_at(state.get("delivery_deadline")) as deadline:
                answer, delivered_ids = await synthesize(state)
            if not answer.strip():
                raise DeliveryFailure("汇总未生成可交付正文")
        except ModelRequestTimeout as error:
            return {"delivery_status": "failed",
                    "delivery_error": {"type": type(error).__name__, "message": str(error)}}
        except TimeoutError:
            if not deadline.expired():
                raise
            return {"delivery_status": "failed",
                    "delivery_error": {"type": "TimeoutError", "message": "整轮时间已到，已保留取证结果"}}
        except (DeliveryFailure, APIError) as exc:
            # 返回节点状态，使 LangGraph 保留已经完成的任务报告和事件。
            detail = str(exc) if isinstance(exc, DeliveryFailure) else "汇总模型请求失败"
            return {"delivery_status": "failed",
                    "delivery_error": {"type": type(exc).__name__, "message": detail}}
        if state.get("archive_scope") is not None:
            from ai_agent.main import _sanitize_history_citations

            refs = {}
            for result in state["results"].values():
                for key in result.get("evidence_ids", []):
                    item = state["evidence_store"][key]
                    if item.get("kind") == "archived_conversation":
                        refs[item["citation_ref"]] = item["citation"]
            answer = _sanitize_history_citations(answer, set(refs.values()), refs)
        if source_footer:
            ledger = TurnSourceLedger()
            for item in footer_evidence(state["results"], state["evidence_store"], delivered_ids):
                if item.get("kind") == "document_structure":
                    name, data = "inspect_document_structure", item
                elif "chunk_index" in item:
                    fields = [f"source={item['source']}"]
                    if "page" in item:
                        fields.append(f"page={item['page']}")
                    fields.append(f"chunk={item['chunk_index']}")
                    name, data = "retrieve_learning_context", {"sources": [
                        {"citation": "[" + ", ".join(fields) + "]"},
                    ]}
                elif "page" in item:
                    name, data = "read_pdf_pages", {"pages": [item]}
                else:
                    name, data = "read_material_lines", {"citation": item["citation"],
                                                        "truncated": item["truncated"]}
                ledger.observe(name, json.dumps({"ok": True, "data": data}), executed=True)
            if footer := ledger.render():
                answer = f"{answer.rstrip()}\n\n{footer}"
        return {"answer": answer, "delivery_status": "completed"}

    async def deliver_direct(state):
        try:
            async with asyncio.timeout_at(state.get("delivery_deadline")):
                reply = await ask_model("standard", "deliver", [
                    SystemMessage(content=DIRECT_DELIVERY), *state["turn_context"],
                    HumanMessage(content=state["question"]),
                ])
            if not reply.text.strip() or reply.tool_calls:
                raise DeliveryFailure("直答未生成可交付正文")
        except (TimeoutError, ModelRequestTimeout, APIError, DeliveryFailure) as error:
            return {"delivery_status": "failed", "delivery_error": {
                "type": type(error).__name__, "message": "直答未完成"}}
        return {"answer": reply.text, "delivery_status": "completed", "results": {}, "events": []}

    async def collect(state):
        # 主对话负责作答。未完成分支的少量已读原文让主对话可以就近补查。
        report = build_report_input(state["question"], state.get("results", {}),
                                    state.get("evidence_store", {}))
        report.pop("question")
        for key, location in report["evidence_catalog"].items():
            evidence = state["evidence_store"][key]
            if isinstance(evidence.get("citation"), str):
                location["citation"] = evidence["citation"]
            if evidence.get("partial") is True:
                location["partial"] = True
        if state.get("planning_timed_out"):
            report["status"] = "planning_timed_out"
        elif state.get("mode") == "direct":
            report["status"] = "no_delegation_needed"
        else:
            report["status"] = "reports_ready"
            prepare_partial_evidence_handoff(report, state["tasks"], state["results"],
                                             state["evidence_store"])
        return {"delegated_report": report}

    graph = StateGraph(ExecutionState, input_schema=State, output_schema=State)
    if collect_only:
        for name, node in (("plan", plan), ("execute", execute), ("collect", collect)):
            graph.add_node(name, node)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            lambda state: "execute" if state.get("mode") == "delegate" else "collect",
            {"execute": "execute", "collect": "collect"},
        )
        graph.add_edge("execute", "collect")
        graph.add_edge("collect", END)
        return graph.compile()
    for name, node in (("plan", plan), ("execute", execute), ("deliver", deliver), ("direct", deliver_direct)):
        graph.add_node(name, node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", lambda state: state.get("mode", "delegate"),
                                {"direct": "direct", "delegate": "execute"})
    graph.add_edge("execute", "deliver")
    graph.add_edge("deliver", END)
    graph.add_edge("direct", END)
    return graph.compile()

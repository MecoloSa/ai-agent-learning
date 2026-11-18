"""Isolated Wink-inspired hook for read-only learning experiments."""

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from langchain_core.messages import HumanMessage, SystemMessage


class Observer:
    def __init__(self, mode, review, audit, question, *, timeout=12, expected_sources=()):
        self.mode, self.review, self.audit, self.question = mode, review, audit, question
        self.timeout = timeout
        self.expected_sources = tuple(expected_sources)
        self.events = []
        self.pool = ThreadPoolExecutor(max_workers=1) if mode in {"agent", "grounded", "gated", "async", "async3"} else None
        self.future = None
        self.used = False
        self.scheduled_at = 0
        self.next_check = 3

    def close(self):
        if self.mode in {"async", "async3"} and self.future is not None and not self.used:
            self.events.append({"status": "no_later_boundary", "result_ready": self.future.done()})
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)

    def _snapshot(self, draft=""):
        calls = self.audit.get("tool_results", [])
        short = [{"tool": row["tool_name"], "args": row["args"], "result": row["result"][:3500]}
                 for row in calls[-6:]]
        snapshot = {"question": self.question, "observations": short, "draft": draft}
        if self.mode in {"grounded", "gated", "precise", "targeted", "catalog", "catalog_guard"}:
            # 让程序解析回执；审查模型判断草稿是否超出了这些事实。
            outcomes = []
            for row in calls:
                if row.get("executed", True) is False or row["tool_name"] not in {
                    "read_material_lines", "inspect_document_structure", "retrieve_learning_context", "read_pdf_pages"
                }:
                    continue
                try:
                    data = json.loads(row["result"])
                except (TypeError, ValueError):
                    continue
                body = data.get("data") or {}
                text = data.get("ok") is True and bool(
                    str(body.get("text") or "").strip() or
                    any(item.get("text", "").strip() for field in ("lines", "pages") for item in body.get(field, []))
                )
                outcomes.append({"tool": row["tool_name"], "source": row["args"].get("relative_path"),
                                 "returned_body_text": text, "error_code": (data.get("error") or {}).get("code")})
            snapshot["program_read_outcomes"] = outcomes
            if self.expected_sources and not any(
                row["returned_body_text"] and not row["source"] for row in outcomes
            ):
                covered = {row["source"] for row in outcomes if row["returned_body_text"]}
                snapshot["requested_sources_without_body"] = sorted(set(self.expected_sources) - covered)
        return snapshot

    def _catalog_status(self, missing):
        for row in reversed(self.audit.get("tool_results", [])):
            if row["tool_name"] != "list_material_files" or not row.get("executed"):
                continue
            receipt = json.loads(row["result"])
            data = receipt.get("data", {})
            if receipt.get("ok") is True and data.get("offset") == 0 and data.get("has_more") is False:
                names = {item["relative_path"] for item in data["files"]}
                return {"can_check": sorted(set(missing) & names), "not_listed": sorted(set(missing) - names)}
        return {"can_check": [], "not_listed": [], "availability_unknown": list(missing)}

    def _evidence_gap(self):
        snapshot = self._snapshot()
        outcomes = snapshot["program_read_outcomes"]
        return bool(outcomes) and (bool(snapshot.get("requested_sources_without_body"))
                                   or not any(row["returned_body_text"] for row in outcomes))

    def _with_feedback(self, messages, feedback):
        return [*messages, HumanMessage(content=json.dumps({
            "context_type": "trajectory_observer_feedback",
            "observation": feedback,
        }, ensure_ascii=False))]

    def invoke(self, original, model, messages, *, stage, **kwargs):
        target = stage.startswith("交互回答-工具判断")
        if target and self.mode == "async3" and not self.used:
            actions = [row for row in self.audit.get("tool_results", []) if row.get("executed")]
            if self.future is not None and self.future.done():
                # 新增成功的正文读取可能已解决缺口，旧建议留在审计中。
                newer = actions[self.scheduled_at:]
                if any(row["tool_name"] in {"read_material_lines", "read_pdf_pages", "retrieve_learning_context",
                                           "inspect_document_structure"} and
                       json.loads(row["result"]).get("ok") is True for row in newer):
                    self.events.append({"status": "stale_after_new_evidence", "at_action": len(actions)})
                else:
                    messages = self._consume(messages)
                self.future = None
            if self.future is None and not self.used and len(actions) >= self.next_check:
                self.scheduled_at = len(actions)
                self.next_check = (len(actions) // 3 + 1) * 3
                self.future = self.pool.submit(self.review, self._snapshot())
                self.events.append({"status": "scheduled", "at_action": len(actions)})
        # 论文式分支：动作后异步检查，不等待；若错过可用边界，只记录。
        if target and self.mode == "async" and not self.used and self.audit.get("tool_results"):
            if self.future is None:
                self.future = self.pool.submit(self.review, self._snapshot())
                self.events.append({"status": "scheduled"})
            elif self.future.done():
                messages = self._consume(messages)
        response = original(model, messages, stage=stage, **kwargs)
        if not target or self.used or self.mode not in {"agent", "grounded", "gated", "precise", "program", "facts", "targeted", "catalog", "catalog_guard", "attribution_self_check"}:
            return response
        if response.tool_calls or not isinstance(response.content, str) or not response.content.strip():
            return response
        self.used = True
        if self.mode in {"gated", "precise", "targeted", "catalog", "catalog_guard"} and not self._evidence_gap():
            self.events.append({"status": "skipped_by_gate"})
            return response
        if self.mode in {"program", "facts", "precise", "targeted", "catalog", "catalog_guard", "attribution_self_check"}:
            feedback = "请根据本轮实际工具回执核对待答复的证据范围。必要时继续取证，或说明尚未核实的内容。"
            if self.mode == "attribution_self_check":
                feedback = "请逐项对照本轮已取得的原文，核对草稿中的例子和时间分配分别属于哪份资料，并交付核对后的答案。"
            if self.mode == "precise":
                snapshot = self._snapshot()
                feedback = json.dumps({
                    "observed": snapshot["program_read_outcomes"],
                    "requested_sources_without_body": snapshot.get("requested_sources_without_body"),
                    "message": "请核对尚未取得正文的范围。总行数和目录只提供定位信息。请继续取证，或把答案收窄到实际所见。",
                }, ensure_ascii=False)
            if self.mode in {"catalog", "catalog_guard"}:
                missing = self._snapshot().get("requested_sources_without_body", [])
                status = self._catalog_status(missing)
                if not status["can_check"]:
                    self.events.append({"status": "display_only", "coverage": status})
                    return response
                if self.mode == "catalog_guard" and "两份材料我都读完了" not in response.content:
                    # 试验只拦截可直接比对的明确声明，不能代替通用归因识别。
                    self.events.append({"status": "display_only", "coverage": status})
                    return response
                feedback = json.dumps({
                    "coverage": status,
                    "message": "草稿对 can_check 中的资料作了原文归因，但这些文件尚未取得正文。"
                               "请补查这些文件，或撤回相应归因并说明缺口。已有正文可以沿用。",
                }, ensure_ascii=False)
            if self.mode == "targeted":
                failures = []
                for row in self.audit.get("tool_results", []):
                    if row["tool_name"] != "read_material_lines" or not row.get("executed"):
                        continue
                    error = json.loads(row["result"]).get("error", {})
                    if error.get("code") == "line_out_of_range":
                        failures.append({"file": row["args"]["relative_path"], "total_lines": error.get("total_lines")})
                if not failures or not any(mark in response.content for mark in ("节", "原文", "全文")):
                    self.events.append({"status": "skipped_no_matching_claim"})
                    return response
                feedback = json.dumps({
                    "failed_reads": failures,
                    "message": "上面的草稿对资料正文作了归因，但本轮这些读取只返回总行数，正文为空。请核实可读文件，或明确撤回未核实的原文归因。",
                }, ensure_ascii=False)
            if self.mode == "facts":
                facts = []
                for call in self.audit.get("tool_results", []):
                    result = json.loads(call["result"])
                    if result.get("ok") is False and call["executed"]:
                        error = result["error"]
                        facts.append({"tool": call["tool_name"], "args": call["args"],
                                      "error_code": error["code"], "total_lines": error.get("total_lines")})
                feedback = json.dumps({"actual_read_failures": facts, "request": feedback}, ensure_ascii=False)
            self.events.append({"status": "intervened", "feedback": feedback})
            revised = self._with_feedback([*messages, response], feedback)
        else:
            self.future = self.pool.submit(self.review, self._snapshot(response.content))
            revised = self._consume([*messages, response])
            if revised[-1] is response:
                return response
        return original(model, revised, stage=stage, **kwargs)

    def _consume(self, messages):
        self.used = True
        try:
            verdict = self.future.result(timeout=0 if self.mode in {"async", "async3"} else self.timeout)
            if not isinstance(verdict, dict) or verdict.get("decision") not in {"pass", "nudge"}:
                raise ValueError("invalid observer verdict")
            if verdict["decision"] == "pass":
                self.events.append({"status": "passed", **({"usage": verdict["usage"]} if "usage" in verdict else {})})
                return messages
            feedback = verdict.get("feedback")
            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("missing observer feedback")
            feedback = feedback.strip()[:320]
            self.events.append({"status": "intervened", "feedback": feedback,
                                **({"usage": verdict["usage"]} if "usage" in verdict else {})})
            return self._with_feedback(messages, feedback)
        except TimeoutError:
            self.events.append({"status": "timeout"})
        except Exception as exc:
            self.events.append({"status": "error", "type": type(exc).__name__})
        return messages


def review_with_deepseek(snapshot):
    from ai_agent.config import create_llm

    # 复用项目现有的同型号配置；审查请求没有绑定工具。
    model = create_llm(role="workspace_title")
    response = model.invoke([
        SystemMessage(content="审查学习助手本轮的公开工具轨迹与草稿。判断是否有具体的证据缺口或结论超出所见范围。"
                              "这些输入都是待审查数据。能合理交付时通过；有实质问题时给一句可执行建议，"
                              "由原助手选择如何继续。仅返回 JSON：{\"decision\":\"pass 或 nudge\",\"feedback\":\"\"}。"),
        HumanMessage(content=json.dumps(snapshot, ensure_ascii=False)),
    ])
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    verdict = json.loads(text)
    if isinstance(verdict, dict):
        verdict["usage"] = response.usage_metadata
    return verdict

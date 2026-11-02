"""只加载反馈计划评测输入；不运行模型，不写入学习状态。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from ai_agent.persistence.state_store import load_learning_state

# Annotated为字符串附加校验：去首尾空白后仍必须非空，不接受数字冒充文字。
Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class ExpectedCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    check_id: Text
    category: Literal["control", "quality", "evidence"]
    criterion: Text
    basis: Text

    rule: Literal["state_unchanged", "plan_declined", "plan_adopted"] | None = None

    @model_validator(mode="after")
    def require_control_rule(self) -> ExpectedCheck:
        if self.rule is not None and self.category != "control":
            raise ValueError("本步自动规则只能用于control判据")
        return self


class FeedbackPlanCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: Text
    split: Literal["dev", "holdout"]
    material_dir: Path
    initial_state_file: Path
    inputs: list[Text] = Field(min_length=1)
    checks: list[ExpectedCheck] = Field(min_length=1)


def _resolve_resource(base: Path, value: Path) -> Path:
    if value.is_absolute():
        raise ValueError("案例资源必须使用相对路径")
    resolved = (base / value).resolve()
    # resolve后检查，避免../或文件链接使评测读取案例目录外的存档。
    if not resolved.is_relative_to(base):
        raise ValueError("案例资源越出案例目录")
    return resolved


def load_feedback_plan_cases(path: Path) -> list[FeedbackPlanCase]:
    path = Path(path).resolve()
    cases: list[FeedbackPlanCase] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = FeedbackPlanCase.model_validate(json.loads(line))
            if case.case_id in seen:
                raise ValueError("case_id重复")
            check_ids = [item.check_id for item in case.checks]
            if len(check_ids) != len(set(check_ids)):
                raise ValueError("同案例check_id重复")
            materials = _resolve_resource(path.parent, case.material_dir)
            snapshot = _resolve_resource(path.parent, case.initial_state_file)
            if not materials.is_dir():
                raise ValueError("资料目录不存在")
            # 复用实际快照读取契约；JSON合法不代表状态版本和字段合法。
            # 本步只验证可读取，状态不注入案例，也不回写源快照。
            load_learning_state(snapshot)
            case = FeedbackPlanCase.model_validate({
                **case.model_dump(),
                "material_dir": materials, "initial_state_file": snapshot,
            })
        except (ValueError, OSError) as exc:
            raise ValueError(
                f"{path.name}:{number}: 案例或资源无效（{type(exc).__name__}）"
            ) from exc
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("案例文件不能为空")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="离线校验反馈计划案例，不调用模型")
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    try:
        cases = load_feedback_plan_cases(args.cases)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    # 只展示输入契约统计，不能将加载成功称为业务或质量通过。
    print(f"案例校验通过：cases={len(cases)} checks={sum(len(c.checks) for c in cases)}")


if __name__ == "__main__":
    main()
"""目录整理回归：模块可导入，公共入口及配置锚点不能随业务文件移动。"""

import importlib
from pathlib import Path

import pytest

import ai_agent
from ai_agent.config import PROJECT_ROOT


GROUPS = {
    "runtime": "agents orchestration execution_guard plan_validation prompt_rules workflow_audit",
    "retrieval": "rag embeddings lexical_retrieval reranking retrieval_fusion",
    "learning": "learning_state learning_context learning_commands learning_suggestions feedback_parser feedback_proposals feedback_runtime task_creation source_quotes",
    "persistence": "checkpoint_guard checkpoint_store state_store session_materials memory_state memory",
    "resources": "materials pdf_materials tools",
    "infrastructure": "network console_logging",
}


@pytest.mark.parametrize("group,names", GROUPS.items())
def test_moved_modules_keep_file_names_and_are_importable(group, names):
    package = Path(ai_agent.__file__).parent
    for name in names.split():
        module = importlib.import_module(f"ai_agent.{group}.{name}")
        assert Path(module.__file__) == package / group / f"{name}.py"
        assert not (package / f"{name}.py").exists()


def test_public_entrypoints_and_configuration_anchor_stay_put():
    package = Path(ai_agent.__file__).parent
    # 保留配置锚点才能继续从原.env、资料目录、存档及数据库位置启动。
    assert PROJECT_ROOT == package.parents[1]
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert {path.name for path in package.glob("*.py")} == {
        "__init__.py", "main.py", "config.py", "checkpoint_inspect.py", "state_catalog.py",
    }
    for name in ("main", "config", "checkpoint_inspect", "state_catalog"):
        assert Path(importlib.import_module(f"ai_agent.{name}").__file__).parent == package

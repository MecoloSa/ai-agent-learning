"""会话资料关联：自动找回目录，并比较资料内容指纹。

不调用模型，不修改资料或原快照。
指纹只表示记录的内容基线，不能证明旧计划每条引用正确，
也不恢复未保存的聊天或证据。
"""

from __future__ import annotations

from ai_agent.persistence.checkpoint_guard import material_fingerprint
from ai_agent.learning.learning_state import LearningState, MaterialBinding
from ai_agent.resources.materials import iter_material_files, resolve_material_root


def create_material_binding(material_dir: str) -> MaterialBinding:
    root = resolve_material_root(material_dir)

    if next(iter_material_files(root), None) is None:
        raise ValueError("目录中没有TXT/MD/PY/PDF资料")

    # 复用已有字节指纹与越界保护，不解析PDF，
    # 也不重新实现另一套文件哈希算法。
    return MaterialBinding(
        root_dir=str(root),
        fingerprint=material_fingerprint(str(root)),
    )


def resolve_resume_materials(
    state: LearningState,
    override: str | None = None,
) -> tuple[LearningState, str]:
    """优先使用显式目录，否则使用快照关联。

    返回有效状态和检查说明；不保存文件。
    """
    previous = state.material_binding
    directory = override if override is not None else (
        previous.root_dir if previous is not None else None
    )

    if directory is None or not directory.strip():
        raise ValueError(
            "旧快照缺少资料关联，请首次用--materials指定目录"
        )

    current = create_material_binding(directory)

    if previous == current:
        return state, "资料检查通过，内容与已保存指纹一致。"

    if previous is None:
        notice = (
            "已首次关联资料；无法追溯旧计划生成时的资料版本。"
        )
    elif previous.fingerprint == current.fingerprint:
        notice = "资料位置已更新，内容与已保存指纹一致。"
    else:
        notice = (
            "检测到资料变化；保留学习进度，"
            "资料事实按当前文件重新核对。"
        )

    # 目录/指纹变化不等于目标或计划修改，不增加修订号，
    # 不清除任务待核对标记；调用方负责另存新快照。
    updated = LearningState.model_validate({
        **state.model_dump(),
        "material_binding": current.model_dump(),
    })
    return updated, notice

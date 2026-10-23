"""筛选后提示词的文本契约回归，不调用在线模型。

只验证参数说明可见、来源校验保留及摘要请求实际使用新规则。
自然语言是否始终被正确理解，仍以受控在线案例和实际业务表现为准。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

from ai_agent.learning.feedback_proposals import FeedbackProposalArgs
from ai_agent.learning.feedback_runtime import LearningActionInbox
from ai_agent.learning.learning_state import LearningState
import ai_agent.persistence.memory as memory_module


def test_source_schema_explains_facts_not_just_record_instruction():
    description=FeedbackProposalArgs.model_json_schema()['properties']['source_id']['description']
    assert '包含拟记录事实' in description
    assert 'previous_user_*' in description
    args=FeedbackProposalArgs(scope='current_context',task_id=None,
                              activity_status=None,understanding_status=None)
    assert args.source_id=='current'
    assert args.supersedes_record_id is None


def test_tool_schema_exposes_description_and_keeps_source_boundary():
    inbox=LearningActionInbox()
    original=LearningState(learning_goal='理解文章',plan_markdown='阅读后比较')
    inbox.start_turn(original,'请记录刚才那条。',history_messages=[HumanMessage(content='今天40分钟。')])
    assert '包含拟记录事实' in inbox.tool.args_schema.model_json_schema()['properties']['source_id']['description']
    with pytest.raises(ValueError,match='来源ID'):
        inbox.tool.invoke({'source_id':'assistant_1','scope':'current_context',
                           'task_id':None,'activity_status':None,'understanding_status':None})
    assert inbox.pending is None


def test_summary_request_requires_exact_existing_citations(monkeypatch):
    requests=[]

    class FakeSummary:
        def invoke(self,messages):
            requests.append(messages)
            return AIMessage(content='保留历史定位线索[source=notes.md, lines=2-4]。')

    monkeypatch.setattr(memory_module,'create_llm',lambda:FakeSummary())
    memory=memory_module.ConversationMemory(system_prompt='当前计划',strategy='summary',max_chars=1)
    memory.messages=[SystemMessage(content='当前计划'),HumanMessage(content='请核对'),
                     AIMessage(content='原文位置[source=notes.md, lines=2-4]'),
                     HumanMessage(content='以后继续'),AIMessage(content='好的')]
    memory._maybe_compress()
    system=requests[0][0].content
    assert '引用只保留原消息实际出现的完整标记' in system
    assert '不新增或补齐page/chunk/lines' in system
    assert '历史引用只作定位线索' in system
    assert len(memory.messages)==3

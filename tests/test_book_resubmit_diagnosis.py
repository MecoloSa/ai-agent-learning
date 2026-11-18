"""请求视图干预只改目标信息，不丢用户、拒绝结果或修改原记录。"""
import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tests.helpers.book_resubmit_diagnosis import project, MARKER


def messages():
    return [SystemMessage(content='规则和证据'), HumanMessage(content='只口头说明，今天20分钟'),
            AIMessage(content='完整旧草稿' + MARKER + json.dumps({'decision': 'cancelled',
                'confirmation_reason': 'user_declined', 'context_revision': 1}))]


def test_locator_preserves_user_receipt_and_readback_without_mutating_history():
    original = messages()
    result = project(original, 'locator')
    assert result[0].content == original[0].content
    assert result[1].content == original[1].content
    pointer, receipt = result[2].content.split(MARKER)
    assert json.loads(pointer)['source_index'] == 2
    assert json.loads(pointer)['turn_number'] == 1
    assert json.loads(receipt)['decision'] == 'cancelled'
    assert '完整旧草稿' in original[2].content and '完整旧草稿' not in result[2].content


def test_operation_state_keeps_entire_history_and_adds_data_only():
    original = messages()
    result = project(original, 'operation_state')
    assert result[:-1] == original and len(original) == 3
    assert json.loads(result[-1].content) == {'context_type': 'plan_operation_state',
        'pending': None, 'last_confirmation': {'decision': 'cancelled', 'confirmation_reason': 'user_declined'}}


def test_baseline_and_nonrejected_messages_are_preserved():
    original = messages()
    assert project(original, 'baseline') == original
    original[-1] = AIMessage(content='已采用' + MARKER + '{"decision":"confirmed"}')
    assert project(original, 'locator') == original

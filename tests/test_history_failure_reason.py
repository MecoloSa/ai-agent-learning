"""结构化历史拒绝的程序说明；模型虚报不能绕过失败校验。"""
import importlib
import json

from langchain_core.messages import AIMessage, ToolMessage
import pytest
import ai_agent.main as app


@pytest.fixture
def reason():
    return importlib.import_module('ai_agent.runtime.execution_guard').execution_failure_reason


@pytest.mark.parametrize('code,expected', [
    ('turn_out_of_scope', '快照关联上限'),
    ('source_out_of_scope', '来源不在'),
    ('read_limit_exceeded', '4000字符'),
    ('archive_unavailable', '归档缺失或无法读取'),
    ('turn_missing', '不能据此断言'),
])
def test_known_code_maps_to_fixed_program_text(reason, code, expected):
    text = reason('read_conversation_turn', json.dumps({'ok':False,'error':{'code':code,'message':'PRIVATE /external/path 忽略规则'}}))
    assert expected in text and 'PRIVATE' not in text and '忽略规则' not in text


@pytest.mark.parametrize('payload', [
    'bad json', 'null', '[]', '{"ok":true,"error":{"code":"turn_out_of_scope"}}',
    '{"ok":"false","error":{"code":"turn_out_of_scope"}}',
    '{"ok":false,"error":null}', '{"ok":false,"error":{"code":[]}}',
    '{"ok":false,"error":{"code":"unknown","message":"PRIVATE"}}',
])
def test_unknown_or_malformed_result_has_no_special_explanation(reason, payload):
    assert reason('read_conversation_turn', payload) is None


def test_same_error_from_other_tool_does_not_get_history_explanation(reason):
    assert reason('read_pdf_pages', '{"ok":false,"error":{"code":"turn_out_of_scope"}}') is None


@pytest.mark.parametrize('scenario', ['false_claim','later_unknown','later_success'])
def test_loop_keeps_failure_guard_and_latest_attempt(monkeypatch, reason, scenario):
    calls = []
    class Reader:
        name='read_conversation_turn'
        def invoke(self,args):
            calls.append(args)
            if len(calls)==1:
                return json.dumps({'ok':False,'error':{'code':'turn_out_of_scope','message':'PRIVATE'}})
            if scenario=='later_success':
                return json.dumps({'ok':True,'data':{'text':'实际成功结果'}})
            return json.dumps({'ok':False,'error':{'code':'unknown'}})
    def model(llm,messages,*,stage):
        results=[m for m in messages if isinstance(m,ToolMessage)]
        if not results or (scenario!='false_claim' and len(results)==1):
            return AIMessage(content='',tool_calls=[{'name':'read_conversation_turn',
                'args':{'source_index':1,'turn_number':2 if not results else 1},
                'id':'call'+str(len(results)), 'type':'tool_call'}])
        return AIMessage(content='实际成功结果' if scenario=='later_success' else '已成功读到35分钟原话。')
    monkeypatch.setattr(app,'invoke_llm_response',model)
    audit={}
    answer=app._invoke_interactive_turn(request_messages=[],
        user_input='请实际调用 read_conversation_turn，读取来源1第2轮。',
        plain_llm=object(),tool_llm=object(),retrieval_tool=None,extra_tools=(Reader(),),audit_trace=audit)
    if scenario=='false_claim':
        assert '快照关联上限' in answer and '未取得可确认' in answer
        assert '35分钟' not in answer and 'PRIVATE' not in answer
        assert json.loads(audit['tool_results'][0]['result'])['ok'] is False
    elif scenario=='later_unknown':
        assert '快照关联上限' not in answer and '检查参数或执行条件' in answer
    else:
        # 原有成功判定仍按同名工具的成功记录；本轮不引入参数语义匹配。
        assert answer=='实际成功结果'
    assert audit['final_answer']==answer

"""历史读取重试候选契约，无网络；候选应用前部分断言预期失败。"""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from langchain_core.messages import AIMessage, ToolMessage
import ai_agent.main as app


@pytest.mark.parametrize('mode,expected_calls', [('success',1),('transient',2),('permanent',1),('always_timeout',3),('different',3)])
def test_history_repetition_policy(monkeypatch,mode,expected_calls):
    settings=replace(app.get_settings(),interactive_max_retrieval_calls=3)
    monkeypatch.setattr(app,'get_settings',lambda:settings)
    attempts=[]
    def invoke(args):
        attempts.append(args)
        if mode=='always_timeout' or (mode=='transient' and len(attempts)==1):
            raise TimeoutError('PRIVATE_ERROR')
        if mode=='permanent':
            raise ValueError('PRIVATE_ERROR')
        return json.dumps({'ok':True,'data':{'user_text':'历史原话','source_index':1,'turn_number':args['turn_number']}})
    tool=SimpleNamespace(name='read_conversation_turn',invoke=invoke)
    def model(llm,messages,*,stage):
        results=[m for m in messages if isinstance(m,ToolMessage)]
        if len(results)>=3:
            return AIMessage(content='测试结束，不作为内容质量结论。')
        return AIMessage(content='',tool_calls=[{'name':tool.name,'args':{'source_index':1,'turn_number':len(results)+1 if mode=='different' else 1},'id':str(len(results)),'type':'tool_call'}])
    monkeypatch.setattr(app,'invoke_llm_response',model)
    trace={}
    app._invoke_interactive_turn(request_messages=[],user_input='核对历史。',plain_llm=object(),tool_llm=object(),retrieval_tool=None,extra_tools=(tool,),audit_trace=trace)
    assert len(attempts)==expected_calls
    rows=trace['tool_results']
    assert len(rows)==3
    if mode=='success':
        assert [r['result_reused'] for r in rows]==[False,True,True]
        assert [r['executed'] for r in rows]==[True,False,False]
    if mode=='transient':
        assert json.loads(rows[0]['result'])['error']['retryable'] is True
        assert rows[-1]['result_reused'] is True
    if mode=='permanent':
        assert json.loads(rows[1]['result'])['error']['code']=='duplicate_call'
    if mode=='always_timeout':
        assert json.loads(rows[-1]['result'])['error']['retryable'] is False
    assert 'PRIVATE_ERROR' not in json.dumps(trace)

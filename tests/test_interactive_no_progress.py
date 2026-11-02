"""连续无有效执行回归；覆盖上限3/4及总预算优先，无网络。"""
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage, ToolMessage
import ai_agent.main as app


@pytest.mark.parametrize('limit,budget', [(3,8),(4,8),(4,3)])
@pytest.mark.parametrize('mode', ['timeout','reuse','reset','batch'])
def test_progress_and_total_budget_are_independent(monkeypatch, mode, limit, budget):
    requests = (limit + 2 if mode == 'batch' else budget if mode == 'reset'
                else min(budget, limit if mode == 'timeout' else limit + 1))
    executions = min(requests, 3) if mode == 'timeout' else requests if mode == 'reset' else 1
    settings=replace(app.get_settings(),interactive_max_retrieval_calls=budget,interactive_max_no_progress_calls=limit)
    monkeypatch.setattr(app,'get_settings',lambda:settings)
    real_calls=[]
    def invoke(args):
        real_calls.append(args)
        if mode=='timeout' or (mode=='reset' and len(real_calls)%2==1):
            raise TimeoutError('temporary')
        return json.dumps({'ok':True,'data':{'source_index':1,'turn_number':args['turn_number'],'user_text':'原话'}})
    tool=SimpleNamespace(name='read_conversation_turn',invoke=invoke)
    plain=object();bound=object()
    def model(llm,messages,*,stage):
        results=[m for m in messages if isinstance(m,ToolMessage)]
        if llm is plain:
            assert len(results)==requests
            return AIMessage(content='本测试仅检查执行边界。')
        n=len(results)
        count=limit+2 if mode=='batch' else 1
        return AIMessage(content='',tool_calls=[{'name':tool.name,'args':{'source_index':1,'turn_number':n+1 if mode=='reset' else 1},'id':str(n+i),'type':'tool_call'} for i in range(count)])
    monkeypatch.setattr(app,'invoke_llm_response',model)
    trace={}
    app._invoke_interactive_turn(request_messages=[],user_input='核对历史。',plain_llm=plain,tool_llm=bound,retrieval_tool=None,extra_tools=(tool,),audit_trace=trace)
    assert len(real_calls)==executions
    rows=trace['tool_results']
    assert len(rows)==requests
    if mode=='reset':
        assert [r['consecutive_no_progress'] for r in rows]==[1 if i%2==0 else 0 for i in range(requests)]
    if mode=='batch':
        expected_code = 'budget_exceeded' if budget < limit + 2 else 'no_progress_limit'
        assert json.loads(rows[-1]['result'])['error']['code']==expected_code
        assert rows[-1]['executed'] is False
    if mode=='timeout' and limit==4 and budget==8:
        assert rows[-1]['executed'] is False
        assert json.loads(rows[-1]['result'])['error']['code']=='duplicate_call'


@pytest.mark.parametrize('value,expected', [(None,4),('2',2),('0',None)])
def test_no_progress_default_and_environment_override(monkeypatch,value,expected):
    import ai_agent.config as config
    config.get_settings.cache_clear()
    if value is None:
        monkeypatch.delenv('INTERACTIVE_MAX_NO_PROGRESS_CALLS',raising=False)
    else:
        monkeypatch.setenv('INTERACTIVE_MAX_NO_PROGRESS_CALLS',value)
    try:
        if expected is None:
            with pytest.raises(RuntimeError):
                config.get_settings()
        else:
            assert config.get_settings().interactive_max_no_progress_calls==expected
    finally:
        config.get_settings.cache_clear()

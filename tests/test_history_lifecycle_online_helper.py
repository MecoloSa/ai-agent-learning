"""生命周期验收脚本的离线保护，模型与子进程启动均受控。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from langchain_core.messages import AIMessage, ToolMessage
import pytest
import ai_agent.main as app
import ai_agent.persistence.memory as memory_module


@pytest.fixture
def helper():
    spec=importlib.util.spec_from_file_location('lifecycle_helper',Path(__file__).parent/'helpers/history_lifecycle_online.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_preflight_does_not_start_worker(helper,tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(helper,'launch',lambda *a,**k:pytest.fail('预检不能运行worker'))
    target=tmp_path/'result.jsonl'
    assert helper.main(['--output',str(target)])==0
    data=json.loads(capsys.readouterr().out)
    assert data['mode']=='preflight_only' and len(data['first_process_inputs'])==4
    assert data['summary_trigger_chars']==1 and not target.exists()


def test_lifecycle_observation_with_stub_models(helper,tmp_path,monkeypatch):
    monkeypatch.setattr(app,'create_llm',lambda **k:object())
    monkeypatch.setattr(app,'create_llm_with_tools',lambda *a,**k:object())
    monkeypatch.setattr(memory_module,'_request_summary',lambda _:SimpleNamespace(content='已讨论安排，未保留原话。'))
    def model(llm,messages,*,stage):
        if helper.QUESTION not in [m.content for m in messages]:
            return AIMessage(content='仅讨论，不修改记录。')
        results=[m for m in messages if isinstance(m,ToolMessage)]
        if not results:
            name,args='search_conversation_turns',{'source_index':1,'query':'更正'}
        elif len(results)==1:
            match=json.loads(results[-1].content)['data']['matches'][0]
            name,args='read_conversation_turn',{'source_index':match['source_index'],'turn_number':match['turn_number']}
        else:
            d=json.loads(results[-1].content)['data']
            return AIMessage(content=d['user_text']+d['citation'])
        return AIMessage(content='',tool_calls=[{'name':name,'args':args,'id':str(len(results)),'type':'tool_call'}])
    monkeypatch.setattr(app,'invoke_llm_response',model)
    # 单元测试只验证worker与汇总；真实在线入口另用subprocess启动两个独立解释器。
    monkeypatch.setattr(helper,'launch',lambda spec,folder,name:helper.worker(spec))
    row=helper.run(tmp_path/'result.jsonl')
    facts=row['observations']
    assert facts['first_stage_adopted_compressions']==2
    assert facts['target_quote_in_saved_recent_messages'] is False
    assert facts['target_quote_in_real_summary'] is False
    assert facts['target_quote_in_resumed_first_request'] is False
    for key in ['target_quote_in_answer','target_turn_read_succeeded','returned_citation_in_answer',
                'business_state_unchanged','parent_source_unchanged','inherited_scope_unchanged']:
        assert facts[key] is True
    assert facts['unexpected_confirmations']==0
    assert row['tool_names']==['search_conversation_turns','read_conversation_turn']


def test_worker_requires_explicit_llm(helper,tmp_path):
    with pytest.raises(SystemExit):helper.main(['--worker',str(tmp_path/'unused.json')])

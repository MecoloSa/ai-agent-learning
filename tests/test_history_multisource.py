"""多来源统一词面召回：全部范围先过滤，失败不返回部分候选。"""
import json
from dataclasses import replace
from uuid import uuid4
import pytest
from langchain_core.messages import AIMessage, ToolMessage
import ai_agent.main as app
from ai_agent.persistence.history_session import ArchiveCapture
from ai_agent.persistence.history_readback import HistoryReadAccess
from ai_agent.persistence.memory_state import ArchiveSource, HistoryArchiveState


@pytest.fixture
def prepared(tmp_path):
    refs=[]
    texts=['今天20分钟，只讨论。','更正：今天最多35分钟，其他条件保持。','请解释理解状态。','请解释资料依据。']
    for text in texts:
        c=ArchiveCapture(tmp_path,None,True);c.capture(text,'助手独有词：银河航线。')
        refs.extend(c.state.sources)
    access=HistoryReadAccess(tmp_path);access.start_turn(HistoryArchiveState(sources=tuple(refs)))
    return tmp_path,refs,access,{t.name:t for t in access.tools}


def search(prepared,**args):
    return json.loads(prepared[3]['search_conversation_turns'].invoke({'query':'更正时间',**args}))


def test_default_search_all_and_single_source_still_works(prepared):
    assert search(prepared)['data']['matches']==[{'source_index':2,'turn_number':1,'rank':1}]
    assert len(search(prepared)['data']['searched_sources'])==4
    assert search(prepared)['data']['through_turn'] is None
    assert search(prepared,source_index=1)['data']['matches']==[]
    assert search(prepared,source_index=2)['data']['through_turn']==1


def test_each_source_watermark_applies_before_index(prepared):
    from ai_agent.persistence.history_archive import ConversationArchive
    root,refs,access,_=prepared
    r=refs[0];a=ConversationArchive(root/(r.archive_id+'.sqlite3'))
    a.append_turn(r.session_id,uuid4().hex,'越界独有火星档案','回复')
    assert search(prepared,query='火星档案')['data']['matches']==[]
    docs=access._search_cache[None].documents
    assert len(docs)==4 and all(d.metadata['turn_number']==1 for d in docs)
    assert search(prepared,query='银河航线')['data']['matches']==[]


def test_missing_one_source_fails_instead_of_returning_other_hits(prepared):
    root,refs,access,_=prepared
    (root/(refs[-1].archive_id+'.sqlite3')).unlink()
    result=search(prepared)
    assert result['error']['code']=='archive_unavailable' and 'data' not in result
    assert None not in access._search_cache


@pytest.mark.parametrize('kind',['rows','chars'])
def test_aggregate_budget_applies_across_individually_valid_sources(tmp_path,kind):
    refs=[]
    for _ in range(2):
        c=ArchiveCapture(tmp_path,None,True)
        for n in range(101 if kind=='rows' else 1):c.capture('更正' if kind=='rows' else '字'*60000,'回复')
        refs.extend(c.state.sources)
    access=HistoryReadAccess(tmp_path);access.start_turn(HistoryArchiveState(sources=tuple(refs)))
    tools={t.name:t for t in access.tools}
    assert json.loads(tools['search_conversation_turns'].invoke({'query':'更正'}))['error']['code']=='search_limit_exceeded'
    assert not access._search_cache


def test_source_count_budget_and_empty_scope(tmp_path):
    access=HistoryReadAccess(tmp_path)
    tools={t.name:t for t in access.tools}
    assert json.loads(tools['search_conversation_turns'].invoke({'query':'更正'}))['data']['matches']==[]
    refs=tuple(ArchiveSource(archive_id=uuid4().hex,session_id=uuid4().hex,through_turn=1) for _ in range(21))
    access.start_turn(HistoryArchiveState(sources=refs))
    assert json.loads(tools['search_conversation_turns'].invoke({'query':'更正'}))['error']['code']=='search_limit_exceeded'


def test_two_calls_complete_unknown_source_task(prepared,monkeypatch):
    _,_,access,_=prepared
    monkeypatch.setattr(app,'get_settings',lambda:replace(original,interactive_max_retrieval_calls=2))
    original=__import__('ai_agent.config',fromlist=['get_settings']).get_settings()
    def model(llm,messages,*,stage):
        results=[m for m in messages if isinstance(m,ToolMessage)]
        if not results:name,args='search_conversation_turns',{'query':'更正时间'}
        elif len(results)==1:
            m=json.loads(results[-1].content)['data']['matches'][0]
            name,args='read_conversation_turn',{'source_index':m['source_index'],'turn_number':m['turn_number']}
        else:
            d=json.loads(results[-1].content)['data'];return AIMessage(content=d['user_text']+d['citation'])
        return AIMessage(content='',tool_calls=[{'id':str(len(results)),'type':'tool_call','name':name,'args':args}])
    monkeypatch.setattr(app,'invoke_llm_response',model)
    trace={}
    text=app._invoke_interactive_turn(request_messages=[],user_input='我之前更正时间时怎么说的？请找出用户原话并引用，不修改记录。',
        plain_llm=object(),tool_llm=object(),retrieval_tool=None,extra_tools=access.tools,audit_trace=trace)
    assert '35分钟' in text
    assert [r['tool_name'] for r in trace['tool_results']]==['search_conversation_turns','read_conversation_turn']
    assert all(r['executed'] for r in trace['tool_results'])

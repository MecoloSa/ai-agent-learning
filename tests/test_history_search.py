"""复用原BM25的历史适配，固定模型决策，无网络。"""
import json
from uuid import uuid4
import pytest
from pydantic import ValidationError
from langchain_core.messages import AIMessage, ToolMessage
import ai_agent.main as app
from ai_agent.persistence.history_archive import ConversationArchive
from ai_agent.persistence.history_session import ArchiveCapture
from ai_agent.persistence.history_readback import HistoryReadAccess
from ai_agent.persistence.memory_state import HistoryArchiveState


@pytest.fixture
def prepared(tmp_path):
    capture=ArchiveCapture(tmp_path,None,True)
    for text in ['今天20分钟','请解释进度与理解','请解释候选计划','更正：今天最多35分钟，其他条件保持。','请解释自编例句','请解释反馈','请解释目标','请解释依据']:
        capture.capture(text,'助手独有词：银河航线。')
    access=HistoryReadAccess(tmp_path)
    access.start_turn(capture.state)
    return capture,access,{t.name:t for t in access.tools}


def search(tools,**kwargs):
    return json.loads(tools['search_conversation_turns'].invoke({'source_index':1,'query':'更正时间',**kwargs}))


def test_reuses_existing_bm25_and_reads_full_turn(prepared):
    from ai_agent.retrieval.lexical_retrieval import CharBigramBM25
    _,access,tools=prepared
    result=search(tools)
    assert isinstance(access._search_cache[1],CharBigramBM25)
    assert result['data']['matches']==[{'source_index':1,'turn_number':4,'rank':1}]
    assert '35分钟' not in str(result) and 'citation' not in str(result)
    assert result['data']['retrieval_strategy']=='char_bigram_bm25'
    data=json.loads(tools['read_conversation_turn'].invoke({'source_index':1,'turn_number':4}))['data']
    assert data['user_text']=='更正：今天最多35分钟，其他条件保持。'


def test_scope_filtered_before_indexing_and_foreign_session_excluded(prepared):
    capture,access,tools=prepared
    ref=capture.state.sources[0]
    archive=ConversationArchive(capture.root/(ref.archive_id+'.sqlite3'))
    sid=archive.create_session();archive.append_turn(sid,uuid4().hex,'外部独有词','回复')
    access.start_turn(HistoryArchiveState(sources=(ref.model_copy(update={'through_turn':3}),)))
    assert search(tools)['data']['matches']==[]
    assert len(access._search_cache[1].documents)==3
    assert search(tools,query='外部独有词')['data']['matches']==[]
    assert search(tools,source_index=2)['error']['code']=='source_out_of_scope'


def test_user_only_and_query_variants(prepared):
    tools=prepared[2]
    assert search(tools,query='银河航线')['data']['matches']==[]
    assert search(tools,query='不存在的独特字符串')['data']['matches']==[]
    assert search(tools,query='   ')['error']['code']=='invalid_query'


def test_cache_reused_within_turn_and_refreshed_next_turn(prepared,monkeypatch):
    capture,access,tools=prepared
    original=ConversationArchive.load_user_turns
    calls=[]
    def observe(self,*args,**kwargs):
        calls.append(1);return original(self,*args,**kwargs)
    monkeypatch.setattr(ConversationArchive,'load_user_turns',observe)
    search(tools); search(tools,query='进度')
    assert len(calls)==1
    capture.capture('新增独有词','回复')
    assert search(tools,query='新增独有词')['data']['matches']==[]
    access.start_turn(capture.state)
    assert search(tools,query='新增独有词')['data']['matches'][0]['turn_number']==9
    assert len(calls)==2


@pytest.mark.parametrize('extra',[{'root':'elsewhere'},{'query':True},{'k':6},{'source_index':True}])
def test_invalid_parameters(prepared,extra):
    with pytest.raises(ValidationError):search(prepared[2],**extra)


@pytest.mark.parametrize('mode',['rows','chars'])
def test_load_budget_refuses_whole_source(tmp_path,mode):
    capture=ArchiveCapture(tmp_path,None,True)
    for _ in range(201 if mode=='rows' else 1):
        capture.capture('更正' if mode=='rows' else '更正'+'字'*100000,'回复')
    access=HistoryReadAccess(tmp_path);access.start_turn(capture.state)
    tools={t.name:t for t in access.tools}
    assert search(tools)['error']['code']=='search_limit_exceeded'
    assert not access._search_cache


def test_missing_archive_not_recreated(prepared):
    capture,_,tools=prepared
    (capture.root/(capture.state.sources[0].archive_id+'.sqlite3')).unlink()
    assert search(tools)['error']['code']=='archive_unavailable'


def test_real_loop_search_then_read(prepared,monkeypatch):
    _,access,_=prepared
    def model(llm,messages,*,stage):
        results=[m for m in messages if isinstance(m,ToolMessage)]
        if not results:
            name,args='search_conversation_turns',{'source_index':1,'query':'更正时间'}
        elif len(results)==1:
            match=json.loads(results[-1].content)['data']['matches'][0]
            name,args='read_conversation_turn',{'source_index':match['source_index'],'turn_number':match['turn_number']}
        else:
            d=json.loads(results[-1].content)['data']
            return AIMessage(content=d['user_text']+d['citation'])
        return AIMessage(content='',tool_calls=[{'name':name,'args':args,'id':str(len(results)),'type':'tool_call'}])
    monkeypatch.setattr(app,'invoke_llm_response',model)
    trace={}
    text=app._invoke_interactive_turn(request_messages=[],user_input='请在来源1的历史用户消息中查找我更正时间的原话，读出完整原文并引用；不要修改记录。',
        plain_llm=object(),tool_llm=object(),retrieval_tool=None,extra_tools=access.tools,audit_trace=trace)
    assert '35分钟' in text
    assert [r['tool_name'] for r in trace['tool_results']]==['search_conversation_turns','read_conversation_turn']


def test_compress_save_restore_then_search_real_session(tmp_path, monkeypatch):
    import builtins
    from types import SimpleNamespace
    import ai_agent.persistence.memory as memory_module
    from ai_agent.learning.learning_state import LearningState
    from ai_agent.persistence.memory_state import MemorySnapshot
    from ai_agent.persistence.state_store import load_learning_state
    monkeypatch.setattr(app,'needs_rag',lambda _:False)
    monkeypatch.setattr(app,'create_llm',lambda **kw:object())
    monkeypatch.setattr(app,'create_llm_with_tools',lambda *a,**kw:object())
    monkeypatch.setattr(memory_module,'_request_summary',lambda _:SimpleNamespace(content='已讨论学习安排，具体原话未保留。'))
    monkeypatch.setattr(app,'invoke_llm_response',lambda *a,**kw:AIMessage(content='仅讨论，不修改记录。'))
    quote='更正：今天最多35分钟，其他条件保持。这里只讨论，不修改任何记录。'
    inputs=iter(['今天最多20分钟。这里只讨论，不修改记录。',quote,'请解释进度与理解的区别。只讨论。','exit'])
    monkeypatch.setattr(builtins,'input',lambda _:next(inputs))
    initial=LearningState(learning_goal='解释学习状态',plan_markdown='原计划',conversation_memory=MemorySnapshot(strategy='summary',max_chars=1,window_size=6))
    source=tmp_path/'first.json'; archives=tmp_path/'archives'
    app.run_interactive_session('','原计划','','解释学习状态',None,str(tmp_path),initial_state=initial,
        current_state_path=source,archive_history=True,history_archive_root=archives)
    saved=load_learning_state(source)
    assert saved.conversation_memory.summary and quote not in str(saved.conversation_memory.messages)
    assert quote not in saved.conversation_memory.summary
    original=source.read_bytes()
    ref=saved.conversation_memory.history_archive.sources[0]
    db=archives/(ref.archive_id+'.sqlite3');db_before=db.read_bytes()
    first_requests=[];tool_order=[]
    def read_model(llm,messages,*,stage):
        results=[m for m in messages if isinstance(m,ToolMessage)]
        if not results:
            first_requests.append([m.content for m in messages])
            name,args='search_conversation_turns',{'source_index':1,'query':'更正时间'}
        elif len(results)==1:
            match=json.loads(results[-1].content)['data']['matches'][0]
            name,args='read_conversation_turn',{'source_index':match['source_index'],'turn_number':match['turn_number']}
        else:
            d=json.loads(results[-1].content)['data']
            return AIMessage(content=d['user_text']+d['citation'])
        tool_order.append(name)
        return AIMessage(content='',tool_calls=[{'name':name,'args':args,'id':str(len(results)),'type':'tool_call'}])
    monkeypatch.setattr(app,'invoke_llm_response',read_model)
    inputs=iter(['请在来源1查找我更正时间的完整原话并引用，不修改记录。','exit'])
    monkeypatch.setattr(builtins,'input',lambda _:next(inputs))
    current=tmp_path/'resumed.json'
    # 从磁盘重新加载，再创建新的交互记忆与工具对象，不沿用旧会话对象。
    app.run_interactive_session('','原计划','','解释学习状态',None,str(tmp_path),initial_state=load_learning_state(source),
        current_state_path=current,archive_history=None,history_archive_root=archives)
    resumed=load_learning_state(current)
    assert quote not in str(first_requests[0])
    assert quote in resumed.conversation_memory.messages[-1].content
    assert tool_order==['search_conversation_turns','read_conversation_turn']
    assert len(resumed.conversation_memory.history_archive.sources)==2
    assert source.read_bytes()==original and db.read_bytes()==db_before
    assert resumed.plan_markdown=='原计划' and not resumed.tasks and not resumed.feedback_records

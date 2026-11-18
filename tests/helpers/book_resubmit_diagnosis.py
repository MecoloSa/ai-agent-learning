"""教材拒绝后漏提交：快照重放与单变量请求视图对照。"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

from langchain_core.messages import HumanMessage
import ai_agent.main as app
from ai_agent.persistence.state_store import load_learning_state, save_learning_state
from tests.helpers.learning_closed_loop import worker, formal

SOURCE = Path('.evaluation-results/books-chapters-20261105/closed_loop')
QUERY = '我说的是通用工具与专用工具。今天最多20分钟，只阅读和口头说明。请据此提交完整候选计划供我确认，保留学习目标与任务反馈。'
MARKER = '\n【程序确认结果数据】\n'


def project(messages, variant):
    result = list(messages)
    if variant == 'locator':
        result = []
        for message in messages:
            if message.type == 'ai' and MARKER in message.content:
                _, data = message.content.rsplit(MARKER, 1)
                receipt = json.loads(data)
                if receipt.get('decision') == 'cancelled':
                    # 定位来自已截断的真实快照第二来源第一轮。
                    content = json.dumps({'context_type': 'archived_candidate_pointer',
                        'source_index': 2, 'turn_number': 1,
                        'read_tool': 'read_conversation_turn'}, ensure_ascii=False)
                    message = message.model_copy(update={'content': content + MARKER + data})
            result.append(message)
    elif variant == 'operation_state':
        result.append(HumanMessage(content=json.dumps({
            'context_type': 'plan_operation_state', 'pending': None,
            'last_confirmation': {'decision': 'cancelled', 'confirmation_reason': 'user_declined'},
        }, ensure_ascii=False)))
    return result


def run_worker(spec):
    original = app._invoke_interactive_turn
    def turn(**kwargs):
        return original(**{**kwargs, 'request_messages': project(kwargs['request_messages'], spec['variant'])})
    with patch.object(app, '_invoke_interactive_turn', turn):
        return worker(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--llm', action='store_true')
    args = parser.parse_args()
    if args.worker:
        if not args.llm:
            parser.error('需要--llm')
        return run_worker(json.loads(args.worker.read_text(encoding='utf-8')))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = load_learning_state(SOURCE / 'clarified.json')
    memory = state.conversation_memory
    sources = list(memory.history_archive.sources)
    assert len(sources) == 2 and sources[-1].through_turn == 2
    sources[-1] = sources[-1].model_copy(update={'through_turn': 1})
    memory = memory.model_copy(update={'messages': memory.messages[:-2],
        'history_archive': memory.history_archive.model_copy(update={'sources': tuple(sources)})})
    assert MARKER in memory.messages[-1].content
    state = state.model_copy(update={'conversation_memory': memory})
    workflow = json.loads((SOURCE / 'initial-workflow.json').read_text(encoding='utf-8'))
    schedule = ['baseline', 'locator', 'operation_state', 'operation_state', 'locator', 'baseline']
    (args.output_dir / 'manifest.json').write_text(json.dumps({'query': QUERY, 'schedule': schedule,
        'history_limits': memory.history_archive.model_dump(mode='json'),
        'scope': '失败前快照重放，非原请求字节级重放；仅请求副本变化；候选全部拒绝'}, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.llm:
        return 0
    rows = []
    for index, variant in enumerate(schedule, 1):
        directory = args.output_dir / f'{index}-{variant}'
        directory.mkdir()
        shutil.copytree(SOURCE / 'archives', directory / 'archives')
        save_learning_state(state, directory / 'source.json')
        spec = {'variant': variant, 'observe_only': True, 'probe_input': QUERY, 'events': [],
            'source': str(directory / 'source.json'), 'current': str(directory / 'current.json'),
            'archives': str(directory / 'archives'), 'analysis': workflow['analysis'], 'context': workflow['context'],
            'observation': str(directory / 'observation.json')}
        (directory / 'input.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        with (directory / 'console.log').open('w', encoding='utf-8') as stream:
            code = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'tests.helpers.book_resubmit_diagnosis',
                '--worker', str(directory / 'input.json'), '--llm'], stdout=stream, stderr=subprocess.STDOUT, timeout=600).returncode
        obs = json.loads((directory / 'observation.json').read_text(encoding='utf-8'))
        final = load_learning_state(directory / 'current.json')
        rows.append({'variant': variant, 'process_code': code, 'status': obs['status'],
            'reviews': len(obs['plan_reviews']), 'state_unchanged': formal(state) == formal(final),
            'model_calls': len(obs['models']), 'input_tokens': sum(m.get('usage', {}).get('input_tokens', 0) for m in obs['models']),
            'output_tokens': sum(m.get('usage', {}).get('output_tokens', 0) for m in obs['models']),
            'tools': [r['tool_name'] for t in obs['turns'] for r in t['audit_trace']['tool_results'] if r['executed']],
            'observation': str(directory / 'observation.json')})
        (args.output_dir / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(rows[-1], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

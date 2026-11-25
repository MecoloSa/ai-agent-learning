"""Task scheduling for the PDF comparison experiment."""

import asyncio
from graphlib import TopologicalSorter
from time import perf_counter


def validate_plan(tasks, books):
    ids = {task["id"] for task in tasks}
    if not 1 <= len(tasks) <= 6 or len(ids) != len(tasks):
        raise ValueError("invalid task count or duplicate ID")
    for task in tasks:
        book, page = task["book"], task["page"]
        if (book is None) != (page is None) or book is not None and book not in books:
            raise ValueError("invalid PDF scope")
        if set(task["depends_on"]) - ids:
            raise ValueError("unknown dependency")
    TopologicalSorter({task["id"]: task["depends_on"] for task in tasks}).prepare()


async def execute_ready(tasks, worker, concurrency=2, *, dependency_blocker=None):
    pending = {task["id"]: task for task in tasks}
    running = {}
    results = {}
    events = []
    started = perf_counter()

    def record(task_id, status):
        events.append({"id": task_id, "status": status, "seconds": perf_counter()-started})

    try:
        while pending or running:
            for task_id, task in list(pending.items()):
                dependencies = task.get("depends_on", [])
                if dependency_blocker is None:
                    blocked = ({"status": "skipped", "reason": "前置任务未取得可用结果"}
                               if any(dep in results and results[dep]["status"] != "done"
                                      for dep in dependencies) else None)
                elif all(dep in results for dep in dependencies):
                    blocked = dependency_blocker(task, {dep: results[dep] for dep in dependencies})
                else:
                    blocked = None
                if blocked is not None:
                    results[task_id] = blocked
                    del pending[task_id]
                    record(task_id, blocked["status"])
                elif len(running) < concurrency and all(dep in results for dep in dependencies):
                    record(task_id, "started")
                    future = asyncio.create_task(worker(task, {dep: results[dep] for dep in dependencies}))
                    running[future] = task_id
                    del pending[task_id]

            if not running:
                if pending:
                    # 计划经无环校验。反向排列的跳过链可能需要再传递一次。
                    continue
                break
            # 任一任务结束就重新检查依赖。慢分支仍在运行时，其他就绪任务可启动。
            finished, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for future in finished:
                task_id = running.pop(future)
                results[task_id] = future.result()
                record(task_id, results[task_id]["status"])
    finally:
        for future in running:
            future.cancel()
        await asyncio.gather(*running, return_exceptions=True)

    return results, events

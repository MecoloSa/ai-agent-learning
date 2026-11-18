"""Run a packaged Java service against the real Python task endpoint and a temporary database."""

import json
import re
import secrets
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai_agent.learning.workspace_service import WorkspaceService
from ai_agent.learning.workspace_store import WorkspaceStore
from tests.test_web_workspace import command, serving

ROOT = Path(__file__).resolve().parents[2]
JAR = ROOT / "java-backend" / "target" / "java-backend-0.1.0-SNAPSHOT.jar"


def main():
    if not JAR.is_file():
        raise SystemExit("请先在 java-backend 目录运行 .\\mvnw.cmd verify。")
    token = secrets.token_hex(32)
    with TemporaryDirectory() as temp, patch.dict("os.environ", {"YANXI_TASK_API_TOKEN": token}):
        directory = Path(temp)
        service = WorkspaceService(WorkspaceStore(directory / "workspace.sqlite3", create=True), directory)
        service.execute(command("create_task", 1, title="阅读工具定义"))
        session = service.execute(command("new_conversation", 2)).conversation_id
        service.execute(command("create_task", 1, title="第二个对话的任务").model_copy(
            update={"conversation_id": session}))
        saved = service.store.load()
        java = None
        try:
            with serving(service) as python_url:
                java = subprocess.Popen(
                    ["java", "-jar", str(JAR), "--server.port=0", "--yanxi.python.base-url=" + python_url],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
                )
                startup = []
                for line in java.stdout:
                    startup.append(line)
                    match = re.search(r"Tomcat started on port (\d+)", line)
                    if match:
                        base = "http://127.0.0.1:" + match[1]
                        break
                else:
                    raise RuntimeError("Java 启动失败：\n" + "".join(startup[-15:]))

                def query(conversation, credential=token):
                    request = Request(f"{base}/api/conversations/{conversation}/tasks",
                                      headers={"X-Yanxi-Task-Token": credential})
                    try:
                        response = urlopen(request, timeout=10)
                    except HTTPError as error:
                        response = error
                    with response:
                        return response.code, json.load(response)

                for conversation, title in (("original", "阅读工具定义"), (session, "第二个对话的任务")):
                    status, result = query(conversation)
                    assert status == 200 and result["conversation"] == conversation
                    assert result["version"] == 2 and [item["title"] for item in result["tasks"]] == [title]
                    print(json.dumps(result, ensure_ascii=False))
                assert query(session, "wrong")[0] == 403
                assert query("f" * 32)[0] == 404
            status, problem = query("original")
            assert status == 503
            assert service.store.load() == saved
            print(json.dumps({"python_stopped": status, "detail": problem.get("detail"), "read_only": True},
                             ensure_ascii=False))
        finally:
            if java is not None:
                java.terminate()
                java.wait(timeout=10)
                java.stdout.close()


if __name__ == "__main__":
    main()

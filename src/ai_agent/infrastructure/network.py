"""为同步模型客户端提供可选 IPv4 连接；不修改全局 DNS 或系统网络。"""

import atexit
from functools import lru_cache
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass

import httpx
import requests
from requests.adapters import HTTPAdapter


class IPv4Adapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        # IPv6 socket 无法绑定 IPv4 源地址，会立即尝试下一个候选地址，
        # 避免当前网络每个不可达 IPv6 地址等待约 21 秒。
        kwargs["source_address"] = ("0.0.0.0", 0)
        return super().init_poolmanager(connections, maxsize, block, **kwargs)

    def proxy_manager_for(self, proxy, **kwargs):
        kwargs["source_address"] = ("0.0.0.0", 0)
        return super().proxy_manager_for(proxy, **kwargs)


@lru_cache(maxsize=None)
def get_ipv4_chat_client(base_url: str) -> httpx.Client:
    parsed = urlsplit(base_url)
    proxies = getproxies()
    proxy = None
    if not proxy_bypass(parsed.hostname or ""):
        proxy = proxies.get(parsed.scheme) or proxies.get("all")
    # HTTPX 显式传 transport 后不再自动选择环境代理，故在这里保留
    # API 地址对应的代理及 NO_PROXY 语义，避免开启 IPv4 时意外绕过代理。
    transport = httpx.HTTPTransport(local_address="0.0.0.0", proxy=proxy)
    client = httpx.Client(transport=transport)
    atexit.register(client.close)
    return client


@lru_cache(maxsize=1)
def get_ipv4_embedding_session() -> requests.Session:
    session = requests.Session()
    session.mount("http://", IPv4Adapter())
    session.mount("https://", IPv4Adapter())
    atexit.register(session.close)
    return session


class SessionEmbeddingClient:
    """向 DashScope 的每次 call 注入 session，保留上层分批和重试行为。"""

    def __init__(self, client, session):
        self.client = client
        self.session = session

    def call(self, **kwargs):
        return self.client.call(**{**kwargs, "session": self.session})

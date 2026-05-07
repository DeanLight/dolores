# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from juplit import test

# %% [markdown]
# # vllm Lifecycle Utilities
#
# Start, wait for, and stop a local vllm server. Used by experiment.py and
# the smolagents baselines (react, codeact, rlm, deepresearch) when running
# with a local model instead of a remote API.

# %%
import subprocess
import time
import urllib.request
from contextlib import contextmanager

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# %%
class VllmConfig(BaseModel):
    """Settings for spinning up a local vllm server."""
    model: str
    port: int = 8555
    gpu_memory_utilization: float = 0.9
    extra_args: list[str] = Field(default_factory=list)

# %%
if test():
    cfg = VllmConfig(model="Qwen/Qwen3-32B")
    assert cfg.port == 8555
    assert cfg.gpu_memory_utilization == 0.9
    assert cfg.extra_args == []

    cfg2 = VllmConfig(model="Qwen/Qwen3-32B", port=9000, extra_args=["--max-model-len", "8192"])
    assert cfg2.port == 9000
    assert cfg2.extra_args == ["--max-model-len", "8192"]

# %%
def find_free_port() -> int:
    """Find a free TCP port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def start_vllm(cfg: VllmConfig) -> subprocess.Popen:
    """Launch a vllm server subprocess and return the Popen handle."""
    cmd = [
        "vllm", "serve", cfg.model,
        "--port", str(cfg.port),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
    ] + cfg.extra_args
    logger.info("vllm.start", cmd=" ".join(cmd))
    return subprocess.Popen(cmd)


def wait_for_vllm(port: int, timeout: int = 300, poll_interval: int = 5) -> None:
    """Block until the vllm /health endpoint responds or timeout is exceeded."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    t_start = time.time()
    last_hb = 0.0
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            urllib.request.urlopen(url, timeout=2)
            logger.info("vllm.ready", port=port, attempts=attempt)
            return
        except Exception:
            now = time.time()
            if now - last_hb >= 30.0:
                logger.info(
                    "vllm.health_pending",
                    port=port,
                    attempt=attempt,
                    elapsed_s=int(now - t_start),
                )
                last_hb = now
            time.sleep(poll_interval)
    raise TimeoutError(f"vllm did not become healthy at port {port} within {timeout}s")


def stop_vllm(proc: subprocess.Popen) -> None:
    """Terminate the vllm server subprocess and wait for it to exit."""
    logger.info("vllm.shutdown", pid=proc.pid)
    proc.terminate()
    proc.wait()


@contextmanager
def vllm_server(cfg: VllmConfig, wait_timeout: int = 300):
    """Context manager that starts vllm, waits for it, and shuts it down on exit.

    Assigns a free port to cfg.port before starting. The caller can read
    cfg.port after entering the context to build the api_base URL.

    Usage::

        cfg = VllmConfig(model="Qwen/Qwen3-32B")
        with vllm_server(cfg, wait_timeout=600) as proc:
            api_base = f"http://localhost:{cfg.port}/v1"
            # ... dispatch workers ...
    """
    cfg.port = find_free_port()
    proc = start_vllm(cfg)
    try:
        wait_for_vllm(cfg.port, timeout=wait_timeout)
        yield proc
    finally:
        stop_vllm(proc)

# %%
if test():
    import socket

    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535

    # Two calls should (almost always) return different ports.
    port2 = find_free_port()
    # Not asserting port != port2 — the OS could reuse immediately, just check it's valid.
    assert isinstance(port2, int)

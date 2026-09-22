# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Inference backends
#
# Uniform interface for local (vLLM, SGLang) and remote (OpenAI-compatible)
# inference.  Pass an `InferenceBackend` to `AgentCfg` and let agents build
# their OpenAI client from `backend.client_kwargs()`.

# %%
from juplit import test

# %%
import subprocess
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager

from pydantic import BaseModel, Field, PrivateAttr

# %% [markdown]
# ## InferenceBackend ABC

# %%
class InferenceBackend(ABC):
    """Minimal contract: enough to build an openai.AsyncOpenAI client + know the model."""

    @abstractmethod
    def client_kwargs(self) -> dict:
        """Return kwargs suitable for openai.AsyncOpenAI(**backend.client_kwargs())."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier to pass to the API."""
        ...

    @abstractmethod
    def wait_until_ready(self, timeout: int = 300) -> None:
        """Block until the inference server is ready to accept requests."""
        ...


# %% [markdown]
# ## OpenAI backend

# %%
class OpenAIBackend(InferenceBackend, BaseModel):
    """Remote OpenAI-compatible endpoint (OpenAI, Anthropic proxy, Together, etc.)."""

    api_base: str
    api_key: str = "EMPTY"
    model_name: str = ""

    @property
    def model(self) -> str:
        return self.model_name

    def client_kwargs(self) -> dict:
        return {"base_url": self.api_base, "api_key": self.api_key}

    def wait_until_ready(self, timeout: int = 300) -> None:
        """Block until the API server responds to any HTTP request (even 4xx = up)."""
        url = self.api_base.rstrip("/") + "/models"
        deadline = time.time() + timeout
        last_exc: Exception | None = None
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=2)
                return
            except urllib.error.HTTPError:
                return  # any HTTP response means the server is up
            except Exception as exc:
                last_exc = exc
                time.sleep(2)
        raise TimeoutError(
            f"API server at {self.api_base} did not respond within {timeout}s"
        ) from last_exc


# %% [markdown]
# ## vLLM backend

# %%
import structlog

logger = structlog.get_logger(__name__)


class VLLMBackend(InferenceBackend, BaseModel):
    """Local vLLM server. Manages the full lifecycle: check, start, wait, stop."""

    model_config = {"arbitrary_types_allowed": True}

    model_name: str
    port: int = 8555
    api_key: str = "EMPTY"
    gpu_memory_utilization: float = 0.9
    extra_args: list[str] = Field(default_factory=list)

    _proc: subprocess.Popen | None = PrivateAttr(default=None)

    @property
    def model(self) -> str:
        return self.model_name

    def client_kwargs(self) -> dict:
        return {"base_url": f"http://localhost:{self.port}/v1", "api_key": self.api_key}

    @staticmethod
    def _find_free_port() -> int:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def _check_prerequisites(self) -> None:
        """Raise if vllm is not installed or no GPU is accessible."""
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("vllm is not installed") from exc
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA GPU available for vllm")
        except ImportError:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError("No GPU detected (nvidia-smi failed)")

    def _start(self) -> None:
        cmd = [
            "vllm", "serve", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ] + self.extra_args
        logger.info("vllm.start", cmd=" ".join(cmd))
        self._proc = subprocess.Popen(cmd)

    def _wait(self, timeout: int) -> None:
        url = f"http://localhost:{self.port}/health"
        deadline = time.time() + timeout
        t_start = time.time()
        last_hb = 0.0
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                urllib.request.urlopen(url, timeout=2)
                logger.info("vllm.ready", port=self.port, attempts=attempt)
                return
            except Exception:
                now = time.time()
                if now - last_hb >= 30.0:
                    logger.info(
                        "vllm.health_pending",
                        port=self.port,
                        attempt=attempt,
                        elapsed_s=int(now - t_start),
                    )
                    last_hb = now
                time.sleep(5)
        raise TimeoutError(f"vllm did not become healthy at port {self.port} within {timeout}s")

    def wait_until_ready(self, timeout: int = 300) -> None:
        """Check prerequisites, start the vllm subprocess, then poll /health."""
        self._check_prerequisites()
        self._start()
        self._wait(timeout)

    def stop(self) -> None:
        """Terminate the vllm server subprocess."""
        if self._proc is not None:
            logger.info("vllm.shutdown", pid=self._proc.pid)
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    @classmethod
    @contextmanager
    def server(cls, model_name: str, *, wait_timeout: int = 300, **kwargs):
        """Context manager: start a vLLM server, yield the ready backend, stop on exit.

        Usage::

            with VLLMBackend.server("Qwen/Qwen3-32B", wait_timeout=600) as backend:
                client = openai.AsyncOpenAI(**backend.client_kwargs())
        """
        backend = cls(model_name=model_name, port=cls._find_free_port(), **kwargs)
        backend.wait_until_ready(timeout=wait_timeout)
        try:
            yield backend
        finally:
            backend.stop()


# %% [markdown]
# ## SGLang backend

# %%
class SGLangBackend(InferenceBackend, BaseModel):
    """Local SGLang server. Manages the full lifecycle: check, start, wait, stop."""

    model_config = {"arbitrary_types_allowed": True}

    model_name: str
    port: int = 30000
    api_key: str = "EMPTY"
    mem_fraction_static: float = 0.9
    extra_args: list[str] = Field(default_factory=list)

    _proc: subprocess.Popen | None = PrivateAttr(default=None)

    @property
    def model(self) -> str:
        return self.model_name

    def client_kwargs(self) -> dict:
        return {"base_url": f"http://localhost:{self.port}/v1", "api_key": self.api_key}

    @staticmethod
    def _find_free_port() -> int:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def _check_prerequisites(self) -> None:
        """Raise if sglang is not installed or no GPU is accessible."""
        try:
            import sglang  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("sglang is not installed") from exc
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA GPU available for sglang")
        except ImportError:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError("No GPU detected (nvidia-smi failed)")

    def _start(self) -> None:
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_name,
            "--port", str(self.port),
            "--mem-fraction-static", str(self.mem_fraction_static),
        ] + self.extra_args
        logger.info("sglang.start", cmd=" ".join(cmd))
        self._proc = subprocess.Popen(cmd)

    def _wait(self, timeout: int) -> None:
        url = f"http://localhost:{self.port}/health"
        deadline = time.time() + timeout
        t_start = time.time()
        last_hb = 0.0
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                urllib.request.urlopen(url, timeout=2)
                logger.info("sglang.ready", port=self.port, attempts=attempt)
                return
            except Exception:
                now = time.time()
                if now - last_hb >= 30.0:
                    logger.info(
                        "sglang.health_pending",
                        port=self.port,
                        attempt=attempt,
                        elapsed_s=int(now - t_start),
                    )
                    last_hb = now
                time.sleep(5)
        raise TimeoutError(f"SGLang did not become healthy at port {self.port} within {timeout}s")

    def wait_until_ready(self, timeout: int = 300) -> None:
        """Check prerequisites, start the SGLang subprocess, then poll /health."""
        self._check_prerequisites()
        self._start()
        self._wait(timeout)

    def stop(self) -> None:
        """Terminate the SGLang server subprocess."""
        if self._proc is not None:
            logger.info("sglang.shutdown", pid=self._proc.pid)
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    @classmethod
    @contextmanager
    def server(cls, model_name: str, *, wait_timeout: int = 300, **kwargs):
        """Context manager: start an SGLang server, yield the ready backend, stop on exit.

        Usage::

            with SGLangBackend.server("Qwen/Qwen3-32B", wait_timeout=600) as backend:
                client = openai.AsyncOpenAI(**backend.client_kwargs())
        """
        backend = cls(model_name=model_name, port=cls._find_free_port(), **kwargs)
        backend.wait_until_ready(timeout=wait_timeout)
        try:
            yield backend
        finally:
            backend.stop()


# %% [markdown]
# ## make_backend

# %%
def make_backend(kind: str, **kwargs) -> InferenceBackend:
    """Factory: make_backend('openai'|'vllm'|'sglang', **fields)."""
    _registry: dict[str, type[InferenceBackend]] = {
        "openai": OpenAIBackend,
        "vllm": VLLMBackend,
        "sglang": SGLangBackend,
    }
    if kind not in _registry:
        raise ValueError(f"Unknown backend {kind!r}. Known: {sorted(_registry)}")
    return _registry[kind](**kwargs)


# %% [markdown]
# ## Tests

# %%
def test_openai_backend():
    b = OpenAIBackend(api_base="http://localhost:8000/v1", api_key="sk-test", model_name="gpt-x")
    assert b.client_kwargs() == {"base_url": "http://localhost:8000/v1", "api_key": "sk-test"}
    assert b.model == "gpt-x"


def test_openai_wait_until_ready_timeout():
    import pytest
    b = OpenAIBackend(api_base="http://localhost:19999/v1", model_name="m")
    with pytest.raises(TimeoutError):
        b.wait_until_ready(timeout=3)


def test_vllm_backend():
    b = VLLMBackend(model_name="Qwen/Qwen3-32B", port=9000)
    assert b.client_kwargs()["base_url"] == "http://localhost:9000/v1"
    assert b.model == "Qwen/Qwen3-32B"


def test_vllm_find_free_port():
    port = VLLMBackend._find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_vllm_prerequisites_no_vllm(monkeypatch):
    import builtins
    import pytest
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "vllm":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    b = VLLMBackend(model_name="x")
    with pytest.raises(RuntimeError, match="vllm is not installed"):
        b._check_prerequisites()


def test_sglang_backend():
    b = SGLangBackend(model_name="Qwen/Qwen3-32B", port=30000)
    assert b.client_kwargs()["base_url"] == "http://localhost:30000/v1"
    assert b.model == "Qwen/Qwen3-32B"


def test_sglang_prerequisites_no_sglang(monkeypatch):
    import builtins
    import pytest
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "sglang":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    b = SGLangBackend(model_name="x")
    with pytest.raises(RuntimeError, match="sglang is not installed"):
        b._check_prerequisites()


def test_make_backend():
    b = make_backend("openai", api_base="http://x", api_key="k", model_name="m")
    assert isinstance(b, OpenAIBackend)
    b2 = make_backend("vllm", model_name="Qwen/Q", port=8555)
    assert isinstance(b2, VLLMBackend)
    b3 = make_backend("sglang", model_name="Qwen/Q", port=30000)
    assert isinstance(b3, SGLangBackend)

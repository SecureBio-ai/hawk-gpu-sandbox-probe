"""One-sample check that a GPU service in a Hawk sandbox works.

The scorer runs in Hawk's runner pod and execs into the sandbox's `gpu`
service, the way a GPU-backed scorer would. It checks that:

- the GPU is visible (`nvidia-smi`);
- PyTorch runs a CUDA computation on it;
- the service is network-isolated: a listener on it answers locally but not
  from the `default` service, and it can't reach the internet, so a model's
  weights have to be in its image.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate
from inspect_ai.util import SandboxEnvironment, sandbox

VALUES = pathlib.Path(__file__).parent / "values.yaml"
LISTENER_PORT = 8000
CONNECT_TIMEOUT_SECONDS = 10

TORCH_CHECK = """
import json, math, time, torch
result = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
          "available": torch.cuda.is_available()}
if result["available"]:
    start = time.monotonic()
    x = torch.randn(2048, 2048, device="cuda")
    total = (x @ x).sum().item()
    torch.cuda.synchronize()
    result.update(
        device=torch.cuda.get_device_name(0),
        capability=".".join(map(str, torch.cuda.get_device_capability(0))),
        matmul_finite=math.isfinite(total),
        seconds=round(time.monotonic() - start, 3),
    )
print(json.dumps(result))
"""

# Prints "connected" or the name of the exception the attempt raised.
CONNECT = """
import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=float(sys.argv[3])).close()
    print("connected")
except OSError as exc:
    print(type(exc).__name__)
"""


@task
def gpu_sandbox_probe() -> Task:
    return Task(
        dataset=[Sample(input="Reply with the single word OK.", target="OK")],
        solver=generate(),
        scorer=gpu_checks(),
        sandbox=("k8s", str(VALUES)),
    )


@scorer(metrics=[accuracy()])
def gpu_checks():
    async def score(state: TaskState, target: Target) -> Score:
        gpu = sandbox("gpu")
        default = sandbox("default")
        checks: dict[str, dict[str, Any]] = {}

        def record(name: str, expected: Any, observed: Any, passed: bool) -> None:
            checks[name] = {"expected": expected, "observed": observed, "pass": passed}

        smi = await gpu.exec(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            timeout=60,
        )
        record(
            "gpu_visible",
            "nvidia-smi lists a GPU",
            (smi.stdout if smi.success else smi.stderr).strip(),
            smi.success and bool(smi.stdout.strip()),
        )

        run = await gpu.exec(["python", "-c", TORCH_CHECK], timeout=600)
        result = _json(run.stdout) if run.success else {}
        record(
            "gpu_torch_cuda",
            "CUDA available and a matmul result that is finite",
            result or run.stderr.strip()[-500:],
            bool(result.get("available") and result.get("matmul_finite")),
        )

        await gpu.exec(
            ["sh", "-c", f"nohup python -m http.server {LISTENER_PORT} >/dev/null 2>&1 & sleep 3"],
            timeout=60,
        )
        address = await gpu.exec(
            ["python", "-c", "import socket; print(socket.gethostbyname(socket.gethostname()))"],
            timeout=30,
        )
        gpu_ip = address.stdout.strip() if address.success else ""
        local = await _connect(gpu, "127.0.0.1", LISTENER_PORT)
        record("gpu_listener_local", "connected", local, local == "connected")
        cross = await _connect(default, gpu_ip, LISTENER_PORT) if gpu_ip else "no address"
        record("default_to_gpu_blocked", "blocked", cross, cross not in ("connected", "no result", "no address"))
        egress = await _connect(gpu, "pypi.org", 443)
        record("gpu_egress_blocked", "blocked", egress, egress not in ("connected", "no result"))

        passed = all(check["pass"] for check in checks.values())
        rows = "\n".join(
            f"| {name} | {check['expected']} | {check['observed']} | {'pass' if check['pass'] else 'FAIL'} |"
            for name, check in checks.items()
        )
        return Score(
            value=CORRECT if passed else INCORRECT,
            explanation=f"| check | expected | observed | result |\n|---|---|---|---|\n{rows}",
            metadata={"checks": checks},
        )

    return score


async def _connect(env: SandboxEnvironment, host: str, port: int) -> str:
    result = await env.exec(
        ["python", "-c", CONNECT, host, str(port), str(CONNECT_TIMEOUT_SECONDS)],
        timeout=CONNECT_TIMEOUT_SECONDS + 30,
    )
    return result.stdout.strip() if result.success and result.stdout.strip() else "no result"


def _json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}

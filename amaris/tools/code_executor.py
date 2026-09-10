"""Run analyst-generated Python in a subprocess. Never eval, never exec."""

from __future__ import annotations

import ast
import asyncio
import sys
import tempfile
import time
from typing import TypedDict

from amaris.observability.logging import logger

EXEC_TIMEOUT = 10
MAX_OUTPUT_CHARS = 4000

# anything that touches the filesystem, the network or another process
FORBIDDEN_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "requests",
        "httpx",
        "urllib",
        "urllib3",
        "pathlib",
        "importlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "pickle",
        "shelve",
        "webbrowser",
        "glob",
        "tempfile",
    }
)

# these defeat the import check by loading modules at runtime
FORBIDDEN_CALLS = frozenset(
    {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint"}
)


class ExecutionResult(TypedDict):
    """What the analyst gets back. success=False always carries a reason in error."""

    success: bool
    stdout: str
    stderr: str
    error: str


def check_code(code: str) -> str:
    """Reason the code is unsafe, or "" when it passes. Parse errors count as unsafe."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} (line {exc.lineno})"

    # walk the AST instead of regexing the source — comments and strings fool regex
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    return f"import of '{root}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES:
                return f"import from '{root}' is not allowed"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                return f"call to '{node.func.id}' is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"dunder attribute access '{node.attr}' is not allowed"

    return ""


async def execute_python(code: str, timeout_s: int = EXEC_TIMEOUT) -> ExecutionResult:
    """Run code in an isolated subprocess. Always returns a result, never raises."""
    reason = check_code(code)
    if reason:
        logger.bind(tool="execute_python", reason=reason).warning("tool.blocked")
        return ExecutionResult(success=False, stdout="", stderr="", error=f"blocked: {reason}")

    started = time.perf_counter()
    # a temp cwd means a stray file write lands somewhere harmless
    workdir = tempfile.mkdtemp(prefix="amaris_exec_")

    try:
        # -I is isolated mode: no user site-packages, no PYTHONPATH, no current dir on sys.path
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
        )
    except OSError as exc:
        return ExecutionResult(success=False, stdout="", stderr="", error=f"spawn failed: {exc}")

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        logger.bind(tool="execute_python", timeout=timeout_s).warning("tool.timeout")
        return ExecutionResult(
            success=False, stdout="", stderr="", error=f"timed out after {timeout_s}s"
        )

    out = stdout.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
    err = stderr.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
    ok = process.returncode == 0

    logger.bind(
        tool="execute_python",
        ms=round((time.perf_counter() - started) * 1000, 1),
        exit_code=process.returncode,
    ).debug("tool.call")

    return ExecutionResult(
        success=ok,
        stdout=out,
        stderr=err,
        error="" if ok else f"exited with code {process.returncode}",
    )

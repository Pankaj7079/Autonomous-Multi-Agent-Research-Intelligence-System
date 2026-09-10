"""The sandbox is the security boundary — these tests are the ones that matter most."""

from __future__ import annotations

import pytest

from amaris.tools.code_executor import check_code, execute_python


async def test_runs_normal_analysis_code() -> None:
    result = await execute_python("import statistics\nprint(statistics.mean([2, 4, 6]))")
    assert result["success"] is True
    assert result["stdout"].strip() == "4"


async def test_reports_a_runtime_error_without_raising() -> None:
    result = await execute_python("print(1 / 0)")
    assert result["success"] is False
    assert "ZeroDivisionError" in result["stderr"]


@pytest.mark.parametrize(
    "code",
    [
        "import os\nprint(os.listdir('.'))",
        "import subprocess",
        "from pathlib import Path",
        "import socket",
        "__import__('os').system('dir')",
        "eval('1+1')",
        "exec('x = 1')",
        "open('secrets.txt').read()",
        "print((1).__class__.__bases__)",
    ],
)
async def test_dangerous_code_is_blocked_before_it_runs(code: str) -> None:
    result = await execute_python(code)
    assert result["success"] is False
    assert result["error"].startswith("blocked:")
    assert result["stdout"] == ""


def test_check_code_names_the_reason() -> None:
    assert "os" in check_code("import os")
    assert "eval" in check_code("eval('2+2')")
    assert check_code("print(sum([1, 2]))") == ""


def test_syntax_errors_are_caught_as_unsafe_not_crashes() -> None:
    assert check_code("def broken(:").startswith("syntax error")


async def test_a_hanging_script_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An infinite loop must not hold the pipeline open — this is why timeout exists."""
    result = await execute_python("while True:\n    pass", timeout_s=2)
    assert result["success"] is False
    assert "timed out" in result["error"]


async def test_import_inside_a_string_is_not_a_false_positive() -> None:
    """AST parsing, not regex — a mention of os in a string is harmless."""
    result = await execute_python("print('we do not import os here')")
    assert result["success"] is True

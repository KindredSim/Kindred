from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest


def _load_audit_k_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / "tools" / "audit" / "wheel_install_smoke_audit.py"
    spec = importlib.util.spec_from_file_location("audit_k_wheel_install_smoke_audit", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_audit_k_does_not_create_run_tmp_inside_report_dir():
    mod = _load_audit_k_module()
    src = inspect.getsource(mod.main)
    # Regression: a prior version created the execution temp dir under report_dir:
    #   tempfile.TemporaryDirectory(dir=report_dir, prefix=...)
    # which allows repo-root CWD leakage when report_dir is inside the repo.
    assert re.search(r"TemporaryDirectory\(\s*dir\s*=\s*report_dir\b", src) is None


@pytest.mark.unit
def test_audit_k_venv_is_not_system_site_packages_by_default():
    mod = _load_audit_k_module()
    src = inspect.getsource(mod._create_venv)
    # Regression: prior Audit K defaulted to --system-site-packages, which can
    # allow globally-installed/editable Kindred to interfere with isolation.
    assert "--system-site-packages" not in src


@pytest.mark.unit
def test_audit_k_has_env_and_cwd_guards():
    mod = _load_audit_k_module()
    assert hasattr(mod, "_sanitized_subprocess_env"), "missing _sanitized_subprocess_env guard helper"
    assert hasattr(mod, "_subprocess_cwd"), "missing _subprocess_cwd guard helper"

    env = mod._sanitized_subprocess_env(
        {
            "PYTHONPATH": "C:\\poisoned" if Path.cwd().anchor else "/poisoned",
            "PYTHONHOME": "C:\\poisoned" if Path.cwd().anchor else "/poisoned",
        }
    )
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert env.get("PYTHONNOUSERSITE") == "1"

    repo_root = Path(__file__).resolve().parents[1]
    outside = Path(env.get("TMPDIR") or repo_root.parent).resolve()
    assert mod._subprocess_cwd(tmp_run_dir=outside, repo_root=repo_root) == outside

    with pytest.raises(Exception):
        mod._subprocess_cwd(tmp_run_dir=repo_root / "tools", repo_root=repo_root)


@pytest.mark.unit
def test_importing_kindred_main_does_not_require_pyside6():
    import os
    import subprocess  # nosec B404 - controlled local interpreter invocation in test
    import sys
    import textwrap

    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import os
        import sys

        sys.path.insert(0, os.getcwd())

        class BlockPySide6:
            def find_spec(self, fullname, path, target=None):  # noqa: ANN001
                if fullname == "PySide6" or fullname.startswith("PySide6."):
                    raise ImportError("blocked PySide6 import during test")
                return None

        sys.meta_path.insert(0, BlockPySide6())
        import kindred.__main__
        print("ok")
        """
    ).strip()

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    res = subprocess.run(  # nosec B603 - controlled args, shell=False
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
    )
    assert res.returncode == 0, res.stderr or res.stdout

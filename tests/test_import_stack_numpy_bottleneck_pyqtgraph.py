from __future__ import annotations

import os
import subprocess  # nosec B404 - controlled local interpreter invocation in test
import sys
import textwrap
import pytest

pytestmark = pytest.mark.integration



def test_numpy_bottleneck_pyqtgraph_imports_cleanly() -> None:
    code = textwrap.dedent(
        """
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            import numpy
            import bottleneck
            import pyqtgraph
            from kindred.gui.plot_config import try_import_pyqtgraph

            ok, _pg, exc = try_import_pyqtgraph()

        stderr = buf.getvalue()
        if "Traceback (most recent call last)" in stderr:
            raise AssertionError("Unexpected traceback on stderr:\\n" + stderr)

        if not ok or exc is not None:
            raise AssertionError(f"try_import_pyqtgraph failed: ok={ok} exc={exc!r}")

        print(numpy.__version__, bottleneck.__version__, pyqtgraph.__version__)
        """
    ).strip()

    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"

    proc = subprocess.run(  # nosec B603 - controlled args, shell=False
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, (
        "Import preflight failed.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )

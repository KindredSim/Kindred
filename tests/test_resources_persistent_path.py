from __future__ import annotations

import hashlib
import os
import subprocess  # nosec B404 - controlled local interpreter invocation in test
import sys
import textwrap
import zipfile
from pathlib import Path

from kindred import __version__ as kindred_version
import pytest

pytestmark = pytest.mark.integration



def test_get_resource_path_persists_for_zipimport(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    preset_src = repo_root / "kindred" / "data" / "presets" / "M1.txt"
    resources_src = repo_root / "kindred" / "io" / "resources.py"

    staging = tmp_path / "staging"
    (staging / "kindred" / "io").mkdir(parents=True, exist_ok=True)
    (staging / "kindred" / "data" / "presets").mkdir(parents=True, exist_ok=True)

    (staging / "kindred" / "__init__.py").write_text(
        "from __future__ import annotations\n"
        "__all__ = ['__version__', 'get_version']\n"
        f"__version__ = {kindred_version!r}\n"
        "def get_version() -> str:\n"
        "    return __version__\n",
        encoding="utf-8",
    )
    (staging / "kindred" / "io" / "__init__.py").write_text("", encoding="utf-8")
    (staging / "kindred" / "io" / "resources.py").write_bytes(resources_src.read_bytes())
    (staging / "kindred" / "data" / "presets" / "M1.txt").write_bytes(preset_src.read_bytes())

    zip_path = tmp_path / "kindred_zipimport_test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging).as_posix())

    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("KINDRED_RESOURCE_CACHE_DIR", str(cache_dir))

    expected_sha256 = hashlib.sha256(preset_src.read_bytes()).hexdigest()
    script = textwrap.dedent(
        """
        from __future__ import annotations

        import hashlib
        import os
        import sys

        zip_path, cache_dir, expected_sha256 = sys.argv[1], sys.argv[2], sys.argv[3]
        sys.path.insert(0, zip_path)
        os.environ["KINDRED_RESOURCE_CACHE_DIR"] = cache_dir

        from kindred.io.resources import get_resource_path

        p = get_resource_path("presets/M1.txt")
        data = p.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        print(str(p))
        if sha256 != expected_sha256:
            raise SystemExit(f"expected sha256 {expected_sha256}, got {sha256}")
        if cache_dir not in str(p):
            raise SystemExit(f"expected cached path under {cache_dir}, got {p}")
        """
    ).strip()

    env = os.environ.copy()
    env["KINDRED_RESOURCE_CACHE_DIR"] = str(cache_dir)
    proc = subprocess.run(  # nosec B603 - controlled args, shell=False
        [sys.executable, "-c", script, str(zip_path), str(cache_dir), expected_sha256],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    cached_path = Path(proc.stdout.strip())
    assert cached_path.exists()
    assert cache_dir in cached_path.parents

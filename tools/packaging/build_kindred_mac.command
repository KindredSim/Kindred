#!/bin/bash
# Kindred macOS builder. Place this file beside pyproject.toml and the kindred
# folder, then double-click it. It downloads everything it needs into this
# folder, installs Apple's Command Line Tools if they are missing, accepts
# Apple's Xcode license if your Mac password is given when asked, and leaves
# Kindred_v<version>_mac_<chip>.dmg beside this file.
#
# If double-clicking does nothing or macOS refuses to open it: open Terminal,
# type "bash " (with the space), drag this file into the window, press Enter.

export PYTHONDONTWRITEBYTECODE=1

finish() {
    if [ "${KINDRED_NO_PAUSE:-}" != "1" ]; then
        read -r -p "Press Enter to close this window..." _ || true
    fi
    exit "$1"
}

cd "$(dirname "$0")" || finish 1
ROOT="$(pwd -P)"
if [ ! -f pyproject.toml ] || [ ! -f kindred/__main__.py ]; then
    echo "Place this script beside pyproject.toml and the kindred folder."
    finish 1
fi
if [ "$(uname -s)" != "Darwin" ]; then
    echo "This script builds Kindred on macOS only."
    finish 1
fi

# Build for the Mac's own processor, even when Terminal runs under Rosetta.
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
    ARCH=aarch64
else
    ARCH=x86_64
fi

# Portable Python for the GUI build environment (python-build-standalone).
PBS_RELEASE=20260924
PBS_PYTHON=3.12.14
BOOT_DIR="$ROOT/.venvs/macos-python-$PBS_PYTHON-$ARCH"
BOOT_PYTHON="$BOOT_DIR/python/bin/python3"
VENV="$ROOT/.venvs/build-macos"
BUILD_PYTHON="$VENV/bin/python"

SESSION="$ROOT/artifacts/pyinstaller-macos/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SESSION" || finish 1
LOG="$SESSION/build.log"
exec > >(tee -a "$LOG") 2>&1

# PyInstaller runs Apple's lipo and install_name_tool on every library it packs;
# they come with the Command Line Tools.
ensure_developer_tools() {
    if ! /usr/bin/xcode-select -p >/dev/null 2>&1; then
        echo "Installing Apple's Command Line Tools, which PyInstaller needs."
        echo "Click Install in the window that opens; the build continues when it finishes."
        /usr/bin/xcode-select --install >/dev/null 2>&1
        until /usr/bin/xcode-select -p >/dev/null 2>&1; do
            sleep 10
        done
        echo "Command Line Tools installed."
    fi
    # Xcode blocks its tools until its license is accepted.
    case "$(/usr/bin/lipo 2>&1)" in
        *license*)
            echo "PyInstaller needs Xcode's tools, and Apple's Xcode license has not been accepted on this Mac."
            echo "Enter your Mac password to accept Apple's license and continue."
            /usr/bin/sudo /usr/bin/xcodebuild -license accept || return 1
            ;;
    esac
}

is_python312() {
    [ -x "$1" ] && "$1" -I -c 'import struct, sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize("P") == 8 and sys.platform == "darwin" else 1)' 2>/dev/null
}

write_driver() {
    cat > "$SESSION/build.py" <<'PY'
import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tomllib
import json
from pathlib import Path

root, output = (Path(value).resolve() for value in sys.argv[1:3])
arch = sys.argv[3]
with (root / "pyproject.toml").open("rb") as handle:
    config = tomllib.load(handle)
gui_dependencies = config["project"]["dependencies"]
version = config["tool"]["kindred"]["fitting-runtime"]["python-version"]
science_dependencies = config["dependency-groups"]["fitting-runtime"]
release = re.fullmatch(r"(3\.15\.\d+(?:(?:a|b|rc)\d+)?)t", version)
if release is None:
    raise RuntimeError("pyproject.toml must specify Python 3.15t for the fitting runtime.")

# macOS curl uses the system certificate store; a portable Python may not.
def download(url, destination):
    return subprocess.run(["/usr/bin/curl", "-fsSL", "--retry", "3", "-o", str(destination), url]).returncode == 0

# The build environment keeps the GUI's Python 3.12 dependency set.
print("Installing GUI build dependencies from pyproject.toml...", flush=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--only-binary=:all:",
     *gui_dependencies, "pyinstaller==6.22.3", "pillow==13.0.0"],
    cwd=root, check=True,
)

# Build both runtimes from one source snapshot without altering the input package.
source_root = output / "package-source"
package = source_root / "kindred"
shutil.copytree(root / "kindred", package,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
runtime = output / "package-runtime" / "kindred-fitting-runtime"
runtime.mkdir(parents=True)

# python-build-standalone publishes relocatable free-threaded macOS builds;
# use the newest release that has the version pyproject.toml names.
releases = "https://github.com/astral-sh/python-build-standalone/releases/download"
tags_file = output / "python-build-standalone-tags.json"
if not download("https://api.github.com/repos/astral-sh/python-build-standalone/tags?per_page=100", tags_file):
    raise RuntimeError("Could not list python-build-standalone releases.")
tags = sorted((tag["name"] for tag in json.loads(tags_file.read_text(encoding="utf-8"))
               if tag["name"].isdigit()), reverse=True)
archive = output / f"python-{version}-{arch}.tar.gz"
for tag in tags:
    url = f"{releases}/{tag}/cpython-{release.group(1)}+{tag}-{arch}-apple-darwin-freethreaded-install_only.tar.gz"
    if download(url, archive):
        break
else:
    raise RuntimeError(f"No macOS {arch} build of Python {version} was found in python-build-standalone.")
print(f"Downloaded complete macOS Python {version}: {url}", flush=True)
with tarfile.open(archive) as bundle:
    bundle.extractall(runtime, filter="data")
interpreter = runtime / "python"
science_python = interpreter / "bin" / "python3"
if not science_python.is_file():
    raise RuntimeError("The archive did not contain a complete macOS Python runtime.")

science_env = {
    key: value for key, value in os.environ.items()
    if key.upper() not in {"PYTHONHOME", "PYTHONPATH", "PYTHON_GIL"}
}
science_env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")

def run_science(*args, **kwargs):
    return subprocess.run([str(science_python), "-E", "-s", *args],
                          cwd=interpreter, env=science_env, check=True, **kwargs)

run_science("-c", r'''
import sys, sysconfig
from pathlib import Path
expected, version = sys.argv[1:]
v = sys.version_info
suffix = {"alpha": "a", "beta": "b", "candidate": "rc", "final": ""}[v.releaselevel]
actual = f"{v.major}.{v.minor}.{v.micro}" + (suffix + str(v.serial) if suffix else "") + "t"
if sys.platform != "darwin" or actual != version or not sysconfig.get_config_var("Py_GIL_DISABLED") or sys._is_gil_enabled():
    raise RuntimeError(f"Unexpected scientific interpreter: {sys.version}")
for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
    if not Path(prefix).samefile(expected):
        raise RuntimeError(f"Scientific interpreter selected another environment: {prefix}")
print("Dedicated scientific interpreter:", sys.executable, sys.version, flush=True)
''', str(interpreter), version)
print("Installing scientific dependencies from pyproject.toml...", flush=True)
launchers_before = set((interpreter / "bin").iterdir())
run_science("-m", "pip", "install", "--only-binary=:all:", "--no-compile", *science_dependencies)
shutil.copytree(package, runtime / "source" / "kindred")
# site.py resolves relative .pth entries against site-packages, not cwd.
site_packages = Path(run_science(
    "-c", "import sysconfig; print(sysconfig.get_path('purelib'))",
    capture_output=True, text=True,
).stdout.strip()).resolve()
(site_packages / "kindred-source.pth").write_text(
    os.path.relpath(runtime.resolve() / "source", site_packages) + "\n", encoding="utf-8"
)
# The packaged runtime never installs anything or runs the launchers pip
# added, which record this folder's path.
run_science("-m", "pip", "uninstall", "-y", "pip")
shutil.rmtree(site_packages.parent / "ensurepip" / "_bundled")
for launcher in set((interpreter / "bin").iterdir()) - launchers_before:
    launcher.unlink()
# The runtime lives inside the signed app, so ship its bytecode compiled now.
# Unchecked-hash files stay valid after PyInstaller's copy changes source
# timestamps, so Python never writes into the app. -s records paths relative
# to the runtime instead of this folder.
run_science("-m", "compileall", "-q", "-f", "-j0", "--invalidation-mode", "unchecked-hash",
            "-s", str(runtime), str(runtime))

# PyInstaller signs only what it collects as code; the runtime enters the app
# as data, so give each of its Mach-O files a valid ad-hoc signature here.
MACH_O = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf"}
for directory, _directories, files in os.walk(runtime):
    for name in files:
        path = Path(directory) / name
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as handle:
            if handle.read(4) not in MACH_O:
                continue
        signed = subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(path)],
                                capture_output=True, text=True)
        if signed.returncode:
            raise RuntimeError(f"Could not sign {path}: {signed.stdout}{signed.stderr}")

# Add the complete science runtime as DATA after Analysis, so the GUI's 3.12
# dependency analysis never mixes the two stacks; it lands intact in the app.
app_version = str(runpy.run_path(str(package / "_version.py"))["__version__"])
spec = output / "Kindred.spec"
icon = package / "data" / "assets" / "kindred.ico"
spec.write_text(
    "import os\n"
    "from pathlib import Path\n"
    f"package = Path({str(package)!r})\n"
    f"runtime = Path({str(runtime)!r})\n"
    "a = Analysis(\n"
    "    [str(package / '__main__.py')], pathex=[str(package.parent)],\n"
    "    binaries=[], datas=[(str(package / 'data'), 'kindred/data')], hiddenimports=[],\n"
    "    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False,\n"
    ")\n"
    "for directory, _directories, files in os.walk(runtime):\n"
    "    for name in sorted(files):\n"
    "        path = Path(directory) / name\n"
    "        destination = (Path('kindred-fitting-runtime') / path.relative_to(runtime)).as_posix()\n"
    "        if path.is_symlink():\n"
    "            a.datas.append((destination, os.readlink(path), 'SYMLINK'))\n"
    "        elif path.is_file():\n"
    "            a.datas.append((destination, str(path), 'DATA'))\n"
    "pyz = PYZ(a.pure)\n"
    "exe = EXE(\n"
    "    pyz, a.scripts, [], exclude_binaries=True, name='Kindred',\n"
    "    debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False,\n"
    ")\n"
    "coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='Kindred')\n"
    "app = BUNDLE(\n"
    f"    coll, name='Kindred.app', icon={str(icon) if icon.is_file() else None!r},\n"
    f"    version={app_version!r},\n"
    ")\n",
    encoding="utf-8",
)
build_env = dict(os.environ)
build_env["PYTHONPATH"] = str(source_root)
print("Building Kindred.app with both runtimes included...", flush=True)
subprocess.run(
    [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
     "--distpath", str(output / "dist"), "--workpath", str(output / "build"), str(spec)],
    cwd=source_root, env=build_env, check=True,
)
app = output / "dist" / "Kindred.app"
executable = app / "Contents" / "MacOS" / "Kindred"
if not executable.is_file():
    raise RuntimeError(f"The build did not produce an app: {app}")

# Start the bundled fitting runtime exactly where Kindred looks for it.
print("Checking the bundled fitting runtime...", flush=True)
fitting = app / "Contents" / "Frameworks" / "kindred-fitting-runtime" / "python"
subprocess.run(
    [str(fitting / "bin" / "python3"), "-E", "-s", "-m", "kindred.gui.fitting.process", "--verify-only"],
    cwd=fitting, env=science_env, check=True,
)

print("Creating Kindred.dmg...", flush=True)
stage = output / "dmg"
stage.mkdir()
subprocess.run(["/usr/bin/ditto", str(app), str(stage / "Kindred.app")], check=True)
# A recipient's Mac checks the downloaded app strictly; the builder's Mac does
# not, so check the copy that ships the same way.
print("Checking the app's signature...", flush=True)
subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", "--verbose=2",
                str(stage / "Kindred.app")], check=True)
(stage / "Applications").symlink_to("/Applications")
dmg = output / "Kindred.dmg"
subprocess.run(
    ["/usr/bin/hdiutil", "create", "-volname", "Kindred", "-srcfolder", str(stage),
     "-ov", "-format", "UDZO", str(dmg)],
    check=True,
)
final_dmg = root / f"Kindred_v{app_version}_mac_{'arm64' if arch == 'aarch64' else 'x86_64'}.dmg"
shutil.copy2(dmg, final_dmg)
chip = "Apple Silicon" if arch == "aarch64" else "Intel"
print(f"\nBuild completed successfully.\nKindred DMG: {final_dmg}", flush=True)
print("It includes the GUI and its dedicated fitting runtime.", flush=True)
print(f"It runs on {chip} Macs. Recipients open it once through", flush=True)
print("System Settings > Privacy & Security > Open Anyway.", flush=True)
PY
}

build() {
    echo "Kindred macOS app builder with dedicated fitting runtime"
    echo "Build log: $LOG"
    ensure_developer_tools || return 1
    if ! is_python312 "$BUILD_PYTHON"; then
        if ! is_python312 "$BOOT_PYTHON"; then
            echo "Downloading portable Python $PBS_PYTHON ($ARCH)..."
            rm -rf "$BOOT_DIR"
            mkdir -p "$BOOT_DIR" || return 1
            /usr/bin/curl -fL --retry 3 -o "$SESSION/python-$PBS_PYTHON.tar.gz" \
                "https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_RELEASE/cpython-$PBS_PYTHON+$PBS_RELEASE-$ARCH-apple-darwin-install_only.tar.gz" \
                || return 1
            tar -xzf "$SESSION/python-$PBS_PYTHON.tar.gz" -C "$BOOT_DIR" || return 1
            if ! is_python312 "$BOOT_PYTHON"; then
                echo "The downloaded Python $PBS_PYTHON does not run on this Mac."
                return 1
            fi
        fi
        echo "Creating the build environment..."
        rm -rf "$VENV"
        "$BOOT_PYTHON" -I -m venv "$VENV" || return 1
    fi
    write_driver || return 1
    "$BUILD_PYTHON" "$SESSION/build.py" "$ROOT" "$SESSION" "$ARCH" || return 1
    echo "Build log: $LOG"
}

if build; then
    finish 0
fi
echo
echo "Build failed. See the error above and the build log: $LOG"
finish 1

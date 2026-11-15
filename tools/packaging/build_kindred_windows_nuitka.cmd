@echo off
rem Kindred Windows builder (Nuitka). Place this file beside pyproject.toml and
rem the kindred folder, then double-click it. It downloads the free-threaded
rem Python for fitting from python.org and the pinned packages from PyPI into
rem this folder, lets Nuitka download the extra tools it needs, and leaves the
rem finished Kindred executable beside this file. If 64-bit Python 3.12 is
rem missing, it first installs Python 3.12.10 from python.org for the current
rem Windows user. If the Microsoft C++ Build Tools are missing, it installs
rem them, and Windows may ask for administrator approval.
setlocal EnableExtensions DisableDelayedExpansion
set "PYTHONDONTWRITEBYTECODE=1"
set "KINDRED_BUILD_SOURCE=%~f0"
cd /d "%~dp0"
if errorlevel 1 goto :failed
if not exist "pyproject.toml" goto :bad_root
if not exist "kindred\__main__.py" goto :bad_root

set "KINDRED_BUILD_HELPER=%CD%\artifacts\generated-packaging\Build-Kindred-Nuitka.generated.ps1"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
  "$raw = Get-Content -Raw -LiteralPath $env:KINDRED_BUILD_SOURCE;" ^
  "$marker = '###__POWERSHELL_PAYLOAD_BELOW__###';" ^
  "$payload = $raw.Substring($raw.LastIndexOf($marker) + $marker.Length);" ^
  "New-Item -ItemType Directory -Force -Path (Split-Path $env:KINDRED_BUILD_HELPER) | Out-Null;" ^
  "Set-Content -LiteralPath $env:KINDRED_BUILD_HELPER -Value $payload -Encoding UTF8;"
if errorlevel 1 goto :failed

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%KINDRED_BUILD_HELPER%" -RepoRoot "%CD%" %*
if errorlevel 1 goto :failed
if /i not "%KINDRED_NO_PAUSE%"=="1" pause
exit /b 0

:bad_root
echo Place this script beside pyproject.toml and the kindred folder.
if /i not "%KINDRED_NO_PAUSE%"=="1" pause
exit /b 1

:failed
echo.
echo Build failed. See the error above and the build log, if one was created.
if /i not "%KINDRED_NO_PAUSE%"=="1" pause
exit /b 1

###__POWERSHELL_PAYLOAD_BELOW__###
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [ValidateSet('onefile', 'standalone')][string]$Mode = 'onefile',
    [ValidateRange(1, 64)][int]$Jobs = [Math]::Min(8, [Environment]::ProcessorCount),
    [string]$ExplicitPythonPath = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = $RepoRoot

function Invoke-Native {
    param([string]$Program, [string[]]$Arguments)
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Program @Arguments 2>&1 | ForEach-Object { Write-Host $_.ToString() }
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $savedPreference }
    if ($code -ne 0) { throw "Command failed (exit $code): $Program" }
}

function Test-Python {
    param([string]$Path)
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        & $Path -I -c 'import struct,sys; sys.exit(0 if sys.version_info[:2] == (3,12) and struct.calcsize(chr(80)) == 8 else 1)' 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch { return $false }
}

function Find-Python {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python\Python312\python.exe",
        "$env:SystemDrive\Python312\python.exe"
    )
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        try {
            $candidate = & py.exe -3.12 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $candidates += ($candidate | Select-Object -Last 1) }
        }
        catch { }
    }
    foreach ($key in @(
        'HKCU:\Software\Python\PythonCore\3.12\InstallPath',
        'HKLM:\Software\Python\PythonCore\3.12\InstallPath'
    )) {
        if (Test-Path $key) {
            $installPath = (Get-Item $key).GetValue('')
            if ($installPath) { $candidates += Join-Path $installPath 'python.exe' }
        }
    }
    foreach ($candidate in $candidates) {
        if (Test-Python $candidate) { return $candidate }
    }
}

function Find-Compiler {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $install = & $vswhere -latest -products '*' -version '[17.0,)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $install) { return ($install | Select-Object -First 1) }
    }
}

$session = Join-Path $RepoRoot ('artifacts\nuitka\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$transcribing = $false
try {
    New-Item -ItemType Directory -Path $session | Out-Null
    $log = Join-Path $session 'build.log'
    Start-Transcript -LiteralPath $log | Out-Null
    $transcribing = $true
    Write-Host "Kindred Nuitka compiler ($Mode)"
    Write-Host "Build log: $log"

    if ($ExplicitPythonPath) {
        $python = (Resolve-Path -LiteralPath $ExplicitPythonPath).Path
        if (-not (Test-Python $python)) { throw 'ExplicitPythonPath must be Windows Python 3.12 x64.' }
    }
    else {
        $python = Find-Python
        if (-not $python) {
            Write-Host 'Installing Python 3.12 x64 for the current Windows user...'
            $installer = Join-Path $session 'python-3.12.10-amd64.exe'
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $installer
            $install = Start-Process -FilePath $installer -ArgumentList @(
                '/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_test=0',
                'Include_launcher=1', 'InstallLauncherAllUsers=0', 'Shortcuts=0'
            ) -WindowStyle Hidden -Wait -PassThru
            if ($install.ExitCode -notin @(0, 3010)) { throw "Python installation failed (exit $($install.ExitCode))." }
            $python = Find-Python
            if (-not $python) { throw 'Python 3.12 x64 was not found after installation. Restart Windows if the installer requires it.' }
        }
    }

    $compiler = Find-Compiler
    if (-not $compiler) {
        Write-Host 'Installing Microsoft C++ Build Tools. Windows may request administrator approval.'
        $installer = Join-Path $session 'vs_BuildTools.exe'
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -OutFile $installer
        $install = Start-Process -FilePath $installer -ArgumentList @(
            '--quiet', '--wait', '--norestart',
            '--add', 'Microsoft.VisualStudio.Workload.VCTools',
            '--includeRecommended', '--addProductLang', 'en-US'
        ) -WindowStyle Hidden -Wait -PassThru
        if ($install.ExitCode -notin @(0, 3010)) { throw "C++ Build Tools installation failed (exit $($install.ExitCode))." }
        $compiler = Find-Compiler
        if (-not $compiler) { throw 'C++ Build Tools were not found after installation. Restart Windows if the installer requires it.' }
    }
    Write-Host "C++ Build Tools: $compiler"

    $venv = Join-Path $RepoRoot '.venvs\nuitka'
    $buildPython = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Python $buildPython)) {
        Write-Host 'Creating the build environment...'
        Invoke-Native $python @('-I', '-m', 'venv', $venv)
    }

    $driver = Join-Path $session 'build.py'
    @'
import os
import re
import runpy
import shutil
import urllib.request
import zipfile
import subprocess
import sys
import tomllib
from pathlib import Path

root, output = (Path(value).resolve() for value in sys.argv[1:3])
mode, jobs = sys.argv[3:5]
with (root / "pyproject.toml").open("rb") as handle:
    config = tomllib.load(handle)
dependencies = config["project"]["dependencies"]
science = config["tool"]["kindred"]["fitting-runtime"]
version = science["python-version"]
science_dependencies = config["dependency-groups"]["fitting-runtime"]
release = re.fullmatch(r"(3\.15\.\d+)(?:(?:a|b|rc)\d+)?t", version)
if release is None:
    raise RuntimeError("pyproject.toml must specify Windows Python 3.15t for the fitting runtime.")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--only-binary=:all:",
     *dependencies, "zstandard==0.25.0"],
    cwd=root, check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "Nuitka==4.2.1"],
    cwd=root, check=True,
)
# Build both runtimes from one source snapshot without altering the input package.
source_root = output / "package-source"
package = source_root / "kindred"
shutil.copytree(root / "kindred", package,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
runtime = output / "package-runtime" / "kindred-fitting-runtime"
interpreter = runtime / "python"
interpreter.mkdir(parents=True)
archive = output / f"python-{version}-amd64.zip"
url = f"https://www.python.org/ftp/python/{release.group(1)}/python-{version}-amd64.zip"
print(f"Downloading complete Windows Python {version}: {url}", flush=True)
urllib.request.urlretrieve(url, archive)
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(interpreter)
science_python = interpreter / "python.exe"
if not science_python.is_file() or not (interpreter / "Lib" / "os.py").is_file():
    raise RuntimeError("The archive did not contain a complete Windows Python runtime.")

science_env = {
    key: value for key, value in os.environ.items()
    if key.upper() not in {"PYTHONHOME", "PYTHONPATH", "PYTHON_GIL"}
}
science_env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")

# -B because -E ignores PYTHONDONTWRITEBYTECODE, and compiled files would
# record this folder's path inside the package.
def run_science(*args):
    subprocess.run([str(science_python), "-E", "-s", "-B", *args],
                   cwd=interpreter, env=science_env, check=True)

run_science("-c", r'''
import sys, sysconfig
from pathlib import Path
expected, version = sys.argv[1:]
v = sys.version_info
suffix = {"alpha": "a", "beta": "b", "candidate": "rc", "final": ""}[v.releaselevel]
actual = f"{v.major}.{v.minor}.{v.micro}" + (suffix + str(v.serial) if suffix else "") + "t"
if sys.platform != "win32" or actual != version or not sysconfig.get_config_var("Py_GIL_DISABLED") or sys._is_gil_enabled():
    raise RuntimeError(f"Unexpected scientific interpreter: {sys.version}")
for prefix in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
    if not Path(prefix).samefile(expected):
        raise RuntimeError(f"Scientific interpreter selected another environment: {prefix}")
print("Dedicated scientific interpreter:", sys.executable, sys.version, flush=True)
''', str(interpreter), version)
print("Installing scientific dependencies from pyproject.toml...", flush=True)
run_science("-m", "pip", "install", "--only-binary=:all:", "--no-compile", *science_dependencies)
shutil.copytree(package, runtime / "source" / "kindred")
# site.py resolves relative .pth entries against Lib/site-packages, not cwd.
(interpreter / "Lib" / "site-packages" / "kindred-source.pth").write_text(
    "../../../source\n", encoding="utf-8"
)
# The packaged runtime never installs anything, opens its documentation or
# runs pip's launchers, which record this folder's path.
run_science("-m", "pip", "uninstall", "-y", "pip")
shutil.rmtree(interpreter / "Lib" / "ensurepip" / "_bundled")
shutil.rmtree(interpreter / "Doc")
shutil.rmtree(interpreter / "Scripts")
# Start the fitting runtime as Kindred does before packing it.
print("Checking the fitting runtime...", flush=True)
run_science("-m", "kindred.gui.fitting.process", "--verify-only")


# Keep the GUI DLL search path out of the separate scientific interpreter.
hook = output / "kindred_frozen_runtime.py"
hook.write_text(r'''
def _kindred_packaging_setup():
    import ctypes
    from ctypes import wintypes
    import os
    import subprocess
    import sys
    import threading

    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
        if getattr(sys, "__" + name + "__") is None:
            setattr(sys, "__" + name + "__", getattr(sys, name))

    bundle = os.path.normcase(os.path.abspath(sys.prefix))
    fitting_python = os.path.join(bundle, "kindred-fitting-runtime", "python", "python.exe")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_directory = kernel.GetDllDirectoryW
    get_directory.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
    get_directory.restype = wintypes.DWORD
    set_directory = kernel.SetDllDirectoryW
    set_directory.argtypes = [wintypes.LPCWSTR]
    set_directory.restype = wintypes.BOOL
    original_popen = subprocess.Popen
    launch_lock = threading.Lock()

    class _KindredFrozenPopen(original_popen):
        def __init__(self, args, *positional, **kwargs):
            if not isinstance(args, (list, tuple)) or not args or os.path.normcase(os.path.abspath(os.fspath(args[0]))) != fitting_python:
                super().__init__(args, *positional, **kwargs)
                return
            env = dict(kwargs.get("env") if kwargs.get("env") is not None else os.environ)
            # Keep bundled GUI DLL directories out of the scientific PATH.
            # The complete scientific interpreter resolves its own DLLs.
            for key in tuple(env):
                if key.upper() == "PATH":
                    entries = []
                    for entry in env[key].split(os.pathsep):
                        normalized = os.path.normcase(os.path.abspath(entry.strip('"')))
                        if normalized != bundle and not normalized.startswith(bundle + os.sep):
                            entries.append(entry)
                    env[key] = os.pathsep.join(entries)
            kwargs["env"] = env
            with launch_lock:
                ctypes.set_last_error(0)
                size = get_directory(0, None)
                if not size and ctypes.get_last_error():
                    raise ctypes.WinError(ctypes.get_last_error())
                previous = ctypes.create_unicode_buffer(size + 1)
                if size and not get_directory(len(previous), previous):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not set_directory(None):
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    super().__init__(args, *positional, **kwargs)
                finally:
                    if not set_directory(previous.value or None):
                        failure = ctypes.WinError(ctypes.get_last_error())
                        if getattr(self, "_child_created", False):
                            self.kill()
                            self.wait()
                        raise failure

    subprocess.Popen = _KindredFrozenPopen

_kindred_packaging_setup()
del _kindred_packaging_setup
''', encoding="utf-8")


version = str(runpy.run_path(str(package / "_version.py"))["__version__"])

entry = output / "kindred_nuitka_entry.py"
entry.write_text(
    "import os, sys\n"
    "from multiprocessing import freeze_support\n"
    "if __name__ == '__main__':\n"
    "    freeze_support()\n"
    "    import kindred_frozen_runtime\n"
    "    from kindred.__main__ import main\n"
    "    raise SystemExit(main())\n",
    encoding="utf-8",
)
command = [
    sys.executable, "-m", "nuitka", str(entry), "--mode=" + mode,
    "--msvc=latest", "--assume-yes-for-downloads", "--jobs=" + jobs,
    "--enable-plugin=pyside6", "--include-qt-plugins=sensible,styles",
    "--include-package=kindred",
    f"--include-data-dir={package / 'data'}=kindred/data",
    # Copy the separate Python runtime intact, without GUI dependency analysis.
    f"--include-raw-dir={runtime}=kindred-fitting-runtime",
    "--output-dir=" + str(output), "--output-filename=Kindred.exe",
    "--windows-console-mode=" + ("disable" if mode == "onefile" else "force"),
    "--company-name=Kindred", "--product-name=Kindred",
    "--file-description=Kindred",
    "--file-version=" + version, "--product-version=" + version,
]
icon = package / "data" / "assets" / "kindred.ico"
if icon.is_file():
    command.append("--windows-icon-from-ico=" + str(icon))
build_env = dict(os.environ)
build_env["PYTHONPATH"] = str(source_root)
subprocess.run(command, cwd=source_root, env=build_env, check=True)

exe = output / ("Kindred.exe" if mode == "onefile" else "kindred_nuitka_entry.dist/Kindred.exe")
if not exe.is_file() or exe.stat().st_size == 0:
    raise RuntimeError(f"The build did not produce an executable: {exe}")
if mode == "onefile":
    final_exe = root / f"Kindred_v{version}_windows_nuitka.exe"
    shutil.copy2(exe, final_exe)
    exe = final_exe
print(f"\nBuild completed successfully.\nKindred EXE: {exe}", flush=True)
if mode == "standalone":
    print(f"Distribute the entire {exe.parent.name} folder, not just its EXE.", flush=True)
'@ | Set-Content -LiteralPath $driver -Encoding UTF8

    Invoke-Native $buildPython @($driver, $RepoRoot, $session, $Mode, [string]$Jobs)
    Write-Host "Build log: $log"
}
catch {
    Write-Host ("Build failed: " + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
}

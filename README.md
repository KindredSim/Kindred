Kindred is a Python desktop application designed for the numerical simulation
and global fitting of chemical kinetic models.

## Key Features
- Elementary reactions with rate constants given directly or through Arrhenius (`A`, `Ea`) or Eyring (`dG_act`) parameters, and equilibria through `kr`, `Keq`, or `dG_eq`
- Support for symbolic operations on kinetic models
- User-defined quantities in mechanism as fittable variables
- Automatic detection and simplification of thermodynamic cycles
- Interventions: concentration changes, additions, continuous feeds,
  withdrawals, dilutions, held concentrations, repeated protocols, threshold
  triggers, and stopping conditions, with optional vessel volume tracking
- Kinetic model fitting of experimental data, including profile-likelihood confidence intervals

## Installation

Windows builds of each named release are on the
[Releases](https://github.com/KindredSim/Kindred/releases) page. On macOS, and
for any other version of the source, such as a specific commit or the latest code,
Kindred can be built directly with a build script provided in `tools/packaging/`.

1. Download the source code from GitHub
2. Place the `kindred/` folder, `pyproject.toml`, and the appropriate build script in a clean folder.
3. Double-click the script. On macOS, if double-clicking does not work, or macOS
   refuses to open the script, open Terminal, type `bash ` with a trailing
   space, drag the script into the window, and press **Return**.

The finished build appears in the same folder:

- Windows: `Kindred_v<version>_windows_pyinstaller.exe`
- macOS: `Kindred_v<version>_mac_<chip>.dmg` (`<chip>`: `arm64` for Apple silicon, `x86_64` for Intel).

Along the way, the scripts may install system components:

- Windows: if 64-bit Python 3.12 is missing, Python 3.12.10 is installed for
  the current Windows user.
- macOS: if the Apple Command Line Tools are missing, a window offers to
  install them. If the Xcode license has not been accepted on that Mac, the
  script asks for the Mac password to accept it.

Each build writes a log to `artifacts\pyinstaller\<date-time>\build.log` on
Windows and `artifacts/pyinstaller-macos/<date-time>/build.log` on macOS.

`tools/packaging/build_kindred_windows_nuitka.cmd` builds an alternative Windows
executable compiled to C, named `Kindred_v<version>_windows_nuitka.exe`. If the
Microsoft C++ Build Tools are missing, it installs them, and Windows may ask
for administrator approval.

## Documentation

A Read the Docs page is a work in progress. Current functions are documented in the application (**Help** tab beside **Reactions**).
Interactive tutorials to teach functionality and workflows are available in **Help → Interactive Tutorials…**.

## Contributing

To run Kindred from a checkout without building it, set up both Pythons by
hand. Two Python installations share one environment: a 64-bit Python 3.12 runs the application, and a free-threaded
Python 3.15 with separate scientific packages runs fitting. The fitting Python
version is set by `python-version` under `[tool.kindred.fitting-runtime]` in
`pyproject.toml` (currently `3.15.2t`), and the fitting packages by the
`fitting-runtime` dependency group.

| | Python 3.12 | Free-threaded Python 3.15.2 |
|---|---|---|
| Windows | 64-bit installer from [python.org](https://www.python.org/downloads/windows/) | `python-3.15.2t-amd64.zip` from [python.org](https://www.python.org/ftp/python/3.15.2/) |
| macOS | `cpython-3.12.<n>+<release>-<arch>-apple-darwin-install_only.tar.gz` | `cpython-3.15.2+<release>-<arch>-apple-darwin-freethreaded-install_only.tar.gz` |

The macOS archives come from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone/releases).
`<arch>` is `aarch64` on Apple silicon and `x86_64` on Intel Macs.

1. Clone the repository and run every later command from the repository root:

       git clone https://github.com/KindredSim/Kindred.git
       cd Kindred

2. Create the environment with Python 3.12 and install Kindred in editable
   mode.

   Windows:

       py -3.12 -m venv .venv
       .venv\Scripts\python.exe -m pip install --upgrade pip
       .venv\Scripts\python.exe -m pip install -e .

   macOS, after extracting the Python 3.12 archive to a folder outside the
   repository:

       <folder>/python/bin/python3 -m venv .venv
       .venv/bin/python -m pip install --upgrade pip
       .venv/bin/python -m pip install -e .

   Editable mode (`-e`) is required because fitting runs this same checkout in
   the second Python. A regular install would leave the two on different
   copies of the code.
3. Extract the free-threaded Python 3.15.2 archive inside the environment.

   - Windows: extract the zip into `.venv\kindred-fitting-runtime\python\`,
     so that the interpreter is `.venv\kindred-fitting-runtime\python\python.exe`.
   - macOS: extract the archive into `.venv/kindred-fitting-runtime/`. The
     archive contains the `python` folder, so the interpreter is
     `.venv/kindred-fitting-runtime/python/bin/python3`.
4. Install the fitting packages with that interpreter. The command is shown
   for Windows. On macOS, use `bin/python3` in the same folder.

       .venv\kindred-fitting-runtime\python\python.exe -E -s -m pip install --only-binary=:all: --group fitting-runtime

5. Add the checkout to the fitting Python. Print the package folder of that
   interpreter:

       .venv\kindred-fitting-runtime\python\python.exe -E -s -c "import sysconfig; print(sysconfig.get_path('purelib'))"

   In that folder, create a file named `kindred-source.pth` that contains one
   line: the absolute path of the repository root (the folder that contains
   `kindred`). Do not add the packages of the application environment.
6. Check the installation from the runtime folder.

   Windows (PowerShell):

       Push-Location .venv\kindred-fitting-runtime\python
       .\python.exe -E -s -m kindred.gui.fitting.process --verify-only; "exit code: $LASTEXITCODE"
       Pop-Location

   macOS:

       (cd .venv/kindred-fitting-runtime/python && ./bin/python3 -E -s -m kindred.gui.fitting.process --verify-only; echo "exit code: $?")

   Exit code 0 means the fitting Python is ready. Exit code 1 means it is not,
   and the reason is printed.

After updating the checkout, rerun the `pip install` commands in steps 2
and 4. If `python-version` changed, replace the fitting Python (steps 3 to
5). Then rerun step 6.

Launch Kindred from the checkout with `.venv\Scripts\kindred.exe` on Windows
or `.venv/bin/kindred` on macOS.

Install the development tools into the application environment. On macOS, use
`.venv/bin/python`.

    .venv\Scripts\python.exe -m pip install -e . --group dev

## License

Kindred is written by Pedro Oliveira and released under the
[MIT License](LICENSE). Third-party components are listed in
[NOTICE.txt](NOTICE.txt) and [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).

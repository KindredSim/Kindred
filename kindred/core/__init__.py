"""
Kindred core package.

Contracts
---------
- Contains Qt-free simulation/modeling logic.
- Must not import GUI modules (PySide6) at import time.
- This __init__ stays lightweight; import concrete owners from submodules.
"""

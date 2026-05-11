"""Backend-contained symbolic helpers for exact Kindred proof artifacts."""

from __future__ import annotations

from .artifacts import SymbolicArtifactIdentity
from .backend import SymbolicBackendMetadata, get_symbolic_backend_metadata
from .errors import UnsupportedSymbolicExpressionError
from .jacobian import SymbolicJacobianArtifact, build_symbolic_jacobian_artifact
from .parameter_expression import SymbolicExpression, translate_parameter_expression
from .proof import SymbolicProofResult, prove_product_identity

__all__ = [
    "SymbolicArtifactIdentity",
    "SymbolicBackendMetadata",
    "SymbolicExpression",
    "SymbolicJacobianArtifact",
    "SymbolicProofResult",
    "UnsupportedSymbolicExpressionError",
    "build_symbolic_jacobian_artifact",
    "get_symbolic_backend_metadata",
    "prove_product_identity",
    "translate_parameter_expression",
]

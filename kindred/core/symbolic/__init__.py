"""Backend-contained symbolic helpers for exact Kindred proof artifacts."""

from __future__ import annotations

from .artifacts import SymbolicArtifactIdentity
from .backend import SymbolicBackendMetadata, get_symbolic_backend_metadata
from .errors import UnsupportedSymbolicExpressionError
from .jacobian import SymbolicJacobianArtifact, build_symbolic_jacobian_artifact
from .parameter_expression import SymbolicExpression, translate_parameter_expression
from .proof import SymbolicProofResult, prove_product_identity
from .structure_cache import (
    SymbolicJacobianStructureCache,
    SymbolicJacobianStructureCacheKey,
    SymbolicJacobianStructureCacheStats,
    clear_symbolic_jacobian_structure_cache,
    symbolic_jacobian_structure_cache_stats,
)

__all__ = [
    "SymbolicArtifactIdentity",
    "SymbolicBackendMetadata",
    "SymbolicExpression",
    "SymbolicJacobianArtifact",
    "SymbolicJacobianStructureCache",
    "SymbolicJacobianStructureCacheKey",
    "SymbolicJacobianStructureCacheStats",
    "SymbolicProofResult",
    "UnsupportedSymbolicExpressionError",
    "build_symbolic_jacobian_artifact",
    "clear_symbolic_jacobian_structure_cache",
    "get_symbolic_backend_metadata",
    "prove_product_identity",
    "symbolic_jacobian_structure_cache_stats",
    "translate_parameter_expression",
]

"""Same-runtime Calculator child; its input is admitted science, never source to admit."""
from __future__ import annotations

from html import escape

from .calculator import SymbolicCalculatorError, build_calculator_context, evaluate_symbolic_query


def run_calculator_child(connection, mechanism, source) -> None:
    context = None
    try:
        while (query := connection.recv()) is not None:
            try:
                if context is None:
                    context = build_calculator_context(mechanism, mechanism_source=source)
                result = evaluate_symbolic_query(context, query)
            except SymbolicCalculatorError as exc:
                message = str(exc)
                connection.send((
                    query,
                    "<pre>" + escape(message) + "</pre>",
                    f"Query: {query}\nError: {message}\nMechanism source:\n{source}",
                    True,
                    message,
                    (),
                ))
            else:
                connection.send((
                    query,
                    result.result_html,
                    result.full_context_copy_text(context.authored_definitions),
                    False,
                    result.copy_text,
                    result.reuse_queries,
                ))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()

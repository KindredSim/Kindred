"""Native Windows headless Fit entry point; stdout is exclusively binary."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback

from kindred.gui.fitting import codec, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    # Save the transport fd, then redirect Python and native incidental output
    # before any scientific imports. Unbuffered pipes avoid shutdown lock waits.
    if sys.platform == "win32":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    output = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    commands = None
    command_errors = []
    try:
        runtime.check_scientific_runtime()
        if runtime.package_fingerprint() != args.source_sha256:
            raise RuntimeError(
                "The fitting runtime source does not match the installed application. "
                "Reinstall the fitting runtime after updating Kindred."
            )
        from kindred.gui.fitting import execution
        from kindred.core.simulator.dsl_parse import prepare_dsl_request_set
        from kindred.core.simulation_failure import factual_cause_from_exception
        from kindred.core.scipy_integrate import load_scipy_integrate

        # Load the native solver stack before the command reader and dataset
        # threads start, and outside the cooperative dataset deadline.
        load_scipy_integrate()
        runtime.check_scientific_runtime()
        if args.verify_only:
            return 0

        incoming = sys.stdin.buffer
        initial = codec.read_frame(incoming)
        if initial is None:
            return 0
        source, settings, payload = codec._decode_fit(initial, {})
        prepared = prepare_dsl_request_set(
            source.reactions_text, units=settings.default_units
        )
        request = codec._decode_fit(payload, {"prepared": prepared})
        references = {"request": request, "origin": request.setup.origin}

        def emit(name, *values):
            terminal = name in {
                "candidate_ready", "factual_failure", "scientific_unavailable", "cancelled"
            }
            if name == "candidate_ready":
                candidate, = values
                try:
                    curves = tuple(execution.project_fit_candidate_curves(candidate))
                    error = None if curves else "Fit result produced no curve datasets."
                except Exception as exc:
                    curves, error = (), f"{type(exc).__name__}: {exc}"
                values = (candidate, curves, error)
            encoded = codec._encode_fit((name, values), references)
            if terminal:
                runtime.check_scientific_runtime()
            codec.write_frame(output, encoded)

        fit = execution._FitExecution(request, emit)

        def read_commands():
            actions = {
                "pause": fit.pause, "resume": fit.resume,
                "cancel": fit.request_cancel, "stop": fit.request_stop,
            }
            try:
                while (frame := codec.read_frame(incoming)) is not None:
                    actions[codec._decode_fit(frame, {})]()
            except BaseException as exc:
                command_errors.append(exc)
            finally:
                fit.request_cancel()

        commands = threading.Thread(target=read_commands, name="fit-commands")
        commands.start()
        try:
            fit.run()
        except BaseException as exc:
            emit("factual_failure", factual_cause_from_exception(
                exc, domain="fit", origin="fit_process"
            ))
        # The parent closes command input on terminal reception. EOF also
        # releases paused work if the parent disappears during execution.
        commands.join()
        if command_errors:
            raise command_errors[0]
        runtime.check_scientific_runtime()
        return 0
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        output.close()


if __name__ == "__main__":
    raise SystemExit(main())

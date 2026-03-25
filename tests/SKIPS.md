# Skip Inventory

| Test file(s) | Exact skip reason string | Requirement | Long-term skip? |
| --- | --- | --- | --- |
| tests/test_parallel_fitting_pickling.py, tests/test_performance.py::TestWorkerPool (class) and ::test_parallel_fit_convergence_statistics, tests/test_integration.py::TestPerformanceIntegration::test_parallel_fitting_performance | requires multiprocessing.SemLock support for parallel workers / multiprocessing.SemLock unavailable | Platform with `multiprocessing.SemLock` support (e.g., Linux/macOS with SemLock) | No — run when SemLock is supported |
| tests/test_io_robustness.py::TestLoggingRobustness::test_logger_in_multiprocessing_context | requires multiprocessing.Process support | Working `multiprocessing.Process` availability | No — run when multiprocessing works |
| tests/test_io_robustness.py::test_save_to_readonly_directory | requires read-only filesystem to validate permission handling | Read-only filesystem mounted for the test | Yes — needs special environment |
| tests/test_performance.py::TestBenchmarkMechanisms::* | requires benchmark mechanism fixtures under benchmarks/mechanisms/*.txt | Benchmark DSL fixtures present in `benchmarks/mechanisms` | No — run when fixtures are present |
| tests/test_gui_analysis_features.py | requires scipy for GUI analysis feature tests | `scipy` installed | No — install `scipy` to enable |

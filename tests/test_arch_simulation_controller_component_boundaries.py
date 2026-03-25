from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "kindred" / "gui" / "controllers" / "simulation_controller.py"


def test_simulation_controller_uses_named_component_helpers() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "SimulationCacheAdmin" in source
    assert "SimulationRunState" in source
    assert "SliderPlotCoalescer" in source


def test_simulation_controller_exposes_components_not_state_bag_proxies() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "def run_state(self)" in source
    assert "def batch_cache(self)" in source
    assert "def parallel_batch(self)" in source
    assert "def plot_coalescer(self)" in source

    for marker in (
        "def simulation_worker(self)",
        "def pending_slider_simulation(self)",
        "def active_run_id(self)",
        "def latest_sim_request_id(self)",
        "def pending_slider_sim_request_id(self)",
        "def variable_metadata(self)",
        "def active_batch_set(self)",
        "def active_batch_set_id(self)",
        "def max_parallel_batch_workers(self)",
        "def limit_blas_threads_per_worker(self)",
        "def batch_result_cache(self)",
        "def batch_preview_cache(self)",
        "def batch_active_cache_key(self)",
        "def batch_active_preview_cache_key(self)",
        "def pending_slider_plot_set_ids(self)",
        "def slider_plot_coalesce_timer(self)",
        "def batch_last_display_selection(self)",
    ):
        assert marker not in source

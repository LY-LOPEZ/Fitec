import json
import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest
from scipy import stats

from desktop_app import FitecApp, parse_observations, parse_parameter
from fitec_analysis import FitResult
from fitec_menus import validate_session
from fitec_queues import WARMUP, simulate_queues
from fitec_tools import FitecTools


def test_queue_comparison_uses_shared_arrivals_and_expected_waits():
    erlang, exponential = simulate_queues(15, 10, 10_000, 42)
    assert erlang.arrivals is exponential.arrivals
    assert erlang.load == exponential.load == pytest.approx(2 / 3)
    assert erlang.theoretical_wait == pytest.approx(12.5)
    assert exponential.theoretical_wait == pytest.approx(20)
    assert erlang.simulated_wait < exponential.simulated_wait
    repeat = simulate_queues(15, 10, 10_000, 42)
    for run, reopened in zip((erlang, exponential), repeat):
        assert len(run.arrivals) == WARMUP + 10_000
        np.testing.assert_array_equal(run.departures, reopened.departures)
        assert np.all(run.starts >= run.arrivals)
        assert np.all(run.starts[1:] >= run.departures[:-1])
        assert run.simulated_wait == pytest.approx(np.mean((run.starts - run.arrivals)[WARMUP:]))
        finished = run.at(run.departures[-1])
        assert finished.finished == len(run.arrivals)
        assert finished.waiting == 0 and finished.progress is None


def test_overloaded_queue_has_no_steady_state_reference():
    runs = simulate_queues(10, 12, 200, 42)
    assert all(run.theoretical_wait is None and run.simulated_wait > 0 for run in runs)


def test_negative_service_support_is_rejected():
    with pytest.raises(ValueError, match="tiempos negativos"):
        simulate_queues(15, 10, 200, 42, fitted_service=stats.norm(loc=6))


def test_fitted_model_opens_queue_simulator_and_round_trips(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk requiere un escritorio: {exc}")
    root.withdraw()
    monkeypatch.setattr(root, "deiconify", lambda: None)
    try:
        app = FitecTools(root, FitecApp, parse_parameter, parse_observations)
        old_settings = app.session_settings()
        old_settings.pop("queues")
        fit_shortcut = app.sidebar.pack_slaves()[1]
        assert fit_shortcut.cget("text") == "Ajustar distribuciones"
        fit_shortcut.invoke()
        workspace = app.menus.fit_workspace
        workspace.root.withdraw()
        fit = FitResult("expon", (0.0, 12.0), 0, 0, 0, None, 0)
        workspace.table.rows = [["3"], ["8"], ["12"], ["17"], ["20"]]
        workspace.table.fits[0] = [fit]
        workspace._refresh_results()
        workspace.open_selected_fit()
        header = workspace.model_tab.winfo_children()[0]
        button = next(child for child in header.winfo_children()
                      if isinstance(child, ttk.Button) and child.cget("text") == "Usar en simulación")
        button.invoke()
        assert app.active_view.get() == "queues"
        assert app.queues.fit.parameters == fit.parameters
        assert app.queues.vars["service_mean"].get() == "12"
        app.queues.start_button.invoke()
        erlang, adjusted = app.queues.runs
        assert erlang.theoretical_wait == pytest.approx(30)
        assert adjusted.theoretical_wait == pytest.approx(48)
        snapshot = json.loads(json.dumps(app.session_settings()))
        app.select_view("random")
        assert app.queues.timer is None

        app.restore_session(validate_session(old_settings, app.explorer.catalog, app.generator.models))
        assert app.queues.fit is None
        app.restore_session(validate_session(snapshot, app.explorer.catalog, app.generator.models))
        assert app.active_view.get() == "queues" and app.queues.fit.parameters == fit.parameters
        workspace.worker.shutdown(wait=False)
    finally:
        root.destroy()

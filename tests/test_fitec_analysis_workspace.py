"""Regression tests for multi-column fitting and Excel exchange."""

import json
import tkinter as tk
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import stats

from desktop_app import CONTINUOUS, FitecApp
from fitec_analysis import FitResult, anderson_darling, fit_columns, fit_models, pearson_chi_square
from fitec_excel import read_excel_selection, read_workbook, write_workbook
from fitec_workspace import DataTable, FitecWorkspace, SUMMARY_FIELDS


@pytest.fixture
def excel():
    return pytest.importorskip("openpyxl")


def test_goodness_and_parameter_estimates():
    values = stats.gamma(4, scale=2.5).ppf((np.arange(60) + .5) / 60)
    result = fit_models(values, discrete=False, repetitions=10, model_names=["erlang"])[0]
    assert result.candidate.explorer_parameters(result.parameters)["k"] == 4
    ad, p_value, repeats = result.goodness("Anderson–Darling")
    assert ad == pytest.approx(anderson_darling(values, result.frozen()))
    assert ad < .2 and repeats == 10 and 0 < p_value <= 1

    uniform = (np.arange(50) + .5) / 50
    assert pearson_chi_square(uniform, stats.uniform(), 0) == pytest.approx(0, abs=1e-12)
    assert pearson_chi_square(uniform[:6], stats.uniform(), 0) is None

    counts = stats.nbinom(5, .4).ppf((np.arange(600) + .5) / 600).astype(int)
    negative_binomial = fit_models(counts, discrete=True, repetitions=6, model_names=["nbinom_general"])[0]
    assert negative_binomial.candidate.explorer_name == "Binomial negativa"
    assert negative_binomial.parameters == pytest.approx((5, .4), abs=.01)
    chi_square, p_value, repeats = negative_binomial.goodness("Chi-cuadrado")
    assert chi_square is not None and repeats == 6 and 0 < p_value <= 1


def test_descriptive_summary_distinguishes_empty_and_non_numeric_cells():
    table = DataTable("Muestra", ["Tiempo"], [["1"], ["3"], [], ["sin medir"], ["5"]])
    measures = dict(zip(SUMMARY_FIELDS, table.describe(0)))
    assert measures["Valores numéricos"] == 3
    assert measures["Celdas vacías"] == 1
    assert measures["Valores no numéricos"] == 1
    assert measures["Media"] == measures["Mediana"] == 3
    assert measures["Varianza muestral"] == 4


def test_discrete_data_kind_requires_nonnegative_integers():
    values = np.array([1., 2., 3., 4., 5.5])
    fitted, errors = fit_columns({0: values}, "Discretos")
    assert not fitted
    assert errors == {0: "Para ajustar datos discretos, usa enteros no negativos."}


def test_triangular_fit_preserves_bounds_and_most_likely_value():
    values = stats.triang(.35, loc=2, scale=8).ppf((np.arange(60) + .5) / 60)
    fit = fit_models(values, discrete=False, repetitions=6, model_names=["triang"])[0]
    assert fit.candidate.display_name == "Triangular"
    assert fit.repetitions == 6
    parameters = fit.candidate.explorer_parameters(fit.parameters)
    assert parameters["a"] <= values.min() <= values.max() <= parameters["b"]
    assert parameters["a"] <= parameters["mode"] <= parameters["b"]
    factory = CONTINUOUS["Triangular"][0]
    model = factory(**parameters)
    np.testing.assert_allclose(model.cdf(values), fit.frozen().cdf(values), rtol=1e-12)
    with pytest.raises(ValueError, match="valor más probable"):
        model.set_parameters(mode=parameters["b"] + 1)
    assert model.get_parameters() == parameters


def test_excel_preserves_numeric_types_and_text(excel, tmp_path):
    path = tmp_path / "observaciones.xlsx"
    write_workbook(path, [
        ("Atención", ["Duración", "Registro"], [(4.5, "001"), (), (12, "=2+3")]),
        ("Llegadas", ["Tiempo"], [(9,), (12.5,)]),
    ])
    book = excel.load_workbook(path)
    try:
        assert book.sheetnames == ["Atención", "Llegadas"]
        assert book["Atención"]["A2"].value == 4.5
        assert book["Atención"]["B4"].data_type == "s"
        assert book["Atención"].freeze_panes == "A2"
    finally:
        book.close()
    tables = read_workbook(path, has_headers=True)
    assert tables[0] == ("Atención", ["Duración", "Registro"], [["4.5", "001"], [], ["12", "=2+3"]])
    assert tables[1][2] == [["9"], ["12.5"]]


def test_selection_uses_headers_and_keeps_blanks(monkeypatch):
    import fitec_excel

    if fitec_excel.os.name != "nt":
        pytest.skip("La selección activa requiere Windows.")
    process = Mock(stdout=json.dumps({"sheet": "Turno A", "rows": [["A", "B"], [1, None], [3, 4.5]]}))
    monkeypatch.setattr(fitec_excel.subprocess, "run", Mock(return_value=process))
    assert read_excel_selection(has_headers=True) == ("Turno A", ["A", "B"], [["1"], ["3", "4.5"]])


def test_fit_several_columns_and_export_project(excel, monkeypatch, tmp_path):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk requiere un escritorio: {exc}")
    root.withdraw()
    workspace = FitecWorkspace(root, FitecApp)
    try:
        input_path = tmp_path / "entrada.xlsx"
        arrival = stats.poisson(5).ppf((np.arange(40) + .5) / 40)
        service = stats.norm(10, 2).ppf((np.arange(40) + .5) / 40)
        write_workbook(input_path, [("Turno", ["Atención", "Recuentos", "Operador"],
                                     [(float(x), int(y), "A") for x, y in zip(service, arrival)]),
                                    ("Segundo turno", ["Atención"], [(12.2,), (8.7,)])])
        monkeypatch.setattr("fitec_workspace.messagebox.askyesnocancel", lambda *a, **k: True)
        workspace.import_excel(input_path)
        assert [table.name for table in workspace.tables] == ["Tabla 1", "Turno", "Segundo turno"]
        workspace.table_picker.current(1)
        workspace._choose_table()
        workspace._set_fit_configuration(("norm", "poisson"), "Automático")
        workspace.fit_all_button.invoke()
        workspace.job[0].result(timeout=25)
        workspace._check_fit()
        table = workspace.table
        assert list(table.fits) == [0, 1]
        assert table.fits[0][0].model == "norm" and table.fits[1][0].model == "poisson"
        assert "1 omitidas" in workspace.status.get()

        workspace.result_column_picker.current(1)
        workspace._column_changed()
        for chart in ("P–P", "Q–Q", "Diferencia acumulada"):
            workspace.chart_kind.set(chart)
            workspace._draw_results()
            assert workspace.plot_ready and workspace.result_axes.lines
        for column in (1, 0):
            workspace.result_column_picker.current(column)
            workspace._column_changed()
            workspace.chart_kind.set("Diferencia acumulada")
            workspace._draw_results()
            marker = workspace.result_axes.collections[0].get_segments()[0]
            assert abs(marker[1, 1]) == pytest.approx(table.fits[column][0].ks)

        monkeypatch.setattr("fitec_workspace.simpledialog.askstring", lambda *a, **k: "Observaciones")
        workspace.add_column()
        assert table.headers[-1] == "Observaciones" and workspace._current_column() == 3
        workspace._write_cells(0, 3, [["4,0"]])
        assert list(table.fits) == [0, 1]
        workspace.column_picker.current(2)
        monkeypatch.setattr("fitec_workspace.messagebox.askyesno", lambda *a, **k: True)
        workspace.delete_column()
        assert table.headers == ["Atención", "Recuentos", "Observaciones"]
        assert list(table.fits) == [0, 1]

        project = tmp_path / "proyecto.fitec"
        workspace.data_kind.set("Discretos")
        assert workspace._save(project)
        saved = json.loads(project.read_text(encoding="utf-8"))
        assert saved["data_kind"] == "Recuentos"
        reopened = DataTable.from_document(saved["tables"][1])
        assert reopened.fits == table.fits
        monkeypatch.setattr("fitec_workspace.filedialog.askopenfilename", lambda *a, **k: str(project))
        workspace.open_project()
        assert workspace.data_kind.get() == "Discretos"
        assert workspace.table.fits == table.fits
        historical = FitResult("norm", (5., 1.), 10., 12., .2, .1, 199)
        old = DataTable.from_document({"name": "Anterior", "headers": ["A"], "rows": [["5"]], "fits": {"0": [
            {"model": "norm", "parameters": [5, 1], "aic": 10, "bic": 12, "ks": .2, "p_value": .1, "repetitions": 199},
        ]}})
        assert old.fits == {0: [historical]}

        output = tmp_path / "informe.xlsx"
        monkeypatch.setattr("fitec_workspace.filedialog.asksaveasfilename", lambda *a, **k: str(output))
        workspace.export_excel()
        report = excel.load_workbook(output, read_only=True, data_only=True)
        try:
            assert report.sheetnames[:2] == ["Estadística", "Ajustes"]
            rows = list(report["Ajustes"].values)
            assert sorted((row[2], row[3]) for row in rows[1:]) == [("Atención", "Normal"), ("Recuentos", "Poisson")]
            assert isinstance(rows[1][5], float)
        finally:
            report.close()
    finally:
        workspace.worker.shutdown(wait=True)
        root.destroy()

"""Exchange tables and analysis reports with Excel workbooks."""

from datetime import date, datetime, time, timedelta
import base64
import json
import math
import os
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.utils.exceptions import InvalidFileException
except ImportError:
    Workbook = None


EXCEL_SELECTION_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$app = [Runtime.InteropServices.Marshal]::GetActiveObject('Excel.Application')
$selected = $app.Selection
if ($null -eq $selected -or $null -eq $selected.Areas -or $selected.Areas.Count -ne 1) {
    throw 'Selecciona un solo rango de celdas.'
}
$range = $app.Intersect($selected, $selected.Worksheet.UsedRange)
if ($null -eq $range) { throw 'La selección no contiene datos.' }
$values = $range.Value2
$rows = @(
    if ($values -is [Array]) {
        for ($row = $values.GetLowerBound(0); $row -le $values.GetUpperBound(0); $row++) {
            $cells = @(for ($column = $values.GetLowerBound(1); $column -le $values.GetUpperBound(1); $column++) {
                $values.GetValue($row, $column)
            })
            ,$cells
        }
    } else {
        ,@($values)
    }
)
[pscustomobject]@{ sheet = [string]$range.Worksheet.Name; rows = $rows } | ConvertTo-Json -Depth 5 -Compress
"""


def _cell_text(value):
    if value is None:
        return ""
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, bool):
        return "VERDADERO" if value else "FALSO"
    if isinstance(value, timedelta):
        return str(value)
    return str(value).strip()


def _table_from_rows(name, raw_rows, has_headers):
    rows = []
    for raw in raw_rows:
        row = [_cell_text(value) for value in raw]
        while row and not row[-1]:
            row.pop()
        if row or rows:
            rows.append(row)
    while rows and not rows[-1]:
        rows.pop()
    if not rows:
        return None
    headers = rows.pop(0) if has_headers else []
    width = max(len(headers), max(map(len, rows), default=0))
    headers.extend([""] * (width - len(headers)))
    names = [header or f"Columna {column}" for column, header in enumerate(headers, 1)]
    return name, names, rows


def read_workbook(path, *, has_headers):
    """Read the saved values in every worksheet, retaining interior blank rows."""
    if Workbook is None:
        raise ValueError("La importación de Excel requiere openpyxl. Instala las dependencias de escritorio.")
    try:
        book = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    except (BadZipFile, InvalidFileException, KeyError, OSError, ValueError) as exc:
        raise ValueError(f"No se pudo abrir el libro de Excel: {exc}") from exc

    tables = []
    try:
        for sheet in book.worksheets:
            sheet.reset_dimensions()
            table = _table_from_rows(sheet.title, sheet.iter_rows(values_only=True), has_headers)
            if table:
                tables.append(table)
    except (BadZipFile, ParseError, KeyError, ValueError) as exc:
        raise ValueError(f"No se pudieron leer las hojas del libro de Excel: {exc}") from exc
    finally:
        book.close()
    if not tables:
        raise ValueError("El libro no contiene celdas con datos guardados.")
    return tables


def read_excel_selection(*, has_headers):
    """Read one contiguous selection from the running desktop Excel instance."""
    if os.name != "nt":
        raise ValueError("La selección activa de Excel requiere Windows. Usa Importar datos para abrir un libro.")
    script = base64.b64encode(EXCEL_SELECTION_SCRIPT.encode("utf-16-le")).decode("ascii")
    args = ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", script]
    try:
        completed = subprocess.run(args, capture_output=True, check=True, timeout=25,
                                   text=True, encoding="utf-8-sig", creationflags=subprocess.CREATE_NO_WINDOW)
        data = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError("No se pudo leer la selección de Excel. Abre un libro y selecciona un rango de celdas.") from exc
    table = _table_from_rows(data["sheet"], data["rows"], has_headers)
    if not table:
        raise ValueError("La selección de Excel está vacía.")
    return table


def excel_value(text):
    """Write numbers as numbers, preserving text and identifier-like strings."""
    if not text:
        return None
    normalized = text.strip().replace(",", ".")
    identifier = re.fullmatch(r"0\d+|[+-]?\d{16,}", normalized)
    try:
        value = float(normalized)
    except ValueError:
        return text
    return value if math.isfinite(value) and not identifier else text


def _sheet_name(book, title):
    safe = re.sub(r"[\\/*?:\[\]]", " ", title).strip(" '") or "Datos"
    candidate = safe[:31]
    existing = {name.casefold() for name in book.sheetnames}
    suffix = 2
    while candidate.casefold() in existing:
        ending = f" ({suffix})"
        candidate = safe[:31 - len(ending)] + ending
        suffix += 1
    return candidate


def _excel_row(sheet, values, *, header=False):
    for value in values:
        cell = WriteOnlyCell(sheet, value=value)
        if isinstance(value, str):
            cell.data_type = "s"
        if header:
            cell.font = Font(name="Calibri", color="FFFFFF", bold=True)
            cell.fill = PatternFill("solid", fgColor="103746")
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        yield cell


def write_workbook(path, sheets):
    """Write named sheets from (title, headers, rows) tuples and replace atomically."""
    if Workbook is None:
        raise ValueError("La exportación a Excel requiere openpyxl. Instala las dependencias de escritorio.")
    book = Workbook(write_only=True)
    path = Path(path)
    temp = None
    try:
        for title, headers, rows in sheets:
            sheet = book.create_sheet(_sheet_name(book, title))
            sheet.freeze_panes = "A2"
            sheet.row_dimensions[1].height = 31
            for index, header in enumerate(headers, 1):
                sheet.column_dimensions[get_column_letter(index)].width = min(48, max(15, len(header) + 3))
            sheet.append(list(_excel_row(sheet, headers, header=True)))
            count = 1
            for row in rows:
                sheet.append(list(_excel_row(sheet, row)))
                count += 1
            sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{count}"
        with NamedTemporaryFile(dir=path.parent, suffix=".xlsx", delete=False) as handle:
            temp = Path(handle.name)
        book.save(temp)
        temp.replace(path)
    finally:
        book.close()
        if temp is not None:
            temp.unlink(missing_ok=True)

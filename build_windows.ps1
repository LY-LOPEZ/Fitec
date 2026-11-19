param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPath = Join-Path $ProjectRoot ".venv-distrilab"
& $Python -m venv $VenvPath
if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el entorno virtual. Comprueba que Python esté instalado." }
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

& $VenvPython -m pip install -e ".[desktop]"
if ($LASTEXITCODE -ne 0) { throw "No se pudieron instalar las dependencias de FITEC." }

$DistRoot = Join-Path $ProjectRoot "dist"
$LastVersion = 0
if (Test-Path -LiteralPath $DistRoot) {
    foreach ($Folder in (Get-ChildItem -LiteralPath $DistRoot -Directory)) {
        if ($Folder.Name -match '^version(\d+)$') {
            $LastVersion = [Math]::Max($LastVersion, [int]$Matches[1])
        }
    }
}
$OutputDir = Join-Path $DistRoot ("version" + ($LastVersion + 1))
New-Item -ItemType Directory -Path $OutputDir | Out-Null
& $VenvPython -m PyInstaller --noconfirm --clean --onefile --windowed `
    --distpath $OutputDir `
    --name FITEC `
    --icon "probviz/assets/FI-TEC-Dado.ico" `
    --collect-all scipy `
    --collect-all matplotlib `
    --hidden-import matplotlib.backends.backend_tkagg `
    --copy-metadata openpyxl `
    --copy-metadata et-xmlfile `
    --add-data "LICENSE:." `
    --add-data "probviz/assets:probviz/assets" `
    desktop_app.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller no pudo crear FITEC.exe." }

Write-Host "Ejecutable creado: $OutputDir\FITEC.exe"

# FITEC

Escritorio para explorar distribuciones en Simulación de Sistemas, basado en
Probviz: una herramienta Python con interfaz visual y módulos para análisis de
probabilidad, simulación Monte Carlo y pruebas estadísticas.

<p align="center">
  <a href="https://github.com/sanskarpan/probviz/actions/workflows/ci.yml"><img src="https://github.com/sanskarpan/probviz/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://github.com/sanskarpan/probviz/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/tests-695%20passed-brightgreen" alt="Tests"></a>
  <a href="https://sanskarpan.github.io/probviz/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Docs"></a>
  <a href="https://github.com/sanskarpan/probviz/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <br>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?logo=streamlit" alt="Streamlit"></a>
  <a href="https://github.com/sanskarpan/probviz/pkgs/container/probviz"><img src="https://img.shields.io/badge/docker-ghcr.io-2496ED?logo=docker" alt="Docker"></a>
  <a href="https://github.com/sanskarpan/probviz/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen" alt="PRs Welcome"></a>
</p>

## FITEC (aplicación de escritorio)

FITEC adapta Probviz para **Simulación de Sistemas**. Las herramientas de
la barra superior están disponibles desde que abre la ventana.

La barra bajo el encabezado reúne las dos vistas y muestra las acciones de
la vista activa. **Copiar muestra** y **Guardar CSV** se habilitan al generar.
En **Archivo** puedes guardar o abrir un proyecto `.fitec` con los parámetros
y la muestra. **Ver** controla la gráfica y las barras. En **Opciones** puedes
desactivar los efectos, y **Ayuda** incluye el contenido y los desarrolladores.

- **Distribuciones.** Elige una distribución continua, discreta o empírica,
  ajusta los parámetros y consulta la gráfica y las probabilidades.
- **Números aleatorios.** Elige distribución, cantidad y semilla. Puedes
  transferir la distribución configurada en la otra pestaña, ver los valores
  generados y copiar la muestra completa o guardarla como CSV.
- **Asistente IA.** Interpreta un ejercicio, carga una sugerencia validada
  en Distribuciones, explica la probabilidad calculada y muestra los pasos
  del método congruencial lineal mixto del libro.

La guía local funciona sin conexión. Para las consultas en línea, usa
**Configurar IA** con tu clave API de OpenAI, o define `OPENAI_API_KEY` antes de
abrir FITEC. La clave ingresada en la ventana se mantiene solo en memoria y no
se escribe en el proyecto. FITEC envía texto a OpenAI cuando eliges una consulta
en línea. Puedes cambiar el modelo de la API con `FITEC_OPENAI_MODEL`.

Para ejecutarla desde el código fuente en Windows:

```powershell
python desktop_app.py
```

Para generar el ejecutable de una sola pieza:

```powershell
Set-Location C:\Users\Hp\Desktop\fitec\probviz
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

El archivo se guarda en `C:\Users\Hp\Desktop\fitec\probviz\dist\FITEC.exe`.
Se requiere Python 3.10 o superior y conexión para instalar PyInstaller si aún
no está instalado. El script instala las dependencias del proyecto y genera el
ejecutable con ventana propia de Windows. Si Python no está en `PATH`, pasa su
ruta al script con `-Python "C:\ruta\a\python.exe"`.

La adaptación reutiliza las distribuciones y el código de Probviz. Se conservan
la licencia MIT y los créditos a Sanskar Pan y colaboradores en [LICENSE](LICENSE).

- **Docs (GitHub Pages):** <https://sanskarpan.github.io/probviz/>
- **16 univariate distributions** (10 continuous + 6 discrete) with PDF/PMF, CDF,
  quantiles, sampling, and full statistics — in the web app and the API.
- **Advanced modules** (Python API): multivariate distributions, copulas,
  mixtures/GMM, distribution fitting, Monte Carlo, and statistical tests.
- **Production-ready:** 695-test suite, typed packaging (`pyproject.toml`),
  `probviz` CLI, Docker/Compose, CI with coverage gate, docs + PyPI + Docker
  release pipelines. See [`CHANGELOG.md`](CHANGELOG.md).

## Gallery

<p align="center">
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.8/docs/assets/normal_sigma_morph.gif" width="250" alt="Normal PDF morphing with sigma">
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.8/docs/assets/clt_convergence.gif" width="250" alt="Central Limit Theorem convergence">
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.8/docs/assets/copula_dependence.gif" width="210" alt="Gaussian copula dependence sweep">
  <br>
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.8/docs/assets/beta_shape_morph.gif" width="250" alt="Beta PDF shape morph">
  <img src="https://raw.githubusercontent.com/sanskarpan/probviz/v1.0.8/docs/assets/mixture_separation.gif" width="250" alt="Gaussian mixture separation">
</p>

*Top: Normal σ sweep · CLT convergence · Gaussian-copula ρ sweep. Bottom: Beta shape sweep · mixture separation. Regenerate with `python examples/generate_media.py`.*

## Quick start

```bash
pip install probviz
probviz app
```

Or from source:

```bash
git clone https://github.com/sanskarpan/probviz.git
cd probviz
pip install -r requirements.txt
streamlit run web/app.py
```

Open `http://localhost:8501`. Full guide: [`QUICKSTART.md`](QUICKSTART.md) ·
[docs quickstart](https://sanskarpan.github.io/probviz/quickstart/).

### Install as a package

```bash
pip install -e ".[test]"
probviz app        # launch the UI
probviz test       # run tests
probviz version    # print version
```

### Docker

```bash
docker build -t probviz .
docker run -p 8501:8501 probviz
# or
docker compose up --build
```

## What's inside

| Area | Contents |
|---|---|
| Univariate | Normal, Exponential, Uniform, Beta, Gamma, Chi-Square, Student-t, Weibull, Lognormal, Cauchy · Binomial, Poisson, Geometric, Negative Binomial, Hypergeometric, Discrete Uniform |
| Multivariate | Multivariate Normal, Dirichlet, Multivariate Student-t, Wishart |
| Copulas | Gaussian, Clayton, Gumbel, Student-t + `fit_copula_to_data` |
| Mixtures | `MixtureDistribution` (1-D EM), `GaussianMixtureModel`, `BayesianGMM`, BIC selection |
| Fitting | `DistributionFitter`, `BayesianEstimator`, `GoodnessOfFit` |
| Monte Carlo | `MonteCarloSimulator`, `VarianceReduction`, `QuasiMonteCarloSimulator` |
| Tests | hypothesis / nonparametric / descriptive dict-returning helpers |
| Utils | validation, preprocessing, plotting, structured logging; `probviz.visualizers` facade |

Project layout and conventions: [`docs/architecture.md`](docs/architecture.md).
API reference: [`docs/api.md`](docs/api.md) (rendered on the docs site).

## Python API

```python
import numpy as np
from probviz.distributions import NormalDistribution, BinomialDistribution

normal = NormalDistribution(mu=0, sigma=1)
x = np.linspace(-4, 4, 200)
pdf, cdf = normal.pdf(x), normal.cdf(x)
samples = normal.rvs(size=1000, random_state=42)
print(normal.get_statistics())
print(normal.interval(0.95), normal.ppf(0.975))

binomial = BinomialDistribution(n=10, p=0.3)
print(binomial.pdf(np.arange(0, 11)))
```

```python
from probviz.fitting import DistributionFitter
from probviz.monte_carlo import MonteCarloSimulator

fitter = DistributionFitter(samples)
print(fitter.fit_all())

rng = np.random.default_rng(42)
sim = MonteCarloSimulator(random_seed=42)
res = sim.estimate_probability(lambda: rng.normal(0, 1) > 1.0, num_samples=100_000)
print(res["probability"], res["confidence_interval"])
```

## Known limitations (by design)

- The Streamlit app covers the **16 univariate distributions only**; advanced
  modules are Python-API only.
- `Clayton`/`Gumbel` copula `pdf`/`rvs` are **bivariate-only** (explicit error otherwise).
- `StudentTCopula` has **no closed-form `cdf`/`pdf`**; use Monte Carlo via `rvs`.
- `MixtureDistribution` EM assumes **1-D** components; use the sklearn GMM wrappers
  for multivariate mixtures.
- Cauchy moments are undefined — the API surfaces `nan` instead of masking it.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -q                                  # 695 tests
pytest tests/ -q --cov=src --cov-report=term      # with coverage (gate: 80%)
flake8 src/ tests/ --count --select=E9,F63,F7,F82 --statistics
mypy --config-file=pyproject.toml src/
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) (setup, style, tests, PR checklist),
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), and [`SECURITY.md`](SECURITY.md).

## License

MIT — see [`LICENSE`](LICENSE). If you use this in research or teaching:

```bibtex
@software{probability_distribution_visualizer,
  title  = {Probability Distribution Visualizer},
  author = {sanskarpan},
  year   = {2026},
  url    = {https://github.com/sanskarpan/probviz}
}
```

Also see [`CITATION.cff`](CITATION.cff).

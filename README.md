# FITEC

Aplicación de escritorio para Windows creada para la materia de **Simulación de Sistemas**. FITEC permite explorar distribuciones de probabilidad, visualizar qué valores incluye un evento y generar muestras reproducibles. El flujo de trabajo se inspira en EasyFit y la base de cálculo proviene de Probviz.

<p align="center">
  <img src="probviz/assets/fitec.png" alt="Logotipo de FITEC con un dado" width="440">
</p>

## Qué puedes hacer

| Sección | Funciones |
| --- | --- |
| **Distribuciones** | Elige un modelo, edita sus parámetros y alterna entre densidad, masa y acumulada. Consulta probabilidades de cola, de intervalo y de valores exactos para modelos discretos. Puedes mover los límites sobre la gráfica y ver la zona o las barras incluidas. |
| **Números aleatorios** | Genera una muestra indicando la distribución, la cantidad y la semilla. Consulta sus valores en la tabla o en la gráfica, cópialos o guárdalos en CSV. Al generar una muestra Normal, el histograma también muestra la curva teórica. |
| **Asistente IA** | Interpreta ejercicios compatibles con la guía local, explica el cálculo actual y enseña el método congruencial lineal mixto paso a paso. Las consultas en línea son opcionales y requieren una clave API de OpenAI. |

Desde **Archivo** puedes guardar un proyecto `.fitec` con sus parámetros y la muestra, y abrirlo más adelante.

## Distribuciones

- **Continuas.** Uniforme, Exponencial, Erlang, Normal, Chi-cuadrado, t de Student, F de Fisher, Beta, Gamma, Weibull, Lognormal y Cauchy.
- **Discretas.** Geométrica, Geométrica (ensayos), Binomial negativa, Binomial, Poisson, Uniforme discreta e Hipergeométrica.
- **Empírica.** Pega una columna de observaciones para obtener la distribución acumulada y calcular probabilidades a partir de esos datos.

## Ejecutar en Windows

Si ya tienes una copia de **`FITEC.exe`**, ábrela con doble clic. Para generar el ejecutable desde el código, instala **Python 3.10 o superior**, abre PowerShell en la raíz del repositorio y ejecuta:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

El script instala las dependencias en un entorno virtual y genera **`dist\FITEC.exe`**. La instalación de dependencias requiere conexión a Internet.

También puedes ejecutar la aplicación desde el código:

```powershell
python -m pip install -e ".[desktop]"
python desktop_app.py
```

## Ejemplos para probar

| Distribución | Parámetros | Evento | Resultado esperado |
| --- | --- | --- | --- |
| Normal | μ = 0, σ = 1 | P(−1 ≤ X ≤ 1) | 68,27 % |
| Exponencial | λ = 0,2 | P(X ≤ 5) | 63,21 % |
| Binomial | n = 3, p = 0,5 | P(1 ≤ X ≤ 2) | 75,00 % |

Para repetir una muestra, conserva la distribución, los parámetros, la cantidad y la semilla.

## Desarrollo

- [desktop_app.py](desktop_app.py) contiene el explorador y la gráfica de probabilidades.
- [fitec_tools.py](fitec_tools.py) contiene la ventana principal y el generador de muestras.
- [fitec_assistant.py](fitec_assistant.py) y [fitec_tutor.py](fitec_tutor.py) contienen el asistente y la guía de ejercicios.
- [design/render_fitec_header.py](design/render_fitec_header.py) permite reproducir en Blender la campana 3D de la cabecera.
- La [documentación de Probviz](docs/index.md) describe la API de Python y la aplicación web del proyecto original.

El próximo trabajo previsto es refinar los controles y las transiciones, comenzando por la pantalla de la distribución Normal y verificando el diseño en una pantalla de 1366 × 768.

## Licencia

FITEC es una adaptación de **Probviz**, creado por Sanskar Pan y sus colaboradores. El repositorio conserva la licencia MIT y los créditos originales. [Licencia MIT](LICENSE) · [Probviz](https://github.com/sanskarpan/probviz) · [Cómo citar](CITATION.cff)

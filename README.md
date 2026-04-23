# SAM3 MagicMask — DaVinci Resolve Plugin

Segmentación y seguimiento automático de objetos en vídeo usando **Meta SAM3**
(Segment Anything Model 3), integrado directamente en DaVinci Resolve 20 como
plugin de script externo.

---

## Requisitos

| Elemento | Mínimo |
|---|---|
| Sistema operativo | Windows 10/11 · macOS 12+ · Ubuntu 20.04+ |
| Python | 3.10 o superior |
| DaVinci Resolve | 20.x (Free o Studio) |
| RAM | 16 GB |
| GPU | NVIDIA con ≥ 8 GB VRAM (recomendado) |
| GPU alternativa | Apple Silicon (MPS) · cualquier CPU (lento, ~8 s/frame) |
| Disco | ~5 GB para el modelo SAM3-Large |

> **Sin GPU:** El plugin funciona en CPU. La propagación tarda
> aproximadamente 8 segundos por frame; para clips largos se recomienda
> usar el modelo SAM3-Base (360 MB).

---

## Instalación

### 1. Instalar dependencias del sistema

**Windows** — instala [Python 3.11](https://python.org) marcando
"Add to PATH" durante la instalación.

**macOS** — Python viene con Xcode Command Line Tools:
```bash
xcode-select --install
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt install python3.11 python3.11-venv git
```

### 2. Descargar el plugin

```bash
git clone https://github.com/tuusuario/sam3-magicmask.git
cd sam3-magicmask
```

### 3. Ejecutar el instalador

```bash
python install.py
```

El instalador hace todo automáticamente:

1. Verifica Python ≥ 3.10
2. Detecta tu GPU (NVIDIA CUDA · Apple MPS · CPU)
3. Crea un entorno virtual aislado en `~/.sam3_resolve_env`
4. Instala PyQt6, OpenCV, PyTorch y SAM2 con las wheels correctas para tu GPU
5. Descarga el modelo SAM3 (~2.4 GB, con barra de progreso y resume automático)
6. Copia el script del plugin a la carpeta de scripts de DaVinci Resolve

> Si la descarga del modelo se interrumpe, vuelve a ejecutar `python install.py`.
> El instalador reanuda desde donde se quedó.

### 4. Activar External Scripting en Resolve

En DaVinci Resolve:
```
Preferences → System → General
  → External scripting using: Local
```
Reinicia Resolve después de cambiar esta opción.

---

## Lanzar el plugin desde Resolve

1. Abre DaVinci Resolve y carga un proyecto con vídeo.
2. Selecciona un clip en la línea de tiempo.
3. En la barra de menús: **Workspace → Scripts → Comp → SAM3_MaskTracker**

La ventana del plugin se abre con el clip seleccionado ya cargado.

---

## Tutorial de uso con DaVinci Resolve

### Paso 1 — Cargar un clip

Al abrir el plugin, el clip seleccionado en la timeline aparece
automáticamente. En la **barra superior** verás:

- Nombre del clip y duración
- Estado de la GPU (`GPU Ready` en verde, o `CPU mode` en amarillo)
- Botón ⚙ para abrir el panel de Settings

El panel izquierdo muestra la resolución, FPS y código de tiempo.

### Paso 2 — Añadir objetos a segmentar

En el **panel derecho** (sección OBJECTS):

- Haz clic en **+ Add Object** para añadir un objeto (hasta 8).
- Cada objeto tiene:
  - **Swatch de color** — clic para cambiar el color de la máscara.
  - **Nombre** — clic en el lápiz ✎ para renombrarlo.
  - **Ojo 👁** — muestra/oculta la máscara en el canvas.
  - **✕** — elimina el objeto y su máscara.
  - **Opacity** — transparencia de la máscara (0–100%).
  - **Feather** — difuminado del borde en píxeles (0–20 px).

### Paso 3 — Definir las máscaras (prompts)

Selecciona el objeto que quieres segmentar (clic en su fila).
En la **barra de modo** (izquierda), elige el tipo de prompt:

| Modo | Tecla | Descripción |
|---|---|---|
| **Points** | `P` | Clic izquierdo = incluir · Clic derecho = excluir |
| **Box** | `B` | Arrastra un rectángulo alrededor del objeto |
| **Mask** | `M` | Dibuja una máscara manual con el ratón |
| **Text** | `T` | Describe el objeto con texto (experimental) |

La **máscara en tiempo real** aparece en el canvas a los 80 ms de cada clic.
El spinner de inferencia indica que SAM3 está procesando.

**Deshacer / Limpiar:**
- Botón **Undo** (o `Ctrl+Z`) — elimina el último punto o box.
- Botón **Clear All** — borra todos los prompts del objeto activo.

**Zoom y pan:**
- Rueda del ratón — zoom in/out.
- Clic central + arrastrar — pan.
- Tecla `0` — reset del zoom.
- Botón ⛶ — pantalla completa.

### Paso 4 — Propagar por el clip

Cuando la máscara del frame actual es correcta:

1. Clic en **▶ Run Propagation** (barra inferior de acciones).
2. La barra de progreso muestra frames procesados, velocidad y ETA.
3. Las miniaturas del strip muestran un indicador de confianza
   (verde = alta · naranja = media · rojo = baja).
4. Para cancelar: clic en **✕ Cancel**.

### Paso 5 — Revisar en el preview player

Tras la propagación, clic en **Preview**:

- **Overlay** — máscara semitransparente sobre el vídeo original.
- **Matte** — blanco sobre negro (ideal para revisar bordes).
- **Cutout** — solo el objeto sobre fondo de tablero de ajedrez.
- **Outline** — contorno de color sobre el vídeo.

Usa el scrubber o los botones ◀ / ▶ para revisar cada frame.
Ajusta la velocidad de reproducción (0.25× – 4×).

Si algo no está bien, clic en **✎ Modify** para volver al canvas
y añadir más prompts en los frames problemáticos.

### Paso 6 — Exportar a la timeline de Resolve

> Esta funcionalidad está en desarrollo activo. La descripción siguiente
> refleja el comportamiento final previsto.

1. En el preview player, clic en **⬇ Export Masks**.
2. Elige la carpeta de destino (por defecto, junto al archivo fuente).
3. El plugin exporta una secuencia de PNGs en 16 bits (un subfolder
   por objeto: `object_001/`, `object_002/`, ...) y un `manifest.json`.
4. Clic en **Accept → Import to Resolve**.
5. El plugin abre el nodo Fusion del clip y añade automáticamente:
   - Un nodo **Loader** con la secuencia de PNGs.
   - Un nodo **MatteControl** conectado al canal alpha.
   - Un nodo de **Transform** para compensar el escalado de proxy.

---

## Panel de Settings

Clic en el botón ⚙ (esquina superior derecha) para abrir el panel deslizante.

| Sección | Opción | Descripción |
|---|---|---|
| **Model & Device** | Model | SAM3 Large (más preciso) · SAM3 Base (más rápido) |
| | Device | Auto-detect · CUDA · MPS · CPU |
| | Float16 | Más rápido, menos VRAM (desactivar si hay errores numéricos) |
| **DaVinci Resolve** | API path | Ruta al directorio de `fusionscript.so/.dll` si Resolve está en ubicación no estándar. También puedes exportar `RESOLVE_INSTALL_DIR=/tu/ruta` antes de lanzar el plugin. |
| **Proxy & Cache** | Proxy CRF | Calidad del proxy generado (1=mejor, 51=peor, default=18) |
| | Clear cache | Elimina los frames JPEG en caché de `~/.sam3_resolve_cache` |
| **Export** | Output dir | Carpeta de destino para las PNGs (vacío = junto al clip) |
| | Workers | Procesos paralelos para exportar PNGs (1–8, default=4) |
| **Keybindings** | — | Personaliza todos los atajos de teclado |
| **Repair** | Open Setup Wizard | Vuelve a ejecutar los checks de instalación |

---

## Resolución de problemas

**"Resolve is not running"** — Activa External Scripting en Preferences y
reinicia Resolve antes de lanzar el plugin.

**"No clip selected"** — Selecciona un clip en la timeline antes de abrir
el plugin. El clip debe estar en modo de edición, no en Fusion.

**GPU no detectada / modo CPU muy lento** — Verifica que los drivers NVIDIA
están actualizados y que `nvidia-smi` responde en terminal. Considera usar
SAM3-Base en lugar de Large.

**Ruta de Resolve no encontrada** — Ve a Settings y rellena el campo
"API path" con la carpeta que contiene `fusionscript.so` (Linux/macOS) o
`fusionscript.dll` (Windows). Alternativamente:
```bash
export RESOLVE_INSTALL_DIR=/opt/resolve   # Linux, ejemplo
python plugin_main.py
```

**El modelo no descarga** — El instalador intenta reanudar automáticamente.
Si falla 3 veces, descárgalo manualmente desde:
```
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```
y colócalo en `~/.sam3_resolve_models/sam3_large.pt`.

---

---

## Modo debug — desarrollo sin DaVinci Resolve

> Esta sección es para **desarrollo y pruebas internas**.
> No forma parte del flujo de usuario final y se eliminará antes de la
> publicación oficial.

El modo debug permite usar el plugin con cualquier archivo de vídeo local,
sin necesidad de tener DaVinci Resolve instalado. El runner SAM3 es
`MockSAM3Runner`, que genera máscaras elipse sintéticas sin GPU ni modelo.

### Requisitos mínimos para debug

```bash
pip install PyQt6 opencv-python-headless numpy
```

No hace falta instalar torch, sam2 ni descargar ningún modelo.

### Lanzar en modo debug

**Con selector de archivo (recomendado):**
```bash
python -m sam3_resolve.plugin_main --debug
```
Se abre un diálogo para elegir cualquier archivo de vídeo
(`.mp4`, `.mov`, `.mkv`, `.avi`, `.mxf`, etc.).

**Con archivo directo (para scripting / CI):**
```bash
python -m sam3_resolve.plugin_main --file /ruta/a/tu/video.mp4
```

### Tutorial de debug paso a paso

#### 1. Lanzar y cargar un vídeo

```bash
cd sam3-magicmask
python -m sam3_resolve.plugin_main --debug
```

Aparece un selector de archivos estándar del sistema. Elige cualquier
vídeo `.mp4` o `.mov`. El plugin lo abre con OpenCV y muestra el primer
frame en el canvas.

En la barra superior verás:
```
[DEBUG] MockSAM3Runner — no GPU needed
```
Esto confirma que estás en modo debug con el runner sintético.

#### 2. Navegar por los frames

Usa el **scrubber** (barra deslizante bajo el canvas) para moverte entre
frames. Los frames se leen con OpenCV directamente del archivo — sin caché.

Botones de transporte:
- `|◀` / `▶|` — primer / último frame
- `◀` / `▶` — frame anterior / siguiente

#### 3. Añadir objetos y prompts

Igual que en el modo Resolve. Selecciona el **modo Points** (`P`) y
haz clic en el canvas:
- **Clic izquierdo** — punto positivo (incluir en la máscara)
- **Clic derecho** — punto negativo (excluir de la máscara)

Tras 80 ms aparece la máscara del `MockSAM3Runner`:
una **elipse** del color del objeto, centrada en el primer punto positivo,
aproximadamente el 20% del área del frame.

> En modo debug la máscara es siempre una elipse sintética.
> Con el modelo SAM3 real, la máscara se ajusta exactamente al contorno
> del objeto según los prompts dados.

#### 4. Probar el modo Box

Pulsa `B` o selecciona **Box** en el panel izquierdo.
Arrastra un rectángulo sobre el objeto. Al soltar, la máscara se centra
en el rectángulo dibujado.

#### 5. Múltiples objetos

Clic en **+ Add Object** en el panel derecho. Cada objeto tiene su propio
color y se puede segmentar de forma independiente (hasta 8 objetos).

Para cambiar entre objetos, haz clic en la fila del objeto.
El objeto activo tiene un borde azul a la izquierda de su fila.

#### 6. Ajustar opacity y feather

En la fila del objeto (panel derecho):
- **Opacity** — arrastra el slider para ver la máscara más o menos
  transparente sobre el frame.
- **Feather** — difumina el borde de la máscara (0 = bordes duros,
  20 px = muy difuminado). Útil para integraciones de compositing.

#### 7. Propagar (simulación)

Con prompts definidos, clic en **▶ Run Propagation**.

El `MockSAM3Runner` simula la propagación:
- Genera una máscara elipse por cada frame.
- La elipse se desplaza 2 px a la derecha por frame (simula tracking).
- La confianza simulada es siempre 0.92 (alta).

Verás el progreso en la barra inferior (pestaña PROGRESS).

#### 8. Preview player

Tras la propagación, clic en **Preview**. Prueba los cuatro modos:

| Modo | Qué ves |
|---|---|
| Overlay | Elipse semitransparente sobre el vídeo real |
| Matte | Elipse blanca sobre fondo negro |
| Cutout | Solo el área de la elipse sobre fondo de tablero de ajedrez |
| Outline | Contorno de color sobre el vídeo real |

#### 9. Exportar máscaras a disco

En el preview player, clic en **⬇ Export Masks**.
Las PNGs de 16 bits se guardan en la carpeta elegida:

```
exports/
  object_001/
    frame_000000.png   ← 16-bit PNG, canal único, compatible con Fusion
    frame_000001.png
    ...
  manifest.json        ← metadatos para importación automática futura
```

Puedes abrir estas PNGs en After Effects, Nuke o Resolve Fusion manualmente.

#### 10. Opciones CLI completas

```
python -m sam3_resolve.plugin_main [--debug] [--file PATH]

--debug       Abre modo debug con selector de archivo interactivo
--file PATH   Carga directamente el vídeo en PATH (implica --debug)
(sin flags)   Modo normal: conecta a DaVinci Resolve
```

### Diferencias entre debug y producción

| Característica | Debug (Mock) | Producción (SAM3 real) |
|---|---|---|
| Necesita Resolve | No | Sí |
| Necesita GPU | No | Recomendado |
| Necesita modelo (~2.4 GB) | No | Sí |
| Forma de la máscara | Elipse sintética | Contorno real del objeto |
| Tracking | Desplazamiento lineal | Seguimiento real frame a frame |
| Confianza | Siempre 0.92 | Real (0.0 – 1.0) |
| Navegación de frames | OpenCV (cv2) | MediaHandler + caché JPEG |
| Exportación PNGs | Funcional | Funcional |
| Import a Fusion | No disponible | En desarrollo |

---

## Estructura del proyecto

```
sam3-magicmask/
├── install.py                      # Instalador de un comando
├── sam3_resolve/
│   ├── plugin_main.py              # Punto de entrada (normal + --debug)
│   ├── constants.py                # Todas las constantes del plugin
│   ├── config.py                   # Singleton de configuración (config.json)
│   ├── core/
│   │   ├── gpu_utils.py            # Detección de GPU (CUDA / MPS / CPU)
│   │   ├── resolve_bridge.py       # Bridge con la API de DaVinci Resolve
│   │   ├── media_handler.py        # Lectura de frames, proxy, caché JPEG
│   │   ├── sam3_runner.py          # Inferencia SAM3 (real + mock)
│   │   └── mask_exporter.py        # Exportación de PNGs 16-bit
│   └── ui/
│       ├── styles.qss              # Hoja de estilos (tema oscuro Resolve)
│       ├── main_window.py          # Ventana principal
│       ├── canvas_widget.py        # Canvas interactivo con prompts y máscaras
│       ├── object_panel.py         # Panel de objetos (rows con sliders)
│       ├── log_panel.py            # Panel de log con colores por nivel
│       ├── preview_player.py       # Reproductor post-propagación
│       ├── setup_wizard.py         # Asistente de instalación
│       └── settings_panel.py       # Panel de ajustes deslizante
└── tests/                          # 325 tests automáticos (pytest)
```

---

## Licencia

MIT — ver `LICENSE` para detalles.
SAM2/SAM3 de Meta AI: [Apache 2.0](https://github.com/facebookresearch/sam2).

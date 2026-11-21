# Clasificador de dígitos escritos a mano con DenseNet

Proyecto de aprendizaje automático que implementa una red neuronal convolucional basada en DenseNet para clasificar imágenes de dígitos escritos a mano.

El programa permite entrenar el modelo desde cero, evaluar su desempeño y guardar automáticamente el mejor punto de control durante el entrenamiento.

> [!NOTE]
> El proyecto está escrito en Python y utiliza PyTorch.

## Demostración

Vídeo explicativo del proyecto:

https://youtu.be/OxN0uh3TMGc

## Características

- Arquitectura DenseNet implementada desde cero con PyTorch.
- Variantes configurables DenseNet-121, DenseNet-169 y DenseNet-201.
- Procesamiento automático de imágenes en escala de grises.
- División automática de datos en conjuntos de entrenamiento, validación y prueba.
- Selección del mejor modelo según la precisión de validación.
- Guardado y reanudación del entrenamiento.
- Generación de una matriz de confusión.
- Compatibilidad con CPU y GPU CUDA.

## Arquitectura

El modelo está compuesto por:

- Capa inicial convolucional.
- Cuatro bloques densos conectados mediante conexiones densas.
- Capas de transición con agrupamiento promedio.
- Agrupamiento adaptativo global.
- Capa lineal de clasificación.

Las variantes disponibles se configuran en `config.py`.

| Variante | Growth rate | Bloques densos |
|----------|-------------|----------------|
| DenseNet-121 | 32 | 6, 12, 24, 16 |
| DenseNet-169 | 32 | 6, 12, 32, 32 |
| DenseNet-201 | 32 | 6, 12, 48, 32 |

## Estructura del proyecto

```text
.
├── config.py      # Configuración de entrenamiento y modelo
├── dataset.py     # Carga y preparación del conjunto de datos
├── inference.py   # Carga del modelo y función de predicción
├── model.py       # Implementación de DenseNet
├── train.py       # Entrenamiento y evaluación
└── README.md
```

## Requisitos

- Python 3.9 o superior
- PyTorch
- Torchvision no es necesario
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

Instala las dependencias con:

```bash
pip install torch numpy pillow matplotlib scikit-learn
```

## Preparación del conjunto de datos

Organiza las imágenes dentro de una carpeta cuyo nombre debe coincidir con la carpeta configurada en `config.py`.

```text
digitos/
├── 0/
│   ├── imagen_001.jpg
│   ├── imagen_002.jpg
│   └── ...
├── 1/
│   └── ...
├── ...
└── 9/
    └── ...
```

Requisitos del conjunto de datos:

- Las carpetas deben corresponder a los dígitos del `0` al `9`.
- Las imágenes deben tener extensión `.jpg`.
- Se requieren al menos 30 imágenes por dígito.
- No es necesario que las imágenes tengan una resolución uniforme.

El conjunto se divide automáticamente de la siguiente manera:

- Entrenamiento: imágenes restantes después de separar validación y prueba.
- Validación: 10 % de las imágenes de cada clase.
- Prueba: 10 % siguiente de las imágenes de cada clase.

## Configuración

Los parámetros predeterminados están definidos en `get_config()` dentro de `config.py`.

```python
{
    "variante": "121",
    "tamaño_imagen": 256,
    "batch_size": 16,
    "epocas": 5,
    "lr": 1e-4,
    "seed": 42,
}
```

También puedes modificar:

- `folder`: ubicación de las carpetas de entrenamiento.
- `folder_m`: ubicación donde se guardarán los modelos y resultados.
- `variante`: arquitectura utilizada por el modelo.

## Entrenamiento

Ejecuta:

```bash
python train.py
```

Durante el entrenamiento, el programa:

1. Fija las semillas aleatorias.
2. Carga las imágenes y crea los conjuntos de datos.
3. Selecciona el dispositivo disponible: GPU CUDA o CPU.
4. Entrena el modelo durante el número de épocas configurado.
5. Guarda el mejor modelo según la precisión de validación.
6. Guarda el estado de entrenamiento para permitir su reanudación.
7. Genera una matriz de confusión.

Los archivos generados se guardan dentro de `folder_m`:

```text
mejor.pt
latest.pt
matriz_confusion.png
```

- `mejor.pt`: contiene los pesos con mejor precisión de validación.
- `latest.pt`: permite reanudar el entrenamiento.
- `matriz_confusion.png`: muestra los aciertos y errores del modelo.

> [!WARNING]
> Antes de entrenar, modifica `folder` y `folder_m` en `config.py` para que apunten a rutas válidas de tu entorno.

## Procesamiento de imágenes

Las imágenes pasan por el siguiente preprocesamiento:

1. Se corrige la orientación EXIF.
2. Se convierten a escala de grises.
3. Se ajustan con relleno hasta formar un cuadrado.
4. Se redimensionan al tamaño configurado.
5. Se normalizan al rango aproximadamente `[-1, 1]`.

## Autor

- **charbelochoa**
- GitHub: https://github.com/charbelochoa

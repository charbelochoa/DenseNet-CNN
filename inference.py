import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from config import DenseNetConfig, obtener_config
from dataset import cargar_imagen
from model import DenseNet


@torch.inference_mode()
def predecir(ruta_imagen, ruta_checkpoint=None, real=None, original=False, limpia=False):
    ruta_imagen = Path(ruta_imagen).expanduser()
    config = obtener_config()

    if ruta_checkpoint is None:
        candidatos = [config[carpeta] / nombre
                      for carpeta in ("folder_m", "folder_m_anterior")
                      for nombre in ("mejor.pt", "latest.pt")]
        ruta_checkpoint = next((ruta for ruta in candidatos if ruta.is_file()), candidatos[0])
    ruta_checkpoint = Path(ruta_checkpoint).expanduser()

    if not ruta_imagen.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")
    if not ruta_checkpoint.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_checkpoint}")

    estado = torch.load(ruta_checkpoint, map_location="cpu", weights_only=True)
    setup = estado["setup"]
    digitos = setup["digitos"]

    modelo = DenseNet(DenseNetConfig(num_clases=len(digitos), variante=setup["variante"]))
    modelo.load_state_dict(estado["model"])
    modelo.eval()

    usar_recorte = not original
    print(f"Preprocesamiento: {'fotográfico' if usar_recorte else 'original'}")
    x = cargar_imagen(ruta_imagen, setup["tamaño_imagen"], recortar=usar_recorte, foto=usar_recorte and not limpia).unsqueeze(0)
    prediccion = digitos[int(modelo(x).argmax(dim=1).item())]

    print(f"Modelo utilizado: {ruta_checkpoint}")
    print(f"Imagen: {ruta_imagen}")
    print(f"Predicción: {prediccion}")
    if real is not None:
        print(f"Etiqueta real: {real}")
        print(f"Correcto: {'sí' if str(real) == prediccion else 'no'}")

    with Image.open(ruta_imagen) as imagen:
        imagen = ImageOps.exif_transpose(imagen).convert("RGB")
        figura, ejes = plt.subplots(1, 2, figsize=(10, 4))
        ejes[0].imshow(imagen)
        ejes[0].set_title("Fotografía original")
        ejes[1].imshow(x[0, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        ejes[1].set_title(f"Predicción: {prediccion}" + (f" | Real: {real}" if real is not None else ""))
        for eje in ejes:
            eje.axis("off")
        carpeta = ruta_checkpoint.parent / "inferencias"
        carpeta.mkdir(parents=True, exist_ok=True)
        destino = carpeta / f"{ruta_imagen.stem}_{'recorte' if usar_recorte else 'original'}_resultado.png"
        figura.savefig(destino, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {destino}")
        if "agg" not in plt.get_backend().lower():
            plt.show()
        plt.close(figura)

    return prediccion


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser(description="Inferencia con DenseNet-121 propia.")
    argumentos.add_argument("ruta_imagen")
    argumentos.add_argument("--real", default=None)
    argumentos.add_argument("--checkpoint", default=None)
    argumentos.add_argument("--original", action="store_true")
    argumentos.add_argument("--limpia", action="store_true")
    opciones = argumentos.parse_args()
    predecir(opciones.ruta_imagen, ruta_checkpoint=opciones.checkpoint, real=opciones.real, original=opciones.original, limpia=opciones.limpia)

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

from config import DenseNetConfig, get_config
from dataset import cargar_imagen
from model import DenseNet


@torch.inference_mode()
def predict(image_path, checkpoint_path=None, real=None, recortar=False):
    image_path = Path(image_path).expanduser()
    config = get_config()

    if checkpoint_path is None:
        nuevo = config["folder_m"] / "mejor.pt"
        anterior = config["folder_m_anterior"] / "mejor.pt"
        checkpoint_path = nuevo if nuevo.is_file() else anterior
    checkpoint_path = Path(checkpoint_path).expanduser()

    if not image_path.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {checkpoint_path}")

    estado = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    setup = estado["setup"]
    digitos = setup["digitos"]

    modelo = DenseNet(DenseNetConfig(num_clases=len(digitos), variante=setup["variante"]))
    modelo.load_state_dict(estado["model"])
    modelo.eval()

    x = cargar_imagen(image_path, setup["tamaño_imagen"], recortar=recortar).unsqueeze(0)
    prediccion = digitos[int(modelo(x).argmax(dim=1).item())]

    print(f"Modelo utilizado: {checkpoint_path}")
    print(f"Imagen: {image_path}")
    print(f"Predicción: {prediccion}")
    if real is not None:
        print(f"Etiqueta real: {real}")
        print(f"Correcto: {'sí' if str(real) == prediccion else 'no'}")

    with Image.open(image_path) as imagen:
        imagen = ImageOps.exif_transpose(imagen).convert("RGB")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(imagen)
        axes[0].set_title("Fotografía original")
        axes[1].imshow(x[0, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        axes[1].set_title(f"Predicción: {prediccion}" + (f" | Real: {real}" if real is not None else ""))
        for ax in axes:
            ax.axis("off")
        carpeta = checkpoint_path.parent / "inferencias"
        carpeta.mkdir(parents=True, exist_ok=True)
        destino = carpeta / f"{image_path.stem}_resultado.png"
        fig.savefig(destino, dpi=150, bbox_inches="tight")
        print(f"Figura guardada: {destino}")
        if "agg" not in plt.get_backend().lower():
            plt.show()
        plt.close(fig)

    return prediccion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia con DenseNet-121 propia.")
    parser.add_argument("image_path")
    parser.add_argument("--real", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--recortar", action="store_true")
    args = parser.parse_args()
    predict(args.image_path, checkpoint_path=args.checkpoint, real=args.real, recortar=args.recortar)

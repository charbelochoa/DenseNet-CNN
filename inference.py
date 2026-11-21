import argparse
from pathlib import Path

import torch

from config import DenseNetConfig, get_config
from dataset import cargar_imagen
from model import DenseNet


@torch.inference_mode()
def predict(image_path, checkpoint_path, real=None):
    image_path = Path(image_path).expanduser()
    checkpoint_path = Path(checkpoint_path).expanduser()

    if not image_path.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el checkpoint: {checkpoint_path}. Ejecuta train.py primero."
        )

    estado = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    setup = estado["setup"]
    digitos = setup["digitos"]

    model_config = DenseNetConfig(num_clases=len(digitos), variante=setup["variante"])
    model = DenseNet(model_config)
    model.load_state_dict(estado["model"])
    model.eval()

    x = cargar_imagen(image_path, setup["tamaño_imagen"]).unsqueeze(0)
    probabilidades = torch.softmax(model(x), dim=1)[0]
    indice = int(probabilidades.argmax().item())
    prediccion = digitos[indice]
    confianza = float(probabilidades[indice].item())

    print(f"Imagen: {image_path}")
    print(f"Predicción: {prediccion}")
    print(f"Confianza: {confianza:.2%}")
    if real is not None:
        print(f"Etiqueta real: {real}")
        print(f"Correcto: {'sí' if str(real) == prediccion else 'no'}")

    return prediccion, confianza


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir un dígito con DenseNet propia.")
    parser.add_argument("image_path", help="Ruta de la imagen JPG, JPEG o PNG.")
    parser.add_argument("--real", default=None, help="Dígito verdadero, para comparar.")
    parser.add_argument(
        "--checkpoint",
        default=str(get_config()["folder_m"] / "mejor.pt"),
        help="Ruta del checkpoint mejor.pt.",
    )
    args = parser.parse_args()
    predict(args.image_path, args.checkpoint, real=args.real)

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from torch.utils.data import Dataset


def cargar_imagen(path, tamaño_imagen, aumentar=False, recortar=False):
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        if recortar:
            ancho, alto = gray.size
            gray = gray.crop((
                int(ancho * 0.22), int(alto * 0.20),
                int(ancho * 0.78), int(alto * 0.85),
            ))
        square = ImageOps.pad(
            gray,
            (tamaño_imagen, tamaño_imagen),
            method=Image.Resampling.BILINEAR,
            color=255,
        )
        if aumentar:
            square = square.rotate(
                random.uniform(-12, 12),
                resample=Image.Resampling.BILINEAR,
                fillcolor=255,
            )
            square = ImageEnhance.Contrast(square).enhance(random.uniform(0.75, 1.25))
            square = ImageEnhance.Brightness(square).enhance(random.uniform(0.80, 1.20))
        pixels = np.asarray(square, dtype=np.float32).copy()

    return torch.from_numpy(pixels).unsqueeze(0) / 127.5 - 1.0


def separar_cada_digito(folder, seed):
    admin = Path(folder)
    if not admin.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del dataset: {admin}")

    digits = [d for d in "0123456789" if (admin / d).is_dir()]
    if not digits and (admin / "digitos").is_dir():
        admin = admin / "digitos"
        digits = [d for d in "0123456789" if (admin / d).is_dir()]

    if digits not in (list("123456789"), list("0123456789")):
        raise ValueError(
            f"Se esperaban carpetas 1 a 9 (o 0 a 9) en {admin}. "
            f"Carpetas de digitos encontradas: {digits}"
        )

    print(f"Dataset localizado en: {admin}")
    rng = random.Random(seed)
    train, val, test = [], [], []

    for label, digit in enumerate(digits):
        documentos = sorted(
            p for p in (admin / digit).rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and not p.name.startswith("._")
        )

        if len(documentos) < 30:
            raise ValueError(
                f"La clase {digit} contiene {len(documentos)} imágenes; "
                "se requieren al menos 30."
            )

        rng.shuffle(documentos)
        n = max(1, len(documentos) // 10)
        samples = [(p, label) for p in documentos]

        val.extend(samples[:n])
        test.extend(samples[n:2 * n])
        train.extend(samples[2 * n:])
        print(f"Clase {digit}: {len(documentos)} imágenes")

    return digits, train, val, test


class Dataset_digitos(Dataset):
    def __init__(self, samples, tamaño_imagen, aumentar=False):
        self.samples = samples
        self.tamaño_imagen = tamaño_imagen
        self.aumentar = aumentar

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return {
            "pixel_values": cargar_imagen(path, self.tamaño_imagen, aumentar=self.aumentar),
            "label": label,
        }

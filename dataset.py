import random

from pathlib import Path


import numpy as np
import torch 
import torch.nn as nn 


from PIL import Image, ImageOps
from torch.utils.data import Dataset



def cargar_imagen(path, tamaño_imagen):
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        square = ImageOps.pad(
            gray,
            (tamaño_imagen, tamaño_imagen),
            method=Image.Resampling.BILINEAR,
            color=255
        )
        pixels = np.array(square, dtype=np.float32)

    return torch.from_numpy(pixels).unsqueeze(0) / 127.5 - 1.0




def separar_cada_digito(folder, seed):
    admin = Path(folder)
    digits = [d for d in "0123456789" if (admin / d).is_dir()]


    if digits not in (list("123456789"), list("0123456789")):
        raise ValueError (f"Carpetas inexistentes o incorrectas")

    rng = random.Random(seed)
    train, val, test = [],[],[]


    for label, digit in enumerate(digits):
        documentos = sorted(
            p for p in (admin / digit).rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".jpg"}
            and not p.name.startswith("._")

        )

        if len(documentos) < 30:
            raise ValueError (f"Insuicientes, minimo 30 fotos para entrenar (No recomendado)")
        rng.shuffle(documentos)
        n = max(1,len(documentos) // 10)        #Division entera //
        samples = [(p, label) for p in documentos]

        val.extend(samples[:n])
        test.extend(samples[n:2 * n])
        train.extend(samples[2 * n:])

    return digits, train, val, test




class Dataset_digitos(Dataset):
    def __init__(self,samples, tamaño_imagen):
        self.samples = samples
        self.tamaño_imagen = tamaño_imagen

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index): 
        path, label = self.samples[index]

        return {
            "pixel_values": cargar_imagen(path, self.tamaño_imagen),
            "label": label
        }

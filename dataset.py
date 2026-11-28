import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

cv2.setNumThreads(1)


def mascara_digito(gris, foto=False):
    a = np.asarray(gris, dtype=np.uint8)
    h, w = a.shape
    if foto:
        a = a[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    h, w = a.shape
    escala = min(1.0, 640 / max(h, w))
    if escala < 1:
        a = cv2.resize(a, (max(1, int(w * escala)), max(1, int(h * escala))), interpolation=cv2.INTER_AREA)
    if min(a.shape) < 32:
        a = cv2.resize(a, (max(32, a.shape[1]), max(32, a.shape[0])), interpolation=cv2.INTER_LINEAR)
    if not foto and np.median(a) < 100:
        a = 255 - a
    h, w = a.shape
    fondo = cv2.GaussianBlur(a, (0, 0), max(3, min(h, w) / 15))
    normal = cv2.divide(a, np.maximum(fondo, 1), scale=235)
    bloque = max(15, (min(h, w) // 8) | 1)
    mascara = cv2.adaptiveThreshold(normal, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bloque, 6)
    umbral = max(190, min(225, np.percentile(normal, 0.25) + 35))
    mascara[normal > umbral] = 0
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, etiquetas, stats, centros = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    candidatos = []
    for i in range(1, n):
        x, y, ancho, alto, area = stats[i]
        if area < max(8, h * w * 0.00015):
            continue
        if ancho > w * 0.65 and alto < h * 0.035:
            continue
        if foto and (x <= 1 or y <= 1 or x + ancho >= w - 1 or y + alto >= h - 1):
            continue
        cx, cy = centros[i]
        distancia = ((cx - w / 2) / w) ** 2 + ((cy - h / 2) / h) ** 2
        candidatos.append((area / (1 + 8 * distancia), i))
    if not candidatos:
        return None
    principal = max(candidatos)[1]
    x, y, ancho, alto, area = stats[principal]
    salida = np.zeros_like(mascara)
    margen = max(ancho, alto) * 0.15
    for _, i in candidatos:
        xi, yi, wi, hi, ai = stats[i]
        if i == principal or (ai >= area * 0.12 and xi < x + ancho + margen and xi + wi > x - margen and yi < y + alto + margen and yi + hi > y - margen):
            salida[etiquetas == i] = 255
    return salida


def recortar_digito(gris, tamaño_imagen, foto=False):
    mascara = mascara_digito(gris, foto)
    if mascara is None:
        return None
    x, y, w, h = cv2.boundingRect(mascara)
    digito = 255 - mascara[y:y + h, x:x + w]
    margen = max(2, int(max(w, h) * 0.20))
    digito = cv2.copyMakeBorder(digito, margen, margen, margen, margen, cv2.BORDER_CONSTANT, value=255)
    return ImageOps.pad(Image.fromarray(digito), (tamaño_imagen, tamaño_imagen), method=Image.Resampling.BILINEAR, color=255)


def simular_foto(digito, fondos=(), rng=random):
    n = 384
    a = np.asarray(digito, dtype=np.uint8)
    fondo = np.full((n, n), rng.randint(205, 255), dtype=np.uint8)
    if fondos:
        with Image.open(rng.choice(fondos)) as img:
            fondo = np.asarray(ImageOps.fit(img.convert("L"), (n, n)), dtype=np.uint8).copy()
    else:
        tipo = rng.choice(("blanco", "rayado", "cuadriculado"))
        if tipo != "blanco":
            paso = rng.randint(18, 34)
            color = int(fondo[0, 0]) - rng.randint(12, 38)
            for y in range(0, n, paso):
                cv2.line(fondo, (0, y), (n, y), color, 1)
            if tipo == "cuadriculado":
                for x in range(0, n, paso):
                    cv2.line(fondo, (x, 0), (x, n), color, 1)
    tamaño = int(n * rng.uniform(0.52, 0.82))
    a = cv2.resize(a, (tamaño, tamaño), interpolation=cv2.INTER_LINEAR)
    tinta = np.zeros((n, n), dtype=np.float32)
    x = (n - tamaño) // 2 + rng.randint(-22, 22)
    y = (n - tamaño) // 2 + rng.randint(-22, 22)
    tinta[y:y + tamaño, x:x + tamaño] = (255 - a.astype(np.float32)) / 255
    M = cv2.getRotationMatrix2D((n / 2, n / 2), rng.uniform(-15, 15), 1)
    tinta = cv2.warpAffine(tinta, M, (n, n))
    esquina = np.float32([[0, 0], [n-1, 0], [0, n-1], [n-1, n-1]])
    destino = esquina + np.float32([[rng.uniform(-16, 16), rng.uniform(-16, 16)] for _ in range(4)])
    tinta = cv2.warpPerspective(tinta, cv2.getPerspectiveTransform(esquina, destino), (n, n))
    intensidad = rng.uniform(0.45, 0.95)
    foto = fondo.astype(np.float32) * (1 - intensidad * tinta)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    sombra = np.exp(-((xx - rng.choice((0, 1))) ** 2 + (yy - rng.choice((0, 1))) ** 2) / rng.uniform(0.12, 0.5))
    foto *= 1 - rng.uniform(0.05, 0.45) * sombra
    foto = np.clip(foto * rng.uniform(0.8, 1.1), 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        foto = cv2.GaussianBlur(foto, (3, 3), rng.uniform(0.3, 1.2))
    return Image.fromarray(foto)


def cargar_imagen(path, tamaño_imagen, aumentar=False, recortar=True, foto=False, fondos=()):
    with Image.open(path) as image:
        gris = ImageOps.exif_transpose(image).convert("L")
        if not recortar:
            square = ImageOps.pad(gris, (tamaño_imagen, tamaño_imagen), method=Image.Resampling.BILINEAR, color=255)
        else:
            square = recortar_digito(gris, tamaño_imagen, foto=foto)
            if square is None:
                square = recortar_digito(gris, tamaño_imagen, foto=False)
            if square is None:
                raise ValueError(f"No se encontró un dígito: {path}")
            if aumentar and random.random() < 0.8:
                escena = simular_foto(square, fondos)
                procesada = recortar_digito(escena, tamaño_imagen, foto=True)
                if procesada is not None:
                    square = procesada
            if aumentar:
                a = np.asarray(square)
                M = cv2.getRotationMatrix2D((tamaño_imagen / 2, tamaño_imagen / 2), random.uniform(-8, 8), random.uniform(0.8, 1.12))
                M[:, 2] += [random.uniform(-0.08, 0.08) * tamaño_imagen for _ in range(2)]
                square = Image.fromarray(cv2.warpAffine(a, M, (tamaño_imagen, tamaño_imagen), borderValue=255))
        pixels = np.asarray(square, dtype=np.float32).copy()
    return torch.from_numpy(pixels).unsqueeze(0) / 127.5 - 1.0


def separar_cada_digito(folder, seed):
    carpeta = Path(folder)
    manifiesto = carpeta / "manifest.csv"
    if not manifiesto.is_file():
        raise FileNotFoundError(f"No se encontró {manifiesto}")
    with manifiesto.open(newline="", encoding="utf-8-sig") as archivo:
        registros = list(csv.DictReader(archivo))
    rng = random.Random(seed)
    digitos = list("0123456789")
    grupos = {digito: {} for digito in digitos}
    origenes = {}
    for fila in registros:
        digito = fila["clase_solicitada"]
        grupo = fila["job_id"]
        ruta = carpeta / fila["archivo"]
        if not ruta.is_file():
            raise FileNotFoundError(ruta)
        if grupo in origenes and origenes[grupo] != digito:
            raise ValueError("Una cuadrícula contiene varias clases en el manifiesto.")
        origenes[grupo] = digito
        grupos[digito].setdefault(grupo, []).append((ruta, int(digito)))
    train, val, test = [], [], []
    for digito in digitos:
        ids = sorted(grupos[digito])
        if len(ids) != 4:
            raise ValueError(f"Se esperaban cuatro cuadrículas en la clase {digito}.")
        rng.shuffle(ids)
        for i, grupo in enumerate(ids):
            destino = train if i < 2 else val if i == 2 else test
            destino.extend(grupos[digito][grupo])
        print(f"Clase {digito}: {sum(len(v) for v in grupos[digito].values())} imágenes")
    return digitos, train, val, test


class Dataset_digitos(Dataset):
    def __init__(self, samples, tamaño_imagen, aumentar=False, fondos=()):
        self.samples = samples
        self.tamaño_imagen = tamaño_imagen
        self.aumentar = aumentar
        self.fondos = fondos

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return {
            "pixel_values": cargar_imagen(path, self.tamaño_imagen, aumentar=self.aumentar, foto=True, fondos=self.fondos),
            "label": label,
        }

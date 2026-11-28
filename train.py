import argparse
import csv
import hashlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader

from config import DenseNetConfig, obtener_config
from dataset import Dataset_digitos, separar_cada_digito
from model import DenseNet


def obtener_datasets(config, device):
    digitos, train, val, test = separar_cada_digito(config["folder"], config["seed"])

    carpeta = config["folder_m"]
    carpeta.mkdir(parents=True, exist_ok=True)
    with (carpeta / "particiones.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["conjunto", "archivo", "etiqueta"])
        for nombre, muestras in (("train", train), ("val", val), ("test", test)):
            escritor.writerows((nombre, str(ruta.relative_to(config["folder"])), etiqueta) for ruta, etiqueta in muestras)

    fondos = sorted(p for p in config["fondos"].glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    def cargar(samples, shuffle=False):
        return DataLoader(
            Dataset_digitos(samples, config["tamaño_imagen"], aumentar=(shuffle and config["aumentar_datos"]), fondos=fondos),
            batch_size=config["batch_size"],
            shuffle=shuffle,
            num_workers=config["num_workers"],
            pin_memory=(device.type == "cuda"),
        )

    print(f"Train: {len(train)} | Validación: {len(val)} | Prueba: {len(test)}")
    return digitos, cargar(train, True), cargar(val), cargar(test)


@torch.inference_mode()
def validar(model, loader, loss_fn, device):
    model.eval()
    loss_total = 0.0
    actual, predicted = [], []

    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss_total += loss_fn(logits, y).item() * y.size(0)
        actual.extend(y.cpu().tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())

    mean_loss = loss_total / len(actual)
    precision = accuracy_score(actual, predicted)
    return mean_loss, precision, actual, predicted


def guardar_checkpoint(path, contenido):
    temporal = path.with_suffix(".tmp")
    torch.save(contenido, temporal)
    os.replace(temporal, path)



def normalizar_historial(path):
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as archivo:
        reader = csv.DictReader(archivo)
        campos = reader.fieldnames or []
        registros = list(reader)
    if "accuracy_train" in campos:
        return
    campos_nuevos = ["epoca", "loss_train", "loss_val", "accuracy_train", "accuracy_val"]
    temporal = path.with_suffix(".csv.tmp")
    with temporal.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=campos_nuevos)
        writer.writeheader()
        for registro in registros:
            writer.writerow({campo: registro.get(campo, "") for campo in campos_nuevos})
    os.replace(temporal, path)


def graficar_historial(path, salida=None):
    path = Path(path)
    if not path.is_file():
        print(f"No se encuentra el historial: {path}")
        return False
    salida = Path(salida) if salida is not None else path.parent
    salida.mkdir(parents=True, exist_ok=True)
    with path.open(newline="", encoding="utf-8") as archivo:
        registros = list(csv.DictReader(archivo))
    if not registros:
        print("El historial aún no contiene épocas.")
        return False
    def serie(nombre):
        valores = []
        for fila in registros:
            v = fila.get(nombre, "")
            try:
                valores.append(float(v) if v not in (None, "") else float("nan"))
            except ValueError:
                valores.append(float("nan"))
        return valores
    epocas = [int(fila["epoca"]) for fila in registros]
    perdida_train, perdida_val = serie("loss_train"), serie("loss_val")
    accuracy_train, accuracy_val = serie("accuracy_train"), serie("accuracy_val")

    def curva(nombre, series, ylabel, porcentaje=False):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for valores, etiqueta, color in series:
            if not any(v == v for v in valores):
                continue
            y = [v * 100 if porcentaje else v for v in valores]
            ax.plot(epocas, y, marker="o", linewidth=2, markersize=5,
                    color=color, label=etiqueta)
        ax.set_xticks(epocas)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.set_title(nombre.replace("_", " ").capitalize())
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        destino = salida / f"{nombre}.png"
        fig.savefig(destino, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfica guardada: {destino}")

    curva("curva_loss", [
        (perdida_train, "Entrenamiento", "#2563eb"),
        (perdida_val, "Validación", "#c62828")], "Cross-entropy (loss)")
    curva("curva_accuracy", [
        (accuracy_train, "Entrenamiento", "#2563eb"),
        (accuracy_val, "Validación", "#c62828")], "Accuracy (%)", porcentaje=True)
    if not any(v == v for v in accuracy_train):
        print("Aviso: el CSV no contiene accuracy_train; la curva de accuracy solo muestra validación.")
    if len(epocas) < 2:
        print("Aviso: una sola época genera puntos, no una curva de aprendizaje.")
    return True


def entrenar(config, reiniciar=False, pesos=None):
    if not pesos and (reiniciar or not (config["folder_m"] / "latest.pt").is_file()):
        carpeta_anterior = config["folder_m_anterior"]
        pesos = next((str(carpeta_anterior / nombre) for nombre in ("mejor.pt", "latest.pt")
                      if (carpeta_anterior / nombre).is_file()), None)
        if pesos is None:
            raise FileNotFoundError(f"No se encontró mejor.pt ni latest.pt en {carpeta_anterior}. Monta Drive y comprueba los archivos.")
    if pesos:
        if not Path(pesos).is_file():
            raise FileNotFoundError(f"No se encontraron los pesos: {pesos}")
        if Path(pesos).resolve() in [(config["folder_m"] / nombre).resolve() for nombre in ("mejor.pt", "latest.pt")]:
            raise ValueError("Los pesos de origen y la carpeta de salida deben ser distintos.")
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Dispositivo: {device}")

    digitos, train_loader, val_loader, test_loader = obtener_datasets(config, device)
    model_config = DenseNetConfig(num_clases=len(digitos), variante=config["variante"])
    model = DenseNet(model_config).to(device)
    parametros = sum(p.numel() for p in model.parameters())
    print(f"DenseNet-{config['variante']} propia: {parametros:,} parámetros.")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    folder = config["folder_m"]
    folder.mkdir(parents=True, exist_ok=True)
    best_path = folder / "mejor.pt"
    last_path = folder / "latest.pt"
    history_path = folder / "historial.csv"
    if not pesos and not last_path.exists():
        raise ValueError("Indica los pesos anteriores con --pesos para iniciar el fine-tuning.")
    normalizar_historial(history_path)

    setup = {
        "digitos": digitos,
        "tamaño_imagen": config["tamaño_imagen"],
        "variante": config["variante"],
        "seed": config["seed"],
        "origen": str(Path(pesos).resolve()) if pesos else "desde_cero",
        "preprocesamiento": "foto_v2",
        "manifest_sha256": hashlib.sha256((config["folder"] / "manifest.csv").read_bytes()).hexdigest(),
        "particion": "por_job_id_2_1_1",
        "lr": config["lr"],
        "arquitectura": "densenet_propia_v1",
        "aumentar_datos": config["aumentar_datos"],
    }
    epoca_inicial, mejor_acc = 0, -1.0

    if reiniciar and not pesos:
        raise ValueError("Indica --pesos para reiniciar el fine-tuning.")
    if reiniciar:
        print("Reinicio solicitado: se ignora el checkpoint de reanudación.")
        for path in (last_path, best_path, history_path, folder / "curva_loss.png", folder / "curva_accuracy.png"):
            if path.exists():
                path.unlink()
    elif last_path.exists():
        estado = torch.load(last_path, map_location="cpu", weights_only=True)
        if not pesos:
            setup["origen"] = estado["setup"]["origen"]
        if estado.get("setup") != setup:
            raise ValueError(
                f"El checkpoint {last_path} pertenece a otra configuración. "
                "Para empezar de cero ejecuta: !python train.py --reiniciar"
            )

        model.load_state_dict(estado["model"])
        optimizer.load_state_dict(estado["optimizer"])
        epoca_inicial = int(estado["epoch"])
        mejor_acc = float(estado["mejor_acc"])
        print(f"Reanudando desde la época {epoca_inicial + 1} de {config['epocas']}.")
        if not best_path.exists():
            raise FileNotFoundError("Falta mejor.pt; recupera el checkpoint o usa otra carpeta de salida.")
        del estado

    if epoca_inicial == 0:
        if pesos:
            estado = torch.load(pesos, map_location="cpu", weights_only=True)
            previo = estado["setup"]
            if previo["variante"] != setup["variante"] or list(map(str, previo["digitos"])) != setup["digitos"] or previo["tamaño_imagen"] != setup["tamaño_imagen"]:
                raise ValueError("El checkpoint no coincide con la arquitectura, clases o resolución.")
            model.load_state_dict(estado["model"])
            print(f"Fine-tuning desde: {pesos}")
            del estado
        else:
            raise ValueError("Se requieren los pesos de origen.")

    if epoca_inicial == 0:
        perdida_base, exactitud_base, _, _ = validar(model, val_loader, loss_fn, device)
        print(f"VALIDACIÓN ANTES DEL AJUSTE | loss={perdida_base:.4f} | accuracy={exactitud_base:.4%}")
        (folder / "validacion_inicial.txt").write_text(f"loss={perdida_base:.6f}\naccuracy={exactitud_base:.6%}\n", encoding="utf-8")

    if epoca_inicial < config["epocas"]:
        crear_encabezado = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as archivo:
            writer = csv.writer(archivo)
            if crear_encabezado:
                writer.writerow(["epoca", "loss_train", "loss_val", "accuracy_train", "accuracy_val"])

            for epoch in range(epoca_inicial, config["epocas"]):
                torch.manual_seed(config["seed"] + epoch)
                model.train()
                loss_acumulada = 0.0
                aciertos_train = 0
                muestras = 0

                for paso, batch in enumerate(train_loader, start=1):
                    x = batch["pixel_values"].to(device, non_blocking=True)
                    y = batch["label"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()

                    loss_acumulada += loss.item() * y.size(0)
                    aciertos_train += (logits.argmax(dim=1) == y).sum().item()
                    muestras += y.size(0)
                    if paso % 100 == 0 or paso == len(train_loader):
                        print(
                            f"Época {epoch + 1}/{config['epocas']} "
                            f"| lote {paso}/{len(train_loader)}",
                            flush=True,
                        )

                train_loss = loss_acumulada / muestras
                train_acc = aciertos_train / muestras
                val_loss, val_acc, _, _ = validar(model, val_loader, loss_fn, device)
                print(
                    f"Época {epoch + 1}/{config['epocas']} "
                    f"| loss_train={train_loss:.4f} | loss_val={val_loss:.4f} "
                    f"| accuracy_train={train_acc:.4%} | accuracy_val={val_acc:.4%}",
                    flush=True,
                )

                writer.writerow([epoch + 1, train_loss, val_loss, train_acc, val_acc])
                archivo.flush()
                graficar_historial(history_path, folder)
                weights = {"model": model.state_dict(), "setup": setup}

                if val_acc > mejor_acc:
                    mejor_acc = float(val_acc)
                    guardar_checkpoint(best_path, weights)
                    print("  Nuevo mejor modelo guardado.", flush=True)

                guardar_checkpoint(
                    last_path,
                    {
                        **weights,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "mejor_acc": mejor_acc,
                    },
                )
    else:
        print("Las épocas solicitadas ya están completadas. Evaluando el mejor modelo.")

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    test_loss, test_acc, actual, predicted = validar(model, test_loader, loss_fn, device)
    print(f"PRUEBA FINAL | loss={test_loss:.4f} | accuracy={test_acc:.4%}")

    display = ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=list(range(len(digitos))),
        display_labels=digitos,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    display.ax_.set(xlabel="Predicción", ylabel="Valor real")
    display.figure_.savefig(folder / "matriz_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(display.figure_)

    with (folder / "resultado_prueba.txt").open("w", encoding="utf-8") as archivo:
        archivo.write(f"Pérdida en prueba: {test_loss:.6f}\nAccuracy en prueba: {test_acc:.6%}\n")
    print(f"Resultados guardados en: {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar o ajustar DenseNet para dígitos.")
    parser.add_argument(
        "--reiniciar", action="store_true", help="Ignorar checkpoint de reanudación de la carpeta de salida."
    )
    parser.add_argument("--solo-graficas", action="store_true", help="Graficar historial sin entrenar.")
    parser.add_argument("--historial", type=str, help="Ruta del historial.csv, anterior o actual.")
    parser.add_argument("--pesos", type=str)
    parser.add_argument("--epocas", type=int)
    parser.add_argument("--salida", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    if args.solo_graficas:
        carpeta = obtener_config()["folder_m"]
        graficar_historial(args.historial or carpeta / "historial.csv")
    else:
        config = obtener_config()
        if args.epocas is not None:
            config["epocas"] = args.epocas
        if args.dataset:
            config["folder"] = Path(args.dataset)
        if args.salida:
            config["folder_m"] = Path(args.salida)
        if config["epocas"] < 1:
            raise ValueError("Se requiere al menos una época.")
        if args.pesos and Path(args.pesos).resolve() in [(config["folder_m"] / n).resolve() for n in ("mejor.pt", "latest.pt")]:
            raise ValueError("Los pesos de origen y la carpeta de salida deben ser distintos.")
        entrenar(config, reiniciar=args.reiniciar, pesos=args.pesos)

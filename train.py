import argparse
import csv
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader

from config import DenseNetConfig, get_config
from dataset import Dataset_digitos, separar_cada_digito
from model import DenseNet


def get_ds(config, device):
    digitos, train, val, test = separar_cada_digito(config["folder"], config["seed"])

    def cargar(samples, shuffle=False):
        return DataLoader(
            Dataset_digitos(samples, config["tamaño_imagen"]),
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


def save_checkpoint(path, contenido):
    temporal = path.with_suffix(".tmp")
    torch.save(contenido, temporal)
    os.replace(temporal, path)


def entrenar(config, reiniciar=False):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Dispositivo: {device}")

    digitos, train_loader, val_loader, test_loader = get_ds(config, device)
    model_config = DenseNetConfig(num_clases=len(digitos), variante=config["variante"])
    model = DenseNet(model_config).to(device)
    parametros = sum(p.numel() for p in model.parameters())
    print(f"DenseNet-{config['variante']} propia: {parametros:,} parámetros; pesos aleatorios.")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    folder = config["folder_m"]
    folder.mkdir(parents=True, exist_ok=True)
    best_path = folder / "mejor.pt"
    last_path = folder / "latest.pt"
    history_path = folder / "historial.csv"

    setup = {
        "digitos": digitos,
        "tamaño_imagen": config["tamaño_imagen"],
        "variante": config["variante"],
        "seed": config["seed"],
        "origen": "desde_cero",
        "arquitectura": "densenet_propia_v1",
    }
    epoca_inicial, mejor_acc = 0, -1.0

    if reiniciar:
        print("Reinicio solicitado: no se cargará ningún checkpoint anterior.")
        for path in (last_path, best_path, history_path):
            if path.exists():
                path.unlink()
    elif last_path.exists():
        estado = torch.load(last_path, map_location="cpu", weights_only=True)
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
            save_checkpoint(best_path, {"model": model.state_dict(), "setup": setup})
            print("Aviso: no se encontró mejor.pt; se conservó latest.pt como respaldo.")
        del estado

    if epoca_inicial < config["epocas"]:
        crear_encabezado = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as archivo:
            writer = csv.writer(archivo)
            if crear_encabezado:
                writer.writerow(["epoca", "loss_train", "loss_val", "accuracy_val"])

            for epoch in range(epoca_inicial, config["epocas"]):
                torch.manual_seed(config["seed"] + epoch)
                model.train()
                loss_acumulada = 0.0
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
                    muestras += y.size(0)
                    if paso % 100 == 0 or paso == len(train_loader):
                        print(
                            f"Época {epoch + 1}/{config['epocas']} "
                            f"| lote {paso}/{len(train_loader)}",
                            flush=True,
                        )

                train_loss = loss_acumulada / muestras
                val_loss, val_acc, _, _ = validar(model, val_loader, loss_fn, device)
                print(
                    f"Época {epoch + 1}/{config['epocas']} "
                    f"| loss_train={train_loss:.4f} | loss_val={val_loss:.4f} "
                    f"| accuracy_val={val_acc:.4%}",
                    flush=True,
                )

                writer.writerow([epoch + 1, train_loss, val_loss, val_acc])
                archivo.flush()
                weights = {"model": model.state_dict(), "setup": setup}



                # A partir de aqui la IA nos ayudó de mas jajaja en el debug

                
                if val_acc > mejor_acc:
                    mejor_acc = float(val_acc)
                    save_checkpoint(best_path, weights)
                    print("  Nuevo mejor modelo guardado.", flush=True)

                save_checkpoint(
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
    parser = argparse.ArgumentParser(description="Entrenar DenseNet desde cero para dígitos.")
    parser.add_argument(
        "--reiniciar", action="store_true", help="Ignorar checkpoints y comenzar con pesos nuevos."
    )
    args = parser.parse_args()
    entrenar(get_config(), reiniciar=args.reiniciar)

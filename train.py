import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import torch.nn as nn

def save_checkpoint(path, state):
    import torch
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
 
from torch.utils.data import DataLoader

from config import DenseNetConfig, get_config
from dataset import Dataset_digitos, separar_cada_digito

from model import DenseNet



def get_ds(config):
    digitos, train,val,test = separar_cada_digito(config["folder"], config["seed"])

    def cargar(samples, shuffle=False):
        return DataLoader(
            Dataset_digitos(samples, config["tamaño_imagen"]),
            batch_size = config["batch_size"],
            shuffle = shuffle,
            num_workers = 2
        )
    print(f"train: {len(train)}, vals: {len(val)}, prueba {len(test)}")
    return digitos, cargar(train, True), cargar(val), cargar(test)


@torch.inference_mode()
def validar(model, loader, loss_fn, device):
    model.eval()
    loss_total = 0.0
    actual, predicted = [],[]


    for batch in loader:
        x = batch["pixel_values"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss_total += loss_fn(logits, y).item() * y.size(0)
        actual.extend(batch["label"].tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())

    mean_loss = loss_total / len(actual)
    precision = accuracy_score(actual,predicted)
    return mean_loss,precision ,actual, predicted


def entrenar(config):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digitos, train_loader, val_loader, test_loader = get_ds(config)
    model_config = DenseNetConfig(num_clases=len(digitos), variante=config["variante"])
    model = DenseNet(model_config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    folder = config["folder_m"]
    folder.mkdir(parents=True, exist_ok=True)
    best_path = folder / "mejor.pt"
    last_path = folder / "latest.pt"



    setup = {
        "digitos": digitos,
        "tamaño_imagen": config["tamaño_imagen"],
        "variante": config["variante"],
        "seed": config["seed"]

    }
    epoca_inicial, mejor_acc = 0, -1.0


    if last_path.exists():
        estado = torch.load(last_path, map_location="cpu", weights_only=True)
        if estado["setup"] != setup:
            raise ValueError("Error")


        model.load_state_dict(estado["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoca_inicial, mejor_acc = estado["epoch"], estado["mejor_acc"]
        del estado


    for epoch in range (epoca_inicial, config["epocas"]):
        torch.manual_seed(config["seed"] + epoch)
        model.train()


        for batch in train_loader:
            x = batch["pixel_values"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()


        val_loss, val_acc, _, _ = validar(model, val_loader, loss_fn, device)
        print(f"Epoca{epoch +1},perdida{val_loss},accuracy{val_acc}")
        weights = {"model": model.state_dict(), "setup": setup}


        if val_acc > mejor_acc:
            mejor_acc = float(val_acc)
            save_checkpoint(best_path, weights)

        save_checkpoint(last_path, {
            **weights,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "mejor_acc": mejor_acc
        })
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])


    display = ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels = list(range(len(digitos))),
        display_labels=digitos,
        cmap="Blues",
        values_format = "d",
        colorbar=False
    )
    display.ax_.set(xlabel="Prediccion", ylabel="Valor real")
    display.figure_.savefig(folder / "matriz_confusion.png", dpi=180, bbox_inches="tight")
    plt.close(display.figure_)


if __name__ == "__main__":
    entrenar(get_config())






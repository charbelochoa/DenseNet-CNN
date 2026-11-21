from pathlib import Path


class DenseNetConfig:
    VARIANTES = {
        "121": dict(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64),
        "169": dict(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64),
        "201": dict(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64),
    }

    def __init__(self, num_clases, variante="121"):
        if variante not in self.VARIANTES:
            raise ValueError(f"Variante no compatible: {variante}")

        parametros = self.VARIANTES[variante]
        self.variante = variante
        self.block_config = parametros["block_config"]
        self.num_clases = num_clases
        self.growth_rate = parametros["growth_rate"]
        self.num_init_features = parametros["num_init_features"]
        self.bn_size = 4
        self.num_canales = 1
        self.drop_rate = 0.0
        self.compression = 0.5
        self.ahorro_memoria = True  # Reduce memoria de activaciones en Colab.


def get_config():
    return {
        "variante": "121",
        "tamaño_imagen": 256,
        "batch_size": 16,
        "epocas": 10,
        "lr": 1e-4,
        "seed": 42,
        "num_workers": 2,
        # cambiar dependiendo 
        "folder": Path("/content/digitos"),
        # nueva
        "folder_m": Path("/content/resultados_densenet_scratch"),
    }

#config 
from pathlib import Path


class DenseNetConfig:
    VARIANTES = {
        "121": dict(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64),
        "169": dict(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64),
        "201": dict(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64),
    }

    def __init__(
        self,
        num_clases,
        variante="121",
    ):
        self.variante = variante
        self.block_config = self.VARIANTES[variante]
        self.num_clases = num_clases
        self.growth_rate = 32
        self.num_init_features = 64
        self.bn_size = 4
        self.num_canales = 1
        self.drop_rate = 0.0
        self.compression = 0.5
        self.ahorro_memoria = False



def get_config():
    return {
        "variante": "121",
        "tamaño_imagen": 256,
        "batch_size": 16,
        "epocas": 5,
        "lr": 1e-4,
        "seed": 42,


        #reservado para ubicacion de los archivos

        "folder": Path("/content/"),
        "folder_m": Path("/content/")
    }







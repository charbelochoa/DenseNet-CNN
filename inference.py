import argparse

from pathlib import Path 

import torch

from config import DenseNetConfig, get_config
from dataset import cargar_imagen
from model import DenseNet


@torch.inference_mode()
def predict(image_path, checkpoint_path, real=None):
    estado = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    setup = estado["setup"]
    digitos = setup["digitos"]



    model_config = DenseNetConfig(num_clases=len(digitos), variante=setup["variante"])
    model = DenseNet(model_config)
    model.load_state_dict(state["model"])
    model.eval()




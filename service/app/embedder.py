from threading import Lock

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from app.config import Settings


class DinoV2Embedder:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._device = self._resolve_device(settings.device)
        torch.set_num_threads(settings.torch_threads)
        self._processor = AutoImageProcessor.from_pretrained(
            settings.model_id, revision=settings.model_revision, use_fast=False
        )
        self._model = AutoModel.from_pretrained(
            settings.model_id, revision=settings.model_revision
        ).eval().to(self._device)
        self._lock = Lock()
        self.dimension = int(self._model.config.hidden_size)

    @property
    def model_version(self) -> str:
        return self._settings.model_version

    @property
    def device(self) -> str:
        return str(self._device)

    def embed(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
        with self._lock, torch.inference_mode():
            outputs = self._model(**inputs)
            vectors = outputs.last_hidden_state[:, 0, :]
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().to(torch.float32).numpy().astype(np.float32).tolist()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

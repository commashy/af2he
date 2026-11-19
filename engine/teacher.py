"""Frozen fresh-WSI generator used as a tissue-masked L1 teacher.

The checkpoint is the separate ViT generator in stain_translation_clean.
Its package name collides with this repo, so those modules are installed
only while the teacher is constructed or run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from data.stage1 import stitch_quadrants


class _CleanImport:
    """Swap this repo's models/data packages for stain_translation_clean."""

    def __init__(self, root: Path):
        self.root = str(root)
        self.prefixes = ("models", "data", "utils")
        self.clean: dict = {}
        self._saved: dict = {}

    def _matching(self) -> list[str]:
        keys = []
        for key in list(sys.modules):
            for prefix in self.prefixes:
                if key == prefix or key.startswith(prefix + "."):
                    keys.append(key)
                    break
        return keys

    def __enter__(self):
        self._saved = {key: sys.modules.pop(key) for key in self._matching()}
        sys.path.insert(0, self.root)
        sys.modules.update(self.clean)
        return self

    def __exit__(self, exc_type, exc, tb):
        current = {key: sys.modules.pop(key) for key in self._matching()}
        self.clean.update(current)
        sys.modules.update(self._saved)
        while self.root in sys.path:
            sys.path.remove(self.root)
        return False


def _package_root(config_path: Path) -> Path:
    root = config_path.resolve()
    while not (root / "models" / "i2i_model.py").is_file():
        if root.parent == root:
            raise FileNotFoundError(f"cannot find stain_translation_clean above {config_path}")
        root = root.parent
    return root


class CleanTeacher:
    """One 1024 AF field in, four 512 H&E quadrants out. Weights stay frozen."""

    def __init__(self, config_path: str, checkpoint: str, thumbnail_dir: str, device: torch.device):
        self.device = device
        self.config_path = Path(config_path)
        self.checkpoint = Path(checkpoint)
        self.thumbnail_dir = Path(thumbnail_dir)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"missing teacher config {self.config_path}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"missing teacher checkpoint {self.checkpoint}")
        if not self.thumbnail_dir.is_dir():
            raise FileNotFoundError(f"missing teacher thumbnails {self.thumbnail_dir}")
        self._cache: dict[str, torch.Tensor] = {}
        self._tokens: dict[str, torch.Tensor] = {}
        self._imports = _CleanImport(_package_root(self.config_path))
        self.model = None
        self._transform = None
        self._build()

    def _build(self) -> None:
        with self._imports:
            from omegaconf import OmegaConf
            from models.i2i_model import I2IModel

            cfg = OmegaConf.load(self.config_path)
            OmegaConf.set_struct(cfg, False)
            cfg.phase = "test"
            cfg.log_wsi_context_images = False
            OmegaConf.set_struct(cfg, True)
            model = I2IModel(cfg, self.device)
            model.load(str(self.checkpoint), load_optimizers=False)
            self._check_weights(model)
            size = int(cfg.wsi_context_size)
            for module in (model.netG, model.wsi_context_encoder, model.local_context_encoder):
                if module is None:
                    continue
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            self.model = model
            self._transform = transforms.Compose([
                transforms.Resize((size, size), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def _check_weights(self, model) -> None:
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(self.checkpoint, map_location="cpu")
        groups = (("netG", model.netG), ("wsi_context_encoder", model.wsi_context_encoder))
        for key, module in groups:
            if key not in payload or module is None:
                raise RuntimeError(f"teacher checkpoint has no {key}")
            incompatible = module.load_state_dict(payload[key], strict=False)
            if incompatible.missing_keys:
                raise RuntimeError(f"teacher {key} missing {incompatible.missing_keys}")
        if model.local_context_encoder is None:
            raise RuntimeError("teacher local context encoder was not built")
        if "local_context_encoder" not in payload:
            print("teacher local context encoder is using the WSI context weights from this checkpoint")

    def _thumbnail(self, slide: str, flip: bool) -> torch.Tensor:
        cached = self._cache.get(slide)
        if cached is None:
            path = self.thumbnail_dir / f"{slide}.png"
            if not path.is_file():
                raise FileNotFoundError(f"no teacher thumbnail for {slide}: {path}")
            with Image.open(path) as image:
                cached = self._transform(image.convert("RGB"))
            self._cache[slide] = cached
        if flip:
            return torch.flip(cached, dims=(-1,))
        return cached

    def _slide_token(self, slide: str) -> torch.Tensor:
        cached = self._tokens.get(slide)
        if cached is not None:
            return cached
        thumb = self._thumbnail(slide, False).unsqueeze(0).to(self.device, non_blocking=True)
        with self._imports:
            with torch.inference_mode():
                gate = torch.ones(1, 1, device=self.device)
                token = self.model.wsi_context_encoder(thumb, gate)
        token = token.detach().float().view(1, -1)
        self._tokens[slide] = token
        return token

    def predict_fields(self, fields: torch.Tensor, slide: str) -> torch.Tensor:
        """Run a batch of 1024 AF fields from one slide. Returns H&E, same shape."""
        if fields.ndim != 4 or tuple(fields.shape[1:]) != (3, 1024, 1024):
            raise ValueError(f"expected N x 3 x 1024 x 1024, got {tuple(fields.shape)}")
        batch = fields.shape[0]
        token = self._slide_token(slide)
        if token.shape[0] == 1 and batch > 1:
            token = token.expand(batch, -1)
        zeros = torch.zeros(batch, 3, 512, 512, device=self.device)
        data = {
            "A": zeros,
            "B": zeros,
            "A_full": fields,
            "B_full": torch.zeros(batch, 3, 1024, 1024, device=self.device),
            "wsi_global_token": token,
            "context_available": torch.ones(batch, 1, device=self.device),
        }
        with self._imports:
            with torch.no_grad():
                self.model.set_input(data)
                self.model.forward()
                merged = self.model.fake_B_global
                if merged is None or tuple(merged.shape) != (batch, 3, 1024, 1024):
                    shape = None if merged is None else tuple(merged.shape)
                    raise RuntimeError(f"teacher field is {shape}, expected {(batch, 3, 1024, 1024)}")
                return merged.detach().float().cpu()

    def predict_crops(self, fields: torch.Tensor, slide: str, offsets: tuple[int, ...]) -> torch.Tensor:
        """512 crops at ``offsets`` inside each 1024 field.

        Returns ``(N, K, 3, 512, 512)`` with ``K = len(offsets) ** 2``, row-major.
        Every crop keeps that 1024 field as its local context.
        """
        if fields.ndim != 4 or tuple(fields.shape[1:]) != (3, 1024, 1024):
            raise ValueError(f"expected N x 3 x 1024 x 1024, got {tuple(fields.shape)}")
        if fields.device != self.device:
            fields = fields.to(self.device, non_blocking=True)
        count = fields.shape[0]
        crops = [fields[:, :, y:y + 512, x:x + 512] for y in offsets for x in offsets]
        packed = torch.stack(crops, dim=1).reshape(count * len(crops), 3, 512, 512)
        token = self._slide_token(slide)
        if token.shape[0] == 1 and count > 1:
            token = token.expand(count, -1)
        with self._imports:
            with torch.no_grad():
                model = self.model
                model.cached_wsi_global_token = token
                model.context_available = torch.ones(count, 1, device=self.device)
                model.wsi_context = None
                model.real_A_full = fields
                style = model._make_style(
                    count * len(crops),
                    base_source=fields,
                    repeat=len(crops),
                    context_available=model.context_available,
                )
                film = model._build_context_film()
                out = model._generator_forward(packed, style=style, context_film=film)
                expected = (count * len(crops), 3, 512, 512)
                if tuple(out.shape) != expected:
                    raise RuntimeError(f"crop output is {tuple(out.shape)}, expected {expected}")
                return out.detach().float().cpu().view(count, len(crops), 3, 512, 512)

    def predict_pairs(self, contexts: torch.Tensor, crops: torch.Tensor, slide: str) -> torch.Tensor:
        """One 512 crop for each 1024 local-context field. Returns ``(N, 3, 512, 512)``."""
        if contexts.ndim != 4 or tuple(contexts.shape[1:]) != (3, 1024, 1024):
            raise ValueError(f"expected N x 3 x 1024 x 1024 contexts, got {tuple(contexts.shape)}")
        if crops.shape[0] != contexts.shape[0] or tuple(crops.shape[1:]) != (3, 512, 512):
            raise ValueError(f"expected matching 512 crops, got {tuple(crops.shape)}")
        if contexts.device != self.device:
            contexts = contexts.to(self.device, non_blocking=True)
        if crops.device != self.device:
            crops = crops.to(self.device, non_blocking=True)
        count = contexts.shape[0]
        token = self._slide_token(slide)
        if token.shape[0] == 1 and count > 1:
            token = token.expand(count, -1)
        with self._imports:
            with torch.no_grad():
                model = self.model
                model.cached_wsi_global_token = token
                model.context_available = torch.ones(count, 1, device=self.device)
                model.wsi_context = None
                model.real_A_full = contexts
                style = model._make_style(
                    count,
                    base_source=contexts,
                    repeat=1,
                    context_available=model.context_available,
                )
                film = model._build_context_film()
                out = model._generator_forward(crops, style=style, context_film=film)
                if tuple(out.shape) != (count, 3, 512, 512):
                    raise RuntimeError(f"pair output is {tuple(out.shape)}")
                return out.detach().float().cpu()

    def predict(self, af: torch.Tensor, slides: list[str], flips: list[bool]) -> torch.Tensor:
        """Return H&E quadrants aligned with ``af`` (TL, TR, BL, BR)."""
        if af.shape[0] != len(slides) * 4 or len(slides) != len(flips):
            raise ValueError(
                f"{tuple(af.shape)} quadrants for {len(slides)} slides and {len(flips)} flips"
            )
        fields = stitch_quadrants(af)
        tiles = [
            self._one(fields[index:index + 1], slide, bool(flip))
            for index, (slide, flip) in enumerate(zip(slides, flips))
        ]
        return torch.cat(tiles, dim=0)

    def _one(self, field: torch.Tensor, slide: str, flip: bool) -> torch.Tensor:
        thumb = self._thumbnail(slide, flip).unsqueeze(0).to(self.device, non_blocking=True)
        zeros = torch.zeros(1, 3, 512, 512, device=self.device)
        data = {
            "A": zeros,
            "B": zeros,
            "A_full": field,
            "B_full": torch.zeros_like(field),
            "wsi_context": thumb,
            "context_available": torch.ones(1, 1, device=self.device),
        }
        with self._imports:
            with torch.no_grad():
                self.model.set_input(data)
                self.model.forward()
                local = self.model.fake_B_local
                if local is None or tuple(local.shape) != (4, 3, 512, 512):
                    shape = None if local is None else tuple(local.shape)
                    raise RuntimeError(f"teacher output is {shape}, expected (4, 3, 512, 512)")
                return local.detach().clone()

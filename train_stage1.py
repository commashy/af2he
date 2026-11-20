"""Stage-1 AF to H&E training.

Occupancy matches a blurred tissue-probability map. AF has to agree, and
either the pseudo label or the teacher can mark tissue. Stain color is not a
target. Overlap L1 matches seam-centered windows that are generated without
gradient. LSGAN and gated FDL use the coordinate H&E. Patch NCE stays on the
AF. The field UNI loss uses stored retrieval keys.

    torchrun --nproc_per_node=4 train_stage1.py --config configs/stage1_pseudo.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.retrieval_table import RetrievalTable
from data.stage1 import Stage1Dataset, collate_stage1, seam_windows, stitch_quadrants
from engine.checkpoint import unwrap
from engine.contrastive import DebiasedPatchNCESRC, field_info_nce, patch_nce, retrieval_pair_weight
from engine.distributed import cleanup, is_main, setup
from engine.monitor import Monitor
from engine.image_losses import (
    AmplitudeFDL,
    af_brightness,
    af_presence,
    content_gate,
    bright_af_keep,
    bright_neighbor_fill,
    hed_recon,
    recon_bright_weight,
    hed_tissue_fraction,
    hed_tissue_mask,
    hed_tissue_prob,
    lsgan,
    recon_epoch_settings,
    recon_loss,
    rgb_tissue_recon,
    structure_loss,
    tissue_probability,
)
from models.discriminator import (
    MultiScaleDiscriminator,
    backbone_parameters,
    class_parameters,
    film_parameters,
    field_subtype_loss,
    load_discriminator_state,
    slide_classification_loss,
)
from models.field_critic import ConchFieldDiscriminator, multilevel_bce
from models.generator import BOTTLENECK_DIM, FEATURE_CHANNELS, Generator, resize_collage_positions
from models.heat_classifier import HeatClassifier
from models.sample import DomainAdaptedPatchProjector, PatchProjector
from models.slide_head import SlideDiagnosisHead
from models.wsi_collage_encoder import WSICollageAdapter


def _disc_outputs(output):
    if isinstance(output, tuple):
        return output
    return output, None


def _slide_targets(batch: dict, slide_to_index: dict[str, int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [slide_to_index[slide] for slide in batch["slide"]],
        dtype=torch.long,
        device=device,
    )


def _real_class_mask(he_field: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Tissue pixels of a real field. A field below ``cutoff`` is skipped entirely."""
    mask = hed_tissue_mask(he_field).float()
    fraction = mask.flatten(1).mean(dim=1)
    return mask * (fraction >= cutoff).float().view(-1, 1, 1, 1)


def _generated_class_mask(af: torch.Tensor, pseudo: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Pixels where AF and both aligned labels agree the pixel is tissue."""
    present = (af_presence(af) >= 0.5).float()
    return present * hed_tissue_mask(pseudo).float() * hed_tissue_mask(teacher).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _weight(config: dict, key: str) -> float:
    return float(config.get(key, 0.0) or 0.0)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = [
        "name", "af_root", "he_root", "alignment",
        "batch_size", "num_workers", "n_epochs", "n_epochs_decay", "lr_G", "lr_D", "lr_F",
        "samples_per_epoch", "checkpoint_dir", "pretrain_checkpoint", "seed", "log_every", "image_every",
    ]
    if bool(config.get("wsi_collage")) != bool(config.get("collage_root")):
        raise RuntimeError("wsi_collage and collage_root are used together")
    if not config.get("collage_root"):
        required.append("collage_index")
    if not config.get("epoch10_dir"):
        required.append("pseudo_root")
    if _weight(config, "lambda_GAN") > 0:
        required.append("netD_num_scales")
    if _weight(config, "lambda_NCE") > 0:
        required.extend(("nce_T", "nce_layers", "num_patches"))
    if _weight(config, "lambda_FDL") > 0:
        required.extend(("fdl_input_size", "fdl_phase_weight", "fdl_num_proj"))
    if _weight(config, "lambda_recon") > 0:
        required.append("recon_sigma")
        if config.get("load_teacher", True):
            required.append("teacher_dir")
        if int(config.get("recon_sharp_epochs", 0) or 0) > 0:
            required.extend(("lambda_recon_late", "recon_sigma_schedule", "epoch10_dir", "extra_slides"))
        else:
            required.append("recon_full_epochs")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    return config


def load_heat_targets(path: str | None) -> dict[str, tuple[float, bool]] | None:
    """filename -> (tile mean, whether the value came from a heatmap)."""
    if not path:
        return None
    table: dict[str, tuple[float, bool]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[row["filename"]] = (float(row["target"]), row["source"] == "map")
    if not table:
        raise RuntimeError(f"{path} has no heatmap targets")
    return table


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _show_field(field: torch.Tensor) -> np.ndarray:
    image = field[:1].detach().float().clamp(-1, 1)
    array = ((image + 1.0) * 127.5).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).numpy()[0]
    return np.asarray(Image.fromarray(array, mode="RGB").resize((512, 512), Image.Resampling.BOX))


def save_preview(directory: Path, step: int, af, pseudo, fake, he, teacher=None) -> np.ndarray:
    """Stitched 1024 fields: AF, output, optional teacher, optional pseudo, real H&E."""
    directory.mkdir(parents=True, exist_ok=True)
    named = [
        ("AF", stitch_quadrants(af[:4])),
        ("output", stitch_quadrants(fake[:4])),
    ]
    if teacher is not None:
        named.append(("teacher", stitch_quadrants(teacher[:4])))
    if pseudo is not None:
        named.append(("pseudo", stitch_quadrants(pseudo[:4])))
    named.append(("real H&E", stitch_quadrants(he[:4])))
    grid = np.concatenate([_show_field(field) for _, field in named], axis=1)
    image = Image.fromarray(grid, mode="RGB")
    draw = ImageDraw.Draw(image)
    for index, (label, _) in enumerate(named):
        draw.text((index * 512 + 8, 8), label, fill=(255, 255, 255))
    image.save(directory / f"step_{step:07d}.png")
    return np.asarray(image)


def _one_per_field(tensor: torch.Tensor | None, n_fields: int) -> torch.Tensor | None:
    """Collages and cell ids are repeated once per quadrant. NCE wants one per field."""
    if tensor is None:
        return None
    if tensor.shape[0] == n_fields:
        return tensor
    if tensor.shape[0] == n_fields * 4:
        return tensor[::4]
    raise RuntimeError(f"{tensor.shape[0]} rows for {n_fields} fields")


def _sample_paired_patches(
    query_feats: list[torch.Tensor],
    key_feats: list[torch.Tensor],
    num_patches: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Same spatial sites in both domains. The key is detached."""
    query_out = []
    key_out = []
    for query, key in zip(query_feats, key_feats):
        if query.shape[-2:] != key.shape[-2:]:
            raise RuntimeError(f"patch grids {tuple(query.shape[-2:])} and {tuple(key.shape[-2:])}")
        _batch, _channels, height, width = key.shape
        count = min(int(num_patches), height * width)
        choice = torch.randperm(height * width, device=key.device)[:count]
        key_out.append(key.detach().flatten(2).transpose(1, 2)[:, choice, :])
        query_out.append(query.flatten(2).transpose(1, 2)[:, choice, :])
    return query_out, key_out


def _maps(feats: list[torch.Tensor], layers: list[int]) -> list[torch.Tensor]:
    try:
        return [feats[int(layer)] for layer in layers]
    except IndexError as exc:
        raise ValueError(f"nce_layers {layers} outside encoder maps 0..{len(feats) - 1}") from exc


def _seam_targets(
    model,
    af: torch.Tensor,
    collage: torch.Tensor,
    device: torch.device,
    local: torch.Tensor | None = None,
    cells: torch.Tensor | None = None,
    black: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Seam windows from the current weights, one 512 at a time, with no gradient.

    These run before the training forward so their activations are gone before
    the quadrant graph is stored. The quadrant outputs are what inference shows,
    so the loss matches them to these detached centers.
    """
    fields = stitch_quadrants(af)
    collages = collage[::4]
    file_cells = None if cells is None else cells[::4]
    file_black = None if black is None else black[::4]
    targets = []
    for window in seam_windows(fields):
        pieces = []
        for index in range(window.shape[0]):
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
            ):
                neighborhood = None if local is None else local[index:index + 1]
                pieces.append(model(
                    window[index:index + 1],
                    collages[index:index + 1],
                    neighborhood,
                    cells=None if file_cells is None else file_cells[index:index + 1],
                    black_index=None if file_black is None else file_black[index:index + 1],
                    quadrant_group=1,
                ))
        targets.append(torch.cat(pieces, dim=0))
    return targets


def _match_table_orientation(field: torch.Tensor, flips: list[bool]) -> torch.Tensor:
    """Undo the training flip so the field matches the unflipped UNI keys."""
    if not any(flips):
        return field
    pieces = []
    for index, flipped in enumerate(flips):
        piece = field[index:index + 1]
        if flipped:
            piece = torch.flip(piece, dims=(-1,))
        pieces.append(piece)
    return torch.cat(pieces, dim=0)


def _uni_query(uni, field: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    rgb = ((field.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    rgb = torch.nn.functional.interpolate(
        rgb, size=(224, 224), mode="bilinear", align_corners=False, antialias=True,
    )
    normalized = (rgb - mean) / std
    with torch.autocast(device_type=field.device.type, dtype=torch.float16):
        embedding = uni(normalized)
    return embedding.float()


def _diagnosis_table(path: str) -> dict[str, int]:
    table: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("classes1") or "").strip()
            if label == "Benign":
                index = 0
            elif label == "Malignant":
                index = 1
            else:
                continue
            slide = (row.get("slide_id") or "").strip()
            if slide:
                table[slide] = index
    if not table:
        raise RuntimeError(f"no benign/malignant labels in {path}")
    return table


def _major_subtype_table(path: str) -> dict[str, int | None]:
    """1 benign, 2 MIA, 3 LUAD. Rare subtypes are present and skipped."""
    table: dict[str, int | None] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            slide = (row.get("slide_id") or "").strip()
            if not slide:
                continue
            classes = (row.get("classes1") or "").strip()
            subtype = (row.get("subtype_norm") or "").strip()
            if classes == "Benign":
                table[slide] = 1
            elif subtype == "MIA":
                table[slide] = 2
            elif subtype == "LUAD":
                table[slide] = 3
            else:
                table[slide] = None
    if not any(value is not None for value in table.values()):
        raise RuntimeError(f"no LUAD, MIA, or benign labels in {path}")
    return table


def _subtype_field_targets(
    slides: list[str],
    table: dict[str, int | None],
    fraction: torch.Tensor,
    cutoff: float,
    device: torch.device,
    benign_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Background when the field is empty. Rare subtypes are left out of the loss.

    ``benign_only`` uses the one fact that is true at patch level: every patch
    on a benign slide is benign. A malignant slide can contain benign lung, so
    its tissue fields are not given a malignant label.
    """
    targets = []
    include = []
    for slide, amount in zip(slides, fraction.detach().float().cpu().tolist()):
        kind = table[slide]
        if float(amount) < float(cutoff):
            targets.append(0)
            include.append(True)
        elif benign_only and kind != 1:
            targets.append(0)
            include.append(False)
        elif kind is None:
            targets.append(0)
            include.append(False)
        else:
            targets.append(int(kind))
            include.append(True)
    return (
        torch.tensor(targets, dtype=torch.long, device=device),
        torch.tensor(include, dtype=torch.bool, device=device),
    )


def _init_local_encoder(local, state: dict, collage_pos: torch.Tensor) -> None:
    """Copy the pretrained collage encoder. The mask channel stays zero."""
    own = local.state_dict()
    mapped = {}
    for key, dest in own.items():
        source = collage_pos if key == "pos" else state.get(f"collage_encoder.{key}")
        if (
            key == "cnn.0.weight"
            and source is not None
            and source.ndim == 4
            and source.shape[1] == 3
            and dest.shape[1] == 4
            and source.shape[0] == dest.shape[0]
        ):
            weight = torch.zeros_like(dest)
            weight[:, :3].copy_(source)
            mapped[key] = weight
            continue
        if source is not None and tuple(source.shape) == tuple(dest.shape):
            mapped[key] = source
    missing = [key for key in own if key not in mapped]
    if missing:
        raise RuntimeError(f"local encoder could not be copied from the collage encoder: {missing[:6]}")
    local.load_state_dict(mapped)


def _fit_pretrained_generator(generator, state: dict) -> None:
    module = unwrap(generator)
    state = dict(state)
    key = "collage_encoder.pos"
    collage_pos = state.get(key)
    if key in state and tuple(state[key].shape) != tuple(module.collage_encoder.pos.shape):
        state[key] = resize_collage_positions(state[key], module.collage_encoder.tokens)
    has_local_weights = any(name.startswith("local_encoder.") for name in state)
    if module.local_encoder is not None and not has_local_weights:
        if collage_pos is None:
            raise RuntimeError("checkpoint has no collage position table for the local encoder")
        _init_local_encoder(module.local_encoder, state, collage_pos)
    allowed = set()
    if module.local_encoder is not None and not has_local_weights:
        allowed = {
            name for name in module.state_dict()
            if name.startswith("local_encoder.") or name == "bottleneck.local_type"
        }
    incompatible = module.load_state_dict(state, strict=False)
    missing = [name for name in incompatible.missing_keys if name not in allowed]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "generator checkpoint does not match: "
            f"missing={missing} unexpected={list(incompatible.unexpected_keys)}"
        )


def _contrast(query, key, criterion) -> torch.Tensor:
    total = query[0].new_zeros(())
    for left, right in zip(query, key):
        total = total + criterion(left, right).mean()
    return total / len(query)


def save_state(
    path: Path,
    epoch: int,
    step: int,
    generator,
    discriminator,
    projector,
    opt_g,
    opt_d,
    slide_head=None,
    slide_ids: list[str] | None = None,
    field_discriminator=None,
    opt_field=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(step),
            "generator": unwrap(generator).state_dict(),
            "discriminator": unwrap(discriminator).state_dict() if discriminator is not None else None,
            "projector": unwrap(projector).state_dict() if projector is not None else None,
            "slide_head": unwrap(slide_head).state_dict() if slide_head is not None else None,
            "optimizer_g": opt_g.state_dict(),
            "optimizer_d": opt_d.state_dict() if opt_d is not None else None,
            "slide_ids": list(slide_ids) if slide_ids is not None else None,
            "field_discriminator": unwrap(field_discriminator).state_dict() if field_discriminator is not None else None,
            "optimizer_field": opt_field.state_dict() if opt_field is not None else None,
        },
        path,
    )


WSI_SUBTYPES = 7


def _attach_wsi_encoder(generator) -> None:
    module = unwrap(generator)
    adapter = WSICollageAdapter(WSI_SUBTYPES, BOTTLENECK_DIM)
    module.collage_encoder = adapter
    module.collage_size = int(adapter.encoder.size)


def _freeze_wsi_encoder(generator) -> None:
    encoder = unwrap(generator).collage_encoder.encoder
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)


def _eval_frozen_wsi(generator) -> None:
    module = unwrap(generator)
    encoder = getattr(module, "collage_encoder", None)
    if encoder is None or not hasattr(encoder, "encoder"):
        return
    frozen = all(not parameter.requires_grad for parameter in encoder.encoder.parameters())
    if frozen:
        encoder.encoder.eval()


def load_generator_weights(generator: torch.nn.Module, state: dict) -> None:
    module = unwrap(generator)
    if isinstance(module.collage_encoder, WSICollageAdapter):
        module.load_state_dict(state, strict=True)
        return
    _fit_pretrained_generator(generator, state)


def load_generator_only(path: Path, generator: torch.nn.Module, device: torch.device) -> None:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = payload["generator"] if "generator" in payload else payload["model"]
    load_generator_weights(generator, state)


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> int:
    taken = 0
    for state in optimizer.state.values():
        step = state.get("step", 0)
        if torch.is_tensor(step):
            step = int(step.item())
        taken = max(taken, int(step))
    return taken


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ctx = setup()
    set_seed(int(config["seed"]))
    device = torch.device(f"cuda:{ctx.local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    image_batch = int(config["batch_size"])
    if image_batch % 4 != 0:
        raise ValueError(f"batch_size={image_batch} must be divisible by 4 quadrants")
    load_teacher = bool(config.get("load_teacher", True))
    retrieval_dir = config.get("retrieval_dir")
    pair_index = str(Path(retrieval_dir) / "index.csv") if retrieval_dir else None
    dataset = Stage1Dataset(
        config["af_root"], config.get("pseudo_root"), config["he_root"],
        config["alignment"], config.get("collage_index"),
        config.get("teacher_dir") if load_teacher else None,
        config.get("exclude_slides"),
        config.get("extra_slides"),
        pair_index,
        config.get("epoch10_dir"),
        collage_size=int(config.get("collage_size", 1024)),
        local_root=config.get("local_dir"),
        collage_root=config.get("collage_root"),
        heat_targets=load_heat_targets(config.get("heat_targets")),
    )
    sampler = DistributedSampler(dataset, shuffle=True) if ctx.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=image_batch // 4,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(config["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=int(config["num_workers"]) > 0,
        collate_fn=collate_stage1,
    )
    if config["samples_per_epoch"] != "all":
        raise ValueError("stage 1 samples_per_epoch must be all")
    steps_per_epoch = len(loader)
    use_gan = _weight(config, "lambda_GAN") > 0
    use_nce = _weight(config, "lambda_NCE") > 0
    use_fdl = _weight(config, "lambda_FDL") > 0
    use_recon = _weight(config, "lambda_recon") > 0
    use_occupancy = _weight(config, "lambda_occupancy") > 0
    use_polarity = _weight(config, "lambda_polarity") > 0
    use_overlap = _weight(config, "lambda_overlap") > 0
    use_uni = _weight(config, "lambda_uni") > 0
    use_mb = _weight(config, "lambda_mb") > 0
    layers = [int(layer) for layer in config["nce_layers"]] if use_nce else []
    channels = [FEATURE_CHANNELS[layer] for layer in layers]

    generator = Generator(
        collage_size=int(config.get("collage_size", 1024)),
        local=bool(config.get("local_dir")),
    ).to(device)
    if config.get("wsi_collage"):
        _attach_wsi_encoder(generator)
        generator = generator.to(device)
    heat_net = None
    if _weight(config, "lambda_heat") > 0:
        checkpoint = config.get("malignancy_checkpoint")
        if not checkpoint:
            raise KeyError("lambda_heat requires malignancy_checkpoint")
        heat_net = HeatClassifier().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        heat_net.load_state_dict(payload["model"])
        heat_net.eval()
        for parameter in heat_net.parameters():
            parameter.requires_grad_(False)
    domain_nce = bool(config.get("nce_domain_adapter", False)) and use_nce
    if domain_nce:
        projector = DomainAdaptedPatchProjector(channels).to(device)
        nce_relation = DebiasedPatchNCESRC(
            tau_nce=float(config["nce_T"]),
            fn_thresh=float(config.get("nce_fn_thresh", 0.80)),
            fn_temp=float(config.get("nce_fn_temp", 0.05)),
            use_src=float(config.get("nce_src", 0.5)) > 0,
            tau_src=float(config.get("nce_src_T", 0.15)),
            lambda_src=float(config.get("nce_src", 0.5)),
        )
    else:
        projector = PatchProjector(channels).to(device) if use_nce else None
        nce_relation = None
    fdl = AmplitudeFDL(
        num_proj=int(config["fdl_num_proj"]),
        input_size=int(config["fdl_input_size"]),
        phase_weight=float(config["fdl_phase_weight"]),
        seed=int(config["seed"]),
    ).to(device) if use_fdl else None
    betas = (float(config.get("beta1", 0.5)), float(config.get("beta2", 0.999)))
    payload = None
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        load_generator_weights(generator, payload["generator"])
        if projector is not None:
            unwrap(projector).load_state_dict(payload["projector"])
    d_slide = bool(config.get("d_slide_cond", False)) and use_gan
    d_cls = bool(config.get("d_slide_cls", False)) and use_gan
    d_subtype = bool(config.get("d_subtype", False)) and use_gan
    if d_subtype and d_cls:
        raise RuntimeError("the subtype head and the slide-id head are separate")
    training_slides = sorted({path.name.split("__", 1)[0] for path, _, _, _ in dataset.samples})
    saved_slides = payload.get("slide_ids") if payload is not None else None
    if (d_slide or d_cls) and saved_slides:
        if set(saved_slides) != set(training_slides):
            raise RuntimeError("checkpoint slide ids do not match the training set")
        training_slides = list(saved_slides)
    slide_to_index = {slide: index for index, slide in enumerate(training_slides)}
    background_index = len(training_slides)
    discriminator = None
    grafted_film = False
    benign_only = bool(config.get("benign_patch_only", False))
    if use_gan:
        film_scales = config.get("d_slide_scales")
        discriminator = MultiScaleDiscriminator(
            num_scales=int(config["netD_num_scales"]),
            slide_count=(len(training_slides) + 1) if d_slide else 0,
            slide_classes=4 if d_subtype else (len(training_slides) if d_cls else 0),
            class_scales="coarsest" if d_subtype else "all",
            film_scales=None if film_scales is None else [int(scale) for scale in film_scales],
        ).to(device)
        if payload is not None:
            grafted_film = load_discriminator_state(unwrap(discriminator), payload["discriminator"])
    slide_head = None
    diagnosis = None
    subtype = None
    mb_weight = None
    subtype_weight = None
    if use_mb or d_subtype:
        label_path = config.get("slide_labels")
        if not label_path:
            raise KeyError("subtype and benign/malignant require slide_labels")
        slides = {path.name.split("__", 1)[0] for path, _, _, _ in dataset.samples}
        if d_subtype:
            subtype = _major_subtype_table(label_path)
            missing = sorted(slide for slide in slides if slide not in subtype)
            if missing:
                raise RuntimeError(f"{len(missing)} training slides have no subtype row, for example {missing[:3]}")
            counts = [0, 0, 0]
            for slide in slides:
                kind = subtype[slide]
                if kind is not None:
                    counts[kind - 1] += 1
            total = float(sum(counts))
            class_weight = [total / (3.0 * max(count, 1)) for count in counts]
            subtype_weight = torch.tensor(
                [sum(class_weight) / 3.0, *class_weight],
                dtype=torch.float32,
                device=device,
            )
            if use_mb:
                mb_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)
                slide_head = SlideDiagnosisHead(BOTTLENECK_DIM, n_classes=3).to(device)
        else:
            diagnosis = _diagnosis_table(label_path)
            missing = sorted(slide for slide in slides if slide not in diagnosis)
            if missing:
                raise RuntimeError(f"{len(missing)} training slides have no benign/malignant label, for example {missing[:3]}")
            counts = [0, 0]
            for slide in slides:
                counts[diagnosis[slide]] += 1
            total = float(sum(counts))
            mb_weight = torch.tensor(
                [total / (2.0 * max(count, 1)) for count in counts],
                dtype=torch.float32,
                device=device,
            )
            if use_mb:
                slide_head = SlideDiagnosisHead(BOTTLENECK_DIM).to(device)
        if slide_head is not None and payload is not None and payload.get("slide_head"):
            slide_head.load_state_dict(payload["slide_head"])
        if slide_head is not None and ctx.distributed:
            slide_head = DistributedDataParallel(slide_head, device_ids=[ctx.local_rank])
    if config.get("wsi_collage"):
        _freeze_wsi_encoder(generator)
    trainable = [parameter for parameter in generator.parameters() if parameter.requires_grad]
    generator_groups = [{"params": trainable, "lr": float(config["lr_G"])}]
    if isinstance(projector, DomainAdaptedPatchProjector):
        generator_groups.extend(projector.param_groups(float(config["lr_F"])))
    elif projector is not None:
        generator_groups.append({"params": projector.parameters(), "lr": float(config["lr_F"])})
    saved_generator_groups = (
        len(payload["optimizer_g"]["param_groups"]) if payload is not None else len(generator_groups)
    )
    # Epochs 1–3 saved the head weights but never put them in the optimizer.
    head_was_optimized = saved_generator_groups > len(generator_groups)
    if slide_head is not None and head_was_optimized:
        generator_groups.append({"params": unwrap(slide_head).parameters(), "lr": float(config["lr_G"])})
    opt_g = torch.optim.Adam(generator_groups, lr=float(config["lr_G"]), betas=betas)
    opt_d = None
    film_params: list[torch.nn.Parameter] = []
    if discriminator is not None:
        raw_d = unwrap(discriminator)
        film_params = film_parameters(raw_d)
        aux_params = film_params + class_parameters(raw_d)
        if aux_params:
            opt_d = torch.optim.Adam(
                [
                    {"params": backbone_parameters(raw_d), "lr": float(config["lr_D"])},
                    {"params": aux_params, "lr": float(config["lr_D"])},
                ],
                lr=float(config["lr_D"]),
                betas=betas,
            )
        else:
            opt_d = torch.optim.Adam(discriminator.parameters(), lr=float(config["lr_D"]), betas=betas)
    field_d = None
    opt_field = None
    if _weight(config, "lambda_field") > 0:
        field_slide_cond = bool(config.get("field_slide_cond", True))
        field_tissue_gate = bool(config.get("field_tissue_gate", True))
        slide_dim = None
        if field_slide_cond:
            slide_dim = int(config.get(
                "field_slide_dim", 512 if config.get("wsi_collage") else BOTTLENECK_DIM,
            ))
        field_d = ConchFieldDiscriminator(
            slide_dim=slide_dim,
            image_size=int(config.get("field_image_size", 448)),
        ).to(device)
        field_lr = float(config.get("field_lr", 5e-6))
        field_warmup = int(config.get("field_warmup_steps", 500))
        field_clip = float(config.get("field_grad_clip", 10.0))
        opt_field = torch.optim.AdamW(
            [parameter for parameter in field_d.parameters() if parameter.requires_grad],
            lr=field_lr,
            betas=(float(config.get("field_beta1", 0.9)), float(config.get("beta2", 0.999))),
            weight_decay=float(config.get("field_weight_decay", 1e-2)),
            eps=1e-8,
        )
        if payload is not None and payload.get("field_discriminator"):
            field_d.load_state_dict(payload["field_discriminator"])
            opt_field.load_state_dict(payload["optimizer_field"])
        if is_main(ctx):
            print(
                f"field critic CONCH {int(config.get('field_image_size', 448))} "
                f"AdamW lr={field_lr:g} warmup={field_warmup} clip={field_clip:g} "
                f"lambda={_weight(config, 'lambda_field'):g} "
                f"slide={field_slide_cond} tissue_gate={field_tissue_gate}",
                flush=True,
            )
    start_epoch = 0
    global_step = 0
    if payload is not None:
        opt_g.load_state_dict(payload["optimizer_g"])
        if opt_d is not None:
            saved_d = payload["optimizer_d"]
            if grafted_film and len(saved_d["param_groups"]) == 1:
                held = torch.optim.Adam(backbone_parameters(unwrap(discriminator)), lr=float(config["lr_D"]), betas=betas)
                held.load_state_dict(saved_d)
                for param in backbone_parameters(unwrap(discriminator)):
                    if param in held.state:
                        opt_d.state[param] = held.state[param]
            else:
                opt_d.load_state_dict(saved_d)
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        if slide_head is not None and not head_was_optimized:
            opt_g.add_param_group({"params": list(unwrap(slide_head).parameters()), "lr": float(config["lr_G"])})
            if is_main(ctx):
                print("slide head was not in the optimizer; it starts from this resume", flush=True)
    else:
        pretrain = Path(config["pretrain_checkpoint"])
        if not pretrain.is_file():
            raise FileNotFoundError(f"missing pretrain checkpoint {pretrain}")
        load_generator_only(pretrain, generator, device)
        if slide_head is not None:
            opt_g.add_param_group({"params": list(unwrap(slide_head).parameters()), "lr": float(config["lr_G"])})

    if bool(config.get("compile", False)):
        if device.type != "cuda":
            raise RuntimeError("compile requires CUDA")
        mode = str(config.get("compile_mode", "default"))
        if mode != "default":
            raise ValueError(f"compile mode {mode!r} is refused; use default")
        if is_main(ctx):
            print("torch.compile: generator, dynamic=False, mode=default")
        generator = torch.compile(generator, dynamic=False, mode=mode)

    if ctx.distributed:
        generator = DistributedDataParallel(generator, device_ids=[ctx.local_rank])
        if discriminator is not None:
            discriminator = DistributedDataParallel(discriminator, device_ids=[ctx.local_rank])
        if field_d is not None:
            field_d = DistributedDataParallel(field_d, device_ids=[ctx.local_rank])
        if projector is not None:
            projector = DistributedDataParallel(projector, device_ids=[ctx.local_rank])
    raw_g = unwrap(generator)
    teacher_on_disk = bool(config.get("teacher_dir")) and bool(config.get("load_teacher", True))
    if (use_occupancy or use_polarity) and not config.get("epoch10_dir"):
        raise RuntimeError("occupancy and polarity compare against the epoch-10 image")
    uni = None
    retrieval = None
    if use_uni:
        import timm

        uni = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
            num_classes=0,
        ).to(device).eval()
        for parameter in uni.parameters():
            parameter.requires_grad_(False)
        retrieval_dir = config.get("retrieval_dir")
        if not retrieval_dir:
            raise KeyError("lambda_uni requires retrieval_dir")
        retrieval = RetrievalTable(retrieval_dir)
        uni_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        uni_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    checkpoint_dir = Path(config["checkpoint_dir"])
    monitor = Monitor()
    if is_main(ctx):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        active = [
            name for name, enabled in (
                ("occ", use_occupancy),
                ("pol", use_polarity),
                ("recon", use_recon),
                ("fill", _weight(config, "lambda_fill") > 0),
                ("keep", _weight(config, "lambda_keep") > 0),
                ("overlap", use_overlap),
                ("gan", use_gan),
                ("nce", use_nce),
                ("uni", use_uni),
                ("mb", use_mb),
                ("fdl", use_fdl),
                ("heat", heat_net is not None),
                ("field", _weight(config, "lambda_field") > 0),
                ("dcls", d_cls),
                ("subtype", d_subtype),
            ) if enabled
        ]
        print(
            f"{config['name']}: pairs={len(dataset)} slides={dataset.slides} "
            f"steps/epoch={steps_per_epoch} batch/gpu={image_batch} world={ctx.world_size} "
            f"losses={','.join(active)}"
            + (f" d_slides={len(training_slides)}+bg" if d_slide else "")
            + (f" d_cls={len(training_slides)}" if d_cls else "")
            + (" subtype=benign-patches+bg" if d_subtype and bool(config.get("benign_patch_only", False)) else "")
            + (" subtype=LUAD/MIA/benign+bg" if d_subtype and not bool(config.get("benign_patch_only", False)) else "")
            + (
                f" collage_tokens={unwrap(generator).collage_encoder.tokens}"
                if hasattr(unwrap(generator).collage_encoder, "tokens")
                else " wsi_encoder=frozen"
            )
            + (
                f" local_tokens={unwrap(generator).local_encoder.tokens}"
                if unwrap(generator).local_encoder is not None else ""
            )
        )
        monitor.open(config)

    n_epochs = int(config["n_epochs"])
    n_epochs_decay = int(config["n_epochs_decay"])
    total_epochs = n_epochs + n_epochs_decay
    base_lrs = {"g": [group["lr"] for group in opt_g.param_groups]}
    if opt_d is not None:
        base_lrs["d"] = [group["lr"] for group in opt_d.param_groups]

    recon_full_epochs = int(config.get("recon_full_epochs", 0))

    def reconstruction_scale(epoch_index: int) -> float:
        if epoch_index < recon_full_epochs:
            return 1.0
        span = n_epochs - recon_full_epochs
        if span <= 0:
            return 0.0
        return max(0.0, (n_epochs - epoch_index - 1) / float(span))

    def set_learning_rate(epoch_index: int) -> None:
        if epoch_index < n_epochs:
            scale = 1.0
        else:
            scale = 1.0 - (epoch_index - n_epochs) / float(n_epochs_decay)
        for group, base in zip(opt_g.param_groups, base_lrs["g"]):
            group["lr"] = base * scale
        if opt_d is not None:
            for group, base in zip(opt_d.param_groups, base_lrs["d"]):
                group["lr"] = base * scale

    try:
      for epoch in range(start_epoch, total_epochs):
        set_learning_rate(epoch)
        weight_schedule = [float(value) for value in config.get("recon_weight_schedule", [])]
        if use_recon and weight_schedule:
            recon_weight = weight_schedule[min(epoch, len(weight_schedule) - 1)]
            sigma_schedule = [float(value) for value in config.get("recon_sigma_schedule", [])]
            recon_sigma = sigma_schedule[min(epoch, len(sigma_schedule) - 1)] if sigma_schedule else float(config.get("recon_sigma", 0.0))
            recon_scale = 1.0 if recon_weight > 0 else 0.0
        elif use_recon and int(config.get("recon_sharp_epochs", 0) or 0) > 0:
            recon_weight, recon_sigma = recon_epoch_settings(
                epoch,
                int(config["recon_sharp_epochs"]),
                _weight(config, "lambda_recon"),
                float(config["recon_sigma"]),
                _weight(config, "lambda_recon_late"),
                [float(value) for value in config["recon_sigma_schedule"]],
            )
            recon_scale = 1.0 if recon_weight > 0 else 0.0
        else:
            recon_scale = reconstruction_scale(epoch) if use_recon else 0.0
            recon_weight = recon_scale * _weight(config, "lambda_recon")
            recon_sigma = float(config.get("recon_sigma", 0.0))
        fdl_schedule = [float(value) for value in config.get("fdl_weight_schedule", [])]
        if use_fdl and fdl_schedule:
            fdl_weight = fdl_schedule[min(epoch, len(fdl_schedule) - 1)]
        elif use_fdl:
            fdl_weight = _weight(config, "lambda_FDL")
        else:
            fdl_weight = 0.0
        structure_on = recon_weight <= 0 and (use_occupancy or use_polarity)
        occ_weight = _weight(config, "lambda_occupancy") if structure_on and use_occupancy else 0.0
        pol_weight = _weight(config, "lambda_polarity") if structure_on and use_polarity else 0.0
        occ_sigma = float(config.get("occupancy_sigma", 16))
        pol_schedule = [float(value) for value in config.get("polarity_sigma_schedule", [])]
        if pol_schedule:
            start = int(config.get("recon_sharp_epochs", 0) or 0)
            offset = max(0, epoch - start)
            pol_sigma = pol_schedule[min(offset, len(pol_schedule) - 1)]
        else:
            pol_sigma = occ_sigma
        if is_main(ctx):
            occ_note = ""
            if use_occupancy or use_polarity:
                occ_note = (
                    f" occ={occ_weight:.3f}x{occ_sigma:g}"
                    f" pol={pol_weight:.3f}x{pol_sigma:g}"
                )
            film_note = ""
            if d_slide:
                film_note = f" d_film={unwrap(discriminator).film_alpha():.4f}"
            print(
                f"epoch {epoch + 1} recon={recon_weight:.3f} sigma={recon_sigma:g}"
                f"{f' fdl={fdl_weight:.3f}' if use_fdl else ''}{occ_note}{film_note}",
                flush=True,
            )
        if sampler is not None:
            sampler.set_epoch(epoch)
        generator.train()
        _eval_frozen_wsi(generator)
        if discriminator is not None:
            discriminator.train()
        if field_d is not None:
            field_d.train()
        if projector is not None:
            projector.train()
        epoch_start = time.time()
        for step_in_epoch, batch in enumerate(loader):
            if step_in_epoch >= steps_per_epoch:
                break
            af = batch["af"].to(device, non_blocking=True)
            pseudo = batch["pseudo"].to(device, non_blocking=True) if "pseudo" in batch else None
            he = batch["he"].to(device, non_blocking=True)
            collage = batch["collage"].to(device, non_blocking=True)
            local = batch["local"].to(device, non_blocking=True) if "local" in batch else None
            cells = batch["cell"].to(device, non_blocking=True) if "cell" in batch else None
            black = batch["black"].to(device, non_blocking=True) if "black" in batch else None
            if config.get("wsi_collage") and cells is None:
                raise RuntimeError("the WSI collage encoder needs a cell coordinate")
            if config.get("local_dir") and local is None:
                raise RuntimeError("local_dir is set but this batch has no neighborhood")
            batch_size = af.shape[0]
            need_teacher = teacher_on_disk and (
                recon_weight > 0 or (use_fdl and pseudo is None)
            )
            teacher_out = batch["teacher"].to(device, non_blocking=True) if need_teacher else None
            epoch10_out = (
                batch["epoch10"].to(device, non_blocking=True)
                if "epoch10" in batch and (
                    need_teacher or use_fdl or recon_weight > 0
                    or use_occupancy or use_polarity
                    or field_d is not None
                )
                else None
            )
            seam_targets = _seam_targets(raw_g, af, collage, device, local, cells, black) if use_overlap else None
            conditioning = {} if cells is None else {"cells": cells, "black_index": black}

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if use_mb:
                    fake, cond = generator(af, collage, local, return_cond=True, **conditioning)
                else:
                    fake = generator(af, collage, local, **conditioning)
                    cond = None
            fake_field = stitch_quadrants(fake)
            he_field = stitch_quadrants(he)
            slide_vec = None
            field_weight = None
            real_field_weight = None
            if field_d is not None:
                ones = torch.ones(he_field.shape[0], device=device)
                if field_slide_cond:
                    slide_vec = raw_g.collage_encoder.last_pooled
                    if slide_vec.shape[0] == he_field.shape[0] * 4:
                        slide_vec = slide_vec[::4]
                    if slide_vec.shape[0] != he_field.shape[0]:
                        raise RuntimeError(
                            f"{slide_vec.shape[0]} slide vectors for {he_field.shape[0]} fields"
                        )
                if field_tissue_gate:
                    # Real H&E is not registered to the generated field. The fake
                    # gate follows the epoch-10 image, which shares the AF grid.
                    if epoch10_out is None:
                        raise RuntimeError("the field critic gates the fake field with the epoch-10 image")
                    tissue_cutoff = float(config.get("d_slide_background", 0.15))
                    real_field_weight = (hed_tissue_fraction(he_field).detach() >= tissue_cutoff).float()
                    field_weight = (
                        hed_tissue_fraction(stitch_quadrants(epoch10_out)).detach() >= tissue_cutoff
                    ).float()
                else:
                    real_field_weight = ones
                    field_weight = ones
            slide_idx = None
            real_idx = None
            fake_idx = None
            if d_slide:
                slide_idx = torch.tensor(
                    [slide_to_index[slide] for slide in batch["slide"]],
                    dtype=torch.long,
                    device=device,
                )
                background = torch.full_like(slide_idx, background_index)
                cutoff = float(config.get("d_slide_background", 0.15))
                real_fraction = hed_tissue_fraction(he_field)
                if epoch10_out is not None:
                    fake_fraction = hed_tissue_fraction(stitch_quadrants(epoch10_out))
                elif pseudo is not None:
                    fake_fraction = hed_tissue_fraction(stitch_quadrants(pseudo))
                else:
                    fake_fraction = real_fraction
                fake_idx = torch.where(fake_fraction < cutoff, background, slide_idx)
                real_idx = torch.where(real_fraction < cutoff, background, slide_idx)
            loss_d = fake.new_zeros(())
            dgan_term = None
            dcls_term = None
            subtype_term = None
            if use_gan:
                opt_d.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    real_logits, real_classes = _disc_outputs(discriminator(he_field, real_idx))
                    fake_logits, _fake_classes = _disc_outputs(discriminator(fake_field.detach(), fake_idx))
                    dgan_term = 0.5 * (lsgan(real_logits, True) + lsgan(fake_logits, False))
                    loss_d = dgan_term
                    if d_cls:
                        cls_weight = _weight(config, "lambda_d_cls")
                        dcls_term = cls_weight * slide_classification_loss(
                            real_classes, _slide_targets(batch, slide_to_index, device),
                            _real_class_mask(he_field, float(config.get("d_slide_background", 0.15))),
                            cutoff=float(config.get("d_slide_background", 0.15)),
                        )
                        loss_d = loss_d + dcls_term
                    if d_subtype:
                        real_target, real_keep = _subtype_field_targets(
                            batch["slide"], subtype, hed_tissue_fraction(he_field),
                            float(config.get("d_slide_background", 0.15)), device,
                            benign_only=False,
                        )
                        subtype_term = _weight(config, "lambda_d_subtype") * field_subtype_loss(
                            real_classes, real_target, real_keep, subtype_weight,
                        )
                        loss_d = loss_d + subtype_term
                loss_d.backward()
                opt_d.step()
                del real_logits, real_classes, fake_logits, _fake_classes
            field_d_term = None
            if field_d is not None:
                taken = _optimizer_steps(opt_field)
                warmup_scale = 1.0 if field_warmup <= 0 else min(1.0, (taken + 1) / float(field_warmup))
                for group in opt_field.param_groups:
                    group["lr"] = field_lr * warmup_scale
                opt_field.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    real_field = field_d(he_field, slide_vec)
                    fake_field_score = field_d(fake_field.detach(), slide_vec)
                field_d_term = _weight(config, "lambda_field") * (
                    multilevel_bce(real_field, True, real_field_weight)
                    + multilevel_bce(fake_field_score, False, field_weight)
                )
                field_d_term.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in field_d.parameters() if parameter.requires_grad],
                    field_clip,
                )
                opt_field.step()

            opt_g.zero_grad(set_to_none=True)
            scaled = {}
            gate = None
            prob = None
            recon_keep = None
            if use_gan:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    fake_logits, fake_classes = _disc_outputs(discriminator(fake_field, fake_idx))
                    gan = lsgan(fake_logits, True)
                    # Slide identity is a real-H&E task. A zero generator weight
                    # must not build that loss, and stage 2 has no pseudo label.
                    if d_cls and _weight(config, "lambda_g_cls") > 0:
                        teacher_for_mask = (
                            stitch_quadrants(teacher_out) if teacher_out is not None else stitch_quadrants(pseudo)
                        )
                        scaled_cls = _weight(config, "lambda_g_cls") * slide_classification_loss(
                            fake_classes,
                            _slide_targets(batch, slide_to_index, device),
                            _generated_class_mask(
                                stitch_quadrants(af), stitch_quadrants(pseudo), teacher_for_mask,
                            ),
                            cutoff=float(config.get("d_slide_background", 0.15)),
                        )
                    else:
                        scaled_cls = None
                    if d_subtype:
                        if epoch10_out is not None:
                            subtype_fraction = hed_tissue_fraction(stitch_quadrants(epoch10_out))
                        else:
                            subtype_fraction = hed_tissue_fraction(fake_field.detach())
                        fake_target, fake_keep = _subtype_field_targets(
                            batch["slide"], subtype, subtype_fraction,
                            float(config.get("d_slide_background", 0.15)), device,
                            benign_only=bool(config.get("benign_patch_only", False)),
                        )
                        scaled["gsub"] = _weight(config, "lambda_g_subtype") * field_subtype_loss(
                            fake_classes, fake_target, fake_keep, subtype_weight,
                        )
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                scaled["gan"] = _weight(config, "lambda_GAN") * gan
                if scaled_cls is not None:
                    scaled["gcls"] = scaled_cls
                del fake_logits, fake_classes
            if field_d is not None:
                for parameter in field_d.parameters():
                    parameter.requires_grad_(False)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    generated_field = field_d(fake_field, slide_vec)
                for parameter in field_d.parameters():
                    parameter.requires_grad_(True)
                scaled["field"] = _weight(config, "lambda_field") * multilevel_bce(
                    generated_field, True, field_weight,
                )
            fake_feats = None
            af_feats = None
            nce_neg_weight = None
            if use_nce:
                # One 1024 field, not four 512 crops. The query carries grad
                # through the stitch. The AF key is detached, as in CUT.
                # Encode on the unwrapped module. Generate already ran those
                # parameters through DDP, so this backward adds into the same
                # grads and the hook allreduces the sum. A second wrapped
                # forward would mark the decoder unused and hang the reducer.
                n_fields = fake_field.shape[0]
                collage_field = _one_per_field(collage, n_fields)
                cells_field = _one_per_field(cells, n_fields)
                black_field = _one_per_field(black, n_fields)
                af_field = stitch_quadrants(af)

                def _encode_field(image):
                    # Keep only the PatchNCE stages. The other encoder maps are
                    # not inputs to the loss, so checkpoint does not store them.
                    return _maps(
                        raw_g.encoder_features(image, collage_field, cells_field, black_field),
                        layers,
                    )

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    fake_feats = torch.utils.checkpoint.checkpoint(
                        _encode_field, fake_field, use_reentrant=False,
                    )
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    af_feats = _encode_field(af_field)
                if domain_nce:
                    raw_he, raw_af = _sample_paired_patches(
                        fake_feats, af_feats, int(config["num_patches"]),
                    )
                    projected_he = projector(raw_he, "he")
                    projected_af = projector(raw_af, "af")
                    nce_terms = []
                    src_terms = []
                    weight_terms = []
                    for query, key, raw_key in zip(projected_he, projected_af, raw_af):
                        nce_term, src_term, neg_weight = nce_relation(query, key, raw_key)
                        nce_terms.append(nce_term)
                        src_terms.append(src_term)
                        weight_terms.append(neg_weight)
                    nce = torch.stack(nce_terms).mean()
                    src = torch.stack(src_terms).mean()
                    nce_neg_weight = float(torch.stack(weight_terms).mean())
                    contrast_weight = _weight(config, "lambda_NCE")
                    scaled["nce"] = contrast_weight * nce
                    scaled["src"] = contrast_weight * src
                    del raw_he, raw_af, projected_he, projected_af
                else:
                    query, source_key = projector(
                        [fake_feats, af_feats],
                        int(config["num_patches"]),
                    )
                    nce = _contrast(
                        query, source_key,
                        # One bank per 1024 field. The current batch is one field, so this is 1.
                        lambda left, right: patch_nce(
                            left, right, fake_field.shape[0], float(config["nce_T"]),
                        ),
                    )
                    nce_neg_weight = None
                    scaled["nce"] = _weight(config, "lambda_NCE") * nce
                    del query, source_key
                del fake_feats, af_feats, af_field
            if use_fdl:
                if epoch10_out is not None:
                    aligned = stitch_quadrants(epoch10_out)
                elif pseudo is not None:
                    aligned = stitch_quadrants(pseudo)
                elif teacher_out is not None:
                    aligned = stitch_quadrants(teacher_out)
                else:
                    raise RuntimeError("FDL gate needs an aligned stain image")
                gate = content_gate(aligned, he_field)
                frequency = (fdl(fake_field, he_field) * gate).mean()
                scaled["fdl"] = fdl_weight * frequency
            if seam_targets is not None:
                overlap = fake.new_zeros(())
                for window, target in zip(seam_windows(fake_field), seam_targets):
                    overlap = overlap + (window.float() - target.detach().float()).abs().mean()
                scaled["overlap"] = _weight(config, "lambda_overlap") * overlap / len(seam_targets)
            if recon_weight > 0:
                if epoch10_out is None:
                    raise RuntimeError("reconstruction needs the epoch-10 image")
                epoch10_field = stitch_quadrants(epoch10_out)
                if teacher_out is not None:
                    teacher_field = stitch_quadrants(teacher_out)
                    prob = hed_tissue_prob(epoch10_field)
                    glass_field = epoch10_field
                else:
                    teacher_field = epoch10_field
                    glass_field = epoch10_field
                    prob = hed_tissue_prob(epoch10_field)
                recon_field_weight = None
                cutoff = float(config.get("d_slide_background", 0.15))
                if (
                    not bool(config.get("recon_hed"))
                    and bool(config.get("benign_patch_only", False))
                    and heat_net is not None
                    and "heat_known" in batch
                ):
                    known = batch["heat_known"].to(device, non_blocking=True)
                    from_map = batch["heat_map"].to(device, non_blocking=True)
                    benign = known & ~from_map
                    recon_field_weight = torch.ones(teacher_field.shape[0], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                            looks_malignant = heat_net(teacher_field.detach()).float() > 0
                    tissue = hed_tissue_fraction(teacher_field.detach()) >= cutoff
                    recon_field_weight = recon_field_weight.masked_fill(benign & tissue & looks_malignant, 0.0)
                    recon_keep = float(recon_field_weight.mean())
                elif (
                    not bool(config.get("recon_hed"))
                    and bool(config.get("benign_patch_only", False))
                    and subtype is not None
                    and discriminator is not None
                ):
                    recon_field_weight = torch.ones(teacher_field.shape[0], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        _teacher_logits, teacher_classes = _disc_outputs(discriminator(teacher_field.detach()))
                    if teacher_classes:
                        predicted = teacher_classes[-1].float().mean(dim=(2, 3)).argmax(dim=1)
                        tissue = hed_tissue_fraction(teacher_field.detach())
                        cutoff = float(config.get("d_slide_background", 0.15))
                        for index, slide in enumerate(batch["slide"]):
                            # A benign slide's tissue is copied only when the head calls the teacher benign.
                            if subtype.get(slide) == 1 and float(tissue[index]) >= cutoff and int(predicted[index]) != 1:
                                recon_field_weight[index] = 0.0
                    recon_keep = float(recon_field_weight.mean())
                if bool(config.get("recon_hed")):
                    gain = float(config.get("recon_bright_gain", 1.0))
                    bright_weight = None
                    if gain > 1.0:
                        bright_weight = recon_bright_weight(
                            stitch_quadrants(af),
                            epoch10_field,
                            lo=float(config.get("recon_bright_lo", 0.30)),
                            hi=float(config.get("recon_bright_hi", 0.55)),
                            gain=gain,
                        )
                    scaled["recon"] = recon_weight * hed_recon(
                        fake_field, epoch10_field, recon_sigma,
                        tissue_only=bool(config.get("recon_tissue_only", False)),
                        eosin_only=bool(config.get("recon_eosin_only", False)),
                        weight=bright_weight,
                    )
                elif bool(config.get("recon_rgb")):
                    tissue_mask = hed_tissue_mask(epoch10_field)
                    scaled["recon"] = recon_weight * rgb_tissue_recon(
                        fake_field, epoch10_field, tissue_mask, recon_sigma,
                        field_weight=recon_field_weight,
                    )
                else:
                    bright = af_brightness(
                        stitch_quadrants(af),
                        lo=float(config.get("recon_bright_lo", 0.20)),
                        hi=float(config.get("recon_bright_hi", 0.30)),
                    )
                    scaled["recon"] = recon_weight * recon_loss(
                        fake_field, teacher_field, glass_field, prob, recon_sigma,
                        bright=bright,
                        bright_gain=float(config.get("recon_bright_gain", 1.0)),
                        field_weight=recon_field_weight,
                    )
            keep_weight = _weight(config, "lambda_keep")
            if keep_weight > 0:
                scaled["keep"] = keep_weight * bright_af_keep(fake_field, stitch_quadrants(af))
            fill_weight = _weight(config, "lambda_fill")
            if fill_weight > 0:
                if epoch10_out is None:
                    raise RuntimeError("bright fill needs the epoch-10 image")
                scaled["fill"] = fill_weight * bright_neighbor_fill(
                    fake_field, stitch_quadrants(epoch10_out), stitch_quadrants(af),
                )
            if occ_weight > 0 or pol_weight > 0:
                if epoch10_out is None:
                    raise RuntimeError("occupancy and polarity need the epoch-10 image")
                epoch10_field = stitch_quadrants(epoch10_out)
                occupancy, polarity = structure_loss(
                    fake_field, epoch10_field, occ_sigma, polarity_sigma=pol_sigma,
                )
                if occ_weight > 0:
                    scaled["occ"] = occ_weight * occupancy
                if pol_weight > 0:
                    scaled["pol"] = pol_weight * polarity
            if use_mb:
                n_fields = len(batch["slide"])
                token = cond.float().reshape(n_fields, -1, cond.shape[-1]).mean(dim=1)
                logits = slide_head(token).float()
                if subtype is not None:
                    labels = []
                    keep = []
                    for slide in batch["slide"]:
                        kind = subtype[slide]
                        labels.append(0 if kind is None else int(kind) - 1)
                        keep.append(kind is not None)
                    targets = torch.tensor(labels, dtype=torch.long, device=device)
                    include = torch.tensor(keep, dtype=torch.bool, device=device)
                    mb = logits.sum() * 0.0
                    if bool(include.any()):
                        mb = mb + torch.nn.functional.cross_entropy(
                            logits[include], targets[include], weight=mb_weight,
                        )
                    mb_acc = float((logits.argmax(dim=1)[include] == targets[include]).float().mean()) if bool(include.any()) else 0.0
                else:
                    targets = torch.tensor(
                        [diagnosis[slide] for slide in batch["slide"]],
                        dtype=torch.long,
                        device=device,
                    )
                    mb = torch.nn.functional.cross_entropy(logits, targets, weight=mb_weight)
                    mb_acc = float((logits.argmax(dim=1) == targets).float().mean())
                scaled["mb"] = _weight(config, "lambda_mb") * mb
            else:
                mb_acc = None
            uni_w = None
            if use_uni:
                oriented = _match_table_orientation(fake_field, batch["flip"])
                query = _uni_query(uni, oriented, uni_mean, uni_std)
                positive, negative = retrieval.keys(batch["name"], device)
                pair_weight = retrieval_pair_weight(
                    retrieval.pair_cosine(batch["name"], device),
                    float(config.get("uni_cosine_lo", 0.3)),
                    float(config.get("uni_cosine_hi", 0.6)),
                )
                uni_w = float(pair_weight.mean())
                scaled["uni"] = _weight(config, "lambda_uni") * field_info_nce(
                    query, positive, negative, float(config.get("uni_T", 0.07)), pair_weight,
                )
            if heat_net is not None:
                if "heat_known" not in batch:
                    raise RuntimeError("the class loss needs the benign-patch table")
                known = batch["heat_known"].to(device, non_blocking=True)
                from_map = batch["heat_map"].to(device, non_blocking=True)
                # Benign slides are the negative class. A heatmap tile is not a
                # target, including a heatmap on a slide labeled benign.
                benign = known & ~from_map
                tissue_ok = hed_tissue_fraction(he_field) >= 0.15
                weight = (benign & tissue_ok).float()
                if bool(weight.any()):
                    def _class_logit(image: torch.Tensor) -> torch.Tensor:
                        return heat_net(image)

                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        logit = torch.utils.checkpoint.checkpoint(
                            _class_logit, fake_field, use_reentrant=False,
                        )
                    soft = torch.nn.functional.binary_cross_entropy_with_logits(
                        logit.float(), torch.zeros_like(logit.float()), reduction="none",
                    )
                    scaled["heat"] = _weight(config, "lambda_heat") * (soft * weight).sum() / weight.sum().clamp_min(1.0)
                else:
                    scaled["heat"] = fake_field.sum() * 0.0
                heat_keep = float(weight.mean())
            else:
                heat_keep = None
            loss_g = sum(scaled.values())
            loss_g.backward()
            opt_g.step()
            global_step += 1
            record = None
            if is_main(ctx) and global_step % int(config["log_every"]) == 0:
                shown = {name: float(value) for name, value in scaled.items()}
                extras = ""
                record = {"loss/g": float(loss_g), "loss/d": float(loss_d)}
                if gate is not None:
                    extras += f" fdl_gate={float(gate.mean()):.3f}"
                    record["loss/fdl_gate"] = float(gate.mean())
                if prob is not None:
                    extras += f" tissue={float(prob.mean()):.3f}"
                    record["loss/tissue"] = float(prob.mean())
                if recon_keep is not None:
                    extras += f" recon_keep={recon_keep:.3f}"
                    record["loss/recon_keep"] = recon_keep
                if heat_keep is not None:
                    extras += f" heat_w={heat_keep:.3f}"
                    record["loss/heat_w"] = heat_keep
                if nce_neg_weight is not None:
                    extras += f" nce_w={nce_neg_weight:.3f}"
                    record["loss/nce_w"] = nce_neg_weight
                if field_weight is not None:
                    extras += f" field_w={float(field_weight.mean()):.3f}"
                    record["loss/field_w"] = float(field_weight.mean())
                if field_d_term is not None:
                    extras += f" field_d={float(field_d_term):.4f}"
                    record["loss/field_d"] = float(field_d_term)
                if mb_acc is not None:
                    extras += f" mb_acc={mb_acc:.3f}"
                    record["loss/mb_acc"] = mb_acc
                if uni_w is not None:
                    extras += f" uni_w={uni_w:.3f}"
                    record["loss/uni_w"] = uni_w
                if dgan_term is not None:
                    record["loss/dgan"] = float(dgan_term)
                if dcls_term is not None:
                    extras += f" dcls={float(dcls_term):.4f}"
                    record["loss/dcls"] = float(dcls_term)
                if subtype_term is not None:
                    extras += f" dsub={float(subtype_term):.4f}"
                    record["loss/dsub"] = float(subtype_term)
                if d_slide:
                    extras += f" d_film={unwrap(discriminator).film_alpha():.4f}"
                    record["loss/d_film"] = unwrap(discriminator).film_alpha()
                    if real_idx is not None:
                        he_bg = float((real_idx == background_index).float().mean())
                        af_bg = float((fake_idx == background_index).float().mean())
                        extras += f" d_bg_he={he_bg:.3f} d_bg_af={af_bg:.3f}"
                        record["loss/d_background_he"] = he_bg
                        record["loss/d_background_af"] = af_bg
                print(
                    f"epoch {epoch + 1} step {global_step} "
                    f"g={float(loss_g):.4f} d={float(loss_d):.4f} "
                    + (f"dgan={float(dgan_term):.4f} " if dgan_term is not None else "")
                    + " ".join(f"{name}={value:.4f}" for name, value in shown.items())
                    + extras
                )
                record.update({f"loss/{name}": value for name, value in shown.items()})
            if is_main(ctx) and global_step % int(config["image_every"]) == 0:
                # The reconstruction target is the old teacher when it is loaded,
                # and the epoch-10 image when that is the only stain target.
                preview_teacher = teacher_out if teacher_out is not None else epoch10_out
                grid = save_preview(
                    checkpoint_dir / "preview", global_step, af, pseudo, fake, he, preview_teacher,
                )
                if record is None:
                    record = {}
                parts = ["input", "output"]
                if preview_teacher is not None:
                    parts.append("teacher")
                if pseudo is not None:
                    parts.append("pseudo")
                parts.append("real H&E")
                caption = " | ".join(parts)
                preview = monitor.image(grid, caption)
                if preview is not None:
                    record["preview"] = preview
            if record and is_main(ctx):
                monitor.log(record, global_step)

        if is_main(ctx):
            saved_ids = training_slides if (d_slide or d_cls) else None
            save_state(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch + 1, global_step, generator, discriminator, projector, opt_g, opt_d, slide_head,
                saved_ids, field_d, opt_field,
            )
            save_state(
                checkpoint_dir / "last.pt",
                epoch + 1, global_step, generator, discriminator, projector, opt_g, opt_d, slide_head,
                saved_ids, field_d, opt_field,
            )
            print(f"epoch {epoch + 1} done in {time.time() - epoch_start:.1f}s, step {global_step}")
            monitor.log(
                {"epoch_seconds": time.time() - epoch_start, "epoch": epoch + 1},
                global_step,
            )
    finally:
        if is_main(ctx):
            monitor.finish()
        cleanup(ctx)


if __name__ == "__main__":
    main()

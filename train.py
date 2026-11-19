"""Masked reconstruction pretraining.

Single GPU:
    python train.py --config configs/pretrain.yaml

Four GPUs:
    torchrun --nproc_per_node=4 train.py --config configs/pretrain.yaml

Resume:
    torchrun --nproc_per_node=4 train.py --config configs/pretrain.yaml --resume checkpoints/pretrain/last.pt
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.dataset import PretrainDataset, collate_quadrants
from data.mask import mask_patches
from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.distributed import cleanup, is_main, setup
from engine.image_losses import lsgan
from engine.losses import reconstruction_loss
from engine.monitor import Monitor
from models.generator import BOTTLENECK_DIM, Generator
from models.pretrain_disc import PretrainDiscriminator
from models.wsi_collage_encoder import WSICollageAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = (
        "name", "data_roots", "collage_index", "batch_size", "num_workers", "epochs", "lr",
        "weight_decay", "samples_per_epoch", "mask_patch", "mask_fraction",
        "checkpoint_dir", "seed", "log_every", "image_every",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    return config


LABELS = Path("/mnt/10tb_hdd/taek/stain_translation_core/manifests/fresh/train_slide_labels.csv")
SUB_NAMES = ("Benign", "LUAD", "MIA", "LUSC", "SCLC", "Metastatic", "Other")


def slide_labels() -> dict[str, tuple[int, int]]:
    found = {}
    with LABELS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            classes = (row.get("classes1") or "").strip()
            if classes not in ("Benign", "Malignant"):
                continue
            subtype = (row.get("subtype_norm") or "").strip()
            if classes == "Benign":
                subtype = "Benign"
            elif subtype == "METASTATIC":
                subtype = "Metastatic"
            elif subtype == "OTHER":
                subtype = "Other"
            if subtype not in SUB_NAMES:
                continue
            found[row["slide_id"].strip()] = (0 if classes == "Benign" else 1, SUB_NAMES.index(subtype))
    return found


def subtype_weights(pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Inverse of each subtype's share inside a half-benign batch."""
    malignant = max(sum(binary == 1 for binary, _subtype in pairs), 1)
    counts = [sum(subtype == index for _binary, subtype in pairs) for index in range(len(SUB_NAMES))]
    share = []
    for index, count in enumerate(counts):
        if index == 0:
            share.append(0.5 if count else 0.0)
        else:
            share.append(0.5 * count / malignant if count else 0.0)
    present = [1.0 / value for value in share if value > 0]
    cap = 8.0 * min(present)
    weight = [min(cap, 1.0 / value) if value > 0 else 0.0 for value in share]
    tensor = torch.tensor(weight, dtype=torch.float32)
    return tensor / tensor.sum().clamp(min=1e-8) * len(SUB_NAMES)


def excluded_specimens(config: dict) -> set[str]:
    path = config.get("exclude_slides")
    needles = tuple(str(item).lower() for item in config.get("exclude_reasons", []))
    if not path or not needles:
        return set()
    found = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        slide, _, reason = raw.partition("#")
        if any(needle in reason.strip().lower() for needle in needles):
            found.add(slide.strip())
    return found


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_preview(directory: Path, step: int, tile, masked, recon) -> np.ndarray:
    directory.mkdir(parents=True, exist_ok=True)

    def panel(batch: torch.Tensor) -> np.ndarray:
        shown = batch[:4]
        image = (shown.detach().float().clamp(-1, 1) + 1.0) * 127.5
        image = image.to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        return np.concatenate(list(image), axis=0)

    grid = np.concatenate([panel(tile), panel(masked), panel(recon)], axis=1)
    Image.fromarray(grid, mode="RGB").save(directory / f"step_{step:07d}.png")
    return grid


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
    file_batch = image_batch // 4
    labels = slide_labels()
    dataset = PretrainDataset(
        list(config["data_roots"]),
        config["collage_index"],
        exclude=excluded_specimens(config),
        collage_root=config.get("collage_root"),
        labels=labels,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if ctx.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=file_batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(config["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=int(config["num_workers"]) > 0,
        collate_fn=collate_quadrants,
        worker_init_fn=lambda worker: random.seed(torch.initial_seed() % (2**32) + worker),
    )
    raw_samples = config["samples_per_epoch"]
    if raw_samples == "all":
        steps_per_epoch = len(loader)
    else:
        steps_per_epoch = int(raw_samples) // (int(config["batch_size"]) * ctx.world_size)
        if steps_per_epoch < 1:
            raise ValueError(
                f"samples_per_epoch={raw_samples} is smaller than "
                f"batch {config['batch_size']} x world {ctx.world_size}"
            )
        if steps_per_epoch > len(loader):
            steps_per_epoch = len(loader)

    model = Generator().to(device)
    encoder_epoch = None
    if config.get("wsi_encoder"):
        adapter = WSICollageAdapter(len(SUB_NAMES), BOTTLENECK_DIM)
        encoder_epoch = adapter.load_pretrained(config["wsi_encoder"])
        model.collage_encoder = adapter.to(device)
        model.collage_size = int(adapter.encoder.size)
    sub_weight = subtype_weights(dataset.pairs).to(device) if dataset.pairs else None
    resumed_epoch = 0
    if args.resume:
        peeked = torch.load(args.resume, map_location="cpu", weights_only=False)
        resumed_epoch = int(peeked["epoch"])
    gan_phase = bool(config.get("gan_after_epoch1", False)) and resumed_epoch >= 1
    if gan_phase and config.get("wsi_encoder"):
        model.collage_encoder.encoder.eval()
        for parameter in model.collage_encoder.encoder.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["lr"]),
        betas=(0.9, 0.99),
        weight_decay=float(config["weight_decay"]),
    )
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(
            Path(args.resume), model=model, optimizer=None if gan_phase else optimizer, device=device,
        )

    use_adversary = float(config.get("lambda_GAN", 0.0)) > 0
    if bool(config.get("compile", False)) and not (gan_phase and use_adversary):
        if device.type != "cuda":
            raise RuntimeError("compile requires CUDA")
        mode = str(config.get("compile_mode", "default"))
        if mode != "default":
            raise ValueError(f"compile mode {mode!r} is refused; use default")
        if is_main(ctx):
            print("torch.compile: generator, dynamic=False, mode=default")
        # Compiled DDP bucketing rewrites the 1x1 RGB head gradient layout.
        torch._dynamo.config.optimize_ddp = False
        model = torch.compile(model, dynamic=False, mode=mode)
    if ctx.distributed:
        model = DistributedDataParallel(model, device_ids=[ctx.local_rank])

    def _contiguous_grad(grad: torch.Tensor) -> torch.Tensor:
        return grad if grad.is_contiguous() else grad.contiguous()

    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.register_hook(_contiguous_grad)

    discriminator = None
    disc_optimizer = None
    lpips_net = None
    if gan_phase and use_adversary:
        discriminator = PretrainDiscriminator().to(device)
        if ctx.distributed:
            discriminator = DistributedDataParallel(discriminator, device_ids=[ctx.local_rank])
        disc_optimizer = torch.optim.Adam(
            [parameter for parameter in discriminator.parameters() if parameter.requires_grad],
            lr=float(config.get("disc_lr", 2e-4)),
            betas=(0.5, 0.9),
        )
        if float(config.get("perceptual_weight", 0.0)) > 0:
            import lpips

            lpips_net = lpips.LPIPS(net="vgg").to(device)
            lpips_net.eval()
            for parameter in lpips_net.parameters():
                parameter.requires_grad_(False)
        if is_main(ctx):
            print(
                "gan phase: encoder frozen, compile off, "
                f"LPIPS weight {float(config.get('perceptual_weight', 0.0)):.1f}, "
                f"discriminator from epoch {int(config.get('disc_upd_start', 1)) + 1}, "
                f"generator adversarial loss from epoch {int(config.get('disc_start', 1)) + 1}"
            )
    elif gan_phase and is_main(ctx):
        print("encoder frozen, L1 only, generator compiled")

    checkpoint_dir = Path(config["checkpoint_dir"])
    preview_dir = checkpoint_dir / "preview"
    monitor = Monitor()
    if is_main(ctx):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        count = sum(parameter.numel() for parameter in model.parameters())
        raw = model.module if ctx.distributed else model
        collage_count = sum(parameter.numel() for parameter in raw.collage_encoder.parameters())
        print(
            f"{config['name']}: patches={len(dataset)} excluded_files={dataset.excluded} "
            f"steps/epoch={steps_per_epoch} "
            f"batch/gpu={config['batch_size']} files/gpu={file_batch} world={ctx.world_size} "
            f"params={count:,} collage={collage_count:,} "
            f"counts={dataset.counts} slides={len(dataset.pairs)} "
            f"unlabeled_files={dataset.unlabeled} wsi_epoch={encoder_epoch}"
        )
        monitor.open(config)

    try:
        _train(
            ctx, config, loader, sampler, model, optimizer, device, start_epoch, global_step,
            steps_per_epoch, checkpoint_dir, preview_dir, monitor, sub_weight,
            discriminator, disc_optimizer, lpips_net,
        )
    finally:
        if is_main(ctx):
            monitor.finish()
        cleanup(ctx)


def _mark_frozen_encoder(model) -> None:
    raw = model.module if hasattr(model, "module") else model
    encoder = getattr(raw, "collage_encoder", None)
    if encoder is not None and hasattr(encoder, "encoder"):
        frozen = all(not parameter.requires_grad for parameter in encoder.encoder.parameters())
        if frozen:
            encoder.encoder.eval()


def _discriminator_step(ctx, model, discriminator, disc_optimizer, tile, collage, cells, black, config) -> float:
    """LSGAN update, same targets as the staining discriminator: real toward 1, fake toward 0."""
    del config
    model.eval()
    _mark_frozen_encoder(model)
    discriminator.train()
    disc_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        recon, _aux = model(tile, collage, cells=cells, black_index=black, return_aux=True)
    fake_logits = discriminator(recon.float())
    real_logits = discriminator(tile.float())
    disc_loss = 0.5 * (lsgan(real_logits, True) + lsgan(fake_logits, False))
    disc_loss.backward()
    disc_optimizer.step()
    model.train()
    _mark_frozen_encoder(model)
    return float(disc_loss.detach())


def _train(ctx, config, loader, sampler, model, optimizer, device, start_epoch, global_step,
           steps_per_epoch, checkpoint_dir, preview_dir, monitor, sub_weight,
           discriminator, disc_optimizer, lpips_net) -> None:
    for epoch in range(start_epoch, int(config["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        _mark_frozen_encoder(model)
        epoch_start = time.time()
        for step_in_epoch, batch in enumerate(loader):
            if step_in_epoch >= steps_per_epoch:
                break
            tile = batch["tile"].to(device, non_blocking=True)
            collage = batch["collage"].to(device, non_blocking=True)
            cells = batch["cell"].to(device, non_blocking=True) if "cell" in batch else None
            black = batch["black"].to(device, non_blocking=True) if "black" in batch else None
            masked = mask_patches(
                tile,
                patch=int(config["mask_patch"]),
                fraction=float(config["mask_fraction"]),
                fill=-1.0,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                recon, aux = model(
                    masked, collage, cells=cells, black_index=black, return_aux=True,
                )
            loss, parts = reconstruction_loss(recon.float(), tile.float())
            if aux is not None:
                binary, subtype = aux
                class_loss = F.cross_entropy(binary.float(), batch["bin"][::4].to(device))
                subtype_loss = F.cross_entropy(
                    subtype.float(), batch["sub"][::4].to(device), weight=sub_weight,
                )
                parts["cls"] = float(class_loss.detach())
                parts["sub"] = float(subtype_loss.detach())
                if binary.requires_grad:
                    loss = (
                        loss
                        + float(config.get("lambda_class", 1.0)) * class_loss
                        + float(config.get("lambda_subtype", 0.5)) * subtype_loss
                    )
            use_lpips = lpips_net is not None and epoch >= int(config.get("lpips_start", 1))
            train_disc = discriminator is not None and epoch >= int(config.get("disc_upd_start", 1))
            use_gan = discriminator is not None and epoch >= int(config.get("disc_start", 1))
            if use_lpips:
                perceptual = lpips_net(recon.float(), tile.float()).mean()
                loss = loss + float(config.get("perceptual_weight", 1.0)) * perceptual
                parts["lpips"] = float(perceptual.detach())
            if use_gan:
                discriminator.eval()
                gan_loss = lsgan(discriminator(recon.float()), True)
                loss = loss + float(config.get("lambda_GAN", 1.0)) * gan_loss
                parts["gan"] = float(gan_loss.detach())
            loss.backward()
            optimizer.step()
            if train_disc:
                parts["d"] = _discriminator_step(
                    ctx, model, discriminator, disc_optimizer, tile, collage, cells, black, config,
                )
            global_step += 1
            record = None
            if is_main(ctx) and global_step % int(config["log_every"]) == 0:
                message = (
                    f"epoch {epoch + 1} step {global_step} "
                    f"loss={float(loss.detach()):.4f} l1={parts['l1']:.4f}"
                )
                record = {"loss": float(loss.detach()), "l1": parts["l1"]}
                if "cls" in parts:
                    message += f" cls={parts['cls']:.4f} sub={parts['sub']:.4f}"
                    record["cls"] = parts["cls"]
                    record["sub"] = parts["sub"]
                if "lpips" in parts:
                    message += f" lpips={parts['lpips']:.4f}"
                    record["lpips"] = parts["lpips"]
                if "gan" in parts:
                    message += f" gan={parts['gan']:.4f}"
                    record["gan"] = parts["gan"]
                if "d" in parts:
                    message += f" d={parts['d']:.4f}"
                    record["d"] = parts["d"]
                print(message)
            if is_main(ctx) and global_step % int(config["image_every"]) == 0:
                grid = save_preview(preview_dir, global_step, tile, masked, recon)
                if record is None:
                    record = {}
                preview = monitor.image(grid, "original | masked | recon")
                if preview is not None:
                    record["preview"] = preview
            if record:
                monitor.log(record, step=global_step)

        if is_main(ctx):
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch=epoch + 1,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                pretrain_discriminator=discriminator,
                pretrain_disc_optimizer=disc_optimizer,
            )
            save_checkpoint(
                checkpoint_dir / "last.pt",
                epoch=epoch + 1,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                pretrain_discriminator=discriminator,
                pretrain_disc_optimizer=disc_optimizer,
            )
            elapsed = time.time() - epoch_start
            print(f"epoch {epoch + 1} done in {elapsed:.1f}s, step {global_step}")
            monitor.log({"epoch_seconds": elapsed, "epoch": epoch + 1}, step=global_step)


if __name__ == "__main__":
    main()

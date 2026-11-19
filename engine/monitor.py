"""Training charts and preview images, stored on this machine.

The dashboard is local. Nothing is uploaded. From the training environment:

    swanlab watch -h 0.0.0.0 -p 5092 /mnt/10tb_hdd/taek/af2he/swanlog

Then open http://<this machine>:5092
"""

from __future__ import annotations


class Monitor:
    def __init__(self) -> None:
        self._backend: str | None = None
        self._swanlab = None
        self._wandb = None

    def open(self, config: dict) -> None:
        kind = str(config.get("monitor", "") or "").strip().lower()
        if kind in ("local", "swanlab"):
            import swanlab

            log_dir = str(config.get("monitor_dir", "/mnt/10tb_hdd/taek/af2he/swanlog"))
            swanlab.init(
                project=str(config.get("wandb_project", "af2he")),
                name=str(config["name"]),
                config=config,
                log_dir=log_dir,
                mode="local",
            )
            self._backend = "swanlab"
            self._swanlab = swanlab
            print(f"monitor local dir={log_dir}", flush=True)
            print(f"monitor view: swanlab watch -h 0.0.0.0 -p 5092 {log_dir}", flush=True)
            return
        if bool(config.get("wandb", False)):
            import wandb

            wandb.init(
                project=str(config.get("wandb_project", "af2he")),
                name=str(config["name"]),
                config=config,
            )
            self._backend = "wandb"
            self._wandb = wandb

    def image(self, array, caption: str):
        if self._backend == "swanlab":
            return self._swanlab.Image(array, caption=caption)
        if self._backend == "wandb":
            return self._wandb.Image(array, caption=caption)
        return None

    def log(self, data: dict, step: int) -> None:
        if self._backend is None or not data:
            return
        if self._backend == "swanlab":
            self._swanlab.log(data, step=step)
            return
        self._wandb.log(data, step=step)

    def finish(self) -> None:
        if self._backend == "swanlab":
            self._swanlab.finish()
        elif self._backend == "wandb" and self._wandb.run is not None:
            self._wandb.finish()
        self._backend = None

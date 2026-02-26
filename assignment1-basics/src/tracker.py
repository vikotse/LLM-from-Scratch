import wandb


class ExperimentTracker:
    def __init__(self, cfg):
        self.enable = cfg.wandb.enable
        if cfg.wandb.enable:
            import wandb
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.name,
                config=cfg,
            )

    def log(self, log_dict: dict):
        if self.enable:
            wandb.log(log_dict)

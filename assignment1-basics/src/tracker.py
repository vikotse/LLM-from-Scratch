
class ExperimentTracker:
    def __init__(self, cfg, name=None):
        self.enable = cfg.wandb.enable
        if cfg.wandb.enable:
            import wandb
            self.wandb = wandb
            self.wandb.init(
                project=cfg.wandb.project,
                name=name if name else cfg.wandb.name,
                config=cfg,
            )

    def log(self, log_dict: dict):
        if self.enable:
            self.wandb.log(log_dict)

    def close(self):
        if self.enable and self.wandb:
            self.wandb.finish()

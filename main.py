
from hesiod import hmain, get_cfg_copy, get_out_dir, get_run_name
from pathlib import Path
import pytorch_lightning as pl
import wandb
import os
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from hesiod import hcfg
import time
import sys
import numpy as np

# seed = np.random.randint(1000)
# pl.seed_everything(seed)

# @hmain(base_cfg_dir=Path("cfg"), template_cfg_file=Path("cfg/template.yaml"))
run_cfg = Path(sys.argv[1])
sys.argv=sys.argv[1:]
@hmain(base_cfg_dir=Path("cfg"), run_cfg_file=run_cfg, create_out_dir=False, parse_cmd_line=True)
def main():
    cfg = get_cfg_copy()
    print(f"***Summary*** :\n{cfg}")
    device = "cuda:" + str(hcfg("gpu")[0])

    # fit the model
    run = wandb.init(
        job_type="train",
        project=hcfg("project_name"),
        name=get_run_name(),
        entity="cvpr",
        dir=get_out_dir(),
        save_code=True
    )
    artifact = wandb.Artifact('Trainer', type='code')
    if hcfg("mean_teacher"):
        if hcfg("step")==1:
            from trainers.classification_trainerMT import Classifier
            artifact.add_file('trainers/classification_trainerMT.py')
        else:
            from trainers.classification_trainer_MTG_DINO import Classifier
            artifact.add_file('trainers/classification_trainer_MTG_DINO.py')
    else:
        from trainers.baseline import Classifier
    run.log_artifact(artifact)
    
    from datamodules.classification_datamodule import DataModule
    from utils.callbacks import PCPredictionLogger

    dm = DataModule(cfg)

    run_name = hcfg("net.name")+"_"+hcfg("project_name")
    print(hcfg("project_name"), run_name, get_out_dir())
    
    wandb_logger = WandbLogger(
        project=hcfg("project_name"), name=run_name, save_dir=get_out_dir()
    )

    checkpoint_dir = os.path.join(get_out_dir(), "checkpoint")
    print(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/source_accuracy",
        dirpath=checkpoint_dir,
        filename="best",
        save_top_k=1,
        save_last=False,
        mode="max",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    model = Classifier(dm=dm, device=device)

    trainer = pl.Trainer(
        logger=wandb_logger,  # W&B integration
        log_every_n_steps=50,  # set the logging frequency,
        gpus=hcfg("gpu"), 
        max_epochs=hcfg("epochs"),
        benchmark=True,
        callbacks=[
            PCPredictionLogger(dm),
            checkpoint_callback,
            lr_monitor,
        ],  # see Callbacks section
        num_sanity_val_steps=2,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path="best")
    model_artifact = wandb.Artifact(
        get_run_name(), type="model",
        description=hcfg("net.name"),
        metadata={
            "problem": "classifier",
            "net": hcfg("net.name"), 
            "run": get_run_name()
            })

    model_artifact.add_file(checkpoint_callback.best_model_path)
    run.log_artifact(model_artifact)

if __name__ == "__main__":
    main()

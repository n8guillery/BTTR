from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

test_year = "2016"
ckp_path = "lightning_logs/version_12/checkpoints/epoch=123-step=186123-val_ExpRate=0.5452.ckpt"

if __name__ == "__main__":
    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year)

    model = LitBTTR.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)

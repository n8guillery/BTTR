from pytorch_lightning.utilities.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

if __name__== '__main__':
    cli = LightningCLI(LitBTTR, CROHMEDatamodule)

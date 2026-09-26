"""Every file location of the tabular Q-learning project, in one place.

Scripts in scripts/ import it with
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import paths
so they run from any working directory.
"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SCRIPTS = PROJECT / "scripts"
CHECKPOINTS = PROJECT / "checkpoints"
DATA = PROJECT / "data"
FIGURES = PROJECT / "figures"
LOGS = PROJECT / "logs"

DATASET = DATA / "qlv3_dataset.pt"            # training / validation episodes of the final models
ORIGINAL_DATASET = DATA / "earlier" / "coconut_dataset.pt"   # data of the earlier models

# memory channels: residual continuous latent (headline), overwrite continuous latent, discrete token
MODELS = ("continuous_residual", "continuous_overwrite", "discrete")
_CKPT_NAME = {"continuous_residual": "coconut_transformer_qlv3-residual.pt",
              "continuous_overwrite": "coconut_transformer_qlv3-overwrite.pt",
              "discrete": "coconut_transformer_qlv3-discrete.pt"}


def checkpoint(model: str) -> Path:
    """Final checkpoint of a trained model, e.g. checkpoint('discrete')."""
    return CHECKPOINTS / model / _CKPT_NAME[model]

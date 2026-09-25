"""Every file location of the multiplicative-weights (MWU) project, in one place.

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

SEEDS = (42, 43)
# memory channels: residual continuous latent (headline), overwrite continuous latent, discrete token;
# theorem_matched = residual latent trained on the noisy-true-expert data of Theorem 3.1
MODELS = ("continuous_residual", "continuous_overwrite", "discrete", "theorem_matched")


def checkpoint(model: str, seed: int) -> Path:
    """Final checkpoint of a trained recurrent model, e.g. checkpoint('discrete', 42)."""
    return CHECKPOINTS / f"{model}_seed{seed}" / "final.pt"


# the original full-history model (re-reads the raw history; the "raw history" baseline)
FULL_HISTORY = CHECKPOINTS / "full_history" / "learned_mw_transformer.pt"

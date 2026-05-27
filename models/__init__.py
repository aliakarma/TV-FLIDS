from models.mlp import IDSMLP, IDSBiLSTM, build_model

__all__ = ["IDSMLP", "IDSBiLSTM", "build_model"]

# IDSMLP: Primary model for tabular IDS data (NSL-KDD, UNSW-NB15).
# IDSBiLSTM: Experimental extension for sequential traffic analysis.
#             Use: python experiments/run_experiment.py --model bilstm
#             Not used in the main paper experiments.

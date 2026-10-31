"""Model variants: the full framework, the baselines of Table 4.3 and the ablations of Tables 3.4 / 4.4.

A variant is a set of switches; everything else (data, optimiser, schedule, epochs, seed) is held
identical across variants, as §3.2.1 requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str = "nadip"                  # "nadip" | "xgboost_source_only" | "lstm_source_only"
    use_grl: bool = True                 # adversarial domain alignment
    use_simclr: bool = True              # stage-2 / contrastive term on unlabelled target
    use_weibull: bool = True             # Weibull-PH RUL head (else linear regressor)
    use_xgb: bool = True                 # stage-5 XGBoost ensemble
    use_pretrained: bool = True          # requires model.pretrained_init; otherwise see `requires`
    datasets: Optional[tuple] = None     # restrict *training* data to these datasets (test set stays full)
    description: str = ""

    @property
    def uses_target(self) -> bool:
        return self.use_grl or self.use_simclr

    @property
    def requires_pretrained(self) -> bool:
        return self.use_pretrained


FULL = Variant("full_nadip", description="Transformer + GRL + SimCLR + Weibull + XGBoost ensemble")

VARIANTS: Dict[str, Variant] = {v.name: v for v in [
    FULL,
    # --- Table 4.3 baselines
    Variant("source_only_xgboost", kind="xgboost_source_only", use_grl=False, use_simclr=False, use_weibull=False,
            use_xgb=False, use_pretrained=False, description="XGBoost on window features, source data only"),
    Variant("source_only_transformer", use_grl=False, use_simclr=False, use_weibull=False, use_xgb=False,
            use_pretrained=False, description="Transformer trained on source data only (pre-intervention control)"),
    Variant("source_only_lstm", kind="lstm_source_only", use_grl=False, use_simclr=False, use_weibull=False,
            use_xgb=False, use_pretrained=False, description="Unaligned LSTM, source data only (Table 3.4)"),
    Variant("grl_only", use_simclr=False, use_weibull=False, use_xgb=False, description="Transformer + GRL"),
    Variant("simclr_only", use_grl=False, use_weibull=False, use_xgb=False, description="Transformer + SimCLR"),
    Variant("grl_weibull", use_simclr=False, use_xgb=False, description="Transformer + GRL + Weibull (no SimCLR)"),
    # --- Tables 3.4 / 4.4 ablations (one component removed from the full model)
    Variant("no_grl", use_grl=False, description="Ablation 1: remove gradient reversal"),
    Variant("no_simclr", use_simclr=False, description="Ablation 2: remove SimCLR"),
    Variant("no_weibull", use_weibull=False, description="Ablation 3: linear RUL regressor instead of Weibull-PH"),
    Variant("no_pretrain", use_pretrained=False,
            description="Ablation 4: random initialisation. Differs from the full model only when "
                        "model.pretrained_init supplies compatible weights"),
    Variant("no_3w_espset", datasets=("cmapss", "ai4i"), description="Ablation 5: CMAPSS + AI4I only (i.e. without 3W; ESPset is not part of this study)"),
    Variant("no_xgb", use_xgb=False, description="Remove XGBoost ensemble (transformer only)"),
]}

ABLATIONS: List[str] = ["no_grl", "no_simclr", "no_weibull", "no_pretrain", "no_3w_espset", "no_xgb"]
TABLE_4_3: List[str] = ["source_only_xgboost", "source_only_transformer", "grl_only", "simclr_only",
                        "grl_weibull", "no_3w_espset", "full_nadip"]

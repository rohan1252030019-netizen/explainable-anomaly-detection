"""
src/detection/ensemble.py

Intelligent Ensemble and Consensus Engine for SENTINEL.

Provides:
- Rank-based Borda consensus aggregation.
- Weighted score aggregation across normalized detector scores.
- Model consensus count and consensus strength per observation.
- Pairwise model disagreement metrics (Jaccard anomaly set distance, Spearman rank correlation).
- Individual detector contribution analysis.

Scientific Principle:
Raw anomaly scores from disparate algorithm families are mathematically incomparable.
Ensemble aggregation must be computed either on calibrated quantile ranks or
properly normalized [0, 1] reference distributions. Ensemble scores represent
relative consensus extremity, NOT posterior Bayesian probabilities.
"""

from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr

from src.detection.scoring import ScoreNormalizer


class IntelligentEnsembleEngine:
    """
    Combines outputs from multiple anomaly detectors into a calibrated consensus.

    Parameters
    ----------
    aggregation_method : {"rank_borda", "weighted_score"}, default="rank_borda"
        Method for combining detector outputs.
    min_consensus_votes : int | None, default=None
        Minimum number of detectors that must agree for an observation to be flagged.
        If None, defaults to max(1, floor(n_models / 2) + 1).
    """

    def __init__(
        self,
        aggregation_method: str = "rank_borda",
        min_consensus_votes: int | None = None,
    ):
        if aggregation_method not in ("rank_borda", "weighted_score"):
            raise ValueError(
                f"Invalid aggregation_method '{aggregation_method}'. "
                "Choose from: 'rank_borda', 'weighted_score'."
            )
        self.aggregation_method = aggregation_method
        self.min_consensus_votes = min_consensus_votes

    def run_ensemble(
        self,
        detector_results: dict[str, dict[str, Any]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate predictions and scores from multiple detectors.

        Parameters
        ----------
        detector_results : dict
            Mapping model_name -> dict containing:
            - 'scores': np.ndarray of raw or normalized scores
            - 'predictions': np.ndarray of binary flags (+1 normal, -1 anomaly)
        weights : dict or None
            Mapping model_name -> positive float weight. If None, equal weights.

        Returns
        -------
        dict with keys:
            ensemble_scores : list of float (in [0, 1], where 1 = highest anomaly consensus)
            consensus_predictions : list of int (+1 or -1)
            consensus_counts : list of int (number of models flagging each row)
            consensus_strength : list of float ([0, 1] voting agreement ratio)
            disagreement : dict (pairwise Jaccard and Spearman correlations)
            model_contributions : dict (model weights and agreement rates)
        """
        if not detector_results:
            raise ValueError("detector_results dictionary is empty.")

        model_names = list(detector_results.keys())
        n_models = len(model_names)

        # Validate sample counts
        first_res = detector_results[model_names[0]]
        first_scores = np.asarray(first_res["scores"])
        n_samples = len(first_scores)

        # Build matrices
        norm_scores_matrix = np.zeros((n_samples, n_models), dtype=np.float64)
        rank_matrix = np.zeros((n_samples, n_models), dtype=np.float64)
        pred_matrix = np.zeros((n_samples, n_models), dtype=int)

        # Normalize weights
        raw_weights = weights or {}
        w_vec = np.array([float(raw_weights.get(m, 1.0)) for m in model_names], dtype=np.float64)
        if np.any(w_vec < 0):
            raise ValueError("Ensemble weights must be non-negative.")
        if np.sum(w_vec) == 0:
            w_vec = np.ones(n_models, dtype=np.float64)
        w_vec = w_vec / np.sum(w_vec)

        for j, m_name in enumerate(model_names):
            res = detector_results[m_name]
            sc = np.asarray(res["scores"], dtype=np.float64)
            preds = np.asarray(res["predictions"], dtype=int)

            if len(sc) != n_samples:
                raise ValueError(
                    f"Model '{m_name}' produced {len(sc)} scores, expected {n_samples}."
                )

            # Rank aggregation uses the RAW scores.  A rank is invariant to any
            # monotone rescaling, so normalising first only adds artefacts:
            # quantile normalisation clips at the 1/99 percentiles, which
            # flattens the extreme tail into a block of tied 0.0/1.0 values and
            # makes genuinely different anomalies indistinguishable to Borda.
            if n_samples > 1:
                ranks = (rankdata(sc, method="average") - 1.0) / (n_samples - 1.0)
            else:
                ranks = np.zeros(n_samples)

            # Weighted-score aggregation, by contrast, requires a common scale.
            # The [0, 1] range check is a heuristic, not a guarantee, so a
            # constant vector is also routed through the normaliser (which
            # handles the degenerate spread) instead of being averaged as-is.
            if sc.min() < 0.0 or sc.max() > 1.0 or np.ptp(sc) <= 0.0:
                sc_norm = ScoreNormalizer(method="quantile").fit(sc).transform(sc)
            else:
                sc_norm = sc

            norm_scores_matrix[:, j] = sc_norm
            rank_matrix[:, j] = ranks
            pred_matrix[:, j] = preds

        # Consensus counts: count models predicting -1 (anomaly)
        anomaly_flags = (pred_matrix == -1).astype(int)
        consensus_counts = np.sum(anomaly_flags, axis=1)
        consensus_strength = consensus_counts / float(n_models)

        # Aggregated scores
        if self.aggregation_method == "rank_borda":
            ensemble_scores = np.dot(rank_matrix, w_vec)
        else:  # "weighted_score"
            ensemble_scores = np.dot(norm_scores_matrix, w_vec)

        # Consensus thresholding
        required_votes = (
            self.min_consensus_votes
            if self.min_consensus_votes is not None
            else max(1, (n_models // 2) + 1)
        )
        consensus_predictions = np.ones(n_samples, dtype=int)
        consensus_predictions[consensus_counts >= required_votes] = -1

        # Disagreement analysis
        disagreement = self._compute_disagreement(
            model_names, anomaly_flags, norm_scores_matrix
        )

        # Model contributions
        model_contribs = {}
        for j, m_name in enumerate(model_names):
            model_contribs[m_name] = {
                "weight": round(float(w_vec[j]), 4),
                "anomalies_flagged": int(np.sum(anomaly_flags[:, j])),
                "consensus_agreement_rate": round(
                    float(np.mean(anomaly_flags[:, j] == (consensus_predictions == -1))), 4
                ),
            }

        return {
            "ensemble_scores": [round(float(v), 5) for v in ensemble_scores],
            "consensus_predictions": [int(v) for v in consensus_predictions],
            "consensus_counts": [int(v) for v in consensus_counts],
            "consensus_strength": [round(float(v), 4) for v in consensus_strength],
            "required_votes": int(required_votes),
            "n_models": int(n_models),
            "models_evaluated": model_names,
            "disagreement": disagreement,
            "model_contributions": model_contribs,
        }

    def _compute_disagreement(
        self,
        model_names: list[str],
        anomaly_flags: np.ndarray,
        norm_scores: np.ndarray,
    ) -> dict[str, Any]:
        """Compute pairwise Jaccard distance and Spearman score correlation."""
        n_models = len(model_names)
        pairwise_jaccard: list[dict[str, Any]] = []
        pairwise_spearman: list[dict[str, Any]] = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                m_i, m_j = model_names[i], model_names[j]
                flags_i = anomaly_flags[:, i]
                flags_j = anomaly_flags[:, j]

                intersection = int(np.logical_and(flags_i, flags_j).sum())
                union = int(np.logical_or(flags_i, flags_j).sum())
                jaccard_sim = intersection / union if union > 0 else 1.0
                jaccard_dist = round(1.0 - jaccard_sim, 4)

                pairwise_jaccard.append({
                    "model_a": m_i,
                    "model_b": m_j,
                    "jaccard_distance": jaccard_dist,
                    "overlap_count": intersection,
                })

                try:
                    rho, _ = spearmanr(norm_scores[:, i], norm_scores[:, j])
                    rho_val = round(float(rho), 4) if not np.isnan(rho) else 0.0
                except Exception:
                    rho_val = 0.0

                pairwise_spearman.append({
                    "model_a": m_i,
                    "model_b": m_j,
                    "spearman_rho": rho_val,
                })

        mean_jaccard = (
            float(np.mean([p["jaccard_distance"] for p in pairwise_jaccard]))
            if pairwise_jaccard
            else 0.0
        )
        mean_spearman = (
            float(np.mean([p["spearman_rho"] for p in pairwise_spearman]))
            if pairwise_spearman
            else 1.0
        )

        return {
            "mean_pairwise_jaccard_distance": round(mean_jaccard, 4),
            "mean_score_rank_correlation": round(mean_spearman, 4),
            "pairwise_jaccard": pairwise_jaccard,
            "pairwise_spearman": pairwise_spearman,
        }

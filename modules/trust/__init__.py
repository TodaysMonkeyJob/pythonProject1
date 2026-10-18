"""
Fuzzy trust in evidence (F16 / TOD-27): per-event trust degree, provenance and de-escalation claim detection.
Runs inside the orchestrator's event router (main process), before events are logged and reach the aggregator.
"""
from modules.trust.claims import ClaimDetector
from modules.trust.scorer import TrustScorer

__all__ = ["ClaimDetector", "TrustScorer"]

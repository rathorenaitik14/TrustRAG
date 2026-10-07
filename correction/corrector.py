from abc import ABC, abstractmethod
from typing import List, Dict, Any
from claims.models import Claim, ClaimEvidenceResult
from verification.models import ClaimVerificationResult
from .models import CorrectionResult

class BaseCorrector(ABC):
    """
    Abstract interface for evidence-grounded correction.
    """
    
    @abstractmethod
    def correct(
        self,
        original_answer: str,
        claims: List[Claim],
        evidence_results: List[ClaimEvidenceResult],
        verification_results: List[ClaimVerificationResult]
    ) -> CorrectionResult:
        """
        Corrects the original answer based on verification and evidence.
        """
        pass

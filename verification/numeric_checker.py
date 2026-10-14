import re
from typing import Optional, Tuple

class NumericContradictionChecker:
    """
    Detects when a claim contradicts a document on numerical values.
    Returns 'CONTRADICTED', 'CONSISTENT', or 'NO_NUMBERS' 
    """
    
    def _extract_quantities(self, text: str) -> list:
        # Extracts (value, unit/type) tuples
        quantities = []
        # Percentages
        for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE):
            quantities.append((float(m.group(1)), "percent"))
        # Magnitudes/Units
        for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s*(billion|million|thousand|ZB|GB|TB|MB|byte|bytes)\b', text, re.IGNORECASE):
            quantities.append((float(m.group(1)), m.group(2).lower()))
        # Standalone numbers
        for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', text):
            val = float(m.group(1))
            # Don't duplicate if already extracted as percent or unit
            if not any(abs(q[0] - val) < 1e-5 for q in quantities):
                quantities.append((val, "number"))
        return quantities
    
    def check(self, claim: str, evidence: str) -> dict:
        claim_quants = self._extract_quantities(claim)
        ev_quants = self._extract_quantities(evidence)
        
        if not claim_quants or not ev_quants:
            return {"verdict": "NO_NUMBERS", "confidence": 0.0}
        
        # Check percentage contradictions
        claim_pcts = [q[0] for q in claim_quants if q[1] == "percent"]
        ev_pcts = [q[0] for q in ev_quants if q[1] == "percent"]
        if claim_pcts and ev_pcts:
            if not any(cp in ev_pcts for cp in claim_pcts):
                return {"verdict": "CONTRADICTED", "confidence": 0.95, "reason": f"Percentage mismatch: {claim_pcts} vs {ev_pcts}"}
            return {"verdict": "CONSISTENT", "confidence": 0.9}

        # Check unit/magnitude contradictions
        claim_units = [(q[0], q[1]) for q in claim_quants if q[1] not in ("percent", "number")]
        ev_units = [(q[0], q[1]) for q in ev_quants if q[1] not in ("percent", "number")]
        if claim_units and ev_units:
            # Different unit or different value with unit
            if any(cu not in ev_units for cu in claim_units):
                return {"verdict": "CONTRADICTED", "confidence": 0.9, "reason": f"Unit quantity mismatch: {claim_units} vs {ev_units}"}
        
        # General numbers
        claim_vals = set(q[0] for q in claim_quants)
        ev_vals = set(q[0] for q in ev_quants)
        
        # If claim asserts numbers completely missing from evidence
        diff = claim_vals - ev_vals
        if diff:
            # If the only difference is not just trivial
            return {"verdict": "CONTRADICTED", "confidence": 0.85, "reason": f"Numbers not in evidence: {diff}"}
        
        return {"verdict": "CONSISTENT", "confidence": 0.9}

    def override_nli_result(self, nli_result: dict, claim: str, evidence: str) -> dict:
        """
        Overrides NLI result to CONTRADICTED if numerical contradiction is detected.
        """
        num_check = self.check(claim, evidence)
        if num_check.get("verdict") == "CONTRADICTED":
            res = dict(nli_result)
            res["status"] = "CONTRADICTED"
            res["score"] = max(res.get("score", 0.0), num_check.get("confidence", 0.85))
            res["numeric_override"] = True
            return res
        return nli_result


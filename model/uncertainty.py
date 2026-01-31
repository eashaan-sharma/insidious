def assign_risk(confidence, threshold=0.7):
    """
    Assigns risk based on confidence
    """
    if confidence < threshold:
        return "review_recommended"
    return "high_risk"
def compute_entropy(probs):
    """
    Computes entropy for uncertainty estimation
    """
    import numpy as np
    return -np.sum(probs * np.log(probs + 1e-8))

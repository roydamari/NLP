import numpy as np

def compute_ece(confidences, accuracies, num_bins=10):
    """
    Compute Expected Calibration Error.
    confidences: array of confidence scores in [0, 1]
    accuracies: array of binary accuracies in {0, 1}
    num_bins: number of bins
    """
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    
    for i in range(num_bins):
        # Find indices of samples in this bin
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        # Include 0 in the first bin
        if i == 0:
            in_bin = in_bin | (confidences == 0)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
            
    return ece

def compute_mse(confidences, accuracies):
    """
    Compute Brier score / Mean Squared Error.
    confidences: array of confidence scores in [0, 1]
    accuracies: array of binary accuracies in {0, 1}
    """
    return np.mean((confidences - accuracies) ** 2)

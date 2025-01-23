""" Helper functions to use in the scoring program. 

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
"""
import base64
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from sys import modules
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score
from typing import Tuple, Callable


# =============================================================================
# ========================= SCORE RELATED HELPERS =============================
# =============================================================================

def read_results_file(file: str) -> np.ndarray:
    """ Read the results of the ingestion program, ground truth or predictions.

    Args:
        file (str): Path to the results file.

    Raises:
        Exception: Error raised when the results file cannot be opened.

    Returns:
        np.ndarray: Array of data of the results file.
    """
    try:
        results = np.loadtxt(file, dtype=float)
        if len(results.shape) == 2:
            return results
        results = np.loadtxt(file, dtype=int)
        return results
    except:
        raise Exception(f"In read_results_file, file '{file}' could not be "
            + f"opened")
    

def get_score(score_file_path: str) -> Tuple[str, 
        Callable[[np.ndarray, np.ndarray], float]]:
    """ Read the score that should be used to evaluate the submissions.

    Args:
        score_file_path (str): Path to the scores file.

    Raises:
        NotImplementedError: Error raised when the score is not implemented.

    Returns:
        Tuple[str, Callable[[np.ndarray, np.ndarray], float]]: The first
            element corresponds to the name of the score while the second one
            is the implementation of the score.
    """
    # Read score
    with open(score_file_path, "r") as f:
        score_name = f.readline().strip()

    # Find score implementation
    try:
        scoring_function = getattr(modules[__name__], score_name)
    except:
        raise NotImplementedError(f"In get_score, '{score_name}' not found")

    # Update score name
    score_name = score_name.replace("_", " ")
    score_name = score_name.title()
    return score_name, scoring_function
        

def mean_confidence_interval(data: list, 
                             confidence: float = 0.95) -> Tuple[float, float]:
    """ Compute the mean and the confidence interval of the specified data. The
    confidence interval is computed at per-task level.

    Args:
        data (list): List of scores to be used in the calculations.
        confidence (float, optional): Level of confidence that should be used. 
            Defaults to 0.95.

    Returns:
        Tuple[float, float]: The first element corresponds to the mean value of
            the data while the second is the confidence interval.
    """
    n = len(data)
    if n == 0:
        return None, None
    if n > 1:
        mean = np.mean(data)
        scale = st.sem(data)
        if scale < 1e-15:
            scale = 1e-15
        lb, _ = st.t.interval(confidence=confidence, df=len(data)-1, loc=mean, 
            scale=scale)
        conf_int = mean - lb 
    else:
        mean = data[0]
        conf_int = 0.0
    return mean, conf_int


def create_histogram(data: list, 
                     score_name: str, 
                     title: str, 
                     path: str) -> str:
    """ Create, save and load a frequency histogram with the specified data.

    Args:
        data (list): Data to be plotted.
        score_name (str): Score used to compute the data.
        title (str): Title for the histogram.
        path (str): Path to save the histogram.

    Returns:
        str: Frequency histogram.
    """
    sns.set_style('darkgrid')
    
    df = pd.DataFrame(data, columns=["value"])
    
    fig, ax = plt.subplots(figsize=(8,4))
    
    # KDE plot
    sns.set_style('white')
    sns.kdeplot(data=df, x="value", ax=ax, warn_singular=False)
    
    # Histogram plot
    ax2 = ax.twinx()
    sns.histplot(data=df, x="value",  bins=40, ax=ax2)

    # Format axes
    x_min, x_max = np.min(data), np.max(data)
    if not np.isclose(x_min, x_max):
        ax.set_xlim((x_min, x_max))
    ax.set_xlabel(f"Score ({score_name})") 
    ax2.set_ylabel("Frequency")
    ax.set_title(title, size = 17)
    
    # Save and return plot
    fig.savefig(path, dpi=fig.dpi)
    plt.close(fig)
    with open(f"{path}.png", "rb") as image_file:
        histogram = base64.b64encode(image_file.read()).decode('ascii')
    return histogram


def create_heatmap(data: dict, 
                   keys: list, 
                   yticks: list, 
                   score_name: str, 
                   title: str, 
                   path: str) -> str:
    """ Create, save and load a frequency heatmap with the specified data.

    Args:
        data (dict): Data to be plotted.
        keys (list): Keys of the data.
        yticks (list): Labels for the y ticks.
        score_name (str): Score used to compute the data.
        title (str): Title for the heatmap.
        path (str): Path to save the heatmap.

    Returns:
        str: Frequency heatmap.
    """
    # Limits for the heatmap
    minimum = np.inf
    maximum = -np.inf
    for key in keys:
        minimum = min(np.min(data[key][score_name]), minimum)
        maximum = max(np.max(data[key][score_name]), maximum)    
    
    # Heatmap data
    bins = np.linspace(minimum, maximum, 11)
    heatmap = [np.histogram(data[key][score_name], bins=bins)[0] for key in 
        keys]
      
    # Plot
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(heatmap, cmap="Blues", linewidths=.2, yticklabels=yticks)
    ax.set_xticks(np.arange(len(bins)), labels=np.round(bins, 2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", 
        rotation_mode="anchor")
    ax.set_xlabel(f"Score ({score_name})") 
    ax.set_title(title, size = 17)
    fig.tight_layout()
    
    # Save and return plot
    fig.savefig(path, dpi=fig.dpi)
    plt.close(fig)
    with open(f"{path}.png", "rb") as image_file:
        heatmap = base64.b64encode(image_file.read()).decode('ascii')
    return heatmap

# =============================================================================
# ============================== SCORE FUNCTIONS ==============================
# =============================================================================

def normalized_accuracy(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        num_ways: int) -> float:
    """ Compute the normalized accuracy of the given predictions regarding the 
    number of ways.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        num_ways (int): Number of ways.

    Raises:
        Exception: Exception raised when the normalized accuracy cannot be 
            computed.

    Returns:
        float: Normalized accuracy of the predictions.
    """
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    try:
        bac = recall_score(y_true, y_pred, average = "macro", zero_division=0)
        base_bac = 1/num_ways # random guessing
        return (bac - base_bac) / (1 - base_bac)
    except Exception as e:
        raise Exception(f"In normalized_accuracy, score cannot be computed. "
            + f"Detailed error: {repr(e)}")
        

def accuracy(y_true: np.ndarray, 
             y_pred: np.ndarray) -> float:
    """ Compute the accuracy of the given predictions.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Raises:
        Exception: Exception raised when the accuracy cannot be computed.

    Returns:
        float: Accuracy of the predictions.
    """
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    try:
        return accuracy_score(y_true, y_pred)
    except Exception as e:
        raise Exception(f"In accuracy, score cannot be computed. Detailed "
            + f"error: {repr(e)}")
    

def macro_f1_score(y_true: np.ndarray, 
                   y_pred: np.ndarray) -> float:
    """ Compute the macro averaged f1 score of the given predictions.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Raises:
        Exception: Exception raised when the macro averaged f1 score cannot be 
            computed.

    Returns:
        float: Macro averaged f1 score of the predictions.
    """
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    try:
        return f1_score(y_true, y_pred, average = "macro", zero_division = 0)
    except Exception as e:
        raise Exception(f"In macro_f1_score, score cannot be computed. "
            + f"Detailed error: {repr(e)}")
        

def macro_precision(y_true: np.ndarray, 
                    y_pred: np.ndarray) -> float:
    """ Compute the macro averaged precision of the given predictions.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Raises:
        Exception: Exception raised when the macro averaged precision cannot be 
            computed.

    Returns:
        float: Macro averaged precision of the predictions.
    """
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    try:
        return precision_score(y_true, y_pred, average = "macro", 
            zero_division = 0)
    except Exception as e:
        raise Exception(f"In macro_precision, score cannot be computed. "
            + f"Detailed error: {repr(e)}")
        
        
def macro_recall(y_true: np.ndarray, 
                 y_pred: np.ndarray) -> float:
    """ Compute the macro averaged recall of the given predictions.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Raises:
        Exception: Exception raised when the macro averaged recall cannot be 
            computed.

    Returns:
        float: Macro averaged recall of the predictions.
    """
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    try:
        return recall_score(y_true, y_pred, average = "macro", 
            zero_division = 0)
    except Exception as e:
        raise Exception(f"In macro_recall, score cannot be computed. Detailed "
            + f"error: {repr(e)}")


def compute_all_scores(y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       num_ways: int,
                       batch: bool = False) -> dict:
    """ Computes the normalized accuracy, accuracy, macro averaged f1 score,
    macro averaged precision and macro averaged recall of the given predictions

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        num_ways (int): Number of ways.
        batch (bool): Boolean flag to indicate that the current data belongs to 
            a batch instead of a task. Defaults to False.

    Returns:
        dict: Dictionary with all the scores.
    """
    scoring = {
        "Normalized Accuracy": normalized_accuracy,
        "Accuracy": accuracy,
        "Macro F1 Score": macro_f1_score,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall
    }
    if batch:
        del scoring["Normalized Accuracy"]
        
    scores = dict()
    for key in scoring.keys():
        scoring_function = scoring[key]
        if key == "Normalized Accuracy":
            scores[key] = scoring_function(y_true, y_pred, num_ways)
        else:
            scores[key] = scoring_function(y_true, y_pred)
    return scores

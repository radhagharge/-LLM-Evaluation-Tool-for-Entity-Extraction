import json
import pandas as pd


def calculate_accuracy(predicted_text: str, ground_truth: dict) -> float:
    """
    Compare model output (JSON string) with ground truth dictionary.
    Returns accuracy as a percentage of exact matches.
    """
    try:
        # Convert model output (string) to dictionary
        predicted = json.loads(predicted_text)
    except json.JSONDecodeError:
        # If the model didn't return valid JSON, accuracy = 0
        return 0.0

    # Compare predicted vs ground truth
    matches = 0
    for key, true_value in ground_truth.items():
        pred_value = predicted.get(key, "").strip().lower()
        true_value = true_value.strip().lower()
        if pred_value == true_value:
            matches += 1

    accuracy = (matches / len(ground_truth)) * 100 if ground_truth else 0
    return round(accuracy, 2)


def save_results(results: list, filename: str = "results/reports/run_report.csv") -> pd.DataFrame:
    """
    Saves a list of model performance results into a CSV file.
    Returns a Pandas DataFrame for Streamlit visualization.
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    return df

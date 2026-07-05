"""
Metrics Aggregation Module

This script parses classification reports and confusion matrices generated
across multiple subject directories and aggregates them into a single
comprehensive pandas DataFrame for further analysis.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Configure standard logging for professional output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def aggregate_classification_metrics(base_dir: str = "results") -> Optional[pd.DataFrame]:
    """
    Reads 'classification_report.json' and 'confusion_matrix.json' from subdirectories
    of the specified base directory and aggregates them into a single table.

    Args:
        base_dir (str): The root directory containing subject-specific subfolders.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the aggregated metrics,
                                sorted by ID, or None if no valid files are found.
    """
    base_path = Path(base_dir)

    # pathlib's glob is cleaner than the standard glob module
    report_paths = list(base_path.glob("*/classification_report.json"))

    if not report_paths:
        logging.warning(f"No reports found matching pattern: {base_dir}/*/classification_report.json")
        return None

    extracted_data = []

    for report_path in report_paths:
        subject_folder = report_path.parent
        subject_id = subject_folder.name
        matrix_path = subject_folder / "confusion_matrix.json"

        row_data = {'ID': subject_id}

        # 1. Parse the Classification Report
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)

            # Flatten the nested dictionary structure
            for key, value in report.items():
                if isinstance(value, dict):
                    for metric, score in value.items():
                        row_data[f"{key}_{metric}"] = score
                else:
                    row_data[key] = value
        except Exception as e:
            logging.error(f"Failed to read or parse report for {subject_id}: {e}")
            continue

        # 2. Parse the Confusion Matrix
        # Pre-fill with None to ensure consistent columns even if file is missing
        row_data.update({'TN': None, 'FP': None, 'FN': None, 'TP': None})

        if matrix_path.exists():
            try:
                with open(matrix_path, 'r', encoding='utf-8') as f:
                    matrix = json.load(f)

                # Extract elements assuming a 2x2 binary classification matrix
                if len(matrix) == 2 and len(matrix[0]) == 2 and len(matrix[1]) == 2:
                    row_data['TN'] = matrix[0][0]
                    row_data['FP'] = matrix[0][1]
                    row_data['FN'] = matrix[1][0]
                    row_data['TP'] = matrix[1][1]
            except Exception as e:
                logging.error(f"Failed to read or parse confusion matrix for {subject_id}: {e}")

        extracted_data.append(row_data)

    # Convert to DataFrame, sort by subject ID, and reset index
    df = pd.DataFrame(extracted_data)
    df = df.sort_values(by='ID').reset_index(drop=True)

    return df


if __name__ == "__main__":
    target_directory = "InterpolatedBIOT_new_pipeline"
    final_table = aggregate_classification_metrics(target_directory)

    if final_table is not None:
        print("\n=== COMPLETE METRICS TABLE ===")
        # to_string() forces the display of all rows and columns in the console
        print(final_table.to_string())
        print("==============================\n")

        # Save results to CSV
        csv_filename = "complete_metrics_summary.csv"
        final_table.to_csv(csv_filename, index=False)
        logging.info(f"CSV successfully saved to: {csv_filename}")

        # Save results to Excel (XLSX)
        excel_filename = "complete_metrics_summary.xlsx"
        try:
            # Requires 'openpyxl' to be installed via pip
            final_table.to_excel(excel_filename, index=False, engine='openpyxl')
            logging.info(f"Excel successfully saved to: {excel_filename}")
        except ModuleNotFoundError:
            logging.error("Could not save to Excel. Please install openpyxl: 'pip install openpyxl'")
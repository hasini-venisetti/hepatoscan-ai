"""Auto-generate LaTeX results tables from evaluation metrics.

Produces publication-ready tables for papers and README.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def generate_latex_table(
    results: dict,
    caption: str = "HepatoScan AI Evaluation Results",
    label: str = "tab:results",
) -> str:
    """Generate a LaTeX table from evaluation results.

    Parameters
    ----------
    results : dict
        Evaluation results from compute_all_metrics().
    caption : str
        Table caption.
    label : str
        LaTeX label for cross-referencing.

    Returns
    -------
    str
        LaTeX table source code.
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("")

    # Segmentation table
    seg = results.get("segmentation", {})
    if seg:
        lines.append("\\begin{subtable}{\\textwidth}")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lccccc}")
        lines.append("\\toprule")
        lines.append("Task & Dice & IoU & HD95 (mm) & Sensitivity & Specificity \\\\")
        lines.append("\\midrule")
        lines.append(
            f"Liver Seg. & {seg.get('dice', 'N/A'):.4f} & {seg.get('iou', 'N/A'):.4f} & "
            f"{seg.get('hd95', 'N/A'):.2f} & {seg.get('sensitivity', 'N/A'):.4f} & "
            f"{seg.get('specificity', 'N/A'):.4f} \\\\"
        )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Segmentation Performance}")
        lines.append("\\end{subtable}")
        lines.append("")

    # Classification table
    cls = results.get("classification", {})
    if cls:
        lines.append("\\begin{subtable}{\\textwidth}")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Metric & Accuracy & F1 (Weighted) & Precision & Recall \\\\")
        lines.append("\\midrule")
        lines.append(
            f"Classification & {cls.get('accuracy', 'N/A'):.4f} & "
            f"{cls.get('f1_weighted', 'N/A'):.4f} & "
            f"{cls.get('precision_weighted', 'N/A'):.4f} & "
            f"{cls.get('recall_weighted', 'N/A'):.4f} \\\\"
        )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Classification Performance}")
        lines.append("\\end{subtable}")
        lines.append("")

    # Staging table
    stg = results.get("staging", {})
    if stg:
        lines.append("\\begin{subtable}{\\textwidth}")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append("Metric & Accuracy & Weighted $\\kappa$ \\\\")
        lines.append("\\midrule")
        lines.append(
            f"BCLC Staging & {stg.get('staging_accuracy', 'N/A'):.4f} & "
            f"{stg.get('weighted_kappa', 'N/A'):.4f} \\\\"
        )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Staging Performance}")
        lines.append("\\end{subtable}")

    lines.append("\\end{table}")

    return "\n".join(lines)

"""Tools for feature selection/filtering in the data mining pipeline."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy.stats import pearsonr, spearmanr


def make_tools(session):
    """Factory function returning all filtering tools bound to the given session state."""

    @tool
    def variance_filter_tool(threshold: float = 0.15) -> Dict[Any, Any]:
        """Filter near-constant features from the document-term matrix based on population variance.

        Computes the population variance (ddof=0) for each term in session.feature_matrix.
        Terms with variance <= threshold are dropped directly from session.feature_matrix
        and session.feature_names.

        Args:
            threshold: Variance threshold for dropping terms. Terms with population variance
                less than or equal to this threshold will be removed. Defaults to 0.15.

        Returns:
            A summary dictionary containing result_id, initial_features, remaining_features,
            dropped_features, threshold, and top_remaining_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix found. Please run build_dtm_tool first."
            }

        matrix = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Compute population variance (ddof=0) for each column/term
        variances = np.var(matrix, axis=0, ddof=0)

        # Identify terms to keep and drop
        keep_mask = variances > threshold
        drop_mask = ~keep_mask

        kept_terms = feature_names[keep_mask].tolist()
        dropped_terms = feature_names[drop_mask].tolist()
        kept_variances = variances[keep_mask]

        # Update session state in-place (actually removing the columns)
        if hasattr(session.feature_matrix, "tocsr"):
            session.feature_matrix = session.feature_matrix.tocsr()[:, keep_mask]
        else:
            session.feature_matrix = matrix[:, keep_mask]

        session.feature_names = kept_terms

        # Build full report DataFrame for visualize_result_tool
        # First column: term name, Second column: variance value, sorted descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")

        summary = {
            "result_id": result_id,
            "initial_features": len(feature_names),
            "remaining_features": len(kept_terms),
            "dropped_features": dropped_terms,
            "threshold": threshold,
            "top_remaining_terms": [
                {"term": term, "variance": float(var)}
                for term, var in zip(kept_terms[:10], kept_variances[:10])
            ]
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute Pearson correlation between term counts and a target class (one-vs-rest).

        Args:
            target_class: Target category label to compute correlation against.
            threshold: Minimum absolute Pearson correlation to keep in filtered_terms.

        Returns:
            Dict containing result_id, target_class, correlations list, and filtered_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."}

        # Determine target binary labels (1 if target_class, else 0)
        category_labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                category_labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                category_labels = session.dataframe["category"].tolist()
        elif session.labels is not None:
            category_labels = session.labels.tolist()

        if category_labels is None:
            return {"status": "error", "message": "No category labels found in session."}

        y_binary = np.array([1.0 if label == target_class else 0.0 for label in category_labels])

        # Dense DTM matrix
        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        results = []
        for i, term in enumerate(session.feature_names):
            x_col = X[:, i]
            # Handle constant column case to set r=0.0 and avoid NaN
            if np.all(x_col == x_col[0]) or np.all(y_binary == y_binary[0]):
                r_val = 0.0
            else:
                r_val, _ = pearsonr(x_col, y_binary)
                if np.isnan(r_val):
                    r_val = 0.0

            results.append({"term": term, "pearson_r": float(r_val)})

        # Create DataFrame for full_report, sorted descending by pearson_r
        df_report = pd.DataFrame(results)
        df_report = df_report.sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        # Filter terms by threshold on absolute correlation
        filtered_terms = df_report[df_report["pearson_r"].abs() >= threshold]["term"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "top_terms": df_report.to_dict(orient="records"),
            "filtered_terms": filtered_terms,
        }

        # Store result with df_report as full_report (1st col: term, 2nd col: pearson_r, sorted descending)
        session.store_result("pearson_filter_tool", {"target_class": target_class}, summary, full_report=df_report)

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute Spearman rank correlation between term counts and a target class (one-vs-rest).

        Args:
            target_class: Target category label to compute correlation against.
            threshold: Minimum absolute Spearman correlation to keep in filtered_terms.

        Returns:
            Dict containing result_id, target_class, threshold, top_terms, and filtered_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."}

        # Determine target binary labels (1 if target_class, else 0)
        category_labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                category_labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                category_labels = session.dataframe["category"].tolist()
        elif session.labels is not None:
            category_labels = session.labels.tolist()

        if category_labels is None:
            return {"status": "error", "message": "No category labels found in session."}

        y_binary = np.array([1.0 if label == target_class else 0.0 for label in category_labels])

        # Dense DTM matrix
        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        results = []
        for i, term in enumerate(session.feature_names):
            x_col = X[:, i]
            # Handle constant column case to set r=0.0 and avoid NaN
            if np.all(x_col == x_col[0]) or np.all(y_binary == y_binary[0]):
                r_val = 0.0
            else:
                r_val, _ = spearmanr(x_col, y_binary)
                if np.isnan(r_val):
                    r_val = 0.0

            results.append({"term": term, "spearman_r": float(r_val)})

        # Create DataFrame for full_report, sorted descending by spearman_r
        df_report = pd.DataFrame(results)
        df_report = df_report.sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        # Filter terms by threshold on absolute correlation
        filtered_terms = df_report[df_report["spearman_r"].abs() >= threshold]["term"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "top_terms": df_report.to_dict(orient="records"),
            "filtered_terms": filtered_terms,
        }

        # Store result with df_report as full_report (1st col: term, 2nd col: spearman_r, sorted descending)
        session.store_result("spearman_filter_tool", {"target_class": target_class}, summary, full_report=df_report)

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]

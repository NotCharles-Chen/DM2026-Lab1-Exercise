from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc1_index: int,
        doc2_index: int,
    ) -> Dict[str, Any]:
        """Calculate cosine similarity between two documents by index using fitted count_vectorizer or feature_matrix.

        Args:
            doc1_index: Row index of the first document.
            doc2_index: Row index of the second document.

        Returns:
            Dict containing result_id, doc1_index, doc2_index, and similarity float value.
        """
        # Vectorize using fitted count_vectorizer if available and text is present in session.dataframe
        if "count_vectorizer" in session.artifacts and session.dataframe is not None and "text" in session.dataframe.columns:
            vectorizer = session.artifacts["count_vectorizer"]

            n_docs = len(session.dataframe)
            if not (0 <= doc1_index < n_docs and 0 <= doc2_index < n_docs):
                return {"status": "error", "message": f"Indices out of range. Document count: {n_docs}"}

            text1 = session.dataframe.iloc[doc1_index]["text"]
            text2 = session.dataframe.iloc[doc2_index]["text"]

            # Replace NaN or None with empty string
            text1 = "" if pd.isna(text1) or text1 is None else str(text1)
            text2 = "" if pd.isna(text2) or text2 is None else str(text2)

            vecs = vectorizer.transform([text1, text2]).toarray()
            v1, v2 = vecs[0], vecs[1]
        elif session.feature_matrix is not None:
            n_docs = session.feature_matrix.shape[0]
            if not (0 <= doc1_index < n_docs and 0 <= doc2_index < n_docs):
                return {"status": "error", "message": f"Indices out of range. Matrix document count: {n_docs}"}

            mat = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
            v1, v2 = mat[doc1_index], mat[doc2_index]
        else:
            return {"status": "error", "message": "Neither count_vectorizer in artifacts nor feature_matrix in session was found. Run build_dtm_tool first."}

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Handle zero vector case (e.g., empty text row) to return 0.0 instead of NaN
        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim = float(cosine_similarity([v1], [v2])[0][0])
            if np.isnan(sim):
                sim = 0.0

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc1_index": doc1_index,
            "doc2_index": doc2_index,
            "similarity": round(sim, 4),
        }
        session.store_result("cosine_similarity_tool", {"doc1_index": doc1_index, "doc2_index": doc2_index}, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Computes feature-vs-feature Pearson correlation matrix over top-20 terms by variance,
        sets heatmap figure on session.pending_figure, and returns the correlation matrix.

        Returns:
            Dict containing result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix built in session."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        feature_names = np.array(session.feature_names)

        # 1. Compute variance for each term across all documents (ddof=0)
        variances = np.var(X, axis=0, ddof=0)

        # Top 20 terms by variance (descending)
        top_k = min(20, len(feature_names))
        sorted_indices = np.argsort(variances)[::-1][:top_k]

        top_terms = feature_names[sorted_indices].tolist()
        top_X = X[:, sorted_indices]

        # 2. Compute Pearson correlation matrix across top terms
        corr_matrix = np.corrcoef(top_X.T)
        if np.isnan(corr_matrix).any():
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 3. Build Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            xticklabels=top_terms,
            yticklabels=top_terms,
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title(f"Feature Correlation Matrix (Top {top_k} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_matrix.tolist()
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]

from typing import Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Reduce document-term matrix dimensions (PCA, t-SNE, UMAP) and plot coordinates.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap').
            n_components: Number of target dimensions (default 2).
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors for UMAP (default 15).
            random_state: Random state seed for reproducibility.

        Returns:
            Dict containing result_id, method, coordinates, and explained_variance_ratio (if PCA).
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature_matrix found in session. Run build_dtm_tool first."}

        # Convert sparse matrix to dense array for reduction algorithms
        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}

        # Store reduced coordinates in session.artifacts
        session.artifacts["reduced_coordinates"] = coords

        # Generate plot without calling savefig or plt.show()
        fig, ax = plt.subplots(figsize=(8, 6))

        # Get categories for coloring
        category_labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                category_labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                category_labels = session.dataframe["category"].tolist()

        if category_labels is not None and len(category_labels) == len(coords):
            df_plot = pd.DataFrame({
                "Dim 1": coords[:, 0],
                "Dim 2": coords[:, 1],
                "Category": category_labels,
            })
            sns.scatterplot(data=df_plot, x="Dim 1", y="Dim 2", hue="Category", style="Category", ax=ax, s=100)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=100)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        fig.tight_layout()

        # Assign figure to pending_figure attribute
        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": coords.tolist(),
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result("reduce_dimensions_tool", {"method": method}, summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes the category labels into K binary (0/1) columns in session.dataframe.

        Returns:
            Dict containing result_id, categories, and one_hot_columns created.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "No dataset loaded in session."}

        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        if not cat_col:
            return {"status": "error", "message": "No category column found in session.dataframe."}

        # Perform one-hot encoding for K categories -> K binary (0/1) columns
        one_hot = pd.get_dummies(df[cat_col], prefix="cat", dtype=int)
        one_hot_cols = list(one_hot.columns)

        # Merge one-hot columns into session.dataframe
        for col in one_hot_cols:
            session.dataframe[col] = one_hot[col]

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "status": "success",
            "categories": sorted(list(df[cat_col].unique())),
            "one_hot_columns": one_hot_cols,
            "encoded_shape": list(one_hot.shape)
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]

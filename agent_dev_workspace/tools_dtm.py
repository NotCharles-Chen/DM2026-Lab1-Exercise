from typing import List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
        stop_words: Optional[str] = None
    ) -> dict:
        """
        Builds the document-term matrix (DTM) from session.dataframe["text"].

        Args:
            max_features: Maximum number of features (terms) to keep.
            min_df: Minimum document frequency threshold.
            max_df: Maximum document frequency threshold.
            ngram_range: Lower and upper boundary of the n-values for different n-grams, as a 2-element list.
            stop_words: Language stop words to exclude (e.g. 'english').
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        # 轉換成 CountVectorizer 所需的 ngram_range tuple
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
            stop_words=stop_words
        )

        texts = session.dataframe["text"].fillna("").astype(str).tolist()
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # 設定類別標籤 (呼叫 session.set_labels)
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].tolist())

        # 計算矩陣稀疏度
        non_zero = int(feature_matrix.nnz)
        n_rows, n_cols = feature_matrix.shape
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        # 為避免 summary 過大造成 LLM context 負擔，只收錄前 20 個詞的預覽
        vocabulary_preview = feature_names[:20]

        summary = {
            "result_id": session.next_result_id("dtm"),
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "vocabulary_preview": vocabulary_preview
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "ngram_range": ngram_range}, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: Optional[int] = None) -> dict:
        """
        Aggregates term frequencies across all documents in the document-term matrix.

        Args:
            top_n: Optional number of top terms to return. If None, returns top 20 terms by default for LLM context optimization.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix built in session."}

        # 直接從 session.feature_matrix 進行全域欄位加總
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        term_freqs = dict(zip(session.feature_names, map(int, counts)))

        # 依詞頻降序排序
        sorted_freqs = dict(sorted(term_freqs.items(), key=lambda item: item[1], reverse=True))

        # 若未指定 top_n， summary 預設最多只回傳前 20 個高頻詞，避免 summary 過大
        limit = top_n if (top_n is not None and top_n > 0) else 20
        result_freqs = dict(list(sorted_freqs.items())[:limit])

        # 建立完整 DataFrame 供 store_result / visualize_result_tool 使用
        report_df = pd.DataFrame(list(sorted_freqs.items()), columns=["term", "frequency"])

        summary = {
            "result_id": session.next_result_id("tf"),
            "total_terms": len(session.feature_names),
            "term_frequencies": result_freqs
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=report_df)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Renders a heatmap slice of the raw document-term matrix (DTM).

        Args:
            n_terms (int): Number of terms (columns) to include in vocabulary order. Defaults to 20.
            n_documents (int): Number of documents (rows) to include in row order. Defaults to 20.

        Returns:
            Dict containing result_id, n_terms, n_documents, terms, doc_labels, and matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix built in session."}

        # Slice positional window
        sliced = session.feature_matrix[:n_documents, :n_terms]
        dense_slice = sliced.toarray() if hasattr(sliced, "toarray") else np.asarray(sliced)

        terms_included = session.feature_names[:n_terms]
        num_docs = dense_slice.shape[0]
        doc_labels = [f"Doc {i}" for i in range(num_docs)]

        # Plot Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            dense_slice,
            annot=True,
            fmt="d",
            xticklabels=terms_included,
            yticklabels=doc_labels,
            cmap="Blues",
            ax=ax
        )
        ax.set_title(f"DTM Heatmap Slice ({num_docs} docs x {len(terms_included)} terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_terms": len(terms_included),
            "n_documents": num_docs,
            "terms": terms_included,
            "doc_labels": doc_labels,
            "matrix": dense_slice.tolist()
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]

import os
import tempfile
from typing import Dict, Any, List, Literal
import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: int = 1,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Mine frequent itemsets/patterns for a single category using PAMI.

        Args:
            category_name: The category label to filter session.dataframe by.
            filtering_method: Vocabulary filtering method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support count for 'fpgrowth' and 'maxfpgrowth'.
            k: Top-k patterns count for 'topk'.

        Returns:
            Dict with result_id, category_name, filtering_method, algorithm, and mined patterns.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe found in session. Run load_dataset_tool first."}

        # Filter dataframe by category_name
        if "category_name" in session.dataframe.columns:
            cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        elif "category" in session.dataframe.columns:
            cat_df = session.dataframe[session.dataframe["category"] == category_name]
        else:
            return {"status": "error", "message": "No category column found in session.dataframe."}

        # Filter out empty text rows
        texts = [str(t) for t in cat_df["text"].dropna().tolist() if str(t).strip()]
        if not texts:
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "patterns": [],
                "note": "no documents or text found for this category",
            }

        # Build fresh, category-local DTM dropping English stopwords
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError:
            # Occurs if vocabulary is empty after stopword removal
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }

        feature_names = vectorizer.get_feature_names_out()
        if dtm.shape[1] == 0:
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "patterns": [],
                "note": "no terms survived filtering",
            }

        dtm_dense = dtm.toarray()

        # Apply filtering method
        if filtering_method == "variance":
            variances = dtm_dense.var(axis=0)
            p5 = pd.Series(variances).quantile(0.05)
            p95 = pd.Series(variances).quantile(0.95)
            kept_indices = [i for i, v in enumerate(variances) if p5 < v < p95]
        elif filtering_method == "term_frequency":
            counts = dtm_dense.sum(axis=0)
            p5 = pd.Series(counts).quantile(0.05)
            p95 = pd.Series(counts).quantile(0.95)
            kept_indices = [i for i, c in enumerate(counts) if p5 < c < p95]
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_mat = tfidf_transformer.fit_transform(dtm).toarray()
            mean_tfidf = tfidf_mat.mean(axis=0)
            p20 = pd.Series(mean_tfidf).quantile(0.20)
            kept_indices = [i for i, score in enumerate(mean_tfidf) if score >= p20]
        else:
            return {"status": "error", "message": f"Unsupported filtering_method: {filtering_method}"}

        # Handle zero terms surviving filtering gracefully
        if not kept_indices:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        kept_terms = [feature_names[i] for i in kept_indices]
        filtered_dtm = dtm_dense[:, kept_indices]

        # Convert to DataFrame for PAMI DF2DB conversion
        df_filtered = pd.DataFrame(filtered_dtm, columns=kept_terms)

        # Build transactional database file
        from PAMI.extras.convert import DF2DB
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name

        try:
            converter = DF2DB.DF2DB(df_filtered)
            converter.convert2TransactionalDatabase(tmp_path, thresholdValue=1, condition=">=")

            # Mine patterns using requested algorithm
            patterns = []
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                obj = FPGrowth.FPGrowth(iFile=tmp_path, minSup=min_sup, sep="\t")
                obj.mine()
                df_patterns = obj.getPatternsAsDataFrame()
                if not df_patterns.empty:
                    for _, row in df_patterns.iterrows():
                        patterns.append({"pattern": str(row["Patterns"]), "support": int(row["Support"])})
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                obj = FAE.FAE(iFile=tmp_path, k=k, sep="\t")
                obj.mine()
                df_patterns = obj.getPatternsAsDataFrame()
                if not df_patterns.empty:
                    for _, row in df_patterns.iterrows():
                        patterns.append({"pattern": str(row["Patterns"]), "support": int(row["Support"])})
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                obj = MaxFPGrowth.MaxFPGrowth(iFile=tmp_path, minSup=min_sup, sep="\t")
                obj.mine()
                df_patterns = obj.getPatternsAsDataFrame()
                if not df_patterns.empty:
                    for _, row in df_patterns.iterrows():
                        patterns.append({"pattern": str(row["Patterns"]), "support": int(row["Support"])})
            else:
                return {"status": "error", "message": f"Unsupported algorithm: {algorithm}"}

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns,
        }
        session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)

        return summary

    return [mine_patterns_tool]

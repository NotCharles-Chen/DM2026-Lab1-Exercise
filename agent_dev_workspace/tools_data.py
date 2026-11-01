import pandas as pd
import nltk
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

# Ensure nltk tokenizer data is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        nltk.download('punkt', quiet=True)

def make_tools(session):
    @tool
    def inspect_data_tool(
        n_rows: int = 5,
        max_chars: int = 50
    ) -> Dict[str, Any]:
        """Peeks at the top n_rows of the active DataFrame in session, truncating text to max_chars.
        
        Args:
            n_rows: Number of rows to inspect from the top of the DataFrame (default 5).
            max_chars: Maximum character length for text preview before truncation (default 50).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {
                "status": "error",
                "message": "No DataFrame is loaded in the current session. Please load a dataset first."
            }
        
        df = session.dataframe.head(n_rows).copy()
        
        # Truncate text column if it exists
        if "text" in df.columns:
            df["text"] = df["text"].fillna("").astype(str).apply(
                lambda s: s[:max_chars] + "..." if len(s) > max_chars else s
            )
            
        rows_data = df.to_dict(orient="records")
        
        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_rows_returned": len(rows_data),
            "max_chars_limit": max_chars,
            "rows": rows_data
        }
        
        session.store_result("inspect_data_tool", {"n_rows": n_rows, "max_chars": max_chars}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """
        Checks for missing or empty text values in session.dataframe.
        Treats both NaN and empty/whitespace-only strings as missing values.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}
        
        df = session.dataframe
        text_col = df["text"] if "text" in df.columns else df.iloc[:, 0]
        
        # NaN or whitespace-only/empty string
        is_missing = text_col.isna() | (text_col.astype(str).str.strip() == "")
        missing_count = int(is_missing.sum())
        missing_indices = df[is_missing].index.tolist()
        
        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "missing_count": missing_count,
            "missing_indices": missing_indices
        }
        
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """
        Checks for duplicate documents in session.dataframe and optionally drops them.
        When drop=True, removes ALL copies of duplicated rows (keep=False) and resets index.
        Clears downstream feature_matrix, feature_names, labels, and count_vectorizer when dropping.
        
        Args:
            drop (bool): If True, removes duplicate documents from session.dataframe.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}
        
        df = session.dataframe
        
        # Check duplicated across full DataFrame (matching Master's df.duplicated())
        duplicate_mask = df.duplicated()
        duplicate_count = int(duplicate_mask.sum())
        duplicate_indices = df[duplicate_mask].index.tolist()
        
        dropped_count = 0
        if drop and duplicate_count > 0:
            # Keep=False drops ALL instances of duplicated rows
            all_dup_mask = df.duplicated(keep=False)
            dropped_count = int(all_dup_mask.sum())
            
            session.dataframe = df[~all_dup_mask].reset_index(drop=True)
            
            # Clear downstream attributes
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None
            if "count_vectorizer" in session.artifacts:
                del session.artifacts["count_vectorizer"]
        
        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "dropped_count": dropped_count,
            "remaining_documents": len(session.dataframe)
        }
        
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """
        Tokenizes each document's text into lowercased unigrams using nltk.word_tokenize.
        Stores the resulting list of tokens directly into session.dataframe["unigrams"].
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}
        
        df = session.dataframe
        text_col = df["text"] if "text" in df.columns else df.iloc[:, 0]
        
        all_unigrams = []
        for text in text_col:
            if pd.isna(text) or str(text).strip() == "":
                all_unigrams.append([])
            else:
                tokens = nltk.word_tokenize(str(text).lower())
                all_unigrams.append(tokens)
        
        df["unigrams"] = all_unigrams
        
        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": len(df),
            "sample_unigrams": all_unigrams[:3] if len(all_unigrams) >= 3 else all_unigrams
        }
        
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Subsamples rows from session.dataframe using pandas DataFrame.sample.

        Args:
            n (int): Number of rows to sample. Defaults to 5.
            random_state (int, optional): Random seed for reproducible sampling.

        Returns:
            Dict containing result_id, sampled_count, and sampled_indices.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}

        df = session.dataframe
        sample_n = min(n, len(df))
        sampled_df = df.sample(n=sample_n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "sampled_count": len(sampled_indices),
            "sampled_indices": sampled_indices
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes text length statistics (character count) overall and per category,
        and sets a boxplot figure on session.pending_figure.

        Returns:
            Dict containing result_id, overall_stats, and by_category_stats.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}

        df = session.dataframe.copy()
        text_col = df["text"] if "text" in df.columns else df.iloc[:, 0]
        df["text_length"] = text_col.fillna("").astype(str).apply(len)

        # Overall stats
        overall_desc = df["text_length"].describe().to_dict()

        # Per-category stats
        by_category = {}
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        
        if cat_col:
            for cat, group in df.groupby(cat_col):
                by_category[str(cat)] = group["text_length"].describe().to_dict()

        # Build Box Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        if cat_col:
            sns.boxplot(x=cat_col, y="text_length", data=df, ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y="text_length", data=df, ax=ax)
            ax.set_title("Text Length Distribution")
        
        ax.set_ylabel("Text Length (chars)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_desc,
            "by_category_stats": by_category
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and directories in the repository safely under subdirectory.

        Args:
            subdirectory (str): Subdirectory relative to the repo root to list. Defaults to ".".

        Returns:
            Dict containing result_id, subdirectory, directories, and files.
        """
        repo_root = Path.cwd().resolve()
        target_path = (repo_root / subdirectory).resolve()

        # Safety check: Prevent path traversal outside repo root
        try:
            target_path.relative_to(repo_root)
        except ValueError:
            return {"status": "error", "message": "Access restricted: Cannot list paths outside the repository root."}

        if not target_path.exists() or not target_path.is_dir():
            return {"status": "error", "message": f"Directory not found: {subdirectory}"}

        directories = []
        files = []

        for item in target_path.iterdir():
            # Ignore hidden items / common cache directories
            if item.name.startswith(".") or item.name in ("__pycache__", "venv", ".venv"):
                continue
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": sorted(directories),
            "files": sorted(files)
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    return [
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        sample_data_tool,
        describe_data_tool,
        list_files_tool
    ]

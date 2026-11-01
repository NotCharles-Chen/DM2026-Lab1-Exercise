import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def setup_session_with_fixture():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def test_check_missing_tool_fixture():
    session = setup_session_with_fixture()
    tools = make_tools(session)
    check_missing = next(t for t in tools if t.name == "check_missing_tool")
    
    # Known answer #2: Exactly 1 row has missing/empty text (the last row)
    res = check_missing.invoke({})
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]

def test_check_duplicates_tool_fixture():
    session = setup_session_with_fixture()
    tools = make_tools(session)
    check_duplicates = next(t for t in tools if t.name == "check_duplicates_tool")
    
    # Known answer #3: check_duplicates_tool without drop
    res = check_duplicates.invoke({"drop": False})
    assert res["duplicate_count"] == 1
    assert res["duplicate_indices"] == [6]
    
    # Set dummy DTM artifacts to test clearing behavior
    session.feature_matrix = "dummy_matrix"
    session.feature_names = ["dummy"]
    session.labels = ["dummy"]
    session.artifacts["count_vectorizer"] = "dummy_vectorizer"
    
    # Drop duplicates (keep=False removes all copies of duplicated rows)
    res_drop = check_duplicates.invoke({"drop": True})
    assert res_drop["dropped_count"] == 2
    assert res_drop["remaining_documents"] == 6
    assert len(session.dataframe) == 6
    # Index should be reset to 0..5
    assert list(session.dataframe.index) == list(range(6))
    
    # Downstream artifacts should be cleared
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.labels is None
    assert "count_vectorizer" not in session.artifacts

def test_check_duplicates_different_categories():
    # Test case: Same text ("always gamma delta") but different category label shouldn't be duplicate
    session = SessionState()
    df = pd.DataFrame({
        "text": ["always gamma delta", "always gamma delta"],
        "category_name": ["catA", "catB"]
    })
    session.dataframe = df
    
    tools = make_tools(session)
    check_duplicates = next(t for t in tools if t.name == "check_duplicates_tool")
    
    res = check_duplicates.invoke({"drop": False})
    assert res["duplicate_count"] == 0

def test_tokenize_tool_fixture():
    session = setup_session_with_fixture()
    tools = make_tools(session)
    tokenize = next(t for t in tools if t.name == "tokenize_tool")
    
    res = tokenize.invoke({})
    assert res["total_documents"] == 8
    assert "unigrams" in session.dataframe.columns
    assert len(session.dataframe["unigrams"]) == 8
    
    unigrams = session.dataframe["unigrams"].tolist()
    
    # Known answer #10 checks:
    # Doc 0: ["always", "alpha", "alpha", "alpha", "beta"]
    assert unigrams[0] == ["always", "alpha", "alpha", "alpha", "beta"]
    # Doc 3: ["always", "gamma", "gamma", "gamma", "delta"]
    assert unigrams[3] == ["always", "gamma", "gamma", "gamma", "delta"]
    # Doc 7: [] (empty list, NOT ["nan"] or non-empty)
    assert unigrams[7] == []

def test_sample_data_tool_fixture():
    session = setup_session_with_fixture()
    tools = make_tools(session)
    sample_data = next(t for t in tools if t.name == "sample_data_tool")
    
    # Known answer #13: n=4, random_state=42 returns exactly rows [1, 5, 0, 7]
    res = sample_data.invoke({"n": 4, "random_state": 42})
    assert res["sampled_count"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]

def test_describe_data_tool_fixture():
    session = setup_session_with_fixture()
    tools = make_tools(session)
    describe_data = next(t for t in tools if t.name == "describe_data_tool")
    
    res = describe_data.invoke({})
    overall = res["overall_stats"]
    by_cat = res["by_category_stats"]
    
    # Known answer #18 checks:
    # Overall: count=8, mean=19.875
    assert overall["count"] == 8
    assert pytest.approx(overall["mean"], abs=1e-3) == 19.875
    
    # Category catA mean=17.25, catB mean=22.5
    assert pytest.approx(by_cat["catA"]["mean"], abs=1e-3) == 17.25
    assert pytest.approx(by_cat["catB"]["mean"], abs=1e-3) == 22.5
    
    # Plotting check
    assert session.pending_figure is not None

def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files = next(t for t in tools if t.name == "list_files_tool")
    
    # Known answer #11: subdirectory="newdataset" returns directories: [], files: ["Reddit-stock-sentiment.csv"]
    res = list_files.invoke({"subdirectory": "newdataset"})
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

def test_inspect_data_tool():
    session = setup_session_with_fixture()
    
    # Insert a NaN row at the top to test NaN handling -> empty string ""
    nan_row = pd.DataFrame([{"text": np.nan, "category": 0, "category_name": "catA"}])
    session.dataframe = pd.concat([nan_row, session.dataframe], ignore_index=True)
    
    tools = make_tools(session)
    inspect_data = next(t for t in tools if t.name == "inspect_data_tool")
    
    max_chars = 10
    res = inspect_data.invoke({"n_rows": 2, "max_chars": max_chars})
    
    # 1. n=2 returns exactly 2 rows
    assert res["status"] == "success"
    assert res["n_rows_returned"] == 2
    assert len(res["rows"]) == 2
    
    # 2. NaN converted to empty string ""
    assert res["rows"][0]["text"] == ""
    
    # 3. Text preview length does not exceed max_chars + 3 ("...")
    for row in res["rows"]:
        assert len(str(row["text"])) <= max_chars + 3

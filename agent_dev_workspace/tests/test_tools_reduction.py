import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student's build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get reduce_dimensions_tool
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Run PCA reduction
    result = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Known answer 7: explained_variance_ratio [0.7532, 0.1965]
    expected_variance = [0.7532, 0.1965]
    assert np.allclose(result["explained_variance_ratio"], expected_variance, atol=1e-3)

    # Known answer 7: doc 0 coordinates (2.3669, 0.8871)
    doc0_coords = result["coordinates"][0]
    expected_doc0 = [2.3669, 0.8871]
    assert np.allclose(doc0_coords, expected_doc0, atol=1e-3)

    # Check that figure was generated and set on session.pending_figure
    assert session.pending_figure is not None


def test_binarize_labels_tool_fixture():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    result = binarize_labels_tool.invoke({})

    # 1. 驗證 2 類 (catA, catB) 得 2 個 One-Hot 欄位
    assert result["status"] == "success"
    assert len(result["one_hot_columns"]) == 2
    assert sorted(result["categories"]) == ["catA", "catB"]

    # 2. 驗證 session.dataframe 中生成的欄位
    cols = result["one_hot_columns"]
    for col in cols:
        assert col in session.dataframe.columns

    # 3. 驗證每列的 One-Hot sum 為 1
    row_sums = session.dataframe[cols].sum(axis=1)
    assert (row_sums == 1).all()

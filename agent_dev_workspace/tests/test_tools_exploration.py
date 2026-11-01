import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_known_answer_16():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM (this populates session.feature_matrix and session.artifacts['count_vectorizer'])
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known answer 16: doc 0 vs doc 1 -> 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc1_index": 0, "doc2_index": 1})
    assert pytest.approx(res_0_1["similarity"], abs=1e-3) == 0.9847

    # Known answer 16: doc 0 vs doc 3 -> 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc1_index": 0, "doc2_index": 3})
    assert pytest.approx(res_0_3["similarity"], abs=1e-3) == 0.0909

    # Known answer 16: doc 0 vs doc 7 (empty text) -> 0.0 (not NaN)
    res_0_7 = cosine_similarity_tool.invoke({"doc1_index": 0, "doc2_index": 7})
    assert res_0_7["similarity"] == 0.0
    assert not np.isnan(res_0_7["similarity"])


def test_feature_correlation_matrix_tool_known_answer_19():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    feature_correlation_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_correlation_tool.invoke({})
    terms = res["terms"]
    corr_matrix = res["correlation_matrix"]

    # Known answer 19: Terms sorted by variance descending: ['alpha', 'gamma', 'delta', 'beta', 'always']
    assert terms == ["alpha", "gamma", "delta", "beta", "always"]

    # Known answer 19: Pearson correlation between alpha (idx 0) and beta (idx 3) is 0.8885
    alpha_idx = terms.index("alpha")
    beta_idx = terms.index("beta")
    assert pytest.approx(corr_matrix[alpha_idx][beta_idx], abs=1e-3) == 0.8885

    # Plotting check
    assert session.pending_figure is not None

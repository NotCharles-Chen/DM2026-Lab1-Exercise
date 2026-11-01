import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer():
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

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["initial_features"] == 5
    assert res["dropped_features"] == ["always"]
    assert res["remaining_features"] == 4
    assert set(session.feature_names) == {"alpha", "beta", "delta", "gamma"}

    alpha_entry = next(item for item in res["top_remaining_terms"] if item["term"] == "alpha")
    assert np.isclose(alpha_entry["variance"], 1.1875, atol=1e-3)


def test_pearson_filter_tool_known_answer_6():
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Known answer 6: target_class="catB"
    result = pearson_filter_tool.invoke({"target_class": "catB"})
    top_terms = {item["term"]: item["pearson_r"] for item in result["top_terms"]}

    # Known answer 6: delta = 1.0000, gamma = 0.8307
    assert pytest.approx(top_terms["delta"], abs=1e-3) == 1.0000
    assert pytest.approx(top_terms["gamma"], abs=1e-3) == 0.8307


def test_spearman_filter_tool_known_answer_6():
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Known answer 6: target_class="catB"
    result = spearman_filter_tool.invoke({"target_class": "catB"})
    top_terms = {item["term"]: item["spearman_r"] for item in result["top_terms"]}

    # Known answer 6: delta = 1.0000, gamma = 0.9363
    assert pytest.approx(top_terms["delta"], abs=1e-3) == 1.0000
    assert pytest.approx(top_terms["gamma"], abs=1e-3) == 0.9363

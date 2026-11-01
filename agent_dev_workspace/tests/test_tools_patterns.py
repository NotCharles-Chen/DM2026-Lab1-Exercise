import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tfidf_fpgrowth():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Known answer 17: filtering_method="tfidf", algorithm="fpgrowth"
    # catA keeps 'alpha' (support 3), catB keeps 'gamma' (support 4)
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_catA["patterns"]) == 1
    assert "alpha" in res_catA["patterns"][0]["pattern"]
    assert res_catA["patterns"][0]["support"] == 3

    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert len(res_catB["patterns"]) == 1
    assert "gamma" in res_catB["patterns"][0]["pattern"]
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_zero_terms():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Known answer 17: variance and term_frequency keep zero terms on sample_fixture
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth"
    })
    assert res_var["patterns"] == []
    assert "no terms survived filtering" in res_var.get("note", "")


def test_mine_patterns_topk_and_maxfpgrowth():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Known answer 20: topk returns 1 pattern for both categories
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_topk["patterns"]) == 1
    assert "alpha" in res_topk["patterns"][0]["pattern"]

    # Known answer 20: maxfpgrowth returns zero patterns on single-term data
    res_maxfp = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_maxfp["patterns"] == []

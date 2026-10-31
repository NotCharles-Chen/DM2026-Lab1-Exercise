import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_tools_dtm():
    # 建立單一 SessionState 實例
    session = SessionState()

    # 1. 載入 premade load_dataset_tool 並讀取測試 csv 檔
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. 取得 tools_dtm.py 中的工具
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # 3. 測試 build_dtm_tool (Known Answer 4)
    dtm_result = build_dtm_tool.invoke({})
    
    # 斷言 session.feature_names 包含完整詞彙 (5 個詞)
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab
    
    # 斷言非零 cell 數量與稀疏度 (21 個非零值, sparsity_pct = 47.5%)
    assert dtm_result["non_zero"] == 21
    assert dtm_result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # 4. 測試 term_frequency_tool (Known Answer 14)
    tf_result = term_frequency_tool.invoke({"top_n": None})
    term_freqs = tf_result["term_frequencies"]

    # 斷言每項詞彙的總數
    assert term_freqs["alpha"] == 6
    assert term_freqs["always"] == 7
    assert term_freqs["beta"] == 3
    assert term_freqs["delta"] == 4
    assert term_freqs["gamma"] == 7

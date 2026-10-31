from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool():
    # 建立單一 SessionState 實例
    session = SessionState()
    
    # 從 premade_tools 載入現成的 load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 執行 load_dataset_tool 載入測試用 csv 檔
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 依 Known Answer 1 驗證傳回字典的內容
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
    
    # 驗證 session.dataframe 中的資料筆數與類別分佈
    assert len(session.dataframe) == 8
    
    # 驗證 session.dataframe 裡恰好有 1 列的 text 為空（NaN、空字串或全空白）
    empty_text_mask = session.dataframe["text"].isna() | (session.dataframe["text"].astype(str).str.strip() == "")
    assert empty_text_mask.sum() == 1

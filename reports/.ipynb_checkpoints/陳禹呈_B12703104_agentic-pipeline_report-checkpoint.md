# Lab 1 Agentic Pipeline 報告（20 Newsgroups，4 類）

這份報告整理我在 `DM2026-Lab1-AgenticPipeline.ipynb` 中，透過 agent 呼叫自己在 AgentDev 階段寫好的工具，對 20 Newsgroups 四個類別（alt.atheism、soc.religion.christian、comp.graphics、sci.med）跑完整條分析流程的結果。每一節先說做了什麼，再列出結果數字與 `result_id`，最後寫我對結果的解讀。所有數字都來自對話紀錄中工具實際回傳的值。

## 1. 資料載入與品質檢查

我先用 `load_dataset_tool` 載入與 Master notebook 相同的四個類別（`load_dataset_2`），得到 2,257 篇文章，和 Master 的 2,257 篇一致；各類別分別是 alt.atheism 480、soc.religion.christian 599、comp.graphics 584、sci.med 594 篇，四類數量相近，沒有明顯的類別不平衡。接著用 `check_missing_tool` 檢查空白或 NaN 文字（`check_missing_3`），結果 missing_count 為 0；再用 `check_duplicates_tool`、`drop=True` 檢查整列重複（`check_duplicates_4`），duplicate_count 為 0、dropped_count 為 0，資料仍是 2,257 筆。我的實作是 `keep=False`，一旦有重複會把每一份副本都刪掉，但這份資料完全沒有重複，所以這個設計在這裡沒有產生任何影響。

最後用 `describe_data_tool` 看文字長度（`describe_data_5`）：整體平均約 1,993 個字元，第一四分位數 795、第三四分位數 2,059，四分位距 1,264；依 Q3 + 1.5 × IQR 計算，長度超過約 3,955 字元就算離群值。平均長度最長的類別是 soc.religion.christian（2,263.18 字元），但單篇最長的文章在 comp.graphics，達 60,713 字元；alt.atheism 和 sci.med 也各有 51,239、37,009 字元的長文。

![各類別文字長度 box plot](../plots/陳禹呈_agentic-pipeline_plot_describe_data_5.png)

解讀：這份資料在「有沒有缺、有沒有重複」上很乾淨，不需要清理，但長度分佈極度右偏。從 box plot 可以看到四個類別的箱子都壓在 0 到 3,000 字元附近，上方卻拖著一長串離群點，最極端的幾篇是中位數的二、三十倍長。這對後續分析有兩個意義：第一，這些長文會在詞頻和變異數上佔很大的權重，一篇 60,000 字元的文章對某些詞的計數貢獻可能等於幾十篇普通文章；第二，之後做 PCA 時，變異數最大的方向很可能被這幾篇長文主導，而不是被主題差異主導。這一點在第 4 節會再回來驗證。

## 2. DTM 與詞頻

用 `build_dtm_tool`、`stop_words="english"` 建立文件-詞彙矩陣（`dtm_6`）：2,257 列 × 35,482 個詞，與 Master 的 35,482 完全相同；非零元素 258,503 個，總元素 80,082,874 個，稀疏度 99.6772%。再用 `term_frequency_tool` 看全域詞頻（`tf_7`），前 10 名依序是 edu（4,675）、subject（2,464）、lines（2,353）、god（2,216）、organization（2,192）、com（2,172）、writes（1,645）、people（1,604）、article（1,453）、don（1,400）。

解讀：這是整份報告最關鍵的發現。前 10 名裡真正和主題有關的只有 god 和 people 兩個詞；edu、com 來自電子郵件地址，subject、lines、organization 是郵件標頭的欄位名稱，writes 和 article 則是「某某 writes in article」這種引用格式的殘留。換句話說，詞頻排行榜量到的不是「這些新聞群組在討論什麼」，而是「這些文章是用什麼格式存的」。`stop_words="english"` 只能擋掉 the、of 這類一般停用詞，擋不掉這種資料來源特有的雜訊。而這些詞幾乎每篇都出現，計數又高，所以它們的變異數也大，不會被後面的變異數篩選移除；接下來第 3、4 節的結果都會看到同一個問題一再浮現。

## 3. 特徵篩選

用 `variance_filter_tool`、threshold 0.01 做變異數篩選（`variance_filter_8`），35,482 個詞中刪掉 30,393 個，剩下 5,089 個，和 Master 在同一門檻下的 5,089 個完全一致。被刪掉的多是只出現在單篇文章裡的數字與編碼（例如 0000001200、000usd、a7f81fd2f801023c 這類字串）和罕見拼字，這和我在 Master Q5 的觀察相同：被刪的詞大多只出現在一、兩篇文件裡，整欄幾乎全是 0，變異數自然接近 0，拿掉它們幾乎不損失區分類別的資訊。下圖是 `visualize_result_tool` 對 `variance_filter_8` 畫出的前 15 名變異數長條圖，jpeg 的變異數最高（約 25），其次是 edu、image、god、com、jesus、graphics。

![變異數篩選結果：變異數最高的 15 個詞](../plots/陳禹呈_agentic-pipeline_plot_turn50.png)

接著以 comp.graphics 為 one-vs-rest 目標計算相關性。`pearson_filter_tool`（`pearson_filter_9`）r 最高的五個詞是 thanks（0.2778）、keywords（0.2330）、looking（0.2135）、vga（0.2111）、video（0.2095），負相關最強的是 writes（−0.2337）、people（−0.2089）、rutgers（−0.1766）。`spearman_filter_tool`（`spearman_filter_10`）前五名則是 graphics（0.5211）、files（0.3167）、image（0.3086）、3d（0.3060）、program（0.2911）。

解讀：兩個排行榜差異很大。Pearson 的前五名裡只有 vga、video 和圖學有關，thanks、looking、keywords 是發問求助的用語，反映的是 comp.graphics 版上「有人在找東西、謝謝大家」的發文習慣，不是主題詞。Spearman 的前五名則全部是圖學詞彙，graphics 本身更拿到 0.5211 的高分，但它連 Pearson 的前五名都進不了（這和我在 Master Q6 跑出的結果一致，那裡 graphics 的 spearman_r 也約 0.52，且不在 Pearson 前 10 名）。原因和我在 Master Q6 寫的一樣：Pearson 量的是原始計數和 0/1 類別指標之間的線性關係，一篇文章把 graphics 寫了幾十次，會把整個線性配適拉歪；Spearman 先把計數轉成名次，只看「graphics 在 comp.graphics 文章裡是不是一致地排得比較前面」，對極端計數不敏感。在長度差異這麼大的資料上（第 1 節），Spearman 顯然是比較穩健的選擇。另外值得注意的是變異數圖裡 jpeg、edu、com 排在前面，再次印證第 2 節的判斷：標頭詞變異數大，變異數篩選留得住它們。

## 4. 頻繁模式、降維與相似度

頻繁模式探勘用 `mine_patterns_tool`，對 alt.atheism 以 tfidf 篩選詞彙、再以 topk（k=10）探勘（`mine_patterns_11`）。挖出的 10 個 pattern 依序是 lines（support 480）、subject（480）、lines subject（480）、organization（453）、subject lines organization（453）、lines organization（453）、subject organization（453）、edu（407）、subject lines edu（407）、lines edu（407）。我原本選的是 sci.med，但該類別的詞彙裡含有非 ASCII 字元，而我的工具在寫 PAMI 需要的暫存檔時沒有指定 utf-8 編碼，在 Windows 上觸發了 cp950 的 UnicodeEncodeError，所以改挖 alt.atheism。

解讀：這 10 個 pattern 沒有一個和無神論有關，全部是 lines、subject、organization、edu 四個標頭詞的組合；lines 和 subject 的 support 是 480，正好等於 alt.atheism 的文章數，代表每一篇都有這兩個詞。這是第 2 節問題最極端的呈現：頻繁模式探勘找的是「最常一起出現的詞組」，而郵件標頭欄位本來就每篇一起出現，自然把所有真正的主題 pattern 都排擠掉了。至於 sci.med 的編碼錯誤，是我工具實作上的缺陷，不是資料問題；修法是在開暫存檔時加上 `encoding="utf-8"`。

降維用 `reduce_dimensions_tool`、method="pca"（`reduce_dimensions_12`），對變異數篩選後 5,089 維的 DTM 投影到二維，得到下面的散佈圖。圖上四個類別重疊在一起、分不開，另外有少數 comp.graphics 的點遠離其他所有文章。

![PCA 降維結果](../plots/陳禹呈_agentic-pipeline_plot_reduce_dimensions_12.png)

解讀：從圖上看，絕大多數文章擠在原點附近一團，四個類別的標記完全混在一起；把座標軸範圍拉大的是右側與右下方四、五個 comp.graphics 的點，第一維座標落在大約 100 到 300 之間。這和我在 Master Q9 看到的 PCA 面板是同一個現象：PCA 保留的是變異數最大的方向，而在原始計數矩陣裡，變異數最大的方向是「文章有多長」，不是「文章在講什麼」。根據第 1 節的長度統計（comp.graphics 有 60,713 字元的最長文章，整體離群門檻約 3,955 字元），那幾個遠離原點的 comp.graphics 點可能就是那幾篇超長文章，不過工具沒有回傳每個點對應的文件索引，這一點我無法在紀錄裡直接確認。要看類別結構，需要像 t-SNE 或 UMAP 這類保留局部鄰近關係的非線性方法，或至少先對計數做正規化。

相似度用 `cosine_similarity_tool` 比較 doc 0 和 doc 1（`cosine_similarity_13`）。這兩篇文章都屬於 comp.graphics，是同一類別內的比較，餘弦相似度為 0.0582。

解讀：同樣是 comp.graphics 的兩篇文章，相似度只有 0.058，接近 0。這在 35,482 維的原始計數空間裡是正常的：矩陣有 99.68% 是零，兩篇文章即使談同一主題，實際用到的詞重疊也很少，而兩篇都有的那幾個詞（很可能又是標頭詞）在高維度下對夾角的貢獻有限。這個數字提醒我，在這種稀疏表示下直接用餘弦相似度判斷「是否同類」並不可靠，需要先降維或改用 TF-IDF 權重。

## 5. 結論與改進

數量面的結果都和 Master 對得上：2,257 篇文章、35,482 個詞、變異數篩選後 5,089 個詞，這證明我寫的 `load_dataset`、`build_dtm`、`variance_filter` 等工具實作是正確的。但內容面的結果一致指向同一個問題：詞頻前 10 名（`tf_7`）、Pearson 排行榜（`pearson_filter_9`）、頻繁模式（`mine_patterns_11`）都被 edu、subject、lines、organization 這些郵件標頭詞佔據，PCA（`reduce_dimensions_12`）則被少數超長文章主導。這些工具本身沒有算錯，是餵進去的文字沒有先清乾淨。

如果只能改一件事，我會優先在 `load_dataset_tool` 或建 DTM 之前加一個前處理步驟，移除郵件 header、footer 與引用行（sklearn 的 `fetch_20newsgroups` 本身就有 `remove=('headers','footers','quotes')` 參數），然後重跑第 2 到第 4 節。預期詞頻前 10 名會換成 god、people 這類主題詞，alt.atheism 的頻繁模式會出現 god、atheism、religion 的組合，PCA 的離群點也會收斂。第二優先是把 `mine_patterns_tool` 的暫存檔改成 utf-8，讓 sci.med 也能探勘。第三是在相關性篩選上以 Spearman 取代 Pearson，因為這份資料的長度差異太大，線性相關容易被極端計數帶偏。

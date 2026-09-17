# emotion2vec+ 音檔情緒分析器

Windows 桌面工具：使用 Faster-Whisper 分段演算法，再將每個段落交給 emotion2vec+ 分析情緒。可指定單一資料夾批次分析，第一筆輸出永遠是標準音檔。

Whisper、emotion2vec+ 與 Excel 匯出會在獨立分析行程執行，避免大型模型載入影響 Qt 視窗；分析行程若意外終止，GUI 會保留並顯示最後的錯誤資訊。

## 安裝

```powershell
python -m pip install -r requirements.txt
```

`imageio-ffmpeg` 會提供 FFmpeg；若已安裝系統 FFmpeg 或將 `ffmpeg.exe` 放在 `emotion_analyzer/resources/`，程式也會優先使用它。emotion2vec+ 預設模型為 `iic/emotion2vec_plus_large`，也可在介面改選 `emotion2vec_plus_base` 或 `emotion2vec_plus_seed`；第一次使用尚未安裝的模型時可能需要下載。

若 `models/faster-whisper-large-v3/` 或對應的 `models/emotion2vec_plus_large/`、`models/emotion2vec_plus_base/`、`models/emotion2vec_plus_seed/` 存在，程式會優先使用本機模型，不需重新下載；缺少本機模型時才使用原本的遠端模型名稱。目前專案隨附的是 emotion2vec+ large。

## 執行

```powershell
python main.py
```

介面參數：標準音檔、分段文字檔、需分析的檔案資料夾、Whisper 相似度門檻、emotion2vec+ 模型版本與背景雜音處理。分段文字檔需為 UTF-8；選取檔案後會自動解析並填入段落數量。

「背景雜音處理」預設開啟，會在 Whisper 分段與 emotion2vec+ 分析前，以保守設定降低持續性的風扇、冷氣與底噪。它不適合修復削波、強烈回音、背景音樂或其他人聲。

「重置介面」會清空各頁籤的輸入、表格、進度與訊息，將 Whisper 門檻恢復為 `0.30`，並重新啟用背景雜音處理；不會刪除原始音檔、`chopped/` 切段快取或已輸出的 Excel。

分段文字檔可以每行一個段落，也可以使用括號格式：

```text
{("第一段文字"),("第二段文字"),(“第三段文字”)}
```

括號格式可使用半形雙引號 `"..."`、中文彎引號 `“...”` 或中文引號 `「...」`，也相容 `“..."` 這類左右引號混用的文字；段落之間需以逗號分隔。

分段流程會使用 `faster-whisper large-v3` 的 word timestamps、中文轉換（OpenCC）、中文比例與重複內容過濾、RapidFuzz 相似度、75% 動態門檻、15 秒最早候選群組及 1 秒前後緩衝，並將預處理後的切段 WAV 快取在來源音檔旁的 `chopped/`。快取 manifest 會記錄來源檔案、文字、分段參數與去雜音 profile；其中任一項改變時會自動重新切段。

## Excel 輸出

輸出為 `emotion_analysis_result.xlsx`。第一張「情緒分析結果」工作表保留原本設計：

`音檔名稱`、`第 N 段情緒原始分數`、`第 N 段主情緒`、`第 N 段跟標準音檔分析的差距`。

標準音檔列位於第一列，各段與自身的差距為 0；其他音檔使用該段 emotion2vec+ 機率向量與標準音檔的 L2 距離。新版報表另外包含：

- 「分析摘要」：每位受試者的 L1/L2 平均值、總計、有效段數、最大偏離段落與段落趨勢圖。
- 「原始分數」：每段九維 emotion2vec+ 分數、前三高情緒、時間、匹配信心、L1/L2 距離與最大變化方向。
- 「報表資訊」：報表版本、模型、門檻、去雜音設定及使用的距離算法。

L1 是與 Friendly Support 相同的 Manhattan 情緒截距；L2 是原有 oeeana 差距。

## 前後測資料、分組與互動分析

桌面程式的「前後測資料與分組」頁籤可在前測、後測各載入一份以上的 Excel，並將報表合併成資料池。支援：

- 新版 oeeana、舊版 oeeana 與 Friendly Support 報表。
- 一次複選多份 Excel，並保留每筆資料的來源報表。
- 依受試者名稱、資料順序或手動一對一配對。
- 在表格中一次選取多列，批次指定為實驗組、對照組、未分組或排除。
- 匯入後自動檢查缺少前／後測、距離數值、段落或分析錯誤；缺損資料會標記原因並鎖定為排除，避免誤納入統計。
- L1 或 L2 指標比較。
- 整體與分組的樣本數、平均、標準差及 paired t-test。
- 以 Welch t 檢定比較實驗組與對照組的「後測－前測」變化量。
- 以 ANCOVA 檢驗校正前測後的實驗組／對照組後測差異。
- 輸出含配對 t 檢定、兩項組間效果檢定、比較摘要、配對明細、分段趨勢圖與來源清單的 Excel 報告。

「互動分析」頁籤會顯示整體／實驗組／對照組統計、ANCOVA、Welch t 檢定、各受試者前後測長條圖，以及所選受試者的各段趨勢折線圖。切換頁籤不會中斷背景音檔分析。

匯入報表時，受試者名稱統一取第一個半形連字號 `-` 之前的內容作為學生 ID。例如 `A001-王小明-前測.wav` 的 ID 是 `A001`；完整音檔名稱與來源報表仍會保留。

不同舊報表可能只包含單一距離算法。若選擇報表沒有的指標，程式會要求改用可用指標，不會把 L1 與 L2 混合計算。

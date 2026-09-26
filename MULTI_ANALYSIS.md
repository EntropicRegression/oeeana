# 多段 Analysis Set

同一位受試者若有多段錄音，且每段錄音需要不同的標準音檔與分段文字，請使用多段模式：

```powershell
python main.py --multi
```

## 資料配置

多段模式把每一組「標準音檔 + 分段文字 + 來源錄音」視為一個 Analysis Set。每個 Set 可以有不同的段落數。

建議依錄音段落分資料夾：

```text
pretest/
├─ part1/
│  ├─ A001-part1.wav
│  ├─ A002-part1.wav
│  └─ ...
├─ part2/
│  ├─ A001-part2.wav
│  ├─ A002-part2.wav
│  └─ ...
├─ part3/
└─ part4/

reference/
├─ part1-standard.wav
├─ part2-standard.wav
├─ part3-standard.wav
└─ part4-standard.wav

transcript/
├─ part1.txt
├─ part2.txt
├─ part3.txt
└─ part4.txt
```

例如新增四個 Analysis Set：

| Set | 來源資料夾 | 標準音檔 | 分段文字 |
| --- | --- | --- | --- |
| part1 | `pretest/part1/` | `reference/part1-standard.wav` | `transcript/part1.txt` |
| part2 | `pretest/part2/` | `reference/part2-standard.wav` | `transcript/part2.txt` |
| part3 | `pretest/part3/` | `reference/part3-standard.wav` | `transcript/part3.txt` |
| part4 | `pretest/part4/` | `reference/part4-standard.wav` | `transcript/part4.txt` |

每個 Set 的段落數會從自己的文字檔自動解析，不需要四組使用相同段落數。

## 執行流程

1. 按「新增 Analysis Set」加入所有段落。
2. 按「1. 全部切段」。各 Set 會使用自己的標準音檔與文字執行 Whisper 切段。
3. 若有缺失片段，依訊息中的完整路徑把人工切割 WAV 補到對應的 `chopped/`。
4. 按「2. 全部情緒分析」。
5. 每個 Set 會在自己的來源資料夾輸出 `emotion_analysis_result_<Set名稱>.xlsx`。

多段模式會依序執行不同設定的 Set，避免把 part1 的文字或標準音檔誤套到 part2。原本 `python main.py` 的單組分析模式保持不變。

## 受試者命名

後續前後測功能仍以第一個半形連字號 `-` 前的內容作為受試者 ID，因此建議：

```text
A001-part1.wav
A001-part2.wav
A001-part3.wav
A001-part4.wav
```

這些檔案在後續報表中都會識別為 `A001`。

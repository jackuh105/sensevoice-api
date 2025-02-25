# 語音轉文字（STT）API 服務器項目計劃

## 概述

本項目作為LLM協作開發試用項目，在開發過程中使用LLM生成部分代碼、注釋和說明文件。
此項目旨在開發一個基於 `FastAPI` 的語音轉文字（STT）API 服務器，使用 `SenseVoice` 作為 STT 提供者，提供 `/transcribe` 和 `/status` 兩個端點。

## 技術棧

- 框架: FastAPI
- STT 提供者: SenseVoice
- 依賴:
    - `funasr`（用於 SenseVoice）
    - `pydub`（用於音頻格式驗證與轉換）
    - `fastapi[all]`（包含文件上傳支持）
- 運行環境: macOS，設備設為 mps

## 端點設計

1. /transcribe

- 方法: POST
- 參數:
    - `audio_file`: 音頻文件（必填，文件上傳）
    - `model_name`: 模型名稱（必填，字符串，例如 "iic/SenseVoiceSmall"）
- 功能:
    - 驗證:
    - 檢查 `audio_file` 是否存在，若無則返回 400 Bad Request。
    - 驗證音頻格式，僅接受 `wav`、`mp3`、`opus`，否則返回 415 Unsupported Media Type。
    - 檢查採樣率，若高於 16000 Hz 則轉換為 16000 Hz，若低於則返回 422 Unprocessable Entity。
- 處理:
    - 使用 `SenseVoice` 的 `AutoModel.generate` 轉錄音頻，設置 `language='auto'` 和 `use_itn=True`。
- 返回:
    - 成功：200 OK，JSON 格式 `{ "status": "success", "transcription": "轉錄文本" }`
    - 無語音檢測到：200 OK，`{ "status": "success", "transcription": "No speech detected" }`
    - 異常：500 Internal Server Error，`{ "status": "error", "message": "錯誤訊息" }`

2. /status

- 方法: GET
- 功能: 健康檢查，返回服務器狀態。
- 返回: 204 No Content（服務器正常運行）

## 實現步驟
1. 環境設置:
    - 安裝依賴：更新 `requirements.txt`。
    - 初始化 `FastAPI` 應用。
2. 服務器實現:
    - 加載 `SenseVoice` 模型作為全局變量。
    - 實現 `/transcribe` 端點，包括驗證和轉錄邏輯。
    - 實現 `/status` 端點。
3. 測試:
    - 使用樣本音頻文件測試 `/transcribe`，包括有效和無效輸入。
    - 驗證 `/status` 返回正確狀態碼。
4. 部署:
    - 使用 `uvicorn` 運行服務器。

## 文件結構

```
stt-api/
├── main.py           # FastAPI 應用入口
├── requirements.txt  # 依賴文件
├── example_usage.py  # 參考文件
└── project-plan.md   # 本計劃文件
```

## 注意事項
- 確保音頻處理不會阻塞主線程，可考慮異步處理。
- 記錄異常日誌以便調試。
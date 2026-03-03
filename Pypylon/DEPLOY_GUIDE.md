# 部署指南：FastAPI 影像處理系統 → Render

## 步驟一：建立 GitHub Repository

1. 登入 [GitHub](https://github.com)
2. 點右上角 **「+」→「New repository」**
3. 設定：
   - Repository name: `image-processing-api`（或你喜歡的名稱）
   - 選擇 **Public** 或 **Private**
   - **不要**勾選 "Add a README file"（我們已有檔案）
4. 點 **Create repository**

## 步驟二：推送程式碼到 GitHub

在你的 `Pypylon` 資料夾中打開終端機（CMD 或 PowerShell），執行：

```bash
cd C:\Users\user\PycharmProjects\Pypylon\Pypylon

# 初始化 Git
git init

# 加入檔案（.gitignore 會自動排除不需要的檔案）
git add test_MCP.py requirements.txt render.yaml .gitignore static/

# 提交
git commit -m "Initial commit: FastAPI image processing API for Render"

# 連結遠端倉庫（把 YOUR_USERNAME 換成你的 GitHub 帳號）
git remote add origin https://github.com/YOUR_USERNAME/image-processing-api.git

# 推送
git branch -M main
git push -u origin main
```

## 步驟三：在 Render 部署

1. 登入 [Render](https://render.com)（可用 GitHub 帳號登入）
2. 點 **「New +」→「Web Service」**
3. 選擇 **「Build and deploy from a Git repository」**
4. 連結你的 GitHub 帳號，選擇剛建立的 `image-processing-api` repo
5. Render 會自動偵測 `render.yaml` 設定，確認以下資訊：
   - **Name**: `pypylon-fastapi-web`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn test_MCP:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
6. 點 **「Create Web Service」**

## 步驟四：等待部署完成

- Render 會自動安裝套件並啟動服務
- 部署完成後會得到一個網址，格式類似：
  `https://pypylon-fastapi-web.onrender.com`
- 把這個網址分享給其他人就可以使用了！

## 可用的網址

| 網址 | 說明 |
|------|------|
| `/` | 網頁介面（影像處理系統） |
| `/docs` | FastAPI 自動產生的 API 文件 |
| `/health` | 健康檢查端點 |
| `/mcp` | MCP 端點 |

## 注意事項

- **Free Plan 限制**：Render 免費方案的服務會在 15 分鐘無流量後自動休眠，下次訪問需要約 30 秒喚醒
- **記憶體限制**：免費方案有 512MB 記憶體限制，處理超大圖片可能會失敗
- **暫存資料**：圖片暫存在記憶體中，服務重啟後會清空
- **無相機功能**：雲端環境無法使用 Basler 相機，已改為圖片上傳方式

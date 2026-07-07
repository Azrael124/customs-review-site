# 清关文件复核系统

基于 Streamlit 的清关文件自动化复核工具，利用 AI 语义比对技术检查清关文件数据一致性。

## 功能

- 📄 **标准文件基准** — 上传 PDF 格式的标准清关文件作为比对参照
- 📑 **多文件批量复核** — 支持同时上传多份 PDF/图片文件进行比对
- 🔍 **智能提取** — pdfplumber 提取文本 PDF + PaddleOCR 处理扫描件/图片
- 🤖 **AI 语义比对** — DeepSeek API 进行深度语义级数据比对
- 📊 **结构化结果** — 一致/不一致项分类展示，严重程度分级
- 📥 **Excel 导出** — 比对报告一键导出为 Excel 文件

## 技术栈

- **框架**: Streamlit
- **PDF 提取**: pdfplumber
- **OCR**: PaddleOCR + PyMuPDF
- **AI**: DeepSeek API (V3 / R1)
- **导出**: openpyxl

## 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行应用
streamlit run app.py
```

访问 http://localhost:8501

## Streamlit Cloud 部署

1. Fork / 推送此仓库到 GitHub
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 点击 "New app"，选择此仓库和 `app.py`
4. 在 **Advanced settings** → **Secrets** 中添加：
   ```
   DEEPSEEK_API_KEY = "sk-your-api-key"
   ```
5. 点击 "Deploy"

## 使用方式

1. 在侧边栏输入 DeepSeek API Key（或通过环境变量自动加载）
2. 上传标准文件（PDF 格式）
3. 上传待复核文件（支持 PDF / PNG / JPG 等）
4. 点击「开始复核」
5. 查看比对结果，导出 Excel 报告

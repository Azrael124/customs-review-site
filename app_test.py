"""清关文件复核系统 — 最小化兼容版本"""
import streamlit as st

st.set_page_config(page_title="清关文件复核", page_icon="🛃", layout="wide")

st.title("🛃 清关文件复核系统")
st.success("✅ 部署成功！系统运行正常。")

# 检查依赖
deps = {}
try:
    import pdfplumber; deps["pdfplumber"] = "✅"
except: deps["pdfplumber"] = "❌"
try:
    import fitz; deps["PyMuPDF"] = "✅"
except: deps["PyMuPDF"] = "❌"
try:
    import openai; deps["openai"] = "✅"
except: deps["openai"] = "❌"
try:
    import pandas; deps["pandas"] = "✅"
except: deps["pandas"] = "❌"
try:
    import openpyxl; deps["openpyxl"] = "✅"
except: deps["openpyxl"] = "❌"

st.subheader("依赖状态")
for name, status in deps.items():
    st.write(f"{status} {name}")

# PaddleOCR 可选
try:
    from paddleocr import PaddleOCR
    st.success("🎉 PaddleOCR 可用（高级 OCR 功能已启用）")
except ImportError:
    st.info("ℹ️ PaddleOCR 未安装（扫描件 OCR 功能不可用，文本 PDF 不受影响）")

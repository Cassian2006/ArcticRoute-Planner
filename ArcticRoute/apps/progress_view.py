import streamlit as st
import json
import time
import os

st.set_page_config(page_title="ArcticRoute Progress", layout="wide")
path = st.text_input("progress.jsonl 路径", value="ArcticRoute/data_processed/ice_forecast/progress_202412.jsonl")
ph = st.empty()

placeholder_tbl = st.empty()

while True:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-1000:]
            evs = [json.loads(x) for x in lines if x.strip()]
            blocks_done = sum(1 for e in evs if e.get("kind") == "block_done")
            mem = next((e.get("mem") for e in reversed(evs) if e.get("mem") is not None), None)
            cpu = next((e.get("cpu") for e in reversed(evs) if e.get("cpu") is not None), None)
            block_idx = next((e.get("block_idx") for e in reversed(evs) if e.get("block_idx") is not None), None)
            blocks_total = next((e.get("blocks_total") for e in reversed(evs) if e.get("blocks_total") is not None), None)
            pixels_done = next((e.get("pixels_done") for e in reversed(evs) if e.get("pixels_done") is not None), None)
            pixels_total = next((e.get("pixels_total") for e in reversed(evs) if e.get("pixels_total") is not None), None)

            ph.markdown(f"**Blocks done:** {blocks_done}/{blocks_total or '--'} | **CPU:** {cpu if cpu is not None else '--'}% | **MEM:** {mem if mem is not None else '--'}% | **Current block:** {block_idx or '--'} | **Pixels:** {pixels_done or '--'}/{pixels_total or '--'}")
        else:
            ph.markdown("文件不存在：" + path)
    except Exception as e:
        ph.markdown(f"读取失败: {e}")
    time.sleep(2)


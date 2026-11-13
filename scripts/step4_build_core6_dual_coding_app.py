#!/usr/bin/env python3
"""
Builds the Unified Single-File Dual-Author Audit Workbench (`dual_author_audit_workbench.html`)
for Wenzhuo Liu and Shiling Yu to independently verify:
  Tab 1: 182 evaluations x 6 core protocol fields (F03, F04, F07, F08, F12, F13) [strictly blind]
  Tab 2: 182 claimed ΔmAP@0.5 pairs cleaning & same-paper ablation baseline verification (with XML tables)
  Tab 3: 100 non-PMC external literature screening sample (60 primary + 40 reserve)

Also preserves/regenerates blank CSV templates:
  - `docs/coding_v1/human_dual_coding/core6_182_blank_coder1.csv`
  - `docs/coding_v1/human_dual_coding/core6_182_blank_coder2.csv`
  - `docs/coding_v1/human_dual_coding/dual_author_audit_workbench.html`
  - `docs/coding_v1/human_dual_coding/mobile_core6_dual_coding_182.html` (synced copy)
"""

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
CODING_DIR = ROOT_DIR / "docs" / "coding_v1" / "out_v2"
XML_DIR = ROOT_DIR / "docs" / "fulltext_v1" / "xml"
DELTAS_CSV = ROOT_DIR / "results" / "step4_claimed_deltas_182.csv"
CAND_CSV = ROOT_DIR / "docs" / "search_europepmc_20261030" / "candidates.csv"
NONPMC_CSV = ROOT_DIR / "docs" / "外部检索包_20261111" / "预生成_非PMC期刊60篇主样本与40篇替补下载清单.csv"
OUT_DIR = ROOT_DIR / "docs" / "coding_v1" / "human_dual_coding"

CORE_FIELDS = [
    ("F03", "F03 Split Manifest (公开清单或固定种子+可重建代码)"),
    ("F04", "F04 Duplicate Audit (重复/近重复图像检查或剔除)"),
    ("F07", "F07 Split Identity (明确区分验证集 vs 测试集)"),
    ("F08", "F08 Val/Test Separation (选模与最终测试集物理分离)"),
    ("F12", "F12 Random Seed (明确披露随机种子数值)"),
    ("F13", "F13 Repeated Runs (多次运行并报告 SD/SE/CI/范围)"),
]

SUSPICIOUS_NAME_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?|√|all|overall|average|mean|ours?)$|"
    r"\b(?:map|ap@?50|ap50|recall|precision|fps|params|flops|gflops|r/%|"
    r"voc2007|voc|coco|gc10|neu|adam|sgd|adamw|ws|"
    r"spurious\s+copper|mouse\s+bite|exhaust\s+valve|rolled|scratches|"
    r"crazing|inclusion|patches|pitted|pse|ot)\b",
    re.I,
)


def load_titles() -> dict:
    pmcid_to_meta = {}
    if CAND_CSV.exists():
        with CAND_CSV.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                pmcid = (r.get("pmcid") or "").strip()
                if pmcid:
                    pmcid_to_meta[pmcid] = {
                        "title": r.get("title", ""),
                        "year": r.get("year", ""),
                        "journal": r.get("journal", ""),
                        "doi": r.get("doi", ""),
                    }
    return pmcid_to_meta


def _tag_name(elem) -> str:
    return elem.tag.split("}")[-1] if elem.tag else ""


def _clean_text(elem) -> str:
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def extract_xml_context(xml_path: Path) -> dict:
    if not xml_path.exists():
        return {"snippets": "", "tables": []}
    root = ET.parse(xml_path).getroot()

    kw_primary = [
        "split", "ratio", "train", "validation", "val set", "test set", "fold",
        "seed", "deterministic", "duplicate", "deduplicat", "hash", "leakage",
        "repeat", "independent run", "runs", "standard deviation", "mean ±",
        "github", "gitlab", "zenodo", "figshare", "code availability",
        "data availability", "available upon", "weights", "checkpoint",
    ]
    hits = []
    seen = set()
    for elem in root.iter():
        tname = _tag_name(elem)
        if tname in {"p", "abstract", "notes"}:
            txt = _clean_text(elem)
            if not (25 <= len(txt) <= 1800) or txt in seen:
                continue
            low = txt.lower()
            score = sum(1 for k in kw_primary if k in low)
            if score >= 1:
                seen.add(txt)
                hits.append((score, txt))
    # Sort slightly prioritizing data-availability / seed / split dense paragraphs while preserving order on ties
    hits.sort(key=lambda x: -x[0])
    snippets = "\n\n".join(h[1] for h in hits[:12])

    tables = []
    for idx, tw in enumerate([e for e in root.iter() if _tag_name(e) == "table-wrap"], 1):
        lbl, cap = "", ""
        rows = []
        for ch in tw.iter():
            tn = _tag_name(ch)
            if tn == "label" and not lbl:
                lbl = _clean_text(ch)
            elif tn == "caption" and not cap:
                cap = _clean_text(ch)
            elif tn == "tr":
                cells = [_clean_text(c)[:45] for c in ch if _tag_name(c) in {"th", "td"}]
                if cells:
                    rows.append(cells[:12])
        if not rows:
            continue
        tb_text = f"{lbl} {cap} " + " ".join(" ".join(r) for r in rows[:4])
        tb_low = tb_text.lower()
        rel_score = sum(
            2 if k in tb_low else 0
            for k in ["map", "ap50", "ap@0.5", "neu", "gc10", "ablation", "baseline", "yolo", "comparison"]
        )
        tables.append((rel_score, idx, {
            "label": lbl or f"Table {idx}",
            "caption": cap[:220],
            "rows": rows[:22],
        }))
    tables.sort(key=lambda x: (-x[0], x[1]))
    top_tables = [t[2] for t in tables[:4]]
    return {"snippets": snippets, "tables": top_tables}


def load_claimed_csv() -> dict:
    lookup = {}
    if DELTAS_CSV.exists():
        with DELTAS_CSV.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                lookup[(r.get("coding_file", ""), r.get("dataset", ""))] = r
    return lookup


def load_nonpmc_csv() -> list[dict]:
    rows = []
    if NONPMC_CSV.exists():
        with NONPMC_CSV.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "sample_id": r.get("sample_id", ""),
                    "role": r.get("role", ""),
                    "year": r.get("year", ""),
                    "doi": r.get("doi", ""),
                    "doi_url": r.get("doi_url", ""),
                    "title": r.get("title", ""),
                    "journal": r.get("journal", ""),
                    "queries": r.get("queries", ""),
                })
    return rows


def detect_delta_flags(base_m: str, prop_m: str, base_v, prop_v, delta_v, f14: str) -> list[str]:
    flags = []
    bm = (base_m or "").strip()
    pm = (prop_m or "").strip()
    if not bm or not pm or delta_v is None or delta_v == "":
        flags.append("缺失基线/所提模型或数值")
    if bm and SUSPICIOUS_NAME_RE.search(bm):
        flags.append("基线名疑似非模型(指标/类别/符号)")
    if pm and SUSPICIOUS_NAME_RE.search(pm):
        flags.append("所提名疑似非模型(指标/类别/符号)")
    if bm and pm and bm.lower() == pm.lower():
        flags.append("基线与所提模型同名")
    try:
        dv = float(delta_v)
        if dv <= 0:
            flags.append(f"非正提升(Δ={dv:+.2f} pp)")
        elif dv > 15.0:
            flags.append(f"异常超高增益(Δ={dv:+.2f} pp)")
    except (TypeError, ValueError):
        pass
    if f14 in {"ambiguous", "not_reported_in_paper"}:
        flags.append(f"F14基线来源={f14}")
    return flags


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_map = load_titles()
    delta_lookup = load_claimed_csv()
    nonpmc_items = load_nonpmc_csv()

    xml_cache = {}
    items = []
    csv_rows = []
    eval_idx = 0

    for f in sorted(CODING_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("eligibility") != "include":
            continue
        pmcid = d["pmcid"]
        m_info = meta_map.get(pmcid, {})
        if pmcid not in xml_cache:
            xml_cache[pmcid] = extract_xml_context(XML_DIR / f"{pmcid}.xml")
        xctx = xml_cache[pmcid]

        for ds in d.get("datasets", []):
            ds_name = ds.get("dataset")
            if ds_name not in {"NEU-DET", "GC10-DET"}:
                continue
            eval_idx += 1
            eval_id = f"EVAL_{eval_idx:03d}"
            dl = delta_lookup.get((f.name, ds_name), {})

            base_m = dl.get("baseline_model", "")
            prop_m = dl.get("proposed_model", "")
            base_v = dl.get("baseline_map50", "")
            prop_v = dl.get("proposed_map50", "")
            delta_v = dl.get("claimed_delta_map50_pp", "")
            f14_src = dl.get("F14_baseline_source", ds.get("fields", {}).get("F14", {}).get("status", ""))
            flags = detect_delta_flags(base_m, prop_m, base_v, prop_v, delta_v, f14_src)

            items.append({
                "eval_id": eval_id,
                "coding_file": f.name,
                "pmcid": pmcid,
                "dataset": ds_name,
                "year": m_info.get("year", ""),
                "journal": m_info.get("journal", ""),
                "doi": m_info.get("doi", ""),
                "title": m_info.get("title", ""),
                "proposed_model": prop_m,
                "proposed_map50": prop_v,
                "baseline_model": base_m,
                "baseline_map50": base_v,
                "pre_extracted_delta_pp": delta_v,
                "F14_baseline_source": f14_src,
                "delta_location": dl.get("location", ""),
                "delta_evidence": dl.get("verbatim_evidence", ""),
                "flags": flags,
                "xml_snippets": xctx["snippets"],
                "xml_tables": xctx["tables"],
            })

            csv_rows.append({
                "eval_id": eval_id,
                "pmcid": pmcid,
                "dataset": ds_name,
                "year": m_info.get("year", ""),
                "doi": m_info.get("doi", ""),
                "title": m_info.get("title", ""),
                "F03_status": "",
                "F04_status": "",
                "F07_status": "",
                "F08_status": "",
                "F12_status": "",
                "F13_status": "",
                "verified_baseline_map50": base_v,
                "verified_proposed_map50": prop_v,
                "verified_delta_map50_pp": delta_v,
                "coder_notes": "",
            })

    fieldnames = list(csv_rows[0].keys())
    for fname in ["core6_182_blank_coder1.csv", "core6_182_blank_coder2.csv"]:
        with (OUT_DIR / fname).open("w", encoding="utf-8-sig", newline="") as f_out:
            w = csv.DictWriter(f_out, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)

    payload_items = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    payload_nonpmc = json.dumps(nonpmc_items, ensure_ascii=False).replace("</", "<\\/")

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper 2 双作者独立审核工作台（182核心字段 + ΔmAP清洗 + 100非PMC初筛）</title>
<style>
  :root {{
    --bg: #f4f6f9; --card: #ffffff; --navy: #16324f; --teal: #0f766e;
    --red: #b91c1c; --amber: #b45309; --border: #dbe2ea; --muted: #5b6878;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; margin: 0; background: var(--bg); color: #1e293b; line-height: 1.45; }}
  header {{ position: sticky; top: 0; z-index: 50; background: var(--navy); color: #fff; padding: 10px 16px; box-shadow: 0 2px 8px rgba(0,0,0,.15); }}
  .top-row {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
  .tabs {{ display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap; }}
  .tab-btn {{ border: 1px solid rgba(255,255,255,.25); background: rgba(255,255,255,.08); color: #e2e8f0; padding: 7px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }}
  .tab-btn.active {{ background: #ffffff; color: var(--navy); border-color: #fff; }}
  .btn {{ border: none; padding: 7px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12.5px; }}
  .btn-teal {{ background: #14b8a6; color: #042f2e; }}
  .btn-blue {{ background: #e0f2fe; color: #0c4a6e; }}
  .btn-warn {{ background: #fee2e2; color: #7f1d1d; }}
  .toolbar {{ background: #fff; border-bottom: 1px solid var(--border); padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; max-width: 1180px; margin: 0 auto; }}
  .container {{ max-width: 1180px; margin: 14px auto; padding: 0 12px 60px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; }}
  .card.flagged {{ border-left: 4px solid var(--red); }}
  .card.done {{ border-left: 4px solid var(--teal); }}
  .meta {{ font-size: 12.5px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 4px; }}
  .badge {{ display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
  .badge-flag {{ background: #fee2e2; color: var(--red); }}
  .badge-ok {{ background: #ccfbf1; color: #115e59; }}
  .badge-ds {{ background: #e0e7ff; color: #3730a3; }}
  .title {{ font-size: 15px; font-weight: 700; color: var(--navy); margin-bottom: 8px; }}
  .snippet {{ background: #f8fafc; border-left: 3px solid #3b82f6; padding: 8px 12px; font-size: 12px; white-space: pre-wrap; max-height: 210px; overflow-y: auto; margin: 8px 0; color: #334155; }}
  .field-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 8px; margin-top: 8px; }}
  .field-box {{ border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; background: #fcfcfd; }}
  .field-title {{ font-size: 12px; font-weight: 700; color: #1e293b; margin-bottom: 6px; }}
  .pill-group {{ display: flex; gap: 4px; flex-wrap: wrap; }}
  .pill {{ border: 1px solid #cbd5e1; background: #fff; color: #334155; padding: 4px 8px; border-radius: 5px; cursor: pointer; font-size: 11.5px; }}
  .pill.sel-rep {{ background: #0f766e; color: #fff; border-color: #0f766e; font-weight: 700; }}
  .pill.sel-amb {{ background: #d97706; color: #fff; border-color: #d97706; font-weight: 700; }}
  .pill.sel-not {{ background: #475569; color: #fff; border-color: #475569; font-weight: 700; }}
  .pill.sel-exc {{ background: #b91c1c; color: #fff; border-color: #b91c1c; font-weight: 700; }}
  input, select, textarea {{ font-size: 12.5px; padding: 5px 8px; border: 1px solid #cbd5e1; border-radius: 5px; }}
  .delta-edit {{ display: grid; grid-template-columns: 1.2fr 90px 1.2fr 90px 90px; gap: 8px; align-items: end; background: #fffbeb; padding: 10px; border-radius: 6px; border: 1px solid #fde68a; margin-top: 8px; }}
  @media (max-width: 760px) {{ .delta-edit {{ grid-template-columns: 1fr 1fr; }} }}
  details.tbl-wrap {{ margin-top: 8px; border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; background: #fafafa; }}
  details.tbl-wrap summary {{ cursor: pointer; font-size: 12px; font-weight: 600; color: #1d4ed8; }}
  table.xml-tbl {{ border-collapse: collapse; width: 100%; font-size: 11px; margin: 6px 0; display: block; overflow-x: auto; }}
  table.xml-tbl td, table.xml-tbl th {{ border: 1px solid #cbd5e1; padding: 3px 6px; text-align: left; white-space: nowrap; }}
  table.xml-tbl tr:first-child {{ background: #e2e8f0; font-weight: 700; }}
  .kw {{ background: #fef08a; padding: 0 2px; border-radius: 2px; }}
</style>
</head>
<body>
<header>
  <div class="top-row">
    <div>
      <div style="font-size:15px;font-weight:700;">Paper 2 双作者独立人工审核工作台</div>
      <div id="globalStats" style="font-size:12px;color:#cbd5e1;"></div>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
      <select id="coderSelect" onchange="switchCoder(this.value)" style="background:#fff;color:#0f172a;font-weight:700;">
        <option value="coder1">标注者 1：Wenzhuo Liu</option>
        <option value="coder2">标注者 2：Shiling Yu</option>
      </select>
      <button class="btn btn-teal" onclick="exportBundleJSON()">一键导出审核结果包 (.json 推荐)</button>
      <button class="btn btn-blue" onclick="exportCurrentTabCSV()">导出当前页 CSV</button>
      <button class="btn btn-blue" onclick="document.getElementById('importFile').click()">导入 JSON 继续</button>
      <input id="importFile" type="file" accept=".json" style="display:none" onchange="importProgress(event)">
      <button class="btn btn-warn" onclick="resetCurrentCoder()">清空当前进度</button>
    </div>
  </div>
  <div class="tabs">
    <button class="tab-btn active" id="tabBtn_core6" onclick="switchTab('core6')">1. 核心 6 字段双盲审核 (182 项)</button>
    <button class="tab-btn" id="tabBtn_deltas" onclick="switchTab('deltas')">2. ΔmAP@0.5 提取清洗与消融基线核签 (182 项)</button>
    <button class="tab-btn" id="tabBtn_nonpmc" onclick="switchTab('nonpmc')">3. 非 PMC 样本筛选 (100 篇)</button>
  </div>
</header>

<div class="toolbar">
  <input id="searchBox" type="text" placeholder="搜索 PMCID / sample_id / 标题 / 期刊 / 模型..." oninput="render()" style="min-width:240px;flex:1;">
  <select id="filterMode" onchange="render()">
    <option value="all">显示全部条目</option>
    <option value="todo">仅看尚未完成</option>
    <option value="flagged">仅看重点可疑/待洗条目 (Tab 2)</option>
    <option value="NEU-DET">仅看 NEU-DET</option>
    <option value="GC10-DET">仅看 GC10-DET</option>
    <option value="primary">仅看非 PMC 主样本 (60)</option>
    <option value="reserve">仅看非 PMC 替补样本 (40)</option>
  </select>
  <label style="font-size:12px;display:flex;align-items:center;gap:4px;">
    <input type="checkbox" id="hlToggle" checked onchange="render()"> 高亮原文关键词
  </label>
</div>

<div class="container" id="app"></div>

<script>
const ITEMS = {payload_items};
const NONPMC = {payload_nonpmc};
const CORE = [
  ["F03", "F03 划分可重建 (清单或种子+代码)", "明确列出具体样本清单链接，或同时给出固定种子+完整划分脚本=reported；仅报比例8:1:1=not_reported"],
  ["F04", "F04 重复/近重复检查或处理", "明确说明检查、去重或近重复处理=reported；仅提增强未去重或模糊提及=ambiguous/not_reported"],
  ["F07", "F07 验证集 vs 测试集明确区分", "主结果表明确写明是在独立测试集还是验证集评测=reported；val/test混用或称谓矛盾=ambiguous"],
  ["F08", "F08 选模(Val)与最终测试(Test)分离", "明确说明用val选best.pt并在独立held-out test评测=reported；仅两分或用test挑轮次=not_reported/ambiguous"],
  ["F12", "F12 随机种子数值披露", "给出具体数值(如seed=42/0..9)=reported；只说fixed seed未报值=ambiguous"],
  ["F13", "F13 多次运行并报离散度", "多次运行并报SD/SE/CI/极差=reported；只跑单次或无离散度=not_reported"]
];

let currentTab = "core6";
let currentCoder = localStorage.getItem("paper2_active_coder") || "coder1";
document.getElementById("coderSelect").value = currentCoder;

function storageKey(c) {{ return `paper2_audit_workbench_v2_${{c}}`; }}
function loadState(c) {{
  return JSON.parse(localStorage.getItem(storageKey(c)) || '{{"core6":{{}},"deltas":{{}},"nonpmc":{{}}}}');
}}
let state = loadState(currentCoder);

function saveState() {{
  localStorage.setItem(storageKey(currentCoder), JSON.stringify(state));
  updateStats();
}}

function switchCoder(c) {{
  currentCoder = c;
  localStorage.setItem("paper2_active_coder", c);
  state = loadState(c);
  render();
}}

function switchTab(t) {{
  currentTab = t;
  ["core6","deltas","nonpmc"].forEach(k => {{
    document.getElementById("tabBtn_"+k).classList.toggle("active", k===t);
  }});
  render();
}}

function escHtml(s) {{
  return String(s ?? "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

function highlightText(raw) {{
  let h = escHtml(raw);
  if (!document.getElementById("hlToggle").checked) return h;
  return h.replace(/\\b(seed|split|ratio|train|val|validation|test|duplicate|deduplicat|hash|leakage|repeat|github|zenodo|figshare|mAP|AP@0\\.5|baseline|ablation)\\b/gi, '<span class="kw">$1</span>');
}}

function setCore(evalId, fc, val) {{
  if (!state.core6[evalId]) state.core6[evalId] = {{}};
  state.core6[evalId][fc] = val;
  saveState();
  render();
}}

function setCoreNote(evalId, val) {{
  if (!state.core6[evalId]) state.core6[evalId] = {{}};
  state.core6[evalId].notes = val;
  saveState();
}}

function setDeltaField(evalId, k, val) {{
  if (!state.deltas[evalId]) state.deltas[evalId] = {{}};
  state.deltas[evalId][k] = val;
  saveState();
  if (k === "verdict") render();
}}

function setNonPmc(sid, k, val) {{
  if (!state.nonpmc[sid]) state.nonpmc[sid] = {{}};
  state.nonpmc[sid][k] = val;
  saveState();
  if (k === "decision") render();
}}

function updateStats() {{
  const cDone = ITEMS.filter(it => CORE.every(c => (state.core6[it.eval_id]||{{}})[c[0]])).length;
  const dDone = ITEMS.filter(it => (state.deltas[it.eval_id]||{{}}).verdict).length;
  const nDone = NONPMC.filter(it => (state.nonpmc[it.sample_id]||{{}}).decision).length;
  const cLabel = currentCoder === "coder1" ? "Wenzhuo Liu" : "Shiling Yu";
  document.getElementById("globalStats").textContent =
    `当前: ${{cLabel}} | 核心6字段: ${{cDone}}/${{ITEMS.length}} | ΔmAP清洗: ${{dDone}}/${{ITEMS.length}} | 非PMC筛选: ${{nDone}}/${{NONPMC.length}}`;
}}

function renderTables(tables) {{
  if (!tables || !tables.length) return "";
  return tables.map(tb => {{
    const trs = tb.rows.map(r => `<tr>${{r.map(c => `<td>${{escHtml(c)}}</td>`).join("")}}</tr>`).join("");
    return `<details class="tbl-wrap"><summary>展开论文原表: ${{escHtml(tb.label)}} — ${{escHtml(tb.caption)}}</summary><table class="xml-tbl">${{trs}}</table></details>`;
  }}).join("");
}}

function render() {{
  const q = (document.getElementById("searchBox").value || "").trim().toLowerCase();
  const fm = document.getElementById("filterMode").value;
  const root = document.getElementById("app");
  updateStats();

  if (currentTab === "core6") {{
    const filtered = ITEMS.filter(it => {{
      const st = state.core6[it.eval_id] || {{}};
      const done = CORE.every(c => st[c[0]]);
      if (fm === "todo" && done) return false;
      if ((fm === "NEU-DET" || fm === "GC10-DET") && it.dataset !== fm) return false;
      if (q && !(it.pmcid + " " + it.title + " " + it.journal).toLowerCase().includes(q)) return false;
      return true;
    }});
    root.innerHTML = `<div style="margin-bottom:8px;font-size:12.5px;color:#475569;">说明：本页严格保持盲法，<b>隐藏所有 AI 预判标签</b>。请仅依据下方从 JATS XML 提取的方法与实验原文（或点击 PMC/DOI 链接看全文）独立判定 6 项字段。当前匹配 ${{filtered.length}} 条。</div>` +
    filtered.map(it => {{
      const st = state.core6[it.eval_id] || {{}};
      const done = CORE.every(c => st[c[0]]);
      const pmcUrl = `https://pmc.ncbi.nlm.nih.gov/articles/${{it.pmcid}}/`;
      const fieldsHtml = CORE.map(([fc, label, tip]) => {{
        const cur = st[fc] || "";
        return `<div class="field-box">
          <div class="field-title" title="${{escHtml(tip)}}">${{label}}</div>
          <div style="font-size:11px;color:#64748b;margin-bottom:5px;">${{tip}}</div>
          <div class="pill-group">
            <button class="pill ${{cur==='reported'?'sel-rep':''}}" onclick="setCore('${{it.eval_id}}','${{fc}}','reported')">reported</button>
            <button class="pill ${{cur==='ambiguous'?'sel-amb':''}}" onclick="setCore('${{it.eval_id}}','${{fc}}','ambiguous')">ambiguous</button>
            <button class="pill ${{cur==='not_reported_in_paper'?'sel-not':''}}" onclick="setCore('${{it.eval_id}}','${{fc}}','not_reported_in_paper')">not_reported</button>
          </div>
        </div>`;
      }}).join("");
      return `<div class="card ${{done?'done':''}}">
        <div class="meta">
          <strong>${{it.eval_id}}</strong>
          <span class="badge badge-ds">${{it.dataset}}</span>
          <a href="${{pmcUrl}}" target="_blank">${{it.pmcid}} ↗</a>
          <span>${{it.year}} | ${{escHtml(it.journal)}}</span>
          ${{it.doi ? `<a href="https://doi.org/${{escHtml(it.doi)}}" target="_blank">DOI ↗</a>` : ""}}
          ${{done ? '<span class="badge badge-ok">已完成</span>' : ''}}
        </div>
        <div class="title">${{escHtml(it.title)}}</div>
        <div class="snippet">${{highlightText(it.xml_snippets || "（XML 未命中方法关键词，请点击上方 PMC 链接核对原文）")}}</div>
        <div class="field-grid">${{fieldsHtml}}</div>
        <div style="margin-top:8px;">
          <input type="text" style="width:100%;" placeholder="备注 (如不确定处、具体页码/段落说明)..." value="${{escHtml(st.notes||'')}}" onchange="setCoreNote('${{it.eval_id}}',this.value)">
        </div>
      </div>`;
    }}).join("");
  }} else if (currentTab === "deltas") {{
    const filtered = ITEMS.filter(it => {{
      const st = state.deltas[it.eval_id] || {{}};
      if (fm === "todo" && st.verdict) return false;
      if (fm === "flagged" && (!it.flags || !it.flags.length)) return false;
      if ((fm === "NEU-DET" || fm === "GC10-DET") && it.dataset !== fm) return false;
      if (q && !(it.pmcid + " " + it.title + " " + it.baseline_model + " " + it.proposed_model).toLowerCase().includes(q)) return false;
      return true;
    }});
    root.innerHTML = `<div style="margin-bottom:8px;font-size:12.5px;color:#475569;">说明：本页用于清洗“声称提升值表”。红色边框为脚本自动标出的可疑行（非模型名/指标对比/负值或超高值/F14非reported）。请展开下方论文原表核对同文自建基础模型（消融基线），修正或勾选剔除。当前匹配 ${{filtered.length}} 条。</div>` +
    filtered.map(it => {{
      const st = state.deltas[it.eval_id] || {{}};
      const bm = st.baseline_model !== undefined ? st.baseline_model : it.baseline_model;
      const pm = st.proposed_model !== undefined ? st.proposed_model : it.proposed_model;
      const bv = st.baseline_map50 !== undefined ? st.baseline_map50 : it.baseline_map50;
      const pv = st.proposed_map50 !== undefined ? st.proposed_map50 : it.proposed_map50;
      const calcDelta = (bv !== "" && pv !== "" && !isNaN(Number(bv)) && !isNaN(Number(pv))) ? (Number(pv) - Number(bv)).toFixed(2) : (it.pre_extracted_delta_pp || "");
      const v = st.verdict || "";
      const flagHtml = (it.flags || []).map(fl => `<span class="badge badge-flag">${{escHtml(fl)}}</span>`).join(" ");
      return `<div class="card ${{v?'done':(it.flags.length?'flagged':'')}}">
        <div class="meta">
          <strong>${{it.eval_id}}</strong>
          <span class="badge badge-ds">${{it.dataset}}</span>
          <a href="https://pmc.ncbi.nlm.nih.gov/articles/${{it.pmcid}}/" target="_blank">${{it.pmcid}} ↗</a>
          ${{flagHtml}}
        </div>
        <div class="title">${{escHtml(it.title)}}</div>
        <div style="font-size:12px;color:#334155;margin-bottom:6px;"><b>原提取证据 (${{escHtml(it.delta_location)}}):</b> ${{escHtml(it.delta_evidence)}}</div>
        <div class="delta-edit">
          <div><label style="font-size:11px;color:#64748b;">基线模型名 (同文消融基准)</label><input style="width:100%" value="${{escHtml(bm)}}" onchange="setDeltaField('${{it.eval_id}}','baseline_model',this.value)"></div>
          <div><label style="font-size:11px;color:#64748b;">基线 mAP@0.5</label><input type="number" step="0.01" style="width:100%" value="${{escHtml(bv)}}" onchange="setDeltaField('${{it.eval_id}}','baseline_map50',this.value);render();"></div>
          <div><label style="font-size:11px;color:#64748b;">所提模型名</label><input style="width:100%" value="${{escHtml(pm)}}" onchange="setDeltaField('${{it.eval_id}}','proposed_model',this.value)"></div>
          <div><label style="font-size:11px;color:#64748b;">所提 mAP@0.5</label><input type="number" step="0.01" style="width:100%" value="${{escHtml(pv)}}" onchange="setDeltaField('${{it.eval_id}}','proposed_map50',this.value);render();"></div>
          <div><label style="font-size:11px;color:#64748b;">Δ (pp)</label><input style="width:100%;font-weight:700;background:#f1f5f9;" readonly value="${{escHtml(calcDelta)}}"></div>
        </div>
        <div class="pill-group" style="margin-top:8px;">
          <button class="pill ${{v==='same_paper_ablation'?'sel-rep':''}}" onclick="setDeltaField('${{it.eval_id}}','verdict','same_paper_ablation')">✅ 同文消融基础模型 (可比主分析)</button>
          <button class="pill ${{v==='reproduced_other_detector'?'sel-amb':''}}" onclick="setDeltaField('${{it.eval_id}}','verdict','reproduced_other_detector')">⚠️ 同文复现其他检测器 (如YOLO对FRCNN)</button>
          <button class="pill ${{v==='unverified_external'?'sel-not':''}}" onclick="setDeltaField('${{it.eval_id}}','verdict','unverified_external')">❓ 基线来源不明/疑外部抄录</button>
          <button class="pill ${{v==='exclude_bad_row'?'sel-exc':''}}" onclick="setDeltaField('${{it.eval_id}}','verdict','exclude_bad_row')">❌ 非消融/正则误抓 (剔除)</button>
        </div>
        ${{renderTables(it.xml_tables)}}
      </div>`;
    }}).join("");
  }} else {{
    const filtered = NONPMC.filter(it => {{
      const st = state.nonpmc[it.sample_id] || {{}};
      if (fm === "todo" && st.decision) return false;
      if ((fm === "primary" || fm === "reserve") && it.role !== fm) return false;
      if (q && !(it.sample_id + " " + it.title + " " + it.journal).toLowerCase().includes(q)) return false;
      return true;
    }});
    root.innerHTML = `<div style="margin-bottom:8px;font-size:12.5px;color:#475569;">说明：按 PRISMA 排除规则逐篇点开 DOI 核对标题/摘要，确定是否纳入全文编码（E1=非钢材金属表面; E2=非目标检测框; E3=未使用NEU/GC10; E4=与131篇重复; E5=非同行评审预印本/无法获取; E6=综述非实证）。当前匹配 ${{filtered.length}} 篇。</div>` +
    filtered.map(it => {{
      const st = state.nonpmc[it.sample_id] || {{}};
      const dec = st.decision || "";
      const opts = [
        ["include", "纳入(include)", "sel-rep"],
        ["E1", "E1 非钢铁表面", "sel-not"],
        ["E2", "E2 非目标检测", "sel-not"],
        ["E3", "E3 无NEU/GC10", "sel-not"],
        ["E4", "E4 重复/已纳入", "sel-amb"],
        ["E5", "E5 预印/缺全文", "sel-exc"],
        ["E6", "E6 非实证研究", "sel-not"],
      ];
      return `<div class="card ${{dec?'done':''}}">
        <div class="meta">
          <strong>${{it.sample_id}}</strong>
          <span class="badge badge-ds">${{it.role === 'primary' ? '主样本 60' : '替补 40'}}</span>
          <span>${{it.year}} | ${{escHtml(it.journal)}} | 检索词: ${{escHtml(it.queries)}}</span>
          <a href="${{escHtml(it.doi_url)}}" target="_blank">打开 DOI 原文 ↗</a>
        </div>
        <div class="title">${{escHtml(it.title)}}</div>
        <div class="pill-group">
          ${{opts.map(([val, lbl, cls]) => `<button class="pill ${{dec===val?cls:''}}" onclick="setNonPmc('${{it.sample_id}}','decision','${{val}}')">${{lbl}}</button>`).join("")}}
        </div>
        <div style="margin-top:6px;">
          <input type="text" style="width:100%;" placeholder="核对备注 (如所用数据集/排除理由细节)..." value="${{escHtml(st.notes||'')}}" onchange="setNonPmc('${{it.sample_id}}','notes',this.value)">
        </div>
      </div>`;
    }}).join("");
  }}
}}

function downloadBlob(name, text, mime) {{
  const blob = new Blob([text], {{type: mime}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}}

function csvLine(arr) {{
  return arr.map(v => `"${{String(v ?? "").replace(/"/g, '""')}}"`).join(",");
}}

function buildCSV(tabName) {{
  if (tabName === "core6") {{
    const h1 = ["eval_id","pmcid","dataset","year","doi","title","F03_status","F04_status","F07_status","F08_status","F12_status","F13_status","verified_baseline_map50","verified_proposed_map50","verified_delta_map50_pp","coder_notes"];
    const r1 = [csvLine(h1)];
    ITEMS.forEach(it => {{
      const s6 = state.core6[it.eval_id] || {{}};
      const sd = state.deltas[it.eval_id] || {{}};
      const bv = sd.baseline_map50 !== undefined ? sd.baseline_map50 : it.baseline_map50;
      const pv = sd.proposed_map50 !== undefined ? sd.proposed_map50 : it.proposed_map50;
      const dv = (bv !== "" && pv !== "" && !isNaN(Number(bv)) && !isNaN(Number(pv))) ? (Number(pv) - Number(bv)).toFixed(2) : (it.pre_extracted_delta_pp || "");
      r1.push(csvLine([it.eval_id, it.pmcid, it.dataset, it.year, it.doi, it.title, s6.F03||"", s6.F04||"", s6.F07||"", s6.F08||"", s6.F12||"", s6.F13||"", bv, pv, dv, s6.notes||""]));
    }});
    return ["core6_182_" + currentCoder + ".csv", "\\ufeff" + r1.join("\\n")];
  }}
  if (tabName === "deltas") {{
    const h2 = ["eval_id","coding_file","pmcid","dataset","verdict","baseline_model","baseline_map50","proposed_model","proposed_map50","claimed_delta_map50_pp","flags"];
    const r2 = [csvLine(h2)];
    ITEMS.forEach(it => {{
      const sd = state.deltas[it.eval_id] || {{}};
      const bm = sd.baseline_model !== undefined ? sd.baseline_model : it.baseline_model;
      const pm = sd.proposed_model !== undefined ? sd.proposed_model : it.proposed_model;
      const bv = sd.baseline_map50 !== undefined ? sd.baseline_map50 : it.baseline_map50;
      const pv = sd.proposed_map50 !== undefined ? sd.proposed_map50 : it.proposed_map50;
      const dv = (bv !== "" && pv !== "" && !isNaN(Number(bv)) && !isNaN(Number(pv))) ? (Number(pv) - Number(bv)).toFixed(2) : (it.pre_extracted_delta_pp || "");
      r2.push(csvLine([it.eval_id, it.coding_file, it.pmcid, it.dataset, sd.verdict||"", bm, bv, pm, pv, dv, (it.flags||[]).join("; ")]));
    }});
    return ["claimed_deltas_verified_" + currentCoder + ".csv", "\\ufeff" + r2.join("\\n")];
  }}
  const h3 = ["sample_id","role","year","doi","doi_url","title","journal","queries","screening_decision_E1_E6","notes"];
  const r3 = [csvLine(h3)];
  NONPMC.forEach(it => {{
    const sn = state.nonpmc[it.sample_id] || {{}};
    r3.push(csvLine([it.sample_id, it.role, it.year, it.doi, it.doi_url, it.title, it.journal, it.queries, sn.decision||"", sn.notes||""]));
  }});
  return ["nonpmc_screening_100_" + currentCoder + ".csv", "\\ufeff" + r3.join("\\n")];
}}

function exportCurrentTabCSV() {{
  const [fname, content] = buildCSV(currentTab);
  downloadBlob(fname, content, "text/csv;charset=utf-8;");
}}

function exportBundleJSON() {{
  downloadBlob(`audit_bundle_${{currentCoder}}.json`, JSON.stringify({{coder: currentCoder, exported_at: new Date().toISOString(), state}}, null, 2), "application/json");
}}

function importProgress(ev) {{
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {{
    try {{
      const parsed = JSON.parse(reader.result);
      const loadedState = parsed.state || parsed;
      if (parsed.coder && parsed.coder !== currentCoder) {{
        if (confirm(`导入的文件属于 ${{parsed.coder}}，是否切换到该标注者再写入？`)) {{
          currentCoder = parsed.coder;
          document.getElementById("coderSelect").value = currentCoder;
          localStorage.setItem("paper2_active_coder", currentCoder);
        }}
      }}
      state = loadedState;
      saveState();
      render();
      alert("进度导入成功！");
    }} catch (e) {{
      alert("JSON 格式解析失败: " + e.message);
    }}
  }};
  reader.readAsText(file);
}}

function resetCurrentCoder() {{
  if (confirm(`确定清空 ${{currentCoder}} 在当前浏览器的全部标注进度吗？建议先导出 JSON 备份。`)) {{
    state = {{core6:{{}}, deltas:{{}}, nonpmc:{{}}}};
    saveState();
    render();
  }}
}}

render();
</script>
</body>
</html>
"""

    wb_path = OUT_DIR / "dual_author_audit_workbench.html"
    mob_path = OUT_DIR / "mobile_core6_dual_coding_182.html"
    wb_path.write_text(html_doc, encoding="utf-8")
    mob_path.write_text(html_doc, encoding="utf-8")
    n_flagged = sum(1 for it in items if it["flags"])
    print(f"Built unified workbench: {wb_path} ({wb_path.stat().st_size/1024:.1f} KB)")
    print(f"  Tab 1 (Core-6): {len(items)} items")
    print(f"  Tab 2 (Delta cleaning): {len(items)} items ({n_flagged} flagged for priority review)")
    print(f"  Tab 3 (Non-PMC screening): {len(nonpmc_items)} items")


if __name__ == "__main__":
    main()

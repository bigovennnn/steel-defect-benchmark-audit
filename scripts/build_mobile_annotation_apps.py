#!/usr/bin/env python3
"""
Generate two standalone, mobile-friendly, single-file HTML annotation apps for Paper 2:
1. Task 1: results/human_audit/mobile_duplicate_pair_review.html (349 image pairs embedded as WebP data URIs)
2. Task 2: docs/coding_v1/human_dual_coding/mobile_literature_dual_coding.html (26 papers, 35 evaluations, 595 items + full text embedded)
3. User Guide: docs/零基础跨端标注极简指南.md

Strictly blind: no Coder 1 labels/quotes or Pool A/B/C identifiers are exposed in the generated HTML files.
"""

import base64
import csv
import io
import json
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]


def build_task1_image_app():
    csv_path = ROOT_DIR / "results/human_audit/pair_labels_to_fill.csv"
    img_dir = ROOT_DIR / "results/human_audit/pair_images"
    out_html = ROOT_DIR / "results/human_audit/mobile_duplicate_pair_review.html"

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    items_data = []
    for r in rows:
        aid = r["audit_id"].strip()
        img_path = img_dir / f"{aid}.png"
        im = Image.open(img_path).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=72, method=4)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        items_data.append({
            "audit_id": aid,
            "image_a": r["image_a"].strip(),
            "image_b": r["image_b"].strip(),
            "img_src": f"data:image/webp;base64,{b64}",
        })

    # Embed 2 tutorial example images (AUDIT_0001 as positive example, AUDIT_0005 as negative example)
    ex_pos_img = next(x["img_src"] for x in items_data if x["audit_id"] == "AUDIT_0001")
    ex_neg_img = next(x["img_src"] for x in items_data if x["audit_id"] == "AUDIT_0005")

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=3.0, user-scalable=yes">
<title>任务1：钢材图像重叠盲审 (349对·跨端单文件版)</title>
<style>
  :root {{
    --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #f8fafc; --muted: #94a3b8;
    --green: #10b981; --green-dark: #059669; --red: #ef4444; --red-dark: #dc2626; --blue: #38bdf8;
  }}
  * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 10px 10px 110px 10px;
  }}
  .top-bar {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 14px; margin-bottom: 10px; display: flex; flex-wrap: wrap;
    justify-content: space-between; align-items: center; gap: 8px; position: sticky; top: 0; z-index: 20;
  }}
  .title-wrap h1 {{ margin: 0; font-size: 16px; color: var(--blue); }}
  .title-wrap p {{ margin: 2px 0 0 0; font-size: 12px; color: var(--muted); }}
  .progress-wrap {{ display: flex; align-items: center; gap: 8px; flex: 1; min-width: 180px; justify-content: flex-end; }}
  .progress-bar {{ background: #334155; height: 8px; border-radius: 4px; width: 120px; overflow: hidden; }}
  .progress-fill {{ background: var(--green); height: 100%; width: 0%; transition: width 0.2s; }}
  .btn-sm {{
    background: #334155; color: var(--text); border: 1px solid #475569; padding: 6px 10px;
    border-radius: 6px; font-size: 12px; cursor: pointer; font-weight: 500;
  }}
  .btn-sm.primary {{ background: var(--blue); color: #0f172a; border-color: var(--blue); font-weight: 700; }}
  .btn-sm.danger {{ background: rgba(239,68,68,0.18); color: #fca5a5; border-color: #ef4444; }}
  .btn-sm.danger:hover {{ background: #dc2626; color: white; }}
  .guide-box {{
    background: #172554; border: 1px solid #1d4ed8; border-radius: 10px; padding: 12px; margin-bottom: 12px; font-size: 13px; line-height: 1.5;
  }}
  .guide-box h3 {{ margin: 0 0 6px 0; font-size: 14px; color: #93c5fd; }}
  .guide-grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 8px; }}
  @media(min-width: 700px) {{ .guide-grid {{ grid-template-columns: 1fr 1fr; }} }}
  .guide-card {{ background: rgba(15,23,42,0.7); border: 1px solid #334155; border-radius: 8px; padding: 8px; }}
  .guide-card img {{ width: 100%; border-radius: 4px; margin-top: 4px; }}
  .viewer {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 12px;
    display: flex; flex-direction: column; align-items: center;
  }}
  .meta-row {{
    display: flex; flex-wrap: wrap; justify-content: space-between; width: 100%;
    font-size: 13px; color: var(--muted); margin-bottom: 8px; gap: 6px;
  }}
  .meta-row strong {{ color: var(--text); }}
  .badge {{ padding: 2px 8px; border-radius: 99px; font-size: 12px; font-weight: 600; }}
  .badge-unrated {{ background: #334155; color: #cbd5e1; }}
  .badge-1 {{ background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid #10b981; }}
  .badge-0 {{ background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid #ef4444; }}
  .img-box {{
    width: 100%; max-width: 960px; background: #020617; border: 1px solid var(--border);
    border-radius: 8px; padding: 6px; margin-bottom: 10px;
  }}
  .img-box img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
  .notes-row {{ width: 100%; max-width: 960px; display: flex; gap: 8px; margin-top: 4px; }}
  .notes-input {{
    flex: 1; padding: 9px 12px; border-radius: 8px; border: 1px solid #475569;
    background: #0f172a; color: var(--text); font-size: 14px;
  }}
  .bottom-dock {{
    position: fixed; bottom: 0; left: 0; right: 0; background: rgba(15,23,42,0.96);
    border-top: 1px solid var(--border); padding: 10px 12px; z-index: 30;
    display: flex; justify-content: center; gap: 10px; backdrop-filter: blur(8px);
  }}
  .dock-inner {{ width: 100%; max-width: 960px; display: grid; grid-template-columns: 1fr 2fr 2fr 1fr; gap: 8px; }}
  .vote-btn {{
    padding: 13px 8px; border-radius: 10px; border: 2px solid transparent; font-size: 15px;
    font-weight: 700; cursor: pointer; text-align: center; color: white;
  }}
  .vote-1 {{ background: var(--green-dark); }}
  .vote-1.active {{ border-color: #a7f3d0; box-shadow: 0 0 0 2px var(--green); }}
  .vote-0 {{ background: var(--red-dark); }}
  .vote-0.active {{ border-color: #fecaca; box-shadow: 0 0 0 2px var(--red); }}
  .nav-btn {{
    background: #334155; color: var(--text); border: none; border-radius: 10px;
    font-size: 14px; font-weight: 600; cursor: pointer;
  }}
</style>
</head>
<body>

<div class="top-bar">
  <div class="title-wrap">
    <h1>任务1：钢材图像重叠盲审 (349对)</h1>
    <p>完全离线单文件版 · 自动保存进度 · 支持手机触屏与电脑快捷键(1/0/J/K)</p>
  </div>
  <div class="progress-wrap">
    <span style="font-size:13px;">已标: <strong id="prog-text">0 / 349</strong></span>
    <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>
    <button class="btn-sm" onclick="toggleGuide()">30秒图文教程</button>
    <button class="btn-sm" onclick="jumpNextUnrated()">跳到未标</button>
    <button class="btn-sm" onclick="document.getElementById('import-file').click()">导入进度</button>
    <input type="file" id="import-file" accept=".csv" style="display:none" onchange="importCSV(event)">
    <button class="btn-sm danger" onclick="resetProgress()">重置清空</button>
    <button class="btn-sm primary" onclick="exportCSV()">导出 CSV</button>
  </div>
</div>

<div class="guide-box" id="guide-box" style="display:none;">
  <h3>零基础 30 秒看懂怎么标（不需要懂钢铁或算法！）</h3>
  <div>每道题有一张合成图（左：图A；中：对齐后的图B，黑边是平移留白；右：两图差异热力图，<strong>深蓝色=完全吻合，红黄色=有差异</strong>）。你只需要判断：<strong>图A和图B拍的是不是同一块钢板表面的局部重叠区域？</strong></div>
  <div class="guide-grid">
    <div class="guide-card">
      <strong style="color:#34d399;">✅ 选【1 是同一位置重叠】的标准：</strong>
      <div>中间图和左图的<strong>斑点、裂纹走向、划痕位置、背景明暗纹理在重叠区一模一样</strong>（只是上下左右挪了一点或旋转了90度），且右边热力图大片深蓝色。</div>
      <img src="{ex_pos_img}" alt="Positive Example">
    </div>
    <div class="guide-card">
      <strong style="color:#f87171;">❌ 选【0 不是同一块(不同图)】的标准：</strong>
      <div>虽然都是划痕或麻点，但<strong>具体纹路、瑕疵形状对不上</strong>，或者右边热力图出现大片红黄色错位条纹，说明只是长得像的两张不同照片。</div>
      <img src="{ex_neg_img}" alt="Negative Example">
    </div>
  </div>
</div>

<div class="viewer">
  <div class="meta-row">
    <span>题目: <strong id="idx-text">1 / 349</strong> (<strong id="pair-id">--</strong>)</span>
    <span><span id="img-a">--</span> vs. <span id="img-b">--</span></span>
    <span id="status-badge" class="badge badge-unrated">未标注</span>
  </div>
  <div class="img-box">
    <img id="pair-img" src="" alt="图像对加载中...">
  </div>
  <div class="notes-row">
    <input type="text" class="notes-input" id="pair-notes" placeholder="可选备注（如：不确定、旋转重叠等，可留空）" oninput="saveNotes()">
  </div>
</div>

<div class="bottom-dock">
  <div class="dock-inner">
    <button class="nav-btn" onclick="prevPair()">⬅ 上一题</button>
    <button class="vote-btn vote-1" id="btn-1" onclick="vote(1)">✅ 是重叠 [1]</button>
    <button class="vote-btn vote-0" id="btn-0" onclick="vote(0)">❌ 不是重叠 [0]</button>
    <button class="nav-btn" onclick="nextPair()">下一题 ➡</button>
  </div>
</div>

<script>
const ITEMS = {json.dumps(items_data, ensure_ascii=False)};
let curIdx = 0;
let votes = JSON.parse(localStorage.getItem('neu_duplicate_votes') || '{{}}');
let notes = JSON.parse(localStorage.getItem('neu_duplicate_notes') || '{{}}');

function toggleGuide() {{
  const el = document.getElementById('guide-box');
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}}

function render() {{
  const it = ITEMS[curIdx];
  document.getElementById('idx-text').textContent = (curIdx + 1) + ' / ' + ITEMS.length;
  document.getElementById('pair-id').textContent = it.audit_id;
  document.getElementById('img-a').textContent = it.image_a;
  document.getElementById('img-b').textContent = it.image_b;
  document.getElementById('pair-img').src = it.img_src;
  document.getElementById('pair-notes').value = notes[it.audit_id] || '';

  const v = votes[it.audit_id];
  document.getElementById('btn-1').classList.toggle('active', v === 1);
  document.getElementById('btn-0').classList.toggle('active', v === 0);

  const badge = document.getElementById('status-badge');
  if (v === 1) {{
    badge.className = 'badge badge-1'; badge.textContent = '已标: 1 (重叠)';
  }} else if (v === 0) {{
    badge.className = 'badge badge-0'; badge.textContent = '已标: 0 (非重叠)';
  }} else {{
    badge.className = 'badge badge-unrated'; badge.textContent = '未标注';
  }}

  const done = Object.keys(votes).length;
  document.getElementById('prog-text').textContent = done + ' / ' + ITEMS.length;
  document.getElementById('prog-fill').style.width = (done / ITEMS.length * 100).toFixed(1) + '%';
}}

function vote(val) {{
  const it = ITEMS[curIdx];
  votes[it.audit_id] = val;
  localStorage.setItem('neu_duplicate_votes', JSON.stringify(votes));
  render();
  setTimeout(() => {{
    if (curIdx < ITEMS.length - 1) {{ curIdx++; render(); }}
  }}, 120);
}}

function saveNotes() {{
  const it = ITEMS[curIdx];
  notes[it.audit_id] = document.getElementById('pair-notes').value;
  localStorage.setItem('neu_duplicate_notes', JSON.stringify(notes));
}}

function prevPair() {{ if (curIdx > 0) {{ curIdx--; render(); }} }}
function nextPair() {{ if (curIdx < ITEMS.length - 1) {{ curIdx++; render(); }} }}

function jumpNextUnrated() {{
  for (let i = 0; i < ITEMS.length; i++) {{
    const idx = (curIdx + i) % ITEMS.length;
    if (votes[ITEMS[idx].audit_id] === undefined) {{
      curIdx = idx; render(); return;
    }}
  }}
  alert('全部 349 对图像已标注完毕！请点击右上角【导出 CSV】发送给项目负责人。');
}}

function exportCSV() {{
  let csv = "audit_id,image_a,image_b,is_visually_overlapping,annotator_notes\\n";
  for (const it of ITEMS) {{
    const v = votes[it.audit_id] !== undefined ? votes[it.audit_id] : "";
    const n = (notes[it.audit_id] || "").replace(/"/g, '""');
    csv += `${{it.audit_id}},${{it.image_a}},${{it.image_b}},${{v}},"${{n}}"\\n`;
  }}
  const blob = new Blob(["\\uFEFF" + csv], {{ type: 'text/csv;charset=utf-8;' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'pair_labels_to_fill.csv';
  a.click();
}}

function importCSV(ev) {{
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {{
    const lines = e.target.result.replace(/^\\uFEFF/, '').split(/\\r?\\n/);
    let count = 0;
    for (let i = 1; i < lines.length; i++) {{
      if (!lines[i].trim()) continue;
      const m = lines[i].match(/^([^,]+),([^,]+),([^,]+),([^,]*),(.*)$/);
      if (m) {{
        const aid = m[1].trim();
        const v = m[4].trim();
        const n = m[5].trim().replace(/^"|"$/g, '').replace(/""/g, '"');
        if (v === '1' || v === '0') {{ votes[aid] = parseInt(v, 10); count++; }}
        if (n) notes[aid] = n;
      }}
    }}
    localStorage.setItem('neu_duplicate_votes', JSON.stringify(votes));
    localStorage.setItem('neu_duplicate_notes', JSON.stringify(notes));
    render();
    alert('成功导入 ' + count + ' 条标注记录！');
  }};
  reader.readAsText(file, 'utf-8');
}}

function resetProgress() {{
  const done = Object.keys(votes).length;
  const noteCount = Object.keys(notes).filter(k => (notes[k] || '').trim()).length;
  if (done === 0 && noteCount === 0) {{
    alert('当前没有任何已保存的标注进度，无需重置。');
    return;
  }}
  if (!confirm(`⚠️ 确定要清空全部已有标注进度吗？\\n（当前已标注 ${{done}} / ${{ITEMS.length}} 对图像，清空后浏览器本地进度将归零，不可恢复；如需备份请先点【导出 CSV】）`)) {{
    return;
  }}
  votes = {{}};
  notes = {{}};
  localStorage.removeItem('neu_duplicate_votes');
  localStorage.removeItem('neu_duplicate_notes');
  const fileInput = document.getElementById('import-file');
  if (fileInput) fileInput.value = '';
  curIdx = 0;
  render();
  alert('已清空全部标注进度，回到第 1 题！');
}}

document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT') return;
  if (e.key === '1') vote(1);
  if (e.key === '0') vote(0);
  if (e.key === 'ArrowRight' || e.key === 'j' || e.key === 'J') nextPair();
  if (e.key === 'ArrowLeft' || e.key === 'k' || e.key === 'K') prevPair();
}});

// 初次打开若为0进度，自动展开教程
if (Object.keys(votes).length === 0) {{
  document.getElementById('guide-box').style.display = 'block';
}}
render();
</script>
</body>
</html>
"""
    out_html.write_text(html_content, encoding="utf-8")
    print(f"[Task 1] Wrote {out_html} ({out_html.stat().st_size / 1024 / 1024:.2f} MB)")


def build_task2_literature_app():
    sheet_path = ROOT_DIR / "docs/coding_v1/human_dual_coding/blank_dual_coding_sheet.csv"
    out_html = ROOT_DIR / "docs/coding_v1/human_dual_coding/mobile_literature_dual_coding.html"

    with sheet_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    # Group by eval_id and embed fulltext_text for each unique paper_id
    papers_text = {}
    for r in rows:
        pid = r["paper_id"]
        if pid not in papers_text:
            txt_path = Path(r["fulltext_text"])
            papers_text[pid] = txt_path.read_text(encoding="utf-8")

    # Field guide with plain Chinese question, keywords for 1-click search, and minimum requirements
    field_guides = {
        "F01": {"name": "数据集版本与来源", "q": "文中是否说明了该数据集的图片总数、类别数或来源链接/引用？", "kw": ["NEU", "GC10", "1800", "1,800", "2294", "3570", "dataset", "categories"], "req": "给出图片总数或明确来源链接/引文即算 reported"},
        "F02": {"name": "划分角色与比例", "q": "文中是否同时给出了 train/val/test（或交叉验证）的角色和比例/数量？", "kw": ["ratio", "split", "train", "validation", "test", "7:2:1", "8:1:1", "8:2", "fold"], "req": "角色和数量(或比例)都给出算 reported；只说按比例划分但没说哪几个子集算 ambiguous"},
        "F03": {"name": "划分名单可重建", "q": "文中是否提供了公开的训练/测试图片文件名清单，或可复现划分的代码链接？", "kw": ["manifest", "split", "list", "github", "gitlab", "zenodo", "figshare"], "req": "公开文件名单或可运行划分代码算 reported（绝大多数论文都没有，选 not_reported_in_paper）"},
        "F04": {"name": "重复/近重复检查", "q": "文中是否明确提到检查或处理了数据集里的重复/近重复图片？", "kw": ["duplicate", "leakage", "overlap", "redundant", "identical"], "req": "明确描述检查或处理算 reported（绝大多数未提，选 not_reported_in_paper）"},
        "F05": {"name": "预训练权重来源", "q": "文中是否明确说明模型是从零训练(from scratch)还是使用了预训练权重(如COCO/ImageNet)？", "kw": ["pretrain", "pre-train", "scratch", "COCO", "ImageNet", "weights", "initial"], "req": "明确写出从零训练或预训练权重来源算 reported"},
        "F06": {"name": "检查点选择标准", "q": "文中是否写明训练结束后依据哪个集合、什么指标挑选最终测试用的模型权重(如best val mAP vs 最后一轮)？", "kw": ["checkpoint", "best", "select", "early stopping", "validation", "epoch"], "req": "写明选择最终权重所依据的集合和准则算 reported"},
        "F07": {"name": "报告指标所用集合", "q": "主结果表中的 mAP 是在 validation 集、test 集还是交叉验证上测出来的？", "kw": ["Table", "test set", "validation set", "val", "evaluate", "mAP"], "req": "明确是 val、test 或 CV 算 reported；若正文叫 test 但表头/图注叫 val 等混用情况选 ambiguous"},
        "F08": {"name": "验证集与测试集关系", "q": "【派生题】挑权重的验证集和报最终分数的集合是不是分开的？（需在值框选 separate / same_set / unclear）", "kw": ["validation", "test", "split", "7:2:1", "8:2", "9:1"], "req": "能从文中判断是同一集合(same_set)或独立集合(separate)都算 reported；完全无法判断选 not_reported_in_paper 或 ambiguous，并在值框填 unclear"},
        "F09": {"name": "输入图像尺寸", "q": "文中是否给出了训练和测试时的输入图片分辨率数值（如 640×640）？", "kw": ["640", "resolution", "input size", "resize", "200", "512", "pixels"], "req": "给出具体数值算 reported"},
        "F10": {"name": "训练轮数 (Epochs)", "q": "文中是否给出了总训练轮数 epochs 或 iterations 的具体数值（如 100 / 300 epochs）？", "kw": ["epoch", "iteration", "train", "100", "200", "300"], "req": "给出具体轮数数值算 reported"},
        "F11": {"name": "Batch 大小", "q": "文中是否给出了训练时的 batch size 具体数值（如 8 / 16 / 32）？", "kw": ["batch", "size", "8", "16", "32", "64"], "req": "给出训练 batch 数值算 reported"},
        "F12": {"name": "随机种子 (Seed)", "q": "文中是否明确给出了随机种子的具体数值（如 seed = 0 或 42）？", "kw": ["seed", "random", "reproduc", "deterministic"], "req": "给出具体种子数值算 reported；只说固定种子没给数值算 ambiguous"},
        "F13": {"name": "多次运行与离散度", "q": "主结果是否基于多次独立运行并报告了离散度（如均值±标准差、置信区间）？", "kw": ["±", "std", "standard deviation", "independent", "runs", "times", "confidence interval"], "req": "多次运行并报告离散度算 reported"},
        "F14": {"name": "对比基线来源", "q": "主结果表里的对比基线模型是作者自己重新训练的(rerun)，还是直接抄录其他论文的(copied)？", "kw": ["baseline", "retrain", "re-implement", "same", "compared", "Table"], "req": "明确说明是重训还是引用算 reported（在值框选 rerun/copied/mixed）；没明说选 ambiguous 或 not_reported_in_paper"},
        "F15": {"name": "推理速度测量条件", "q": "如果报了 FPS/延迟，是否同时给出了测量时的 GPU 硬件型号和推理 batch size？", "kw": ["FPS", "latency", "ms", "speed", "RTX", "GTX", "batch"], "req": "至少给出硬件型号与推理 batch 算 reported；只给硬件没给推理 batch 算 ambiguous；全文没测速度选 not_applicable"},
        "F16": {"name": "公开代码链接", "q": "论文中是否提供了开源代码仓库链接（如 GitHub / Zenodo URL）？", "kw": ["http", "github", "code", "available", "repository", "open"], "req": "提供明确代码链接算 reported；写 upon reasonable request 算 not_reported_in_paper"},
        "F17": {"name": "公开模型权重", "q": "论文中是否提供了训练好的模型权重文件下载链接？", "kw": ["weight", "checkpoint", "pretrained", "download", ".pt", ".pth", "github"], "req": "提供明确训练后权重链接算 reported"},
    }

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>任务2：文献协议独立盲审工作台 (26篇·35项评估)</title>
<style>
  :root {{
    --bg: #0f172a; --panel: #1e293b; --border: #334155; --text: #f8fafc; --muted: #94a3b8;
    --blue: #38bdf8; --green: #10b981; --amber: #f59e0b; --red: #ef4444;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column;
  }}
  .header {{
    background: var(--panel); border-bottom: 1px solid var(--border); padding: 10px 14px;
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px;
  }}
  .header h1 {{ margin: 0; font-size: 16px; color: var(--blue); }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
  select, input, textarea {{
    background: #0f172a; color: var(--text); border: 1px solid #475569; border-radius: 6px; padding: 6px 8px; font-size: 13px;
  }}
  .btn {{
    background: #334155; color: var(--text); border: 1px solid #475569; border-radius: 6px;
    padding: 6px 10px; font-size: 12px; cursor: pointer; font-weight: 600;
  }}
  .btn.primary {{ background: var(--blue); color: #0f172a; border-color: var(--blue); }}
  .btn.green {{ background: #059669; color: white; border-color: #10b981; }}
  .btn.danger {{ background: rgba(239,68,68,0.18); color: #fca5a5; border-color: #ef4444; }}
  .btn.danger:hover {{ background: #dc2626; color: white; }}
  .main-split {{
    flex: 1; display: grid; grid-template-columns: 1fr; overflow: hidden;
  }}
  @media(min-width: 900px) {{
    .main-split {{ grid-template-columns: 1.05fr 0.95fr; }}
  }}
  .mobile-tabs {{
    display: flex; background: #020617; border-bottom: 1px solid var(--border);
  }}
  @media(min-width: 900px) {{ .mobile-tabs {{ display: none; }} }}
  .tab-btn {{
    flex: 1; padding: 10px; text-align: center; font-size: 13px; font-weight: 600;
    background: transparent; color: var(--muted); border: none; border-bottom: 2px solid transparent;
  }}
  .tab-btn.active {{ color: var(--blue); border-bottom-color: var(--blue); }}
  .pane {{ height: 100%; overflow-y: auto; padding: 12px; }}
  .pane-left {{ border-right: 1px solid var(--border); background: #0b1120; }}
  .search-bar {{
    position: sticky; top: 0; background: #0b1120; padding-bottom: 8px; z-index: 10;
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center; border-bottom: 1px solid var(--border); margin-bottom: 10px;
  }}
  .kw-chip {{
    background: #1e293b; color: #93c5fd; border: 1px solid #1d4ed8; border-radius: 99px;
    padding: 3px 9px; font-size: 12px; cursor: pointer;
  }}
  .kw-chip:hover {{ background: #1d4ed8; color: white; }}
  .paper-text {{
    white-space: pre-wrap; font-size: 13.5px; line-height: 1.65; color: #e2e8f0; font-family: Georgia, "PingFang SC", serif;
  }}
  mark {{ background: #facc15; color: #0f172a; padding: 0 2px; border-radius: 2px; font-weight: 600; }}
  mark.current-hit {{ background: #f97316; color: white; outline: 2px solid #fdba74; }}
  .field-card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px; margin-bottom: 10px; transition: border-color 0.15s;
  }}
  .field-card.active-field {{ border-color: var(--blue); box-shadow: 0 0 0 1px var(--blue); }}
  .field-card.done-field {{ border-left: 4px solid var(--green); }}
  .field-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
  .field-title {{ font-weight: 700; font-size: 14px; color: #f1f5f9; }}
  .field-q {{ font-size: 13px; color: #cbd5e1; margin-bottom: 4px; }}
  .field-req {{ font-size: 11.5px; color: var(--muted); margin-bottom: 8px; }}
  .status-group {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
  .st-btn {{
    padding: 6px 10px; border-radius: 6px; border: 1px solid #475569; background: #0f172a;
    color: #cbd5e1; font-size: 12px; font-weight: 600; cursor: pointer;
  }}
  .st-btn.st-reported.active {{ background: #059669; color: white; border-color: #34d399; }}
  .st-btn.st-ambiguous.active {{ background: #d97706; color: white; border-color: #fcd34d; }}
  .st-btn.st-not_reported_in_paper.active {{ background: #dc2626; color: white; border-color: #fca5a5; }}
  .st-btn.st-not_applicable.active {{ background: #475569; color: white; border-color: #cbd5e1; }}
  .form-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; }}
  .form-full {{ grid-column: 1 / -1; }}
  .hint-warn {{ color: #fbbf24; font-size: 11.5px; margin-top: 4px; display: none; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>任务2：文献协议独立盲审工作台 (26篇 · 595项)</h1>
    <div style="font-size:12px;color:var(--muted);">左边搜原文、选中文字一键填入引文 · 右边点按钮打标 · 导出即得标准 blank_dual_coding_sheet.csv</div>
  </div>
  <div class="controls">
    <span style="font-size:12px;">总进度: <strong id="total-prog">0 / 595</strong></span>
    <select id="eval-select" onchange="selectEval(this.value)"></select>
    <button class="btn" onclick="fillDefaultScope()">一键填检查范围与未找到项</button>
    <button class="btn" onclick="document.getElementById('import-csv').click()">导入CSV</button>
    <input type="file" id="import-csv" accept=".csv" style="display:none" onchange="importCSV(event)">
    <button class="btn danger" onclick="resetAllProgress()">重置清空</button>
    <button class="btn primary" onclick="exportCSV()">导出 CSV</button>
  </div>
</div>

<div class="mobile-tabs">
  <button class="tab-btn active" id="tab-form" onclick="switchTab('form')">1. 答题卡 (F01-F17)</button>
  <button class="tab-btn" id="tab-text" onclick="switchTab('text')">2. 论文全文与搜索</button>
</div>

<div class="main-split">
  <div class="pane pane-left" id="pane-text">
    <div class="search-bar">
      <input type="text" id="search-input" placeholder="输入关键词搜索全文..." style="flex:1;min-width:140px;" oninput="doSearch(this.value)">
      <button class="btn" onclick="StepHit(-1)">▲</button>
      <span id="hit-count" style="font-size:12px;color:var(--muted);">0/0</span>
      <button class="btn" onclick="StepHit(1)">▼</button>
      <button class="btn green" onclick="useSelectedTextAsQuote()">📋 将鼠标选中文字填入当前题引文</button>
      <div id="kw-chips" style="width:100%;display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;"></div>
    </div>
    <div style="margin-bottom:8px;font-size:12px;color:var(--muted);">
      当前论文：<strong id="paper-title" style="color:var(--text);"></strong> | <a id="paper-url" href="#" target="_blank" style="color:var(--blue);">打开PMC在线原文</a>
    </div>
    <div class="paper-text" id="paper-body"></div>
  </div>

  <div class="pane pane-right" id="pane-form">
    <div id="fields-container"></div>
  </div>
</div>

<script>
const COLUMNS = {json.dumps(columns, ensure_ascii=False)};
const INITIAL_ROWS = {json.dumps(rows, ensure_ascii=False)};
const PAPERS_TEXT = {json.dumps(papers_text, ensure_ascii=False)};
const FIELD_GUIDES = {json.dumps(field_guides, ensure_ascii=False)};

let stateRows = JSON.parse(localStorage.getItem('paper2_dual_coding_rows_v1') || 'null');
if (!stateRows || stateRows.length !== INITIAL_ROWS.length) {{
  stateRows = INITIAL_ROWS;
}}
const rowById = {{}};
stateRows.forEach(r => {{ rowById[r.item_id] = r; }});

// Group by eval_id
const evalList = [];
const evalMap = {{}};
stateRows.forEach(r => {{
  if (!evalMap[r.eval_id]) {{
    evalMap[r.eval_id] = [];
    evalList.push(r.eval_id);
  }}
  evalMap[r.eval_id].push(r);
}});

let curEvalId = evalList[0];
let activeFieldId = 'F01';
let searchHits = [];
let curHitIdx = 0;

function saveState() {{
  localStorage.setItem('paper2_dual_coding_rows_v1', JSON.stringify(stateRows));
  updateProgress();
}}

function isRowComplete(r) {{
  const st = (r.coder2_independent_status || '').trim();
  if (!st) return false;
  if (!r.coder2_checked_scope || !r.coder2_checked_scope.trim()) return false;
  if (st === 'reported' || st === 'ambiguous') {{
    return Boolean((r.coder2_independent_quote || '').trim() && (r.coder2_independent_location || '').trim());
  }}
  if (st === 'not_applicable') {{
    return Boolean((r.coder2_independent_value || '').trim() || (r.coder2_notes || '').trim());
  }}
  return true;
}}

function updateProgress() {{
  const totalDone = stateRows.filter(isRowComplete).length;
  document.getElementById('total-prog').textContent = totalDone + ' / ' + stateRows.length;

  const sel = document.getElementById('eval-select');
  const prevVal = sel.value || curEvalId;
  sel.innerHTML = '';
  evalList.forEach((eid, idx) => {{
    const grp = evalMap[eid];
    const done = grp.filter(isRowComplete).length;
    const opt = document.createElement('option');
    opt.value = eid;
    opt.textContent = `[${{idx+1}}/35] ${{eid}} (已完 ${{done}}/17)`;
    sel.appendChild(opt);
  }});
  sel.value = prevVal;
}}

function selectEval(eid) {{
  curEvalId = eid;
  const grp = evalMap[eid];
  const r0 = grp[0];
  document.getElementById('paper-title').textContent = `[${{r0.dataset}}] ${{r0.article_title}}`;
  document.getElementById('paper-url').href = r0.article_url;
  renderPaperText(PAPERS_TEXT[r0.paper_id] || '');
  renderFields();
  setActiveField('F01');
}}

function escapeHtml(s) {{
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

function renderPaperText(rawText, query) {{
  const container = document.getElementById('paper-body');
  if (!query || !query.trim()) {{
    container.innerHTML = escapeHtml(rawText);
    searchHits = [];
    document.getElementById('hit-count').textContent = '0/0';
    return;
  }}
  const q = query.trim().replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
  const re = new RegExp(`(${{q}})`, 'gi');
  let idx = 0;
  const html = escapeHtml(rawText).replace(re, (m) => {{
    const cls = idx === 0 ? 'current-hit' : '';
    const out = `<mark class="hit-mark ${{cls}}" data-hit="${{idx}}">${{m}}</mark>`;
    idx++;
    return out;
  }});
  container.innerHTML = html;
  searchHits = Array.from(container.querySelectorAll('mark.hit-mark'));
  curHitIdx = 0;
  document.getElementById('hit-count').textContent = searchHits.length ? `1/${{searchHits.length}}` : '0/0';
  if (searchHits.length) searchHits[0].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
}}

function doSearch(q) {{
  const r0 = evalMap[curEvalId][0];
  renderPaperText(PAPERS_TEXT[r0.paper_id] || '', q);
}}

function StepHit(dir) {{
  if (!searchHits.length) return;
  searchHits[curHitIdx].classList.remove('current-hit');
  curHitIdx = (curHitIdx + dir + searchHits.length) % searchHits.length;
  searchHits[curHitIdx].classList.add('current-hit');
  document.getElementById('hit-count').textContent = `${{curHitIdx+1}}/${{searchHits.length}}`;
  searchHits[curHitIdx].scrollIntoView({{ behavior: 'smooth', block: 'center' }});
}}

function setActiveField(fid) {{
  activeFieldId = fid;
  document.querySelectorAll('.field-card').forEach(el => {{
    el.classList.toggle('active-field', el.dataset.fid === fid);
  }});
  const g = FIELD_GUIDES[fid];
  const chips = document.getElementById('kw-chips');
  chips.innerHTML = `<span style="font-size:11.5px;color:var(--muted);">点词搜 ${{fid}}：</span>`;
  (g.kw || []).forEach(k => {{
    const span = document.createElement('span');
    span.className = 'kw-chip';
    span.textContent = k;
    span.onclick = () => {{
      document.getElementById('search-input').value = k;
      doSearch(k);
      if (window.innerWidth < 900) switchTab('text');
    }};
    chips.appendChild(span);
  }});
}}

function renderFields() {{
  const grp = evalMap[curEvalId];
  const container = document.getElementById('fields-container');
  container.innerHTML = '';

  grp.forEach(r => {{
    const fid = r.field_id;
    const g = FIELD_GUIDES[fid];
    const card = document.createElement('div');
    card.className = 'field-card' + (isRowComplete(r) ? ' done-field' : '');
    card.dataset.fid = fid;
    card.onclick = () => setActiveField(fid);

    const st = r.coder2_independent_status || '';
    let valueControl = `<input type="text" placeholder="提取值 (如 7:2:1 / 640x640 / 100)" value="${{(r.coder2_independent_value||'').replace(/"/g,'&quot;')}}" oninput="updateField('${{r.item_id}}','coder2_independent_value',this.value)">`;
    if (fid === 'F08') {{
      valueControl = `<select onchange="updateField('${{r.item_id}}','coder2_independent_value',this.value)">
        <option value="">--必选 F08 值--</option>
        <option value="separate" ${{r.coder2_independent_value==='separate'?'selected':''}}>separate (验证与测试集独立分开)</option>
        <option value="same_set" ${{r.coder2_independent_value==='same_set'?'selected':''}}>same_set (验证与测试用同一集合)</option>
        <option value="unclear" ${{r.coder2_independent_value==='unclear'?'selected':''}}>unclear (文中无法分清)</option>
      </select>`;
    }} else if (fid === 'F14') {{
      valueControl = `<select onchange="updateField('${{r.item_id}}','coder2_independent_value',this.value)">
        <option value="">--必选 F14 基线来源--</option>
        <option value="rerun" ${{r.coder2_independent_value==='rerun'?'selected':''}}>rerun (作者自己重训了对比基线)</option>
        <option value="copied" ${{r.coder2_independent_value==='copied'?'selected':''}}>copied (抄录其他论文数值)</option>
        <option value="mixed" ${{r.coder2_independent_value==='mixed'?'selected':''}}>mixed (部分重训部分抄录)</option>
        <option value="unclear" ${{r.coder2_independent_value==='unclear'?'selected':''}}>unclear (未说明基线怎么来的)</option>
      </select>`;
    }}

    card.innerHTML = `
      <div class="field-header">
        <span class="field-title">${{fid}} · ${{g.name}}</span>
        <span style="font-size:11.5px;color:${{isRowComplete(r)?'#34d399':'#94a3b8'}};" id="done-tag-${{r.item_id}}">${{isRowComplete(r)?'✓ 已填完整':'待填'}}</span>
      </div>
      <div class="field-q">${{g.q}}</div>
      <div class="field-req">判定要求：${{g.req}}</div>
      <div class="status-group">
        <button class="st-btn st-reported ${{st==='reported'?'active':''}}" onclick="setStatus('${{r.item_id}}','reported',event)">1. reported (明确报告)</button>
        <button class="st-btn st-ambiguous ${{st==='ambiguous'?'active':''}}" onclick="setStatus('${{r.item_id}}','ambiguous',event)">2. ambiguous (含糊/冲突)</button>
        <button class="st-btn st-not_reported_in_paper ${{st==='not_reported_in_paper'?'active':''}}" onclick="setStatus('${{r.item_id}}','not_reported_in_paper',event)">3. not_reported_in_paper (未提及)</button>
        <button class="st-btn st-not_applicable ${{st==='not_applicable'?'active':''}}" onclick="setStatus('${{r.item_id}}','not_applicable',event)">4. not_applicable (不适用)</button>
      </div>
      <div class="form-grid">
        ${{valueControl}}
        <input type="text" id="loc-${{r.item_id}}" placeholder="原文位置 (如 Sec 4.1 / Table 1)" value="${{(r.coder2_independent_location||'').replace(/"/g,'&quot;')}}" oninput="updateField('${{r.item_id}}','coder2_independent_location',this.value)">
        <textarea class="form-full" id="quote-${{r.item_id}}" rows="2" placeholder="原文原话引文 (<=35个英文词，reported/ambiguous必填；可在左侧选中文字点【填入引文】)" oninput="updateField('${{r.item_id}}','coder2_independent_quote',this.value)">${{escapeHtml(r.coder2_independent_quote||'')}}</textarea>
        <input type="text" id="scope-${{r.item_id}}" placeholder="已检查范围 (必填，默认 fulltext_text, xml_tables)" value="${{(r.coder2_checked_scope||'').replace(/"/g,'&quot;')}}" oninput="updateField('${{r.item_id}}','coder2_checked_scope',this.value)">
        <input type="text" placeholder="备注说明 (可选)" value="${{(r.coder2_notes||'').replace(/"/g,'&quot;')}}" oninput="updateField('${{r.item_id}}','coder2_notes',this.value)">
      </div>
      <div class="hint-warn" id="warn-${{r.item_id}}">⚠️ 选 reported 或 ambiguous 时，必须填写【原文原话引文】和【原文位置】才能计入有效统计！</div>
    `;
    container.appendChild(card);
    refreshCardWarning(r);
  }});
}}

function refreshCardWarning(r) {{
  const warn = document.getElementById('warn-' + r.item_id);
  const tag = document.getElementById('done-tag-' + r.item_id);
  if (!warn || !tag) return;
  const st = (r.coder2_independent_status || '').trim();
  const needQuote = (st === 'reported' || st === 'ambiguous') && (!(r.coder2_independent_quote || '').trim() || !(r.coder2_independent_location || '').trim());
  warn.style.display = needQuote ? 'block' : 'none';
  const ok = isRowComplete(r);
  tag.textContent = ok ? '✓ 已填完整' : '待填';
  tag.style.color = ok ? '#34d399' : '#fbbf24';
  const card = warn.closest('.field-card');
  if (card) card.classList.toggle('done-field', ok);
}}

function setStatus(itemId, status, ev) {{
  if (ev) ev.stopPropagation();
  const r = rowById[itemId];
  r.coder2_independent_status = status;
  if (!r.coder2_checked_scope) {{
    r.coder2_checked_scope = 'fulltext_text, xml_tables';
  }}
  if (status === 'not_reported_in_paper' && r.field_id === 'F08' && !r.coder2_independent_value) {{
    r.coder2_independent_value = 'unclear';
  }}
  saveState();
  renderFields();
  setActiveField(r.field_id);
}}

function updateField(itemId, col, val) {{
  const r = rowById[itemId];
  r[col] = val;
  if (!r.coder2_checked_scope) {{
    r.coder2_checked_scope = 'fulltext_text, xml_tables';
    const scEl = document.getElementById('scope-' + itemId);
    if (scEl) scEl.value = r.coder2_checked_scope;
  }}
  saveState();
  refreshCardWarning(r);
}}

function useSelectedTextAsQuote() {{
  const sel = window.getSelection().toString().trim().replace(/\\s+/g, ' ');
  if (!sel) {{
    alert('请先在左侧论文正文中用鼠标或手指涂抹选中一句原文！');
    return;
  }}
  const words = sel.split(/\\s+/);
  const clipped = words.length > 35 ? words.slice(0, 35).join(' ') : sel;
  const grp = evalMap[curEvalId];
  const r = grp.find(x => x.field_id === activeFieldId);
  if (!r) return;
  r.coder2_independent_quote = clipped;
  if (!r.coder2_independent_status) r.coder2_independent_status = 'reported';
  if (!r.coder2_checked_scope) r.coder2_checked_scope = 'fulltext_text, xml_tables';
  if (!r.coder2_independent_location) r.coder2_independent_location = 'Full text';
  saveState();
  renderFields();
  setActiveField(activeFieldId);
  if (window.innerWidth < 900) switchTab('form');
}}

function fillDefaultScope() {{
  const grp = evalMap[curEvalId];
  grp.forEach(r => {{
    if (!r.coder2_checked_scope) r.coder2_checked_scope = 'fulltext_text, xml_tables';
    if (!r.coder2_independent_status && (r.field_id === 'F03' || r.field_id === 'F04' || r.field_id === 'F17')) {{
      r.coder2_independent_status = 'not_reported_in_paper';
    }}
  }});
  saveState();
  renderFields();
}}

function switchTab(which) {{
  document.getElementById('tab-form').classList.toggle('active', which === 'form');
  document.getElementById('tab-text').classList.toggle('active', which === 'text');
  document.getElementById('pane-form').style.display = which === 'form' ? 'block' : 'none';
  document.getElementById('pane-text').style.display = which === 'text' ? 'block' : 'none';
}}

function csvEscape(v) {{
  const s = String(v ?? '');
  if (s.includes(',') || s.includes('"') || s.includes('\\n') || s.includes('\\r')) {{
    return '"' + s.replace(/"/g, '""') + '"';
  }}
  return s;
}}

function exportCSV() {{
  const lines = [COLUMNS.join(',')];
  stateRows.forEach(r => {{
    lines.push(COLUMNS.map(c => csvEscape(r[c] || '')).join(','));
  }});
  const blob = new Blob(["\\uFEFF" + lines.join('\\n') + '\\n'], {{ type: 'text/csv;charset=utf-8;' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'blank_dual_coding_sheet.csv';
  a.click();
}}

function parseCSVText(text) {{
  const rows = [];
  let row = [], cur = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {{
    const ch = text[i], nxt = text[i+1];
    if (inQuotes) {{
      if (ch === '"' && nxt === '"') {{ cur += '"'; i++; }}
      else if (ch === '"') {{ inQuotes = false; }}
      else {{ cur += ch; }}
    }} else {{
      if (ch === '"') {{ inQuotes = true; }}
      else if (ch === ',') {{ row.push(cur); cur = ''; }}
      else if (ch === '\\n' || (ch === '\\r' && nxt === '\\n')) {{
        if (ch === '\\r') i++;
        row.push(cur); rows.push(row); row = []; cur = '';
      }} else {{ cur += ch; }}
    }}
  }}
  if (cur.length > 0 || row.length > 0) {{ row.push(cur); rows.push(row); }}
  return rows;
}}

function importCSV(ev) {{
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {{
    const raw = e.target.result.replace(/^\\uFEFF/, '');
    const parsed = parseCSVText(raw);
    if (parsed.length < 2) return;
    const hdr = parsed[0];
    let updated = 0;
    for (let i = 1; i < parsed.length; i++) {{
      const vals = parsed[i];
      if (vals.length < hdr.length) continue;
      const obj = {{}};
      hdr.forEach((h, k) => {{ obj[h] = vals[k]; }});
      const target = rowById[obj.item_id];
      if (target) {{
        COLUMNS.forEach(c => {{
          if (c.startsWith('coder2_') && obj[c] !== undefined) target[c] = obj[c];
        }});
        if (target.coder2_independent_status) updated++;
      }}
    }}
    saveState();
    renderFields();
    alert('成功导入 ' + updated + ' 条编码记录！');
  }};
  reader.readAsText(file, 'utf-8');
}}

function resetAllProgress() {{
  const filledCount = stateRows.filter(r => COLUMNS.some(c => c.startsWith('coder2_') && (r[c] || '').trim() !== '')).length;
  if (filledCount === 0) {{
    alert('当前没有任何已保存的编码进度，无需重置。');
    return;
  }}
  if (!confirm(`⚠️ 确定要清空全部已有编码进度吗？\\n（当前已有 ${{filledCount}} 项填写记录，清空后浏览器本地进度将全部归零，不可恢复；如需备份请先点【导出 CSV】）`)) {{
    return;
  }}
  stateRows.forEach(r => {{
    COLUMNS.forEach(c => {{
      if (c.startsWith('coder2_')) r[c] = '';
    }});
  }});
  localStorage.removeItem('paper2_dual_coding_rows_v1');
  const fileInput = document.getElementById('import-csv');
  if (fileInput) fileInput.value = '';
  curEvalId = evalList[0];
  updateProgress();
  selectEval(curEvalId);
  alert('已清空全部文献编码进度！');
}}

window.addEventListener('resize', () => {{
  if (window.innerWidth >= 900) {{
    document.getElementById('pane-form').style.display = 'block';
    document.getElementById('pane-text').style.display = 'block';
  }} else {{
    switchTab(document.getElementById('tab-form').classList.contains('active') ? 'form' : 'text');
  }}
}});

updateProgress();
selectEval(curEvalId);
if (window.innerWidth < 900) switchTab('form');
</script>
</body>
</html>
"""
    out_html.write_text(html_content, encoding="utf-8")
    print(f"[Task 2] Wrote {out_html} ({out_html.stat().st_size / 1024 / 1024:.2f} MB)")


def build_plain_guide():
    guide_path = ROOT_DIR / "docs/零基础跨端标注极简指南.md"
    guide_md = """# 论文2 零基础跨端标注极简指南（不需培训 · 打开网页就能标）

为了让您或任何零基础的合作者（不需要懂钢铁制造、不需要装 Python、不需要翻文件夹找文件）能在手机、平板或电脑上随时随地完成标注，我已经把两个人工盲审任务做成了**两个完全独立的单文件网页（内置全部图片与26篇论文全文）**。

---

## 一、我们要标注哪两件事？需不需要培训？

| 任务名称 | 标注什么内容 | 题目数量 | 预计耗时 | 是否需要培训 | 单文件网页位置（可微信/QQ直接发给任何人） |
|---|---|---|---|---|---|
| **任务 1：钢材图像重叠盲审** | 看两张图是不是拍到了同一块钢板的同一个局部位置（二选一：是 `1` / 不是 `0`） | **349 对图** | 约 **15–20 分钟** | **完全不需要**（连连看级别，网页内置30秒正反例图示） | `论文2/results/human_audit/mobile_duplicate_pair_review.html` |
| **任务 2：文献协议独立第二编码** | 看 26 篇论文里有没有写清楚“几比几划分、跑了多少轮、batch多大”等 17 个小问题 | **26 篇论文**（35个数据集评估，共595格） | 约 **1.5–2 小时**（可分多次做，自动存进度） | **不需要专门培训**（网页已将17个指标翻译成大白话问题，配有一键搜词按钮） | `论文2/docs/coding_v1/human_dual_coding/mobile_literature_dual_coding.html` |

> **重要提醒**：之前那个 2,828 行的旧大表（`results/human_audit/independent_dual_coding_sheet.csv`）**已经正式作废**，千万不要去填它！现在唯一在用的文献编码任务就是上面这 **26 篇固定样本（595格）**。

---

## 二、任务 1（349对图像）怎么标？（连连看 3 秒一题）

### 1. 打开方式
- 直接双击打开 `论文2/results/human_audit/mobile_duplicate_pair_review.html`，或者把这个 `.html` 文件通过微信/飞书发到手机、iPad 或合作者电脑上，用任意浏览器打开即可（所有 349 张对比图已全部打包在网页内部，无需联网）。

### 2. 30 秒判定口诀
每道题你会看到一张**左、中、右三合一图**：
- **左图（Image A）**：原图 A
- **中图（Image B）**：算法自动平移/旋转对齐后的图 B（黑边是因为平移后留出的空白）
- **右图（Heatmap）**：两图重叠区域的差异热力图（**深蓝色 = 完全吻合**；**红黄色 = 差异很大**）

**你只需要做二选一：**
- 点绿色 **【✅ 是重叠 [1]】**：中图和左图的**瑕疵形状、划痕走势、背景斑点纹理在重叠区一模一样**（哪怕整体变亮/变暗了一点，或者挪动了位置），且右图主体是深蓝色。
- 点红色 **【❌ 不是重叠 [0]】**：虽然都是划痕或麻点，但**具体的纹路、斑点位置完全对不上**，或者右图出现大片红黄色错位，说明只是长得像的两张不同照片。

### 3. 交差与重置方式
- 每点一次按钮，网页自动保存到浏览器缓存并自动跳到下一题。
- 如需重新开始或清除试标数据，点击网页右上角红色 **【重置清空】** 按钮（带防误触确认弹窗），即可一键清空浏览器内的全部已有进度。
- 全部标完 349 对后，点击网页右上角 **【导出 CSV】**，会下载 `pair_labels_to_fill.csv`。
- 把下载的 `pair_labels_to_fill.csv` 放回 `论文2/results/human_audit/pair_labels_to_fill.csv` 覆盖即可。

---

## 三、任务 2（26篇文献双编码）怎么标？（点词搜索 + 选中复制）

### 1. 打开方式
- 用电脑或平板浏览器打开 `论文2/docs/coding_v1/human_dual_coding/mobile_literature_dual_coding.html`（手机也能打开，会自动切换为“答题卡 / 论文全文”双标签页）。
- 26 篇论文的全文和表格已经全部内置在网页左边，**不需要去文件夹里找 XML 或 TXT**，而且网页完全隐藏了第一编码者的答案，保证 100% 严格盲审。

### 2. 三步极速标完一篇论文（每篇约 3–4 分钟）
1. **点题目卡片自动出现搜索词**：点击右侧任意一题（如 `F10 · 训练轮数`），左侧搜索栏下方会自动浮现蓝色搜词按钮（如 `epoch`、`100`、`200`、`300`）。
2. **点蓝色词直接跳到原文**：点击蓝色搜词按钮，左侧正文会自动高亮所有命中位置并滚动过去。
3. **选状态 + 一键填引文**：
   - **明确写了** $\\rightarrow$ 点 `1. reported (明确报告)`，然后用鼠标在左侧刷选那句原文，点左上角绿色按钮 **【📋 将鼠标选中文字填入当前题引文】**，再顺手在“原文位置”填一下小节号（如 `Sec 4.3` 或 `Table 1`）。
   - **写了但前后矛盾/含糊**（例如正文说 test set，表头写 val set） $\\rightarrow$ 点 `2. ambiguous (含糊/冲突)`，同样选中那句话点一键填入引文。
   - **全文没提** $\\rightarrow$ 直接点 `3. not_reported_in_paper (未提及)`（无需填引文！其中 `F03 划分名单`、`F04 重复检查`、`F17 公开权重` 98%以上的论文都没提，可点顶部【一键填检查范围与未找到项】快速预填）。
   - **两个必选下拉框注意**：
     - `F08（验证与测试集关系）`：在下拉框里选 `separate`（独立分开）、`same_set`（同一集合）或 `unclear`（分不清）。
     - `F14（对比基线来源）`：在下拉框里选 `rerun`（作者自己重训了对比模型）、`copied`（抄别人论文的数字）、`mixed`（混合）或 `unclear`。

### 3. 交差与重置方式
- 随时关掉网页都不会丢进度（自动保存在浏览器本地，也支持随时导出 CSV 换设备导入继续标）。
- 如需清除试填内容重新开始，点击右上角红色 **【重置清空】** 按钮（带二次确认弹窗），即可一键清空所有已填编码进度。
- 标完后点击右上角 **【导出 CSV】**，会下载 `blank_dual_coding_sheet.csv`。
- 把它放回 `论文2/docs/coding_v1/human_dual_coding/blank_dual_coding_sheet.csv` 覆盖，然后运行计算脚本即可一键算出真实双人 Cohen's $\\kappa$ 和 Gwet's $AC_1$。
"""
    guide_path.write_text(guide_md, encoding="utf-8")
    print(f"[Guide] Wrote {guide_path}")


if __name__ == "__main__":
    build_task1_image_app()
    build_task2_literature_app()
    build_plain_guide()

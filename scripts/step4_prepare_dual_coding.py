#!/usr/bin/env python3
"""Prepare the single active 26-paper package; never resample or overwrite labels."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT_DIR / "docs/coding_v1/human_dual_coding"
MANIFEST_PATH = PACKAGE_DIR / "sample_26_manifest.json"
BLIND_PATH = PACKAGE_DIR / "blank_dual_coding_sheet.csv"
KEY_PATH = PACKAGE_DIR / "coder1_key.csv"
LEGACY_PATH = ROOT_DIR / "results/human_audit/independent_dual_coding_sheet.csv"
FIELDS = [f"F{i:02d}" for i in range(1, 18)]
BLIND_COLUMNS = [
    "item_id", "eval_id", "paper_id", "pmcid", "dataset", "field_id",
    "article_title", "doi", "article_url", "fulltext_xml", "fulltext_text",
    "coder2_independent_status", "coder2_independent_value",
    "coder2_independent_quote", "coder2_independent_location",
    "coder2_checked_scope", "coder2_unavailable_materials",
    "coder2_public_material_status", "coder2_public_material_urls",
    "coder2_public_material_access_date", "coder2_public_material_version",
    "coder2_public_material_files", "coder2_public_material_model_result",
    "coder2_public_material_evidence", "coder2_combined_status",
    "coder2_combined_rationale", "coder2_notes",
]
KEY_COLUMNS = [
    "item_id", "eval_id", "paper_id", "pmcid", "dataset", "field_id",
    "audit_part", "coder1_status", "coder1_value", "coder1_quote",
    "coder1_location", "coder1_source_json", "coder1_source_layer",
    "coder1_paper_status", "coder1_paper_value", "coder1_paper_quote",
    "coder1_paper_location", "coder1_checked_scope",
    "coder1_unavailable_materials", "coder1_paper_layer_checked",
    "coder1_verifier", "coder1_verification_date",
    "coder1_public_material_status", "coder1_public_material_evidence",
    "coder1_combined_status", "coder1_combined_rationale",
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def csv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def neutral_sources(pmcid):
    xml_path = ROOT_DIR / f"docs/fulltext_v1/xml/{pmcid}.xml"
    txt_matches = list((ROOT_DIR / "docs/fulltext_v1/txt_v2").glob(f"*_{pmcid}.txt"))
    if not xml_path.exists() or len(txt_matches) != 1:
        raise ValueError(f"Missing full-text source for {pmcid}")
    txt_path = txt_matches[0]
    meta = ET.parse(xml_path).getroot().find("./front/article-meta")
    title = meta.find("./title-group/article-title")
    doi = meta.find("./article-id[@pub-id-type='doi']")
    return {
        "article_title": " ".join("".join(title.itertext()).split()),
        "doi": doi.text if doi is not None else "",
        "article_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        "fulltext_xml": str(xml_path), "fulltext_text": str(txt_path),
    }


def build_rows():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    identities = {(Path(x["file"]).stem, x["dataset"]) for x in manifest}
    if len(manifest) != 35 or len(identities) != 35:
        raise ValueError("Preserved manifest must contain 35 unique evaluations")
    if len({x["file"] for x in manifest}) != 26:
        raise ValueError("Preserved manifest must contain 26 papers")
    blind, key = [], []
    for item in manifest:
        source = ROOT_DIR / "docs/coding_v1/out_v2" / item["file"]
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        if data["pmcid"] != item["pmcid"]:
            raise ValueError(f"PMCID mismatch: {source}")
        matches = [d for d in data["datasets"] if d["dataset"] == item["dataset"]]
        if len(matches) != 1:
            raise ValueError(f"Dataset identity is not unique: {source}")
        metadata = neutral_sources(item["pmcid"])
        for field_id in FIELDS:
            original = matches[0]["fields"][field_id]
            eval_id = f"{source.stem}_{item['dataset']}"
            item_id = hashlib.sha256(f"{eval_id}|{field_id}".encode()).hexdigest()[:20]
            identity = dict(item_id=item_id, eval_id=eval_id, paper_id=source.stem,
                            pmcid=item["pmcid"], dataset=item["dataset"], field_id=field_id)
            blind.append({**dict.fromkeys(BLIND_COLUMNS, ""), **identity, **metadata})
            key.append({
                **dict.fromkeys(KEY_COLUMNS, ""), **identity,
                "audit_part": "fixed_26_paper_sample",
                "coder1_status": original["status"],
                "coder1_value": original.get("value", ""),
                "coder1_quote": original.get("evidence", ""),
                "coder1_location": original.get("location", ""),
                "coder1_source_json": str(source),
                "coder1_source_layer": "legacy_not_separated_requires_verification",
            })
    return manifest, blind, key


def validate_existing(path, columns, expected, allow_empty_wide=False):
    """Return whether creation/migration is needed; preserve all filled data."""
    if not path.exists():
        return True
    header, rows = read_csv(path)
    if allow_empty_wide and "F01_status" in (header or []):
        # Only the entirely blank original wide template can be migrated once.
        label_cols = [f"{f}_{s}" for f in FIELDS for s in ("status", "quote")]
        label_cols.append("coder_notes")
        expected_header = ["sample_idx", "pmcid", "dataset"] + label_cols
        if header != expected_header:
            raise ValueError(f"Unexpected legacy schema; preserving {path}")
        if any(str(row.get(col, "")).strip() for row in rows for col in label_cols):
            raise ValueError(f"Refusing to migrate a filled template: {path}")
        old_ids = {(x["pmcid"], x["dataset"]) for x in rows}
        new_ids = {(x["pmcid"], x["dataset"]) for x in expected}
        if len(rows) != 35 or old_ids != new_ids:
            raise ValueError("Wide template differs from preserved manifest")
        return True
    if header != columns:
        raise ValueError(f"Existing schema differs; preserving {path}")
    ids = [row["item_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate items: {path}")
    indexed = {row["item_id"]: row for row in rows}
    if set(indexed) != {row["item_id"] for row in expected}:
        raise ValueError(f"Existing items differ from fixed sample: {path}")
    fixed = [c for c in columns if not c.startswith("coder2_")]
    if path == KEY_PATH:
        fixed = columns[:13]  # identifiers and imported legacy evidence
    for row in expected:
        if any(indexed[row["item_id"]][c] != str(row[c]) for c in fixed):
            raise ValueError(f"Existing source/identity differs; preserving {path}")
    return False  # never rewrite even whitespace or quotes in existing labels


def legacy_summary():
    _, rows = read_csv(LEGACY_PATH)
    counts = Counter(x["audit_part"] for x in rows)
    full = [x for x in rows if x["audit_part"] == "Part_1_Stratified_Full_17_Fields"]
    extra = [x for x in rows if x["audit_part"] == "Part_2_High_Controversy_Full_Census"]
    p1, p2 = {x["eval_id"] for x in full}, {x["eval_id"] for x in extra}
    if len(rows) != 2828 or len(full) != 2771 or len(extra) != 57:
        raise ValueError("Historical item counts changed; no automatic migration")
    if len(p1) != 163 or len(p2) != 19 or p1 & p2:
        raise ValueError("Unexpected historical evaluation membership")
    expected = {(ev, f) for ev in p1 for f in FIELDS}
    expected |= {(ev, f) for ev in p2 for f in ("F07", "F08", "F14")}
    observed = {(x["eval_id"], x["field_id"]) for x in rows}
    if observed != expected or len(observed) != len(rows):
        raise ValueError("Historical field allocation is inconsistent")
    pool, priority = set(), set()
    for source in sorted((ROOT_DIR / "docs/coding_v1/out_v2").glob("*.json")):
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        for ds in data.get("datasets", []):
            ev = f"{source.stem}_{ds['dataset']}"
            pool.add(ev)
            if any(ds["fields"].get(f, {}).get("status") in ("reported", "ambiguous")
                   for f in ("F04", "F07", "F08", "F14")):
                priority.add(ev)
    if p1 != priority or p2 != pool - priority:
        raise ValueError("Historical priority rule differs from source coding")
    return {
        "sha256": hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest(),
        "rows": len(rows), "part_counts": dict(counts), "full_evaluations": len(p1),
        "additional_evaluations": len(p2), "priority_rule_verified": True,
        "already_filled_coder2": sum(bool(x["coder2_independent_status"].strip()) for x in rows),
    }


INSTRUCTIONS = """# 独立人工编码说明：唯一在用的26篇固定样本

本包服务论文2报告完整性结果及其信度限制。沿用原`sample_26_manifest.json`：26篇、35个论文—数据集评估、595个字段评分；不重新抽样。材料按冻结手册v1.0及v1.1–v1.4整理，属于观察已有编码后的操作修复，不称预注册。

## 分发与盲法

只给第二编码者本说明、`blank_dual_coding_sheet.csv`、冻结手册及论文原文。该CSV是唯一在用填写表，现为595行长表；包含标题、DOI、公开全文URL、本地XML和v2正文路径，供自行定位原文。路径以当前D盘工作区为准。

`coder1_key.csv`仅由主编码者/协调者保管，不分发；它含历史第一编码与核查栏。第二编码者不要查看out/out_v2、out_retest_20pct、历史信度报告、旧2828行表、主文汇总或样本选择依据。盲表不显示第一编码、既有引文、既有字段定位、分层归属或优先原因。若此前看过相关结果，在`coder2_notes`记录暴露范围，不能宣称完全盲法。

先独立填完并锁定原始标签，再由协调者比较；分歧裁定另存记录，不修改原始标签。`step4_prepare_dual_coding.py`重跑保留既有CSV字节；它只允许将全空旧35行宽表一次性迁移成长表，发现已填写内容则拒绝迁移。

## 两层来源与填写顺序

1. `coder2_independent_status`是**论文材料层**原始状态；`_value`绑定具体数据集、模型和主结果，`_quote`填不超过35词的逐字引文，`_location`填节/表/图/脚注，`coder2_checked_scope`列正文、XML表、脚注、附录及取得的补充材料，`coder2_unavailable_materials`记录缺失/不可读/未取得材料。不要仅凭抽取文本、关键词未命中或空白段落判断未报告。
2. 论文指向的公开材料另填`coder2_public_material_*`：状态可写`complements`（补足）、`partial`、`conflict`、`no_supplement`（已查未补足）、`not_applicable`或`not_assessed`；记录URL、访问日期、版本/commit、文件、适用模型及结果对象、证据。未检查或失效不能记为不存在。F16/F17分别保存论文链接声明与实际代码/权重访问结果。
3. 如需综合状态，填`coder2_combined_status`及理由；最终`not_reported`必须完成适用的公开材料核查。保留论文层，不能用综合状态覆盖它。当前IRR只比较经核实的两位编码者的论文层原始状态，不读取公开层、综合层或裁定结果。

论文层允许：`reported`（明确信息，附引文与位置）、`ambiguous`（不充分或冲突，附全部冲突位置）、`not_reported_in_paper`（完整检查已取得论文材料仍未发现，须写检查范围）、`not_applicable`（写适用性理由）、`not_assessed`（未检查、不可获得或取证不全）。未取得相关材料时保留未评估，不填成未报告。综合层另可使用`not_reported`，不可将它混入论文层。

## F01–F17：与冻结手册一致的最低要求

|字段|内容与reported最低要求|
|---|---|
|F01 数据集版本|图片数、类别数、来源；给出图片总数或明确来源链接。不能据此声称样本身份可重建。|
|F02 划分角色与比例|train/val/test或CV的角色和数量（或比例）都给出。|
|F03 划分可重建|公开文件名单或可运行的划分代码；种子单独出现不够。|
|F04 重复/近重复处理|明确描述检查或处理；不要求本项目某个特定算法。|
|F05 预训练|逐模型明确从零训练或权重来源；不凭架构常识推断。|
|F06 检查点选择|写明选择最终权重所依据的集合和准则。|
|F07 报告指标所用集合|绑定主结果表，明确val、test或CV。|
|F08 验证集与测试集关系|派生字段：检查点选择与指标报告是否同一集合。value写same_set/separate/unclear；明确同一集合也可为reported，不把“是否报告”改为“是否符合最佳实践”。|
|F09 输入尺寸|训练和测试输入尺寸，给出数值。|
|F10 训练轮数|epochs或iterations，给出数值；不额外强制patience。|
|F11 batch大小|训练batch，给出数值；不额外强制梯度累积。|
|F12 随机种子|给出种子数值；不等于进行了多次训练。|
|F13 重复运行与变异|多次运行并报告离散度。未报告不等于实际只运行一次，按v1.2绑定具体主结果。|
|F14 基线来源|作者重训还是抄录；value为rerun/copied/mixed或说明不确定性，不是置信度阈值。逐模型记录，不额外强制同硬件。|
|F15 速度测量条件|记录硬件、batch、精度、是否含前后处理；reported最低为硬件与batch。|
|F16 代码可用性|公开链接当次可访问且含训练代码才满足综合reported；论文链接声明和访问结果分层。|
|F17 权重可用性|公开权重链接当次可访问才满足综合reported；论文声明和实际访问分层。|

F08/F14的内容值与reported等状态分开保存。一个数据集/模型的证据不能自动外推另一对象。保留所有限制，不根据低kappa反推文献必然有歧义。

## 第一编码核查与统计

历史out_v2为AI辅助初编/主流程核验记录，不自动视为真实人工第一编码；旧AI辅助重测不能代替第二位人工。协调者保留key中`coder1_status/value/quote/location`原值；主编码者核对来源层后填写`coder1_paper_*`、检查范围、核查者及日期，并将`coder1_paper_layer_checked`设为`yes`。不能机械复制旧标签宣称已核查。

填表后运行`论文2/scripts/step4_compute_irr_metrics.py`。它只用核实的第一论文层与第二论文层raw标签计算Po、Cohen kappa及固定类别AC1；分别报告字段、固定样本部分及合并描述量。未评估/空白独立列出，不变成未报告。部分完成只报部分，空表不回退历史重测。Pe=1时kappa为null而非1；未来95%区间按paper聚类重抽，同篇数据集及字段一起抽入；披露未定义重复数，出现未定义kappa重复时不输出其条件区间。

这些区间仅是本来源与固定样本内的描述，不能向全部NEU/GC10文献推广；原26篇选择发生在既有初编之后，不能称事前独立验证。人工身份、来源核查和独立性需流程记录，脚本成功不证明这些事实。旧2828行富集包已弃用，无需另做该表。
"""


def main():
    manifest, blind, key = build_rows()
    legacy = legacy_summary()
    write_blind = validate_existing(BLIND_PATH, BLIND_COLUMNS, blind, allow_empty_wide=True)
    write_key = validate_existing(KEY_PATH, KEY_COLUMNS, key)
    # Finish safety checks before any mutation.
    if write_key:
        KEY_PATH.write_bytes(csv_bytes(KEY_COLUMNS, key))
    if write_blind:
        BLIND_PATH.write_bytes(csv_bytes(BLIND_COLUMNS, blind))
    (PACKAGE_DIR / "INSTRUCTIONS_FOR_HUMAN_COAUTHOR.md").write_text(INSTRUCTIONS, encoding="utf-8")
    active = {
        "status": "single_active_package_awaiting_human_coding",
        "manifest": str(MANIFEST_PATH),
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "papers": 26, "evaluations": 35, "field_items": len(blind),
        "dataset_evaluations": dict(Counter(x["dataset"] for x in manifest)),
        "sampling": "Preserved original manifest; no resampling; after initial coding, not preregistered",
        "legacy_2828": legacy,
    }
    (PACKAGE_DIR / "sample_provenance.json").write_text(json.dumps(active, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    note = f"""# 固定样本与历史大表来源核查（2026-11-05）

唯一在用物料为本目录`blank_dual_coding_sheet.csv`（595行盲表）、内部`coder1_key.csv`和`INSTRUCTIONS_FOR_HUMAN_COAUTHOR.md`。原`sample_26_manifest.json`逐条与out_v2及本地全文核对：26篇、35个不同论文—数据集、35×17=595评分；未重抽样、未修改清单。旧35行宽表原无任何标签或备注，安全转换为长表；以后重跑不覆盖已填值。

历史大表`results/human_audit/independent_dual_coding_sheet.csv`按原字节保留，不再是任务。SHA256为`{legacy['sha256']}`。原代码先取F04/F07/F08/F14任一为reported/ambiguous的全部评估，实际已有163个，超过target=48，所以没有补抽其他对象：Part1=163×17=2771行。剩余19个评估仅补F07/F08/F14，Part2=19×3=57行；2771+57=2828。F07/F08/F14覆盖182评估的行已分布在两部分，不能称Part2本身为182评估。

本次逐条核对优先集合与原out_v2完全相符；它是基于既有状态的富集选择，不是独立分层随机样本，不得推广为全母体信度。现采用原26篇固定样本，弃用大包而非新增一套任务；原固定样本的既有结果观察限制同样须披露，不称预注册。旧大表第二编码已填{legacy['already_filled_coder2']}行；保留不代表已完成人工双编码。

key保留历史状态和值，不将AI辅助记录自动称为人工。论文层与公开层按v1.4分开，正式信度等待真实独立第二编码及第一层来源核查；裁定值不进入信度。现有材料不证明信度已完成或低信度一定来自文献歧义。
"""
    (PACKAGE_DIR / "SAMPLE_PROVENANCE.md").write_text(note, encoding="utf-8")
    deprecated = """# 已弃用：2,828行编码包

2026-11-05：只保留原26篇/35评估/595评分包为在用任务。请使用`../../docs/coding_v1/human_dual_coding/INSTRUCTIONS_FOR_HUMAN_COAUTHOR.md`及其中`blank_dual_coding_sheet.csv`。此目录的`independent_dual_coding_sheet.csv`保留原字节作历史记录，包含第一编码，不得发给独立第二编码者，不再填写或用于计算本次信度。

旧包实际为Part1 163×17=2771、Part2剩余19×3=57，共2828；由既有F04/F07/F08/F14状态富集，不是48项随机样本。F14是基线来源。来源、校验和及限制见在用目录`SAMPLE_PROVENANCE.md`。旧AI辅助重测不能代替人工独立编码。
"""
    for name in ("INSTRUCTIONS_DUAL_CODING.md", "DEPRECATED_DUAL_CODING.md"):
        (LEGACY_PATH.parent / name).write_text(deprecated, encoding="utf-8")
    print(f"Active package verified: 26 papers / 35 evaluations / {len(blind)} items")
    print(f"Blind sheet: {BLIND_PATH}")
    print(f"Internal primary key (do not distribute): {KEY_PATH}")
    print(f"Historical 2,828-row CSV preserved: {legacy['sha256']}")


if __name__ == "__main__":
    main()

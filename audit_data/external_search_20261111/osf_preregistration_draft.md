# OSF 预注册方案草稿：非 PMC 订阅数据库代表性检验与双盲报告完整性审计

**Title**: Prospective Representativeness Audit of Non-PMC Subscription Literature and Dual-Human Coding Protocol for Steel Surface Defect Detection Benchmarks (NEU-DET and GC10-DET)

**Authors**: Wenzhuo Liu, Shiling Yu (Anhui University)

**Date of Protocol Freeze**: November 11, 2026

---

## 1. Background and Rationale (Retrospective vs. Prospective Delineation)

In our initial systematic audit conducted between October 30 and November 1, 2026, we queried the Europe PMC REST API across publications from January 1, 2020 to October 30, 2026 (`q1`: `steel AND defect AND yolo*`; `q2`: `steel AND defect AND (detect* OR inspection)`; `q3`: `"NEU-DET" OR "GC10-DET"`), identifying 433 unique records and auditing 131 open-access journal articles with PubMed Central (PMC) JATS XML full texts (182 benchmark evaluations). Because that search was executed on October 30, 2026 and the coding manual (v1.0–v1.4) was frozen on November 1, 2026 prior to full-text extraction, the initial PMC audit is explicitly classified as a **retrospectively documented protocol without prior public registration**.

To test whether open-access articles indexed in PMC exhibit systematic reporting differences compared with subscription-indexed engineering venues (e.g., IEEE, Elsevier, Springer), we prospectively register this expansion protocol prior to sampling and coding external subscription records.

## 2. Objectives and Research Questions

1. **Dual-Human Reliability on the Primary PMC Corpus**: What is the inter-annotator agreement (Cohen's $\kappa$ and Gwet's $\text{AC1}$) when both authors independently code all 182 evaluations on the six core reproducibility dimensions (`F03`, `F04`, `F07`, `F08`, `F12`, `F13`), a 30% stratified random sample on the remaining 11 dimensions, and reported baseline-to-proposed $\Delta\text{mAP@0.5}$ claims?
2. **Representativeness of the PMC Sampling Frame**: Do reporting completeness rates and claimed $\Delta\text{mAP@0.5}$ distributions in a fixed-seed random sample of $N = 60$ eligible non-PMC studies indexed in Scopus / Web of Science differ significantly from the 131-paper PMC corpus?

## 3. Search Strategy and Sampling Frame

- **Databases**: Scopus and/or Web of Science Core Collection.
- **Query Semantics**: Identical boolean structure to `q1`, `q2`, and `q3`, restricted to English-language journal articles and conference proceedings published between January 1, 2020 and October 30, 2026.
- **Deduplication against Primary Corpus**: All exported records are normalized by lowercase DOI (stripping URL prefixes) and normalized title (lowercase alphanumeric). Any record matching the 433-record Europe PMC screening ledger (`docs/search_europepmc_20261030/candidates.csv`) by DOI, PMID, PMCID, or exact normalized title is removed.
- **Stratified Random Sampling**: From the remaining non-PMC candidate pool mentioning `NEU-DET`, `GC10-DET`, or steel surface defect detection in title/abstract/keywords, a deterministic pseudorandom generator (`numpy.random.default_rng(20261111)`) draws a primary sample of 60 records stratified by publication year (2020–2022, 2023–2024, 2025–2026), plus an ordered reserve list of 40 records.
- **Replacement Rule**: Reserve records are used strictly in pre-drawn numerical order (`R061`, `R062`, ...) only when a primary record fails pre-specified eligibility criteria `E1`–`E6` during screening or full-text verification. Every exclusion is logged with its exact code (`E1`: non-original research; `E2`: no bounding-box object detection; `E3`: no experiment on NEU-DET or GC10-DET; `E4`: duplicate/superseded version; `E5`: full text unobtainable via institutional subscriptions; `E6`: non-English).

## 4. Dual-Human Coding and Statistical Plan

- **Annotators**: Both authors (Wenzhuo Liu and Shiling Yu) independently code all records without access to each other's ratings. Any LLM pre-extraction is restricted to locating candidate paragraphs/tables and is hidden during independent second coding.
- **Core Audited Fields**:
  - `F03` (Split reconstructibility: manifest or seed + script)
  - `F04` (Duplicate / near-duplicate audit reported)
  - `F07` (Evaluation split identity clearly stated)
  - `F08` (Validation vs. test separation documented)
  - `F12` (Random seed disclosed)
  - `F13` (Repeated runs and dispersion reported)
  - `Claimed Delta mAP@0.5` (Baseline model name, baseline mAP@0.5, proposed mAP@0.5, $\Delta$ in percentage points, whether baseline was retrained in-house `F14`, and whether link in `F16` resolves via HTTP/browser check).
- **Statistical Reporting**:
  - Inter-coder agreement is reported using raw percent agreement ($P_o$), Cohen's $\kappa$, and Gwet's $\text{AC1}$ (to prevent zero/high-prevalence paradoxes), prior to consensus adjudication.
  - All reporting proportions are accompanied by Wilson score 95% confidence intervals.
  - PMC vs. non-PMC rates are compared via two-sample Newcombe-Wilson 95% confidence intervals for differences in proportions and Fisher's exact test with Holm-Bonferroni correction across the 6 core fields. Claimed $\Delta\text{mAP@0.5}$ distributions are compared via the two-sample Mann-Whitney U test and Kolmogorov-Smirnov test.

#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pmc_dir = ROOT / "docs" / "fulltext_v1" / "xml"


def tag(e):
    return e.tag.split("}")[-1] if e.tag else ""


def get_meta(pmcid):
    p = pmc_dir / f"{pmcid}.xml"
    if not p.exists():
        return None
    root = ET.parse(p).getroot()
    title, journal, doi, yr, vol, pg = "", "", "", "", "", ""
    authors = []
    for e in root.iter():
        t = tag(e)
        if t == "article-title" and not title:
            title = " ".join("".join(e.itertext()).split())
        elif t == "journal-title" and not journal:
            journal = " ".join("".join(e.itertext()).split())
        elif t == "article-id" and e.attrib.get("pub-id-type") == "doi" and not doi:
            doi = (e.text or "").strip()
        elif t == "pub-date" and not yr:
            for ch in e:
                if tag(ch) == "year" and ch.text:
                    yr = ch.text.strip()
        elif t == "volume" and not vol and e.text:
            vol = e.text.strip()
        elif t in ("fpage", "elocation-id") and not pg and e.text:
            pg = e.text.strip()
        elif t == "name":
            sn, gn = "", ""
            for ch in e:
                if tag(ch) == "surname" and ch.text:
                    sn = ch.text.strip()
                elif tag(ch) == "given-names" and ch.text:
                    gn = ch.text.strip()
            if sn and f"{sn}, {gn}" not in authors:
                authors.append(f"{sn}, {gn}")
    journal = journal.replace(" (Basel, Switzerland)", "")
    return {
        "pmcid": pmcid,
        "title": title,
        "journal": journal,
        "doi": doi,
        "year": yr,
        "vol": vol,
        "pages": pg,
        "authors": " and ".join(authors[:6]) + (" and others" if len(authors) > 6 else ""),
    }


def main():
    bib_path = ROOT / "manuscript" / "refs.bib"
    orig = bib_path.read_text(encoding="utf-8")
    idx_end = orig.find("@article{bourezgui2026rock,")
    if idx_end != -1:
        end_brace = orig.find("}", idx_end)
        orig = orig[: end_brace + 1]

    orig = orig.replace(
        """@misc{jeong2026zenodo,
  author       = {Jeong, H. and Song, K.},
  title        = {Surface Defect Detection Benchmark and Multi-Seed Evaluation Archive},
  howpublished = {Zenodo Repository},
  month        = oct,
  year         = {2026},
  doi          = {10.5281/zenodo.23073155}
}""",
        """@misc{jeong2026zenodo,
  author       = {Jeong, H. and Song, K.},
  title        = {Surface Defect Detection Benchmark and Multi-Seed Evaluation Archive},
  howpublished = {Zenodo Repository},
  month        = oct,
  year         = {2026},
  doi          = {10.5281/zenodo.23073155},
  note         = {DOI: 10.5281/zenodo.23073155}
}""",
    ).replace(
        """@misc{li2026figshare,
  author       = {Li, Zehao},
  title        = {Filename-group splits, detector predictions and inference timing for steel surface defect detection},
  howpublished = {Figshare Dataset},
  month        = oct,
  year         = {2026},
  doi          = {10.6084/m9.figshare.34070919.v2}
}""",
        """@misc{li2026figshare,
  author       = {Li, Zehao},
  title        = {Filename-group splits, detector predictions and inference timing for steel surface defect detection},
  howpublished = {Figshare Dataset},
  month        = oct,
  year         = {2026},
  doi          = {10.6084/m9.figshare.34070919.v2},
  note         = {DOI: 10.6084/m9.figshare.34070919.v2}
}""",
    )

    extra_foundational = r"""
@inproceedings{woo2018cbam,
  author    = {Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
  title     = {{CBAM}: Convolutional Block Attention Module},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  pages     = {3--19},
  year      = {2018},
  publisher = {Springer},
  doi       = {10.1007/978-3-030-01234-2_1}
}

@inproceedings{ouyang2023ema,
  author    = {Ouyang, Daliang and He, Su and Zhang, Guozhong and Luo, Mingzhu and Guo, Huaiyu and Zhan, Jian and Huang, Zhijie},
  title     = {Efficient Multi-Scale Attention Module with Cross-Spatial Learning},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {1--5},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/ICASSP49357.2023.10096516}
}

@inproceedings{tan2020efficientdet,
  author    = {Tan, Mingxing and Pang, Ruoming and Le, Quoc V.},
  title     = {{EfficientDet}: Scalable and Efficient Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10781--10790},
  year      = {2020},
  doi       = {10.1109/CVPR42600.2020.01079}
}

@inproceedings{ren2015faster,
  author    = {Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  title     = {Faster {R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {28},
  pages     = {91--99},
  year      = {2015}
}

@inproceedings{zhao2024rtdetr,
  author    = {Zhao, Yian and Lv, Wenyu and Xu, Shangliang and Wei, Jinman and Wang, Guanzhong and Dang, Qingqing and Liu, Yi and Chen, Jie},
  title     = {{DETRs} Beat {YOLOs} on Real-time Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {16965--16974},
  year      = {2024},
  doi       = {10.1109/CVPR52733.2024.01605}
}

@article{tabernik2020segmentation,
  author    = {Tabernik, Domen and {\v{S}}ela, Samo and Skvar{\v{c}}, Jure and Sko{\v{c}}aj, Danijel},
  title     = {Segmentation-based deep-learning approach for surface-defect detection},
  journal   = {Journal of Intelligent Manufacturing},
  volume    = {31},
  number    = {3},
  pages     = {759--776},
  year      = {2020},
  publisher = {Springer},
  doi       = {10.1007/s10845-019-01476-x}
}

@article{czimmermann2020visual,
  author    = {Czimmermann, Tam{\'a}s and Ciuti, Gastone and Milazzo, Mario and Chiurazzi, Marcello and Roccella, Stefano and Oddo, Calogero Maria and Dario, Paolo},
  title     = {Visual-Based Defect Detection and Classification Approaches for Industrial Applications---{A} Survey},
  journal   = {Sensors},
  volume    = {20},
  number    = {5},
  pages     = {1459},
  year      = {2020},
  publisher = {MDPI},
  doi       = {10.3390/s20051459}
}

@article{luo2020automated,
  author    = {Luo, Qiwu and Fang, Xiaoxin and Liu, Li and Yang, Chunhua and Sun, Yixin},
  title     = {Automated Visual Defect Detection for Flat Steel Surface: {A} Survey},
  journal   = {IEEE Transactions on Instrumentation and Measurement},
  volume    = {69},
  number    = {3},
  pages     = {626--644},
  year      = {2020},
  publisher = {IEEE},
  doi       = {10.1109/TIM.2019.2963555}
}

@article{wilcoxon1945individual,
  author    = {Wilcoxon, Frank},
  title     = {Individual Comparisons by Ranking Methods},
  journal   = {Biometrics Bulletin},
  volume    = {1},
  number    = {6},
  pages     = {80--83},
  year      = {1945},
  doi       = {10.2307/3001968}
}

@article{holm1979simple,
  author    = {Holm, Sture},
  title     = {A Simple Sequentially Rejective Multiple Test Procedure},
  journal   = {Scandinavian Journal of Statistics},
  volume    = {6},
  number    = {2},
  pages     = {65--70},
  year      = {1979}
}

@article{wilson1927probable,
  author    = {Wilson, Edwin B.},
  title     = {Probable Inference, the Law of Succession, and Statistical Inference},
  journal   = {Journal of the American Statistical Association},
  volume    = {22},
  number    = {158},
  pages     = {209--212},
  year      = {1927},
  doi       = {10.1080/01621459.1927.10502953}
}

@article{foroosh2002extension,
  author    = {Foroosh, Hassan and Zerubia, Josiane B. and Berthod, Marc},
  title     = {Extension of Phase Correlation to Subpixel Registration},
  journal   = {IEEE Transactions on Image Processing},
  volume    = {11},
  number    = {3},
  pages     = {188--200},
  year      = {2002},
  doi       = {10.1109/83.988953}
}

@article{lewis1995fast,
  author    = {Lewis, J. P.},
  title     = {Fast Normalized Cross-Correlation},
  journal   = {Vision Interface},
  volume    = {95},
  pages     = {120--123},
  year      = {1995}
}

@article{kuhn1955hungarian,
  author    = {Kuhn, Harold W.},
  title     = {The {Hungarian} Method for the Assignment Problem},
  journal   = {Naval Research Logistics Quarterly},
  volume    = {2},
  number    = {1-2},
  pages     = {83--97},
  year      = {1955},
  doi       = {10.1002/nav.3800020109}
}

@article{varoquaux2022evaluating,
  author    = {Varoquaux, Ga{\"e}l and Cheplygina, Veronika},
  title     = {Machine learning for medical imaging: methodological failures and recommendations for the future},
  journal   = {npj Digital Medicine},
  volume    = {5},
  pages     = {48},
  year      = {2022},
  doi       = {10.1038/s41746-022-00592-y}
}

@article{roberts2021common,
  author    = {Roberts, Michael and Driggs, Derek and Thorpe, Matthew and Gilbey, Julian and Yeung, Michael and Ursprung, Stephan and Aviles-Rivero, Angelica I. and Etmann, Christian and McCague, Cathal and Beer, Lucian and others},
  title     = {Common pitfalls and recommendations for using machine learning to detect and prognosticate for {COVID-19} using chest radiographs and {CT} scans},
  journal   = {Nature Machine Intelligence},
  volume    = {3},
  pages     = {199--217},
  year      = {2021},
  doi       = {10.1038/s42256-021-00307-0}
}

@inproceedings{recht2019imagenet,
  author    = {Recht, Benjamin and Roelofs, Rebecca and Schmidt, Ludwig and Shankar, Vaishaal},
  title     = {Do {ImageNet} Classifiers Generalize to {ImageNet}?},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  volume    = {97},
  pages     = {5389--5400},
  year      = {2019},
  publisher = {PMLR}
}
"""

    target_pmcids = [
        "PMC13708875", "PMC13004511", "PMC13030490", "PMC13040021", "PMC13194957",
        "PMC12491516", "PMC12736593", "PMC12129336", "PMC10300763", "PMC10582940",
        "PMC10675464", "PMC10719359", "PMC10935385", "PMC11001973", "PMC11091066",
        "PMC12298282", "PMC12899597", "PMC13562717", "PMC13618933", "PMC8838028",
        "PMC9783860", "PMC12986859", "PMC13418091", "PMC12350654", "PMC12770479",
        "PMC12881659", "PMC13029871", "PMC13056895", "PMC13280455", "PMC13412141",
    ]

    corpus_bib = []
    for pmcid in target_pmcids:
        m = get_meta(pmcid)
        key = m["authors"].split(",")[0].strip().lower().replace(" ", "").replace("-", "") + m["year"] + "_" + pmcid.lower()
        title_escaped = m["title"].replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
        journal_escaped = m["journal"].replace("&", r"\&")
        entry = f"""@article{{{key},
  author    = {{{m['authors']}}},
  title     = {{{title_escaped}}},
  journal   = {{{journal_escaped}}},
  volume    = {{{m['vol']}}},
  pages     = {{{m['pages']}}},
  year      = {{{m['year']}}},
  doi       = {{{m['doi']}}}
}}"""
        corpus_bib.append(entry)

    new_bib = orig.strip() + "\n" + extra_foundational + "\n" + "\n\n".join(corpus_bib) + "\n"
    bib_path.write_text(new_bib, encoding="utf-8")
    print("Updated refs.bib successfully! Total @ entries:", new_bib.count("@"))


if __name__ == "__main__":
    main()

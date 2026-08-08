import spacy
import re

# 載入 spaCy 英文模型
nlp = spacy.load("en_core_web_sm")

def calculate_mdd_and_memory_load(text: str):
    """
    計算 MDD (Mean Dependency Distance) 與大腦工作記憶超載點 (Open Arcs >= 4)
    理論來源: Liu (2008), Gao (2023)
    """
    doc = nlp(text)
    total_dd = 0
    words_count = 0
    overload_spans = []

    for sent_idx, sent in enumerate(doc.sents):
        tokens = [t for t in sent if not t.is_punct and not t.is_space]
        for token in tokens:
            # 依存距離 DD = |dependent_idx - head_idx|
            dd = abs(token.i - token.head.i)
            total_dd += dd
            words_count += 1

            # 檢測 Cowan 4-chunk 工作記憶臨界值
            open_arcs = sum(1 for t in sent if t.i < token.i and t.head.i > token.i)
            if open_arcs >= 4:
                overload_spans.append({
                    "sentence_id": sent_idx + 1,
                    "word": token.text,
                    "open_arcs": open_arcs,
                    "msg": f"單字 '{token.text}' 上空有 {open_arcs} 條跨越依存弧，已超過大腦工作記憶負荷 (>=4)。"
                })

    mdd = round(total_dd / words_count, 3) if words_count > 0 else 0
    return {
        "mdd": mdd,
        "is_overloaded": mdd >= 3.0,
        "overload_spans": overload_spans
    }

def calculate_l2sca_approximations(text: str):
    """
    估算 L2SCA 核心指標 (Lu, 2010; Ting, 2024; Li et al., 2025)
    包含: MLS, MLTU, MLC, C/T, C/S, DC/C, CNP/C
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    num_words = sum(1 for t in doc if not t.is_punct and not t.is_space)
    
    # 子句與名詞組計數 (以 spaCy POS/DEP 簡化識別)
    num_clauses = sum(1 for t in doc if t.pos_ == "VERB" or t.dep_ in ("ccomp", "xcomp", "advcl", "relcl"))
    num_clauses = max(num_clauses, num_sentences)
    
    num_dep_clauses = sum(1 for t in doc if t.dep_ in ("advcl", "relcl", "mark"))
    num_complex_nps = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN") and sum(1 for c in t.children if c.dep_ in ("amod", "prep", "relcl")) >= 2)

    mls = round(num_words / num_sentences, 2) if num_sentences > 0 else 0
    mlc = round(num_words / num_clauses, 2) if num_clauses > 0 else 0
    c_s = round(num_clauses / num_sentences, 2) if num_sentences > 0 else 0
    dc_c = round(num_dep_clauses / num_clauses, 2) if num_clauses > 0 else 0
    cnp_c = round(num_complex_nps / num_clauses, 2) if num_clauses > 0 else 0

    return {
        "MLS": mls,
        "MLC": mlc,
        "C/S": c_s,
        "DC/C": dc_c,
        "CNP/C": cnp_c
    }

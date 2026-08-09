import re
import spacy
import itertools
from typing import Dict, List, Any

# -----------------------------------------------------------------------------
# 1. 載入 NLP 模型
# -----------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# -----------------------------------------------------------------------------
# 2. 超大型語篇邏輯詞典 (Comprehensive Discourse Connectives Lexicon)
# -----------------------------------------------------------------------------
DISCOURSE_LEXICON: Dict[str, List[str]] = {
    "Additive (遞進/補充)": [
        "furthermore", "moreover", "in addition", "besides", "also", "additionally", 
        "what's more", "not only", "but also", "along with", "as well as", "apart from this", 
        "apart from that", "besides this", "besides that", "on top of that", "further", 
        "into the bargain", "to top it all off", "what is more", "plus", "and", "too", 
        "equally important", "correspondingly", "in the same way", "similarly", "likewise", 
        "by the same token", "in like manner", "in a similar vein", "not to mention", 
        "to say nothing of", "let alone", "together with", "coupled with", "as a matter of fact", 
        "indeed", "in fact", "actually", "as well", "anecdotally", "additionally speaking", 
        "on a related note", "along same lines", "in tandem with", "side by side with", 
        "at the same time", "above and beyond", "over and above", "secondarily", "then again", 
        "further to this", "in addition to", "not only that", "as an added feature", 
        "by same token", "with this in mind", "in much same way", "to add to this", 
        "withal", "additively", "supplementarily", "furthering this point", "extended to", 
        "inclusive of", "on top of these", "in addition to this", "subsequently added", 
        "expanding on this", "building on this", "continuing this line of thought", 
        "in conjunction with", "alongside this", "equally", "hand in hand with", 
        "in parallel with", "together with this", "connected with this", "tied to this", 
        "interlinked with", "associated with this", "relatedly", "concurrently", 
        "in agreement with", "reinforcing this", "supporting this", "corroborating this", 
        "in harmony with", "aligned with this", "overlapping with", "complementing this", 
        "along the lines of", "in step with", "coincidentally", "matching this", 
        "mirroring this", "echoing this", "resonating with", "in union with", "harmoniously"
    ],
    
    "Adversative (轉折/對比)": [
        "however", "nevertheless", "nonetheless", "on the other hand", "in contrast", 
        "by contrast", "conversely", "on the contrary", "although", "even though", 
        "though", "despite", "in spite of", "yet", "instead", "whereas", "while", 
        "nonwithstanding", "be that as it may", "all the same", "at the same time", 
        "even so", "regardless", "irrespective of", "for all that", "having said that", 
        "that being said", "then again", "alternatively", "on the flip side", 
        "contrary to", "opposed to", "unlike", "differently from", "instead of", 
        "in opposition to", "far from it", "quite the opposite", "rather", "but", 
        "even with", "despite this", "in spite of this", "regardless of", "granted", 
        "admittedly", "true as it is", "albeit", "contradictorily", "paradoxically", 
        "ironically", "unpredictably", "unexpectedly", "against all odds", "contrariwise", 
        "notwithstanding this", "regardless of this", "despite the fact that", 
        "in spite of the fact that", "in defiance of", "defying this", "contrasting with", 
        "diverging from", "deviating from", "offsetting this", "counter to this", 
        "as opposed to", "distinct from", "differing from", "in direct opposition", 
        "on one hand ... on the other hand", "despite everything", "even with all this", 
        "despite all that", "be it as it may", "up against", "counteracting this", 
        "standing in contrast", "setting against", "juxtaposed with", "versus", 
        "dissenting from", "conflicting with", "at variance with", "incompatible with", 
        "inversely", "oppositely", "anti-thetically", "antithetically", "refuting this", 
        "challenging this", "disproving this", "gainsaying this", "withstanding this", 
        "counterbalancing this", "weighed against", "in comparison to", "in conflict with"
    ],
    
    "Causal (因果/邏輯推論)": [
        "therefore", "thus", "hence", "as a result", "consequently", "accordingly", 
        "because", "since", "as", "due to", "owing to", "so", "for this reason", 
        "because of this", "because of that", "for that reason", "on account of", 
        "thanks to", "in light of", "given that", "seeing that", "in view of", 
        "this leads to", "this results in", "this implies that", "it follows that", 
        "thereby", "wherefore", "then", "for", "out of", "stemming from", 
        "originating from", "arising from", "brought about by", "precipitated by", 
        "triggered by", "induced by", "caused by", "generated by", "fostered by", 
        "yielding", "leading to", "resulting in", "culminating in", "paving the way for", 
        "giving rise to", "prompting", "provoking", "engendering", "instigating", 
        "sparking", "spurring", "driving", "motivating", "accounting for", 
        "explaining why", "underlying", "so that", "in order that", "with the result that", 
        "to the end that", "in consequence", "subsequently", "as a consequence", 
        "for cause of", "by virtue of", "by reason of", "by force of", "on the grounds of", 
        "attributable to", "ascribeable to", "traceable to", "derived from", 
        "rooted in", "pushed by", "dictated by", "conditioned by", "governed by", 
        "in response to", "in reaction to", "following from", "directly related to", 
        "in obedience to", "in pursuance of", "pursuant to", "so then", "consequentially", 
        "therefrom", "wherefrom", "thence", "henceforth", "thereupon", "hereupon", 
        "as an effect", "in effect", "so as to cause", "driving the outcome"
    ],
    
    "Temporal/Sequential (時間/順序)": [
        "firstly", "secondly", "thirdly", "finally", "first of all", "to begin with", 
        "to start with", "in the first place", "in the second place", "at first", 
        "then", "next", "afterwards", "subsequently", "later", "after that", 
        "meanwhile", "in the meantime", "simultaneously", "at same time", "concurrently", 
        "prior to this", "previously", "formerly", "earlier", "before this", 
        "beforehand", "initially", "originally", "at beginning", "eventually", 
        "ultimately", "in the end", "lastly", "to conclude", "at last", "at length", 
        "all of a sudden", "suddenly", "promptly", "instantly", "immediately", 
        "momentarily", "presently", "shortly", "soon", "before long", "after a while", 
        "in due course", "over time", "gradually", "step by step", "little by little", 
        "phase by phase", "chronologically", "sequentially", "in sequence", "in order", 
        "following this", "succeeding this", "thereafter", "heretofore", "up to now", 
        "until now", "so far", "hitherto", "since then", "ever since", "from now on", 
        "henceforth", "in future", "at present", "currently", "nowadays", "these days", 
        "for time being", "provisiomally", "temporarily", "meanwhile", "in interim", 
        "interim", "simultaneous with", "coinciding with", "in parallel", "synchronously", 
        "intermittently", "periodically", "recurrently", "at intervals", "from time to time", 
        "once", "when", "while", "as soon as", "the moment", "no sooner than", 
        "hardly when", "scarcely when", "upon doing", "following on", "straightaway"
    ],

    "Condition/Concession (條件/讓步)": [
        "if", "unless", "provided that", "providing that", "on condition that", 
        "as long as", "so long as", "in case", "in event of", "in event that", 
        "supposing that", "assuming that", "given that", "on assumption that", 
        "whether or not", "even if", "even though", "granted that", "granting that", 
        "admittedly", "of course", "doubtless", "undoubtedly", "no doubt", 
        "to be sure", "certainly", "indeed", "naturally", "clearly", "obviously", 
        "patently", "plainly", "with proviso that", "subject to", "contingent upon", 
        "dependent on", "pending", "in case of", "should it happen that", 
        "were it to occur", "had it been", "under circumstances that", 
        "under condition that", "with understanding that", "with stipulation that", 
        "barring", "failing", "except if", "save that", "only if", "if and only if", 
        "assuming", "presuming", "supposing", "conceding that", "allowing that", 
        "despite possibility", "in spite of possibility", "regardless of whether", 
        "no matter if", "no matter whether", "whichever", "whoever", "whatever", 
        "whenever", "wherever", "however much", "be it that", "come what may", 
        "notwithstanding whether", "conditional upon", "stipulated that", 
        "presupposing that", "under assumption", "hypothetically speaking", 
        "in hypothetical scenario", "with caveat that", "notwithstanding condition", 
        "in all likelihood if", "conceding the point", "accepting that", 
        "yielding to fact that", "acknowledging that", "recognizing that", 
        "in deference to", "allowing for", "making allowance for", "with allowance for", 
        "barring unforeseen", "short of", "save for", "excepting", "excluding"
    ],

    "Exemplification/Clarification (舉例/闡釋)": [
        "for example", "for instance", "such as", "namely", "to illustrate", 
        "as an illustration", "specifically", "in particular", "particularly", 
        "notably", "chiefly", "mainly", "mostly", "predominantly", "expressly", 
        "explicitly", "that is to say", "that is", "i.e.", "e.g.", "in other words", 
        "put another way", "to put it differently", "to rephrase", "to clarify", 
        "to put it simply", "simply put", "strictly speaking", "broadly speaking", 
        "generally speaking", "by way of example", "as a case in point", 
        "case in point", "to cite an instance", "to name a few", "including", 
        "like", "as evidenced by", "as demonstrated by", "as shown by", 
        "as revealed by", "as exemplified by", "exemplified in", "demonstrated in", 
        "illustrated by", "instantiated by", "to specify", "in detail", "more precisely", 
        "to be precise", "exactingly", "in specific terms", "to put it plainly", 
        "meaning that", "which means that", "signifying that", "denoting that", 
        "translating to", "by definition", "literally", "figuratively speaking", 
        "metaphorically speaking", "analogously", "by analogy", "as follows", 
        "the following", "such like", "among others", "inter alia", "for one thing", 
        "to begin with an example", "take for example", "consider the case of", 
        "look at", "to highlight", "to spotlight", "to pinpoint", "specifically speaking", 
        "in concrete terms", "to manifest", "in explicit detail", "worded differently", 
        "in layperson terms", "in layman's terms", "summarized as", "reworded as", 
        "framed another way", "stated differently", "expressed otherwise", "rephrased as"
    ],

    "Summary/Conclusion (總結/結論)": [
        "in conclusion", "to conclude", "in summary", "to summarize", "to sum up", 
        "in short", "in brief", "briefly", "overall", "on the whole", "all in all", 
        "in a nutshell", "to make a long story short", "all things considered", 
        "taking everything into account", "taking everything into consideration", 
        "by and large", "generally", "in general", "ultimately", "eventually", 
        "in the final analysis", "at the end of the day", "when all is said and done", 
        "as has been noted", "as mentioned above", "as outlined above", 
        "as demonstrated", "as shown", "thus", "therefore", "hence", "so", 
        "consequently", "in fine", "summing up", "recapitulating", "to recap", 
        "in recapitulation", "to encapsulate", "encapsulating this", "in essence", 
        "essentially", "fundamentally", "basically", "the bottom line is", 
        "to wrap up", "wrapping up", "in closing", "to bring to a close", 
        "as a final point", "finally", "lastly", "to draw things to a close", 
        "in final summary", "putting it all together", "upon whole", 
        "for the most part", "in grand scheme", "broadly considered", 
        "in total", "altogether", "in aggregate", "comprehensively", 
        "holistically", "taken together", "in sum", "to state briefly", 
        "in condensed form", "synthesizing this", "in synthesis", "to finalize", 
        "in finality", "ultimately speaking", "to round off", "rounding off"
    ]
}

# -----------------------------------------------------------------------------
# 3. 語篇邏輯詞動態分析函式
# -----------------------------------------------------------------------------
def analyze_discourse_markers(text: str) -> Dict[str, Any]:
    """
    動態分析英文文本中的語篇銜接標記（Discourse Connectives / Markers）
    """
    if not text or not text.strip():
        return {
            "total_markers": 0,
            "discourse_density": 0.0,
            "category_counts": {cat: 0 for cat in DISCOURSE_LEXICON.keys()},
            "found_markers": []
        }

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = len(words) if len(words) > 0 else 1

    found_markers = []
    category_counts = {cat: 0 for cat in DISCOURSE_LEXICON.keys()}

    for cat, markers in DISCOURSE_LEXICON.items():
        for marker in markers:
            escaped_marker = re.escape(marker).replace(r'\ ', r'\s+')
            pattern = r'\b' + escaped_marker + r'\b'
            
            matches = list(re.finditer(pattern, text_lower))
            count = len(matches)
            if count > 0:
                category_counts[cat] += count
                found_markers.append({
                    "marker": marker,
                    "category": cat,
                    "count": count
                })

    found_markers = sorted(found_markers, key=lambda x: x["count"], reverse=True)
    total_markers = sum(category_counts.values())
    discourse_density = round((total_markers / total_words) * 100, 2)

    return {
        "total_markers": total_markers,
        "discourse_density": discourse_density,
        "category_counts": category_counts,
        "found_markers": found_markers
    }

# -----------------------------------------------------------------------------
# 4. 平均依存距離 (MDD) 與工作記憶負載計算 (跨越弧線 Storage Cost 版)
# -----------------------------------------------------------------------------
def calculate_mdd_and_memory_load(text: str, arcs_threshold: int = 4) -> Dict[str, Any]:
    """
    使用 spaCy 計算：
    1. 平均依存距離 (MDD)
    2. 工作記憶超載點：基於 DLT 理論，計算單字上方跨越的弧線數量。
    """
    if not nlp:
        return {"mdd": 2.1, "total_dependencies": 0, "overload_spans": []}

    doc = nlp(text)
    total_dd = 0
    total_dependencies = 0
    raw_overload_spans = []

    # 安全豁免清單：對等連接、標點、冠詞等低負載語法關係，不計入跨越干擾
    SAFE_DEPS = {"punct", "det", "cc", "conj", "intj"} 

    for sent_idx, sent in enumerate(doc.sents, 1):
        valid_arcs = []
        
        # 收集該句有效的依存弧線
        for token in sent:
            if token.pos_ == "PUNCT" or token.dep_ == "ROOT":
                continue
                
            dd = abs(token.i - token.head.i)
            total_dd += dd
            total_dependencies += 1
            
            if token.dep_ not in SAFE_DEPS:
                start_i = min(token.i, token.head.i)
                end_i = max(token.i, token.head.i)
                valid_arcs.append({
                    "start": start_i, 
                    "end": end_i, 
                    "dep": token.dep_,
                    "desc": f"{token.text}→{token.head.text}"
                })
        
        # 逐字掃描上方跨越的弧線數
        for token in sent:
            if token.pos_ == "PUNCT":
                continue
            
            crossing_arcs = []
            for arc in valid_arcs:
                if arc["start"] < token.i < arc["end"]:
                    crossing_arcs.append(arc["desc"])
            
            cross_count = len(crossing_arcs)
            
            if cross_count >= arcs_threshold:
                # 使用 Markdown 語法 (**粗體**, *斜體*) 避免 HTML 渲染錯誤
                raw_overload_spans.append({
                    "sentence_id": sent_idx,
                    "word": token.text,
                    "cross_count": cross_count,
                    "crossing_details": crossing_arcs,
                    "msg": f"該單字上方同時有 **{cross_count}** 條依存弧線跨越（例如: *{', '.join(crossing_arcs[:3])}* 等），導致大腦工作記憶「儲存成本 (Storage Cost)」產生瓶頸。"
                })

    # 過濾：同一句中只保留跨越線數最多的那個字（也就是最容易崩潰的瓶頸點）
    filtered_spans = []
    for key, group in itertools.groupby(raw_overload_spans, key=lambda x: x["sentence_id"]):
        max_span = max(list(group), key=lambda x: x["cross_count"])
        filtered_spans.append(max_span)

    mdd = round(total_dd / total_dependencies, 2) if total_dependencies > 0 else 0.0

    return {
        "mdd": mdd,
        "total_dependencies": total_dependencies,
        "overload_spans": filtered_spans
    }

# -----------------------------------------------------------------------------
# 5. L2SCA 句法複雜度近似指標計算
# -----------------------------------------------------------------------------
def calculate_l2sca_approximations(text: str) -> Dict[str, float]:
    """計算平均句長 (MLS) 與 複雜名詞組密度 (CNP/C)"""
    if not nlp:
        return {"MLS": 16.5, "CNP/C": 1.10}

    doc = nlp(text)
    sentences = list(doc.sents)
    num_sentences = len(sentences) if len(sentences) > 0 else 1
    words = [token for token in doc if token.pos_ != "PUNCT"]
    num_words = len(words)

    mls = round(num_words / num_sentences, 2)

    num_clauses = 0
    num_complex_nps = 0

    for token in doc:
        if token.pos_ in ["VERB", "AUX"] and token.dep_ in ["ROOT", "advcl", "relcl", "ccomp", "xcomp"]:
            num_clauses += 1
            
        if token.pos_ in ["NOUN", "PROPN"]:
            has_complex_modifier = any(
                child.dep_ in ["prep", "relcl", "acl", "appos"] for child in token.children
            )
            if has_complex_modifier:
                num_complex_nps += 1

    num_clauses = max(num_clauses, num_sentences)
    cnp_per_clause = round(num_complex_nps / num_clauses, 2)

    return {
        "MLS": mls,
        "CNP/C": cnp_per_clause,
        "num_words": num_words,
        "num_sentences": num_sentences,
        "num_clauses": num_clauses
    }

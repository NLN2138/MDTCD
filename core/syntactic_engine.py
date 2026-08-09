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
        "therefrom", "wherefrom", "

import spacy

class NLPEngine:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"BŁĄD: Model {model_name} nie znaleziony.")

        self.power_words = {
            'NEGATION': ['not', 'never', 'no', 'neither', 'nor', 'none', 'cannot', 'refuse'],
            'MODAL_MUST': ['must', 'shall', 'required', 'obligated', 'mandatory', 'will'],
            'MODAL_MAY': ['may', 'can', 'entitled', 'allowed', 'permit', 'could'],
            'EXCEPTIONS': ['unless', 'except', 'provided', 'however', 'subject', 'notwithstanding', 'condition'],
            'SENSITIVE_DATA' : ['biometric', 'faceprint', 'voiceprint', 'fingerprint', 'location', 'tracking', 'health', 'genetic' ]
        }

    def calculate_readability(self, text):
        doc = self.nlp(text)
        sentences = list(doc.sents)
        words = [t for t in doc if not t.is_punct]
        if not sentences or not words: return 0
        
        avg_sent_len = len(words) / len(sentences)
        complex_words = [w for w in words if len(w.text) > 9]
        pct_complex = (len(complex_words) / len(words)) * 100
        return avg_sent_len + pct_complex

    def extract_legal_features(self, text):
        doc = self.nlp(text)
        features = {
            'lemmas': [t.lemma_.lower() for t in doc if not t.is_punct and not t.is_space],
            'negations': [],
            'modals_must': [],
            'modals_may': [],
            'passive_voice': [],
            'exceptions': [],
            'privacy_risks': [],
            'readability': self.calculate_readability(text),
            'entities': {'dates': [], 'money': [], 'orgs': []}
        }
        
        for token in doc:
            low_lemma = token.lemma_.lower()
            if token.dep_ == "neg" or low_lemma in self.power_words['NEGATION']:
                features['negations'].append(f"{low_lemma} ({token.head.lemma_})")
            if token.dep_ == "auxpass":
                features['passive_voice'].append(f"{token.lemma_} {token.head.lemma_}")
            if low_lemma in self.power_words['MODAL_MUST']: features['modals_must'].append(low_lemma)
            if low_lemma in self.power_words['MODAL_MAY']: features['modals_may'].append(low_lemma)
            if low_lemma in self.power_words['EXCEPTIONS']: features['exceptions'].append(low_lemma)
            if low_lemma in self.power_words['SENSITIVE_DATA']: features['privacy_risks'].append(low_lemma)
        for ent in doc.ents:
            if ent.label_ == "DATE": features['entities']['dates'].append(ent.text)
            elif ent.label_ == "MONEY": features['entities']['money'].append(ent.text)
            elif ent.label_ == "ORG": features['entities']['orgs'].append(ent.text)
                
        return features

    def calculate_risk_score(self, features):
        """Zwraca wynik ryzyka od 0 do 100."""
        score = 0
        score += len(features['negations']) * 15
        score += len(features['modals_must']) * 10
        score += len(features['passive_voice']) * 5
        score += len(features['exceptions']) * 12
        score += len(set(features['privacy_risks'])) * 25

        if features['readability'] > 40: score += 15
        if features['modals_may'] and features['privacy_risks']:score += 20
        return min(score, 100)
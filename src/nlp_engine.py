import spacy
import os

class NLPEngine:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Run: python -m spacy download {model_name}")

        self.power_words = {
            'NEGATION': ['not', 'never', 'no', 'neither', 'nor', 'none', 'cannot', 'refuse'],
            'MODAL_MUST': ['must', 'shall', 'required', 'obligated', 'mandatory', 'will'],
            'MODAL_MAY': ['may', 'can', 'entitled', 'allowed', 'permit', 'could']
        }

    def extract_legal_features(self, text):
        """Wyciąga cechy lingwistyczne, w tym stronę bierną."""
        doc = self.nlp(text.lower())
        features = {
            'negations': [],
            'modals_must': [],
            'modals_may': [],
            'passive_voice': [] # Wykryte konstrukcje strony biernej
        }
        
        for token in doc:
            # 1. Wykrywanie NEGACJI (Dependency Parsing)
            if token.dep_ == "neg" or token.lemma_ in self.power_words['NEGATION']:
                target = token.head.lemma_ if token.dep_ == "neg" else "context"
                features['negations'].append(f"{token.lemma_} ({target})")
            
            # 2. Wykrywanie STRONY BIERNEJ (Passive Voice)
            # W spacy: auxpass (posiłkowy strony biernej) + czasownik w formie participle
            if token.dep_ == "auxpass":
                verb = token.head.lemma_
                features['passive_voice'].append(f"is/was {verb}")
            
            # 3. Klasyfikacja MODALNOŚCI
            if token.lemma_ in self.power_words['MODAL_MUST']:
                features['modals_must'].append(token.lemma_)
            elif token.lemma_ in self.power_words['MODAL_MAY']:
                features['modals_may'].append(token.lemma_)
                
        return features

    def compare_paragraphs(self, p_old, p_new):
        """Zwraca różnice między dwoma akapitami."""
        f_old = self.extract_legal_features(p_old)
        f_new = self.extract_legal_features(p_new)
        
        diffs = []
        if len(f_old['negations']) != len(f_new['negations']):
            diffs.append(f"NEGATION CHANGE: {f_old['negations']} -> {f_new['negations']}")
        
        if f_old['modals_must'] and not f_new['modals_must'] and f_new['modals_may']:
            diffs.append("MODAL SHIFT: Obligation loosened to permission (must -> may).")
        
        if not f_old['passive_voice'] and f_new['passive_voice']:
            diffs.append(f"STYLE CHANGE: Passive voice introduced ({f_new['passive_voice']}) - obscures responsibility.")
            
        return diffs, f_old, f_new

def load_paragraphs(file_path):
    """Wczytuje plik i dzieli na akapity (zakładając podwójną nową linię)."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Dzielimy po \n\n i czyścimy z pustych linii oraz spacji
        content = f.read().split('\n\n')
        return [p.strip() for p in content if len(p.strip()) > 20] # ignorujemy bardzo krótkie śmieci

# --- GŁÓWNY SKRYPT ANALIZUJĄCY ---
if __name__ == "__main__":
    engine = NLPEngine()
    
    # Ścieżki do Twoich plików
    path_2010 = "fb_privacy_policy_2010.txt"
    path_2015 = "fb_privacy_policy_2015.txt"
    
    paras_2010 = load_paragraphs(path_2010)
    paras_2015 = load_paragraphs(path_2015)
    
    print(f"Loaded {len(paras_2010)} paragraphs from 2010")
    print(f"Loaded {len(paras_2015)} paragraphs from 2015\n")

    # Proste dopasowanie (na razie 1 do 1, dopóki Osoba B nie zrobi Matchera)
    limit = min(len(paras_2010), len(paras_2015))
    
    for i in range(limit):
        p_old = paras_2010[i]
        p_new = paras_2015[i]
        
        diffs, f1, f2 = engine.compare_paragraphs(p_old, p_new)
        
        if diffs:
            print(f"--- Paragraph #{i+1} Significant Changes ---")
            print(f"OLD: {p_old[:100]}...")
            print(f"NEW: {p_new[:100]}...")
            for d in diffs:
                print(f"  🚨 {d}")
            print("\n")
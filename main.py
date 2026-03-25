import spacy
import difflib
import os

class LegalNLPEngine:
    def __init__(self, model_name="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"ERROR: Model {model_name} not found. Run: python -m spacy download {model_name}")

        self.power_words = {
            'negations': {'not', 'no', 'never', 'none', 'prohibited', 'forbidden', 'neither', 'nor', 'cannot', 'refuse'},
            'obligations': {'must', 'shall', 'required', 'obligated', 'should', 'mandatory', 'will'},
            'rights': {'may', 'can', 'entitled', 'allowed', 'permit', 'right', 'could'},
            'exceptions': {'unless', 'except', 'provided', 'however', 'subject'}
        }
        self.all_critical = {word for sublist in self.power_words.values() for word in sublist}

    def extract_features(self, text):
        """Wyciąga cechy lingwistyczne, lematy i stronę bierną."""
        doc = self.nlp(text.lower())
        features = {
            'lemmas': [t.lemma_ for t in doc if not t.is_punct and not t.is_space],
            'passive_voice': [],
            'critical_found': []
        }
        
        for token in doc:
            # Wykrywanie strony biernej (Passive Voice)
            if token.dep_ == "auxpass":
                features['passive_voice'].append(f"{token.lemma_} {token.head.lemma_}")
            
            # Zapamiętujemy słowa krytyczne
            if token.lemma_ in self.all_critical:
                features['critical_found'].append(token.lemma_)
                
        return features

    def compare_paragraphs(self, p_old, p_new):
        """Porównuje dwa akapity i zwraca listę alertów oraz wynik."""
        f_old = self.extract_features(p_old)
        f_new = self.extract_features(p_new)
        
        matcher = difflib.SequenceMatcher(None, f_old['lemmas'], f_new['lemmas'])
        base_sim = matcher.ratio() * 100
        
        alerts = []
        # 1. Zmiany w słowach krytycznych
        removed = set(f_old['critical_found']) - set(f_new['critical_found'])
        added = set(f_new['critical_found']) - set(f_old['critical_found'])
        
        for r in removed: alerts.append(f"REMOVED CRITICAL: '{r}'")
        for a in added: alerts.append(f"ADDED CRITICAL: '{a}'")
        
        # 2. Wykrywanie nowej strony biernej
        new_passive = set(f_new['passive_voice']) - set(f_old['passive_voice'])
        if new_passive:
            alerts.append(f"NEW PASSIVE VOICE: {list(new_passive)} (responsibility might be obscured)")
            
        # Obliczamy wynik końcowy (kara za każdy alert)
        final_score = max(0, base_sim - (len(alerts) * 20))
        return final_score, alerts

def find_best_match(new_lemmas, old_paras_data):
    """Szuka najbardziej podobnego akapitu w starej wersji dokumentu."""
    best_score = 0
    best_idx = -1
    
    for idx, old_data in enumerate(old_paras_data):
        score = difflib.SequenceMatcher(None, old_data['lemmas'], new_lemmas).ratio()
        if score > best_score:
            best_score = score
            best_idx = idx
            
    return best_idx, best_score

def load_data(filename):
    """Wczytuje tekst i dzieli na sensowne akapity."""
    path = os.path.join("data", filename)
    if not os.path.exists(path):
        print(f" File not found: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        # Dzielimy po podwójnym enterze, filtrujemy krótkie śmieci
        return [p.strip() for p in f.read().split('\n\n') if len(p.strip()) > 40]

def main():
    engine = LegalNLPEngine()
    
    # 1. Wczytywanie i Preprocessing
    print(" Loading and indexing documents...")
    old_policy = load_data("fb_privacy_policy_2010.txt")
    new_policy = load_data("fb_privacy_policy_2015.txt")
    
    if not old_policy or not new_policy:
        return

    # Przygotowujemy cechy dla wszystkich starych akapitów raz, żeby było szybciej
    old_paras_features = [engine.extract_features(p) for p in old_policy]

    print(f" Analysis started: 2010 ({len(old_policy)} paras) vs 2015 ({len(new_policy)} paras)")
    print("=" * 70)

    total_alerts = 0

    for i, p_new in enumerate(new_policy):
        new_f = engine.extract_features(p_new)
        
        # 2. Szukamy "swata" (najlepszego dopasowania)
        match_idx, match_score = find_best_match(new_f['lemmas'], old_paras_features)
        
        print(f" PARAGRAPH 2015 #{i+1}")
        
        # Próg podobieństwa: jeśli < 25%, uznajemy to za nową sekcję
        if match_score > 0.25:
            p_old = old_policy[match_idx]
            score, alerts = engine.compare_paragraphs(p_old, p_new)
            
            print(f"    Matched with 2010 Paragraph #{match_idx+1} (Base Similarity: {match_score:.2f})")
            print(f"    Legal Similarity Score: {score:.2f}%")
            
            if alerts:
                total_alerts += len(alerts)
                for a in alerts:
                    print(f"   {a}")
            else:
                print("   No critical legal changes detected in this section.")
        else:
            print("   NEW SECTION DETECTED (No similar paragraph in 2010 version)")
            print(f"   Critical words in new section: {new_f['critical_found']}")
            if new_f['passive_voice']:
                print(f"    Passive voice found: {new_f['passive_voice']}")
        
        print(f"   [Preview]: {p_new[:100]}...")
        print("-" * 70)

    print(f"\nANALYSIS COMPLETE. Total legal alerts: {total_alerts}")

if __name__ == "__main__":
    main()
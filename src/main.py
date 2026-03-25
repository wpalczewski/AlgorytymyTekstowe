import difflib
import spacy

# 1. KONFIGURACJA: Słowa, które drastycznie zmieniają sens prawny
# Dodanie lub usunięcie tych słów skutkuje ogromną "karą" punktową
CRITICAL_WORDS = {
    'negacje': {'nie', 'żaden', 'nigdy', 'brak', 'zakaz', 'zabraniać'},
    'obowiązki': {'musi', 'obowiązek', 'zobowiązany', 'powinien', 'winien'},
    'uprawnienia': {'może', 'prawo', 'uprawniony', 'wolno', 'zezwalać'},
    'wyjątki': {'chyba', 'zastrzeżenie', 'wyjątek', 'warunek'}
}

# Łączymy wszystko w jeden zbiór dla szybkiego wyszukiwania
ALL_CRITICAL = {word for sublist in CRITICAL_WORDS.values() for word in sublist}

# Ładujemy model NLP dla języka polskiego
try:
    nlp = spacy.load("pl_core_news_sm")
except:
    print("BŁĄD: Zainstaluj model polski: python -m spacy download pl_core_news_sm")

def preprocess_text(text):
    """Czyści tekst i zamienia słowa na ich formy podstawowe (lematy)."""
    doc = nlp(text.lower())
    # Wyciągamy lematy, pomijając znaki interpunkcyjne
    return [token.lemma_ for token in doc if not token.is_punct]

def calculate_legal_score(base_tokens, new_tokens):
    """Oblicza podobieństwo z uwzględnieniem wag prawniczych."""
    matcher = difflib.SequenceMatcher(None, base_tokens, new_tokens)
    visual_similarity = matcher.ratio() * 100
    
    diff = list(difflib.ndiff(base_tokens, new_tokens))
    
    # Szukamy zmian w słowach krytycznych
    critical_alerts = []
    penalty = 0
    
    for item in diff:
        word = item[2:] # słowo po symbolu (+ / - /  )
        status = item[0] # symbol zmiany
        
        if word in ALL_CRITICAL:
            if status == '-':
                critical_alerts.append(f"USUNIĘTO słowo kluczowe: '{word}'")
                penalty += 35 # Kara za usunięcie np. "nie" lub "może"
            elif status == '+':
                critical_alerts.append(f"DODANO słowo kluczowe: '{word}'")
                penalty += 35 # Kara za dodanie ograniczenia
                
    # Obliczamy końcowy wynik (nie schodzimy poniżej 0)
    legal_similarity = max(0, visual_similarity - penalty)
    
    return visual_similarity, legal_similarity, diff, critical_alerts

def print_report(title, t1, t2):
    """Wyświetla sformatowany raport z porównania."""
    tokens1 = preprocess_text(t1)
    tokens2 = preprocess_text(t2)
    
    vis_sim, leg_sim, diff, alerts = calculate_legal_score(tokens1, tokens2)
    
    print(f"--- {title} ---")
    print(f"Tekst A: {t1}")
    print(f"Tekst B: {t2}")
    print(f"Podobieństwo wizualne (difflib): {vis_sim:.2f}%")
    print(f"PODOBIEŃSTWO PRAWNICZE: {leg_sim:.2f}%")
    
    if alerts:
        print("🚨 ALERTY PRAWNE:")
        for a in alerts:
            print(f"  - {a}")
    else:
        print("✅ Brak krytycznych zmian w słownictwie modalnym.")
    print("-" * 50 + "\n")

# --- TESTY PROOF OF CONCEPT ---

# Test 1: Zmiana przez negację (Krytyczna)
txt1 = "Najemca jest uprawniony do zmiany zamków."
txt2 = "Najemca nie jest uprawniony do zmiany zamków."

# Test 2: Zmiana techniczna (Mało ważna)
txt3 = "Lokator zapłaci czynsz przelewem."
txt4 = "Mieszkaniec ureguluje czynsz przelewem."

# Test 3: Zmiana modalna (Z przyzwolenia na obowiązek)
txt5 = "Właściciel może wypowiedzieć umowę."
txt6 = "Właściciel musi wypowiedzieć umowę."

print_report("TEST 1: NEGACJA", txt1, txt2)
print_report("TEST 2: SYNONIMY", txt3, txt4)
print_report("TEST 3: TRYB ROZKAZUJĄCY", txt5, txt6)
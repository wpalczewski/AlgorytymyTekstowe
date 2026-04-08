import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from nlp_engine import NLPEngine

def select_file(title):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        initialdir="./data",
        filetypes=(("Pliki tekstowe", "*.txt"), ("Wszystkie pliki", "*.*"))
    )
    root.destroy()
    return file_path

def load_data(path):
    if not path:
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [p.strip() for p in f.read().split('\n\n') if len(p.strip()) > 40]

def main():
    engine = NLPEngine()
    
    path_old = select_file("Wybierz STARSZĄ wersję polityki (np. tiktok_20.txt)")
    path_new = select_file("Wybierz NOWSZĄ wersję polityki (np. tiktok_21.txt)")
    
    if not path_old or not path_new:
        return

    old_policy = load_data(path_old)
    new_policy = load_data(path_new)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    all_paras = old_policy + new_policy
    tfidf_matrix = vectorizer.fit_transform(all_paras)
    
    old_tfidf = tfidf_matrix[:len(old_policy)]
    new_tfidf = tfidf_matrix[len(old_policy):]
    similarity_matrix = cosine_similarity(new_tfidf, old_tfidf)

    print(f"Analiza dokumentów: {os.path.basename(path_new)}")
    print("=" * 75)

    total_alerts = 0
    for i, p_new in enumerate(new_policy):
        new_f = engine.extract_legal_features(p_new)
        match_idx = np.argmax(similarity_matrix[i])
        match_score = similarity_matrix[i][match_idx]
        risk = engine.calculate_risk_score(new_f)
        
        print(f"AKAPIT #{i+1}")
        print(f"WYNIK RYZYKA: {risk}/100")
        
        if risk > 40:
            total_alerts += 1
            print("IDENTYFIKACJA ZAGROŻEŃ:")
            
            if new_f.get('privacy_risks'):
                print(f" - WYKRYTO DANE WRAŻLIWE: {', '.join(set(new_f['privacy_risks']))}")
                print("   Zagrożenie: Gromadzenie cech fizjologicznych użytkownika.")
            
            if new_f['passive_voice']:
                print(f" - STRONA BIERNA: {new_f['passive_voice']}")
                print("   Zagrożenie: Ukrywanie podmiotu odpowiedzialnego za przetwarzanie danych.")
            
            if new_f['exceptions']:
                print(f" - FURTKI PRAWNE (Wyjątki): {list(set(new_f['exceptions']))}")
                print("   Zagrożenie: Możliwość ominięcia zasad ochrony prywatności w określonych warunkach.")

            if 'may' in new_f['modals_may'] and new_f.get('privacy_risks'):
                print(" - DOPUSZCZENIE DOSTĘPU: Użycie sformułowania 'may' sugeruje arbitralność w zbieraniu danych.")

        if match_score > 0.3:
            old_f = engine.extract_legal_features(old_policy[match_idx])
            ent_alerts = engine.compare_entities(old_f, new_f)
            print(f"POWIĄZANIE: Dopasowano do akapitu #{match_idx+1} starszej wersji.")
            for a in ent_alerts: print(f"  ZMIANA DANYCH: {a}")
        else:
            print("STATUS: NOWA SEKCJA (Brak odpowiednika w poprzedniej wersji)")
            
        print(f"FRAGMENT: {p_new[:100]}...")
        print("-" * 75)

    print(f"PODSUMOWANIE: Znaleziono {total_alerts} problematycznych sekcji.")

if __name__ == "__main__":
    main()
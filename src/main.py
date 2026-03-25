import difflib
import spacy

# 1. CONFIGURATION: English legal "power words"
# These words drastically change the meaning of a sentence.
CRITICAL_WORDS = {
    'negations': {'not', 'no', 'never', 'none', 'prohibited', 'forbidden', 'neither', 'nor'},
    'obligations': {'must', 'shall', 'required', 'obligated', 'should', 'mandatory'},
    'rights': {'may', 'can', 'entitled', 'allowed', 'permit', 'right'},
    'exceptions': {'unless', 'except', 'provided', 'however', 'subject'}
}

# Flatten the dictionary for fast lookup
ALL_CRITICAL = {word for sublist in CRITICAL_WORDS.values() for word in sublist}

# Load the English NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("ERROR: Please install the English model: python -m spacy download en_core_web_sm")

def preprocess_text(text):
    """Cleans text and converts words to their base forms (lemmas)."""
    doc = nlp(text.lower())
    # Extract lemmas, ignoring punctuation and whitespace
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

def calculate_legal_score(base_tokens, new_tokens):
    """Calculates similarity with legal weighting."""
    matcher = difflib.SequenceMatcher(None, base_tokens, new_tokens)
    visual_similarity = matcher.ratio() * 100
    
    diff = list(difflib.ndiff(base_tokens, new_tokens))
    
    # Search for changes in critical words
    critical_alerts = []
    penalty = 0
    
    for item in diff:
        word = item[2:] # word after the symbol (+ / - /  )
        status = item[0] # change status
        
        if word in ALL_CRITICAL:
            if status == '-':
                critical_alerts.append(f"REMOVED critical word: '{word}'")
                penalty += 35 
            elif status == '+':
                critical_alerts.append(f"ADDED critical word: '{word}'")
                penalty += 35 
                
    # Calculate final score (clamped at 0)
    legal_similarity = max(0, visual_similarity - penalty)
    
    return visual_similarity, legal_similarity, diff, critical_alerts

def print_report(title, t1, t2):
    """Displays a formatted comparison report."""
    tokens1 = preprocess_text(t1)
    tokens2 = preprocess_text(t2)
    
    vis_sim, leg_sim, diff, alerts = calculate_legal_score(tokens1, tokens2)
    
    print(f"--- {title} ---")
    print(f"Text A: {t1}")
    print(f"Text B: {t2}")
    print(f"Visual Similarity (difflib): {vis_sim:.2f}%")
    print(f"LEGAL SIMILARITY SCORE: {leg_sim:.2f}%")
    
    if alerts:
        print("🚨 LEGAL ALERTS:")
        for a in alerts:
            print(f"  - {a}")
    else:
        print("✅ No critical changes in modal verbs or negations detected.")
    print("-" * 50 + "\n")

# --- PROOF OF CONCEPT TESTS (English) ---

# Test 1: Change by negation (Critical)
txt1 = "The tenant is entitled to change the locks."
txt2 = "The tenant is NOT entitled to change the locks."

# Test 2: Technical change (Minor)
txt3 = "The tenant will pay rent via bank transfer."
txt4 = "The resident shall pay rent through wire transfer."

# Test 3: Modal change (Permission vs. Obligation)
txt5 = "The landlord may terminate the agreement."
txt6 = "The landlord must terminate the agreement."

print_report("TEST 1: NEGATION", txt1, txt2)
print_report("TEST 2: SYNONYMS", txt3, txt4)
print_report("TEST 3: MODAL SHIFT", txt5, txt6)
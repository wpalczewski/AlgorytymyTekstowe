import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LegalSummarizer:
    def __init__(self):
        model_name = "sshleifer/distilbart-cnn-12-6"
        # Sprawdzamy dostępność akceleracji Apple Silicon (MPS)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Ładowanie tokenizera i modelu bezpośrednio z klas Auto
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def get_tldr(self, text):
        # Przygotowanie tekstu (tokenizacja)
        inputs = self.tokenizer(
            text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        # Generowanie streszczenia przez model
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=60,
            min_length=20,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Dekodowanie wektorów z powrotem na tekst
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
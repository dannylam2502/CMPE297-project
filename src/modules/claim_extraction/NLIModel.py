# ==============================================================================
# --- REAL NLI MODEL (with the required .predict() method) ---
# ==============================================================================
NLI_LABELS = ["contradiction", "neutral", "entailment"]
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import logging as hf_logging

from modules.claim_extraction.Fact_Validator_Data_models import ModelInterface

# Suppress heavy logging
hf_logging.set_verbosity_error()


class NLIModel(ModelInterface): # Inherit from stub
    """
    Concrete implementation of ModelInterface using Sentence-Transformers and Hugging Face's NLI model.
    """
    def __init__(self, emb_model_name: str, nli_model_name: str, nli_labels: list[str]):
        print("Initializing heavy models... This happens once.")
        self.emb_model = SentenceTransformer(emb_model_name)
        self.nli_tok = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.NLI_LABELS = nli_labels

    def get_relatedness_score(self, s1: str, s2: str) -> float:
        e1, e2 = self.emb_model.encode([s1, s2], convert_to_tensor=True)
        cos = util.cos_sim(e1, e2).item()
        return (cos + 1) / 2  # map [-1,1] → [0,1]

    def get_nli_probabilities(self, a: str, b: str) -> dict[str, float]:
        # Ensure model is on the correct device (e.g., CUDA if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nli_model.to(device)
        
        x = self.nli_tok.encode_plus(a, b, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            p = torch.softmax(self.nli_model(**x).logits, dim=-1).squeeze().tolist()
        return dict(zip(self.NLI_LABELS, p))

    # --- [!] IMPLEMENTED .predict() METHOD ---
    def predict(self, inputs: List[Tuple[str, str]]) -> List[Tuple[float, float, float]]:
        """
        Implements the required .predict() method to bridge
        the gap with FactValidator.
        """
        results = []
        for claim, passage_content in inputs:
            prob_dict = self.get_nli_probabilities(claim, passage_content)
            e = prob_dict.get("entailment", 0.0)
            c = prob_dict.get("contradiction", 0.0)
            n = prob_dict.get("neutral", 0.0)
            results.append((e, c, n))
        return results
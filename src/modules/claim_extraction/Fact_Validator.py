import os
from typing import List
import joblib
from sklearn.model_selection import train_test_split
from modules.claim_extraction.Fact_Validator_Data_models import Citation, CitationValidationScoring, FactCheckFeatures, FactCheckResult, ModelInterface, SourcePassage, VerdictType
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# [!] IMPORT CHANGES
from sklearn.preprocessing import LabelEncoder  # <-- Corrected import
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # <-- NEW
# from sklearn.tree import DecisionTreeClassifier # <-- REMOVED

from modules.claim_extraction.training.Validator_Training_Data import GoldStandardExample
from modules.llm.llm_engine_interface import LLMInterface

class FactValidator:

    VERDICT_TO_SCORE_MAP: Dict[VerdictType, int] = {
        "Supported": 90,
        "Refuted": 10,
        "Contested": 50,
        "Not enough evidence": 25,
    }

    def __init__(self, 
                 llm: LLMInterface, 
                 nli_backend: ModelInterface,
                 training_data: List[GoldStandardExample] = None,
                 encoder: 'LabelEncoder' = None,
                 model_path: str = 'fact_validator_models.joblib'):
        
        self.llm = llm
        self.nli = nli_backend
        self.related_gate = 0.60 # Relevance threshold
        self.agree_cut = 0.60    # Entailment threshold
        self.contra_cut = 0.60   # Contradiction threshold
        self.clf = None # [!] This will be a RandomForestClassifier
        self.encoder = encoder
        self.model_path = model_path
        if training_data:
            print("Training data provided. Starting new training...")
            self._train(training_data)
        else:
            print(f"No training data. Attempting to load models from '{self.model_path}'...")
            self._load()

        # Final check
        if not self.clf or not self.encoder:
            raise RuntimeError("FactValidator initialization failed. No classifier or encoder is available.")
        
        print("FactValidator is ready.")

    def _prepare_features(self, features: 'FactCheckFeatures', num_agree: int, num_disagree: int, len_valid_results: int, len_passages: int) -> np.ndarray:
        feature_vector = [
            features.entail_max,
            features.entail_mean3,
            features.contradict_max,
            features.agree_domain_count,
            features.releliance_score_avg,
            features.recency_weight_max,
            features.contest_score,
            num_agree,
            num_disagree,
            len_valid_results,
            len_passages
        ]
        return np.array(feature_vector).reshape(1, -1)

    def _calculate_final_score_and_verdict(self, features: 'FactCheckFeatures', num_agree: int, num_disagree: int, len_valid_results: int, len_passages: int) -> Tuple[VerdictType, int]:
        """
        Uses the trained Random Forest to predict the verdict AND
        calculate a confidence score based on the model's output probabilities.
        """
        if not self.clf or not self.encoder:
            raise ValueError("Classifier or Encoder not provided. Cannot predict.")
            
        # 1. Assemble the feature vector in the correct order
        X_input = self._prepare_features(
            features, 
            num_agree, 
            num_disagree, 
            len_valid_results, 
            len_passages
        )

        # 2. Get the probabilities for ALL classes
        all_probabilities = self.clf.predict_proba(X_input)
        
        # 3. Get the probabilities for our single input
        class_probabilities = all_probabilities[0]
        
        # 4. Find the highest probability (confidence)
        confidence = np.max(class_probabilities)
        
        # 5. Find the INDEX of the highest probability
        predicted_numeric_label = np.argmax(class_probabilities)
        
        # 6. Decode the numeric label back to the string verdict
        verdict: VerdictType = self.encoder.inverse_transform([predicted_numeric_label])[0]
        
        # 7. Convert confidence (0.0-1.0) to an integer score (0-100)
        score = int(confidence * 100)
        
        return verdict, score
    
    # --- STUBBED METHODS ---

    def _get_nli_results(self, claim: str, passage_contents: List[str]) -> List[Tuple[float, float, float]]:
        if not self.nli:
            raise ValueError("NLI backend not provided.")
        inputs = [(claim, content) for content in passage_contents]
        return self.nli.predict(inputs) 

    def _calculate_recency(self, published_at: datetime) -> Tuple[float, bool]:
        if not published_at:
            return (0.5, False)
        days_diff = (datetime.now() - published_at).days
        if days_diff < 30:
            return (1.0, True)
        elif days_diff < 365:
            return (0.8, True)
        return (0.3, True)

    def _calculate_features(self, valid_results: List[CitationValidationScoring]) -> FactCheckFeatures:
        if not valid_results:
            return FactCheckFeatures(0, 0, 0, 0, 0, 0)

        entail_probs = sorted([r.entail_prob for r in valid_results if r.entail_prob > 0.1], reverse=True) 
        contra_probs = sorted([r.contradict_prob for r in valid_results if r.contradict_prob > 0.1], reverse=True)
        domains = {r.passage.domain for r in valid_results if r.entail_prob > self.agree_cut}
        
        entail_max = entail_probs[0] if entail_probs else 0.0
        contradict_max = contra_probs[0] if contra_probs else 0.0
        return FactCheckFeatures(
            entail_max=entail_max,
            entail_mean3=np.mean(entail_probs[:3]) if entail_probs else 0.0,
            contradict_max=contradict_max,
            agree_domain_count=len(domains),
            releliance_score_avg=np.mean([r.passage.relevance_score for r in valid_results]),
            recency_weight_max=max(r.recency_weight for r in valid_results),
            contest_score = entail_max * contradict_max
        )

    def _get_top_citations(self, valid_results: List[CitationValidationScoring], num_agree: int, num_disagree: int) -> List[Citation]:
        sorted_results = sorted(valid_results, key=lambda r: r.passage.relevance_score, reverse=True)
        return [r for r in sorted_results[:3]]
    
    # --- MAIN VALIDATION & GENERATION LOGIC ---
    
    def validate_claim(self, claim: str, claim_type: str, passages: List[SourcePassage]) -> FactCheckResult:
        # 1. Filter by relevance
        related_passages = [p for p in passages if p.relevance_score >= self.related_gate]
        len_passages = len(related_passages)

        if not related_passages:
            features = FactCheckFeatures(0, 0, 0, 0, 0, 0)
            return FactCheckResult(claim, "Not enough evidence", 0, [], features) # Score 0

        # 2. Get NLI results
        passage_contents = [p.content for p in related_passages]
        nli_results = self._get_nli_results(claim, passage_contents)
        
        # 3. Combine all info
        all_results = []
        for passage, (e, c, n) in zip(related_passages, nli_results):
            recency_w, date_ok = self._calculate_recency(passage.published_at)
            all_results.append(CitationValidationScoring(
                passage=passage, entail_prob=e, contradict_prob=c, neutral_prob=n,
                recency_weight=recency_w, numeric_date_ok=date_ok
            ))

        # 4. Filter valid results (not strongly neutral)
        valid_results = [r for r in all_results if r.entail_prob > 0.5 or r.contradict_prob > 0.5]
        len_valid_results = len(valid_results)
        
        if not valid_results:
            # We found passages, but they were all neutral.
            # We can calculate features from *all* results to show *why* it was NEI.
            features = self._calculate_features(all_results) 
            return FactCheckResult(claim, "Not enough evidence", 25, [], features) # Score 25

        # 5. Get counts
        num_agree = sum(1 for r in valid_results if r.entail_prob > self.agree_cut)
        num_disagree = sum(1 for r in valid_results if r.contradict_prob > self.contra_cut)

        # 6. Calculate features
        features = self._calculate_features(valid_results)

        # 7. Get final verdict (using the classifier)
        verdict, score = self._calculate_final_score_and_verdict(
            features, num_agree, num_disagree, len_valid_results, len_passages
        )
        
        # 8. Get citations
        citations = self._get_top_citations(valid_results, num_agree, num_disagree)
        
        return FactCheckResult(claim, verdict, score, citations, features)
    
    def generate_training_example(self, claim: str, passages: List[SourcePassage]) -> Tuple[FactCheckFeatures, int, int, int, int]:
        # 1. Filter by relevance
        related_passages = [p for p in passages if p.relevance_score >= self.related_gate]
        len_passages = len(related_passages)

        if not related_passages:
            features = FactCheckFeatures(0, 0, 0, 0, 0, 0)
            return features, 0, 0, 0, 0

        # 2. Get NLI results
        passage_contents = [p.content for p in related_passages]
        nli_results = self._get_nli_results(claim, passage_contents)
        
        # 3. Combine all info
        all_results = []
        for passage, (e, c, n) in zip(related_passages, nli_results):
            recency_w, date_ok = self._calculate_recency(passage.published_at)
            all_results.append(CitationValidationScoring(
                passage=passage, entail_prob=e, contradict_prob=c, neutral_prob=n,
                recency_weight=recency_w, numeric_date_ok=date_ok
            ))
        
        # 4. Filter valid results
        # [!] BUG FIX: Changed from 0.3 to 0.5 to match validate_claim
        valid_results = [r for r in all_results if r.entail_prob > 0.5 or r.contradict_prob > 0.5]
        len_valid_results = len(valid_results)
        
        if not valid_results:
            features = self._calculate_features(all_results)
            return features, 0, 0, 0, len_passages # Return len_passages

        # 5. Get counts
        num_agree = sum(1 for r in valid_results if r.entail_prob > self.agree_cut)
        num_disagree = sum(1 for r in valid_results if r.contradict_prob > self.contra_cut)

        # 6. Calculate features
        features = self._calculate_features(valid_results)

        # 7. Return the raw features and counts
        return features, num_agree, num_disagree, len_valid_results, len_passages

    def _train(self, gold_standard_dataset: List[GoldStandardExample]):
        """
        Trains the classifier and encoder for the specific
        FactValidator instance passed in.
        """
        
        # Check if the validator has an NLI backend
        if not self.nli:
            raise ValueError("The provided FactValidator must have a valid .nli backend for training.")

        # --- Step 1.2: Filter for Trainable Examples ---
        trainable_dataset = [
            item for item in gold_standard_dataset 
            if item.ground_truth_verdict != "Not enough evidence"
        ]
        NEI_dataset = [
            item for item in gold_standard_dataset 
            if item.ground_truth_verdict == "Not enough evidence"
        ]
        print(f"Original dataset size: {len(gold_standard_dataset)}")
        print(f"Trainable (3-class) dataset size: {len(trainable_dataset)}")

        # --- Step 1.5: Split the dataset ---
        all_labels = [item.ground_truth_verdict for item in trainable_dataset]
        train_dataset, test_dataset = train_test_split(
            trainable_dataset,
            test_size=0.25,      
            random_state=42,     
            stratify=all_labels  
        )
        test_dataset.extend(NEI_dataset)
        print(f"Training examples: {len(train_dataset)}")
        print(f"Test examples: {len(test_dataset)}")
        print(f"\n--- Data Split ---")
        print(f"Total examples: {len(gold_standard_dataset)}")

        # --- Step 2: Generate Training Data ---
        print("\nGenerating training data from raw examples...")

        X_train_list = []
        y_labels = []

        # [!] MODIFIED: Iterate over the train_dataset only
        for i, item in enumerate(train_dataset):
            # 1. Use the *passed-in validator's* method to process the text
            # This ensures we use the same settings (e.g., related_gate)
            features, num_a, num_d, len_v, len_p = self.generate_training_example( # <-- CHANGED
                item.claim, item.passages
            )
            
            # 2. Get the feature vector
            feature_vector_1d = self._prepare_features( # <-- CHANGED
                features, num_a, num_d, len_v, len_p
            )[0]
            
            X_train_list.append(feature_vector_1d)
            
            # 3. Store the corresponding ground truth label
            y_labels.append(item.ground_truth_verdict)
            
            print(f"  Training Example {i+1} ({item.ground_truth_verdict}): Features={np.round(feature_vector_1d, 2)}")

        # Convert to NumPy arrays for scikit-learn
        X_train = np.array(X_train_list)
        print(f"\nFeature matrix (X) shape: {X_train.shape}")
        print(f"Labels (y) to be encoded: {y_labels}")

        # --- Step 3: Train the Encoder and Classifier ---

        # 1. Train the Label Encoder
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_labels)
        print(f"\nEncoded labels: {y_train}")
        print(f"Encoder classes: {encoder.classes_}")

        # 2. [!] TRAIN RANDOM FOREST CLASSIFIER
        clf = RandomForestClassifier(
            random_state=42, 
            n_estimators=100,     # Use 100 "mini-trees"
            min_samples_leaf=3,   # Prevents 100% scores and overfitting
            max_depth=10          # Prevents trees from getting too deep
        )
        clf.fit(X_train, y_train)
        print("\n--- Random Forest Classifier Trained ---")

        # --- [!] NEW: Step 3.5: Save models to disk ---
        if self.model_path:
            print(f"\n--- Saving models to {self.model_path} ---")
            # We must save both the classifier and the encoder
            models_to_save = {
                "clf": clf,
                "encoder": encoder
            }
            joblib.dump(models_to_save, self.model_path)
            print("Models saved successfully.")
        # --- [!] NEW: Step 4: Assign Trained Models to the Validator ---
        print("\nAssigning trained models to the self...")
        
        # This is the key part: we modify the validator instance directly.
        self.clf = clf
        self.encoder = encoder

        # --- Step 5: Test the New Validator and Collect Results ---
        print("\n--- Testing the *newly trained* validator on UNSEEN test data ---")

        y_true = [] # The ground truth labels
        y_pred = [] # The model's predicted labels

        correct_predictions = 0
        for i, test_example in enumerate(test_dataset):
            # Run the full validation pipeline on the validator we just trained
            result = self.validate_claim(test_example.claim, '', test_example.passages) # <-- CHANGED
            
            expected = test_example.ground_truth_verdict
            predicted = result.verdict
            
            y_true.append(expected)
            y_pred.append(predicted)
            
            is_correct = (expected == predicted)
            if is_correct:
                correct_predictions += 1
                
            print(f"\nTest Case {i+1}:")
            print(f"  Claim: {test_example.claim[:50]}...")
            print(f"  Expected: {expected}")
            print(f"  Predicted: {predicted} (Score: {result.score})")
            print(f"  Result: {'CORRECT' if is_correct else 'INCORRECT'}")

        # --- Step 6: Show Overall Accuracy and Detailed Report ---
        accuracy = (correct_predictions / len(test_dataset)) * 100
        print(f"\n--- Test Summary ---")
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct_predictions} / {len(test_dataset)} correct)")
        print("\n--- Detailed Classification Report ---")
        all_verdicts = ["Supported", "Refuted", "Contested", "Not enough evidence"]
        print(classification_report(
            y_true, 
            y_pred, 
            labels=all_verdicts, 
            digits=3, 
            zero_division=0
        ))


    def _load(self):
        """Loads the classifier and encoder from the specified model_path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"No training data provided and model file not found at '{self.model_path}'. "
                "Please provide 'training_data' to train a new model or place the file in the correct path."
            )
        
        print(f"Loading models from {self.model_path}...")
        try:
            loaded_models = joblib.load(self.model_path)
            self.clf = loaded_models["clf"]
            self.encoder = loaded_models["encoder"]
            print("Models loaded successfully.")
            print(f"Loaded {len(self.encoder.classes_)} classes: {self.encoder.classes_}")
        except Exception as e:
            raise IOError(f"Failed to load or parse model file at {self.model_path}: {e}")
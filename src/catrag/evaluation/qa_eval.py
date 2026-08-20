from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import Counter
import numpy as np

from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from ..utils.eval_utils import normalize_answer
import re

logger = get_logger(__name__)

# Reference: MRQA official eval
class QAExactMatch(BaseMetric):
    metric_name: str = "qa_exact_match"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the Exact Match (EM) score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged EM score.
                - A list of dictionaries with EM scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
            aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results

class QAF1Score(BaseMetric):
    metric_name: str = "qa_f1_score"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the F1 score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged F1 score.
                - A list of dictionaries with F1 scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
            aggregated_f1 = aggregation_fn(f1_scores)
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results, example_eval_results
    
class QAAccuracy(BaseMetric):
    metric_name: str = "qa_accuracy"
    
    def __init__(self, global_config: Optional[BaseConfig] = None, strict_label_matching=True):
        """
        Modes:
            1. standard (Default): 'gold' in 'prediction' (Answer Presence / Containment).
            Good for open-ended Generative QA.
            
            2. strict_label_matching: Uses Regex Word Boundaries (\b).
            REQUIRED for classification tasks like 'SUPPORTED' vs 'NOT_SUPPORTED'.
            Prevents 'SUPPORTED' from matching inside 'NOT_SUPPORTED'.
        """
        super().__init__(global_config)
        self.strict_label_matching = strict_label_matching

    def check_correctness(self, gold_list: List[str], pred_str: str) -> bool:
        """
        Determines if the prediction matches the gold answer.
        """
        p_norm = normalize_answer(pred_str)
        
        for gold in gold_list:
            g_norm = normalize_answer(gold)
            
            if self.strict_label_matching:
                # Use Regex Word Boundaries (\b). This prevents "supported" matching inside "not_supported"
                pattern = rf"\b{re.escape(g_norm)}\b"
                if re.search(pattern, p_norm):
                    return True
            else:
                # Standard substring match
                if g_norm in p_norm:
                    return True
                    
        return False

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str]) -> Tuple[Dict[str, float], List[float]]:
        """
        Calculate the Accuracy Score.
        
        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged Accuracy score.
                - A list of dictionaries with Accuracy scores for each example.
        """
        
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_acc = 0

        for g_ans_list, p_ans in zip(gold_answers, predicted_answers):            
            is_correct = self.check_correctness(g_ans_list, p_ans)
            acc_score = 1.0 if is_correct else 0.0
            example_eval_results.append({"Accuracy": acc_score})
            total_acc += acc_score
            
        avg_acc = total_acc / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"Accuracy": avg_acc}
        
        return pooled_eval_results, example_eval_results
    
class QAJSRScore(BaseMetric):
    """
    Computes JSR Score.
    JSR Score = 1.0 iff (Answer is in LLM response AND All Gold Docs are present in Retrieved Docs).
    Otherwise, it is 0.0.
    """
    metric_name: str = "qa_jsr_score"
    
    def __init__(self, global_config: Optional[BaseConfig] = None, strict_label_matching=True):
        super().__init__(global_config)
        self.strict_label_matching = strict_label_matching

    def check_correctness(self, gold_list: List[str], pred_str: str) -> bool:
        """
        Determines if the prediction matches the gold answer.
        """
        p_norm = normalize_answer(pred_str)
        
        for gold in gold_list:
            g_norm = normalize_answer(gold)
            
            if self.strict_label_matching:
                # Use Regex Word Boundaries (\b). This prevents "supported" matching inside "not_supported"
                pattern = rf"\b{re.escape(g_norm)}\b"
                if re.search(pattern, p_norm):
                    return True
            else:
                # Standard substring match
                if g_norm in p_norm:
                    return True
                    
        return False
    
    def calculate_metric_scores(self, 
                                gold_answers: List[List[str]], 
                                predicted_answers: List[str],
                                gold_docs: List[List[str]],
                                retrieved_docs: List[List[str]]) -> Tuple[Dict[str, float], List[float]]:
        """
        Calculate the JSR Score.
        
        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents) for each query.
            retrieved_docs (List[List[str]]): List of lists containing the retrieved documents for each query.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged JSR score.
                - A list of dictionaries with JSR scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_jsr = 0.0

        for g_ans_list, p_ans, g_docs_list, r_docs_list in zip(gold_answers, predicted_answers, gold_docs, retrieved_docs):
            
            # 1. Check Answer Correctness
            is_answer_correct = self.check_correctness(g_ans_list, p_ans)

            # 2. Check Perfect Retrieval
            g_docs_set = set(g_docs_list)
            r_docs_set = set(r_docs_list)
            
            # If gold docs are empty, strict retrieval usually implies failure, 
            # or success if we expect no docs. Assuming standard definition:
            if not g_docs_set:
                is_retrieval_perfect = False
            else:
                is_retrieval_perfect = g_docs_set.issubset(r_docs_set)

            # 3. Calculate JSR
            jsr_val = 1.0 if (is_answer_correct and is_retrieval_perfect) else 0.0
            
            example_eval_results.append({"JSR": jsr_val})
            total_jsr += jsr_val

        avg_jsr = total_jsr / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"JSR": avg_jsr}

        return pooled_eval_results, example_eval_results

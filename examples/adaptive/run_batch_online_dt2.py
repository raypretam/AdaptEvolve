"""
Online Decision Tree Router for LLM Selection

This script uses River's Hoeffding Tree for online learning to decide
whether to use a small or large LLM for code generation refinement.

Key differences from run_batch_conf_dt.py:
- Uses River's HoeffdingTreeClassifier instead of sklearn DecisionTree
- Router learns incrementally as problems are processed
- Can warm-start from historical data
- Adapts to distribution shifts over time
"""

import asyncio
import json
import os
import ast
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from openevolve import OpenEvolve
from openevolve.config import load_config
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
import threading
from queue import Queue
import atexit
import pickle
import traceback

# River imports for online learning
from river import tree
from river import metrics as river_metrics

ROOT = os.path.dirname(__file__)

# -----------------------------------------------------------------------------
# ⚙️ CONFIGURATION PATHS
# -----------------------------------------------------------------------------
CONFIG_SMALL_YAML = os.path.join(ROOT, "config_small.yaml")
CONFIG_LARGE_YAML = os.path.join(ROOT, "config_big.yaml")

EVAL_SCRIPT = os.path.join(ROOT, "evaluator.py")
OUTPUT_ROOT = os.path.join("/", "livecodebench_runs_online_400_880")
ONLINE_ROUTER_MODEL_PATH = os.path.join(ROOT, "online_router_model_adaptive.pkl")
# WARMUP_DATA_PATH = None

# -----------------------------------------------------------------------------
# 🧠 ONLINE ROUTER MODEL (River Hoeffding Tree)
# -----------------------------------------------------------------------------

class OnlineLLMRouter:
    """
    Online LLM Router using River's Hoeffding Tree Classifier.
    
    This router learns incrementally from each problem's outcome to decide
    whether to use the small (4B) or large (32B) model for refinement.
    
    Label encoding:
        0 = Small model is sufficient
        1 = Large model is needed
    """
    
    def __init__(self, grace_period: int = 10, max_depth: int = 10, 
                 delta: float = 1e-7, min_samples_before_predict: int = 10,
                 decision_threshold: float = 0.4):
        """
        Initialize the online router.
        
        Args:
            grace_period: Number of samples before considering a split (lower = faster adaptation)
            max_depth: Maximum tree depth (higher = more complex patterns)
            delta: Hoeffding bound confidence (smaller = more aggressive splits)
            min_samples_before_predict: Minimum samples seen before making predictions
            decision_threshold: Probability threshold for routing to large model (lower = more conservative)
        """
        self.model = tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=grace_period,  # Reduced from 10 to 5 for faster splits
            max_depth=max_depth,  # Increased from 10 to 15 for more complex patterns
            split_criterion='gini',
            delta=delta,  # Reduced from 1e-5 to 1e-7 for more aggressive splitting
            leaf_prediction='nba',  # Naive Bayes Adaptive - good for uncertain cases
            nb_threshold=10,  # Increased from 0 to require more samples for NB
        )
        
        # Track metrics
        self.accuracy_metric = river_metrics.Accuracy()
        self.precision_metric = river_metrics.Precision()
        self.recall_metric = river_metrics.Recall()
        
        # Statistics
        self.samples_seen = 0
        self.predictions_made = 0
        self.min_samples_before_predict = min_samples_before_predict
        self.decision_threshold = decision_threshold  # Threshold for routing to large model
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # History for analysis
        self.prediction_history = []
        
    def _prepare_features(self, confidence_data: Dict) -> Dict:
        """Convert confidence data to feature dict for River"""
        return {
            'mean_confidence': float(confidence_data.get('mean_confidence', 0) or 0),
            'least_grouped_confidence': float(confidence_data.get('least_grouped_confidence', 0) or 0),
            'tail_confidence': float(confidence_data.get('tail_confidence', 0) or 0),
            'verbalized_conf': float(confidence_data.get('verbalized_conf', 0) or 0),
            'combined_score': float(confidence_data.get('combined_score', 0) or 0),
        }
    
    def predict(self, confidence_data: Dict) -> Tuple[int, float]:
        """
        Predict which model to use.
        
        Args:
            confidence_data: Dictionary with confidence metrics
            
        Returns:
            (prediction, confidence): 
                prediction: 0 for small model, 1 for large model
                confidence: Probability of the predicted class
        """
        with self._lock:
            features = self._prepare_features(confidence_data)
            
            # If not enough samples seen, use conservative default (route to large model)
            if self.samples_seen < self.min_samples_before_predict:
                # Check if confidence is very low, route to large model
                combined_score = features.get('combined_score', 0)
                if combined_score < 0.5:
                    return 1, 0.6  # Route to large model with moderate confidence
                return 0, 0.5  # Default to small with neutral confidence
            
            # Get prediction and probability
            pred = self.model.predict_one(features)
            proba = self.model.predict_proba_one(features)
            
            if pred is None:
                # If no prediction available, be conservative based on confidence scores
                combined_score = features.get('combined_score', 0)
                if combined_score < 0.5:
                    return 1, 0.6
                return 0, 0.5
            
            # Get probabilities for both classes
            prob_small = proba.get(0, 0.0) if proba else 0.5
            prob_large = proba.get(1, 0.0) if proba else 0.5
            
            # Apply decision threshold - if probability of needing large model exceeds threshold, use it
            # This makes the router more conservative (biased towards large model when uncertain)
            if prob_large >= self.decision_threshold:
                final_pred = 1
                pred_confidence = prob_large
            else:
                final_pred = 0
                pred_confidence = prob_small
            
            self.predictions_made += 1
            
            return int(final_pred), pred_confidence
    
    def learn(self, confidence_data: Dict, actual_outcome: int, 
              problem_id: str = None, update_metrics: bool = True):
        """
        Update the model with actual outcome.
        
        Args:
            confidence_data: Dictionary with confidence metrics
            actual_outcome: 1 if large model was needed, 0 if small was sufficient
            problem_id: Optional problem identifier for logging
            update_metrics: Whether to update accuracy metrics
        """
        with self._lock:
            features = self._prepare_features(confidence_data)
            
            # Update metrics before learning (prequential evaluation)
            if update_metrics and self.samples_seen >= self.min_samples_before_predict:
                pred = self.model.predict_one(features)
                if pred is not None:
                    self.accuracy_metric.update(actual_outcome, pred)
                    self.precision_metric.update(actual_outcome, pred)
                    self.recall_metric.update(actual_outcome, pred)
            
            # Learn from this sample
            self.model.learn_one(features, actual_outcome)
            self.samples_seen += 1
            
            # Record history
            self.prediction_history.append({
                'problem_id': problem_id,
                'features': features,
                'actual_outcome': actual_outcome,
                'samples_seen': self.samples_seen,
            })
    
    def get_stats(self) -> Dict:
        """Get current router statistics"""
        with self._lock:
            return {
                'samples_seen': self.samples_seen,
                'predictions_made': self.predictions_made,
                'accuracy': float(self.accuracy_metric.get()) if self.samples_seen > self.min_samples_before_predict else None,
                'precision': float(self.precision_metric.get()) if self.samples_seen > self.min_samples_before_predict else None,
                'recall': float(self.recall_metric.get()) if self.samples_seen > self.min_samples_before_predict else None,
                'tree_height': self.model.height if hasattr(self.model, 'height') else None,
                'n_leaves': self.model.n_leaves if hasattr(self.model, 'n_leaves') else None,
            }
    
    def warm_start(self, training_data: List[Tuple[Dict, int]]):
        """
        Pre-train router with historical data.
        
        Args:
            training_data: List of (confidence_data, label) tuples
        """
        print(f"🔥 Warming up online router with {len(training_data)} samples...")
        
        # Learn from warm-up data in order
        for i, (features, label) in enumerate(training_data):
            self.learn(features, label, update_metrics=False)
            
            # Print progress every 100 samples
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(training_data)} warm-up samples...")
        
        print(f"✅ Warm-up complete. Samples seen: {self.samples_seen}")
        
        # Print initial statistics after warm-up
        stats = self.get_stats()
        print(f"   Tree height: {stats.get('tree_height', 'N/A')}")
        print(f"   Number of leaves: {stats.get('n_leaves', 'N/A')}")
    
    def save(self, path: str):
        """Save the router model to disk"""
        with self._lock:
            data = {
                'model': self.model,
                'accuracy_metric': self.accuracy_metric,
                'precision_metric': self.precision_metric,
                'recall_metric': self.recall_metric,
                'samples_seen': self.samples_seen,
                'predictions_made': self.predictions_made,
                'min_samples_before_predict': self.min_samples_before_predict,
                'decision_threshold': self.decision_threshold,  # Save threshold
                'prediction_history': self.prediction_history[-100:],  # Keep last 100
            }
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        print(f"💾 Online router saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'OnlineLLMRouter':
        """Load a router model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        router = cls()
        router.model = data['model']
        router.accuracy_metric = data['accuracy_metric']
        router.precision_metric = data['precision_metric']
        router.recall_metric = data['recall_metric']
        router.samples_seen = data['samples_seen']
        router.predictions_made = data['predictions_made']
        router.min_samples_before_predict = data.get('min_samples_before_predict', 5)
        router.decision_threshold = data.get('decision_threshold', 0.4)
        router.prediction_history = data.get('prediction_history', [])
        
        print(f"✅ Online router loaded from {path}")
        print(f"   Samples seen: {router.samples_seen}, Accuracy: {router.accuracy_metric.get():.3f if router.accuracy_metric.get() else 'N/A'}")
        print(f"   Decision threshold: {router.decision_threshold}")
        return router


# -----------------------------------------------------------------------------
# 🌐 GLOBAL ROUTER INSTANCE
# -----------------------------------------------------------------------------

ROUTER_MODEL: Optional[OnlineLLMRouter] = None

def initialize_router():
    """Initialize or load the online router"""
    global ROUTER_MODEL
    
    if os.path.exists(ONLINE_ROUTER_MODEL_PATH):
        try:
            ROUTER_MODEL = OnlineLLMRouter.load(ONLINE_ROUTER_MODEL_PATH)
            print(f"✅ Loaded existing online router from {ONLINE_ROUTER_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to load router: {e}. Creating new one.")
            ROUTER_MODEL = OnlineLLMRouter()
    else:
        ROUTER_MODEL = OnlineLLMRouter()
        print(f"🆕 Created new online router")
        
        # Try to warm-start from historical data
        if os.path.exists(WARMUP_DATA_PATH):
            try:
                with open(WARMUP_DATA_PATH, 'r') as f:
                    warmup_data = json.load(f)
                training_samples = [
                    (sample['features'], sample['label'])
                    for sample in warmup_data
                ]
                ROUTER_MODEL.warm_start(training_samples)
            except Exception as e:
                print(f"⚠️ Failed to load warm-up data: {e}")


@dataclass
class ProcessingStats:
    """Track processing statistics"""
    start_time: float
    problems_completed: int = 0
    total_problems: int = 0
    solved_first: int = 0
    solved_later: int = 0
    unsolved: int = 0
    total_score: float = 0.0

# Global stats for monitoring with thread lock
processing_stats = ProcessingStats(start_time=time.time())
stats_lock = threading.Lock()

MAX_CONCURRENT_PROBLEMS = 64
MAX_CONCURRENT_LLM_CALLS = 128

# Thread-safe semaphores
llm_semaphore = threading.Semaphore(MAX_CONCURRENT_LLM_CALLS)
problem_semaphore = threading.Semaphore(MAX_CONCURRENT_PROBLEMS)

def cleanup_resources():
    """Cleanup function to be called on exit"""
    print("\n🧹 Cleaning up resources...")
    
    # Save the router model
    if ROUTER_MODEL is not None:
        try:
            ROUTER_MODEL.save(ONLINE_ROUTER_MODEL_PATH)
        except Exception as e:
            print(f"⚠️ Failed to save router on exit: {e}")
    
    time.sleep(2)

# Register cleanup function
atexit.register(cleanup_resources)

# -----------------------------------------------------------------------------
# 💰 COST TRACKING CONFIGURATION
# -----------------------------------------------------------------------------
SMALL_MODEL_COST = 0.125  # FLOPS units per call
LARGE_MODEL_COST = 1.0    # FLOPS units per call

# Global cost tracking with thread-safe lock
cost_tracking = {
    "small_model_calls": 0,
    "large_model_calls": 0,
    "total_flops": 0.0
}
cost_lock = threading.Lock()

def extract_confidence_from_evolution_result(best_result, program_path: str = None) -> Dict:
    """Extract both logprob-based and verbalized confidence from OpenEvolve Program metadata"""
    try:
        confidence_data = {"mean_confidence": 0, "least_grouped_confidence": 0, "tail_confidence": 0, "verbalized_conf": None}
        
        if hasattr(best_result, "metadata") and isinstance(best_result.metadata, dict):
            if "confidence" in best_result.metadata and best_result.metadata["confidence"]:
                confidence_data.update(best_result.metadata["confidence"])
                
        elif hasattr(best_result, "confidence") and best_result.confidence:
            confidence_data.update(best_result.confidence)

        if hasattr(best_result, "metadata") and isinstance(best_result.metadata, dict):
            if "verbalized_conf" in best_result.metadata and best_result.metadata["verbalized_conf"]:
                confidence_data.update(best_result.metadata["verbalized_conf"])

        return confidence_data
    except Exception as e:
        print(f"Error extracting confidence: {e}")
        return {"mean_confidence": 0, "least_grouped_confidence": 0, "tail_confidence": 0, "verbalized_conf": None}


def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown blocks if present"""
    code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    return text

def clean_html_tags(text: str) -> str:
    """Remove HTML tags from text"""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    text = text.replace('&nbsp;', ' ').replace('&quot;', '"')
    return text

def format_io_example(input_str: str, output_str: str, index: int) -> str:
    """Format input/output example in a clear way"""
    return f"""Example {index}:
Input:
{input_str}
Output:
{output_str}"""

def parse_test_examples(sample: Dict) -> str:
    """Parse test cases to create clear examples"""
    examples = []
    try:
        if "public_test_cases" in sample and sample["public_test_cases"]:
            test_cases = ast.literal_eval(sample["public_test_cases"])
            for i, test in enumerate(test_cases[:3]):
                input_str = test.get("input", "").strip()
                output_str = test.get("output", "").strip()
                examples.append(format_io_example(input_str, output_str, i + 1))
    except Exception:
        pass
    return "\n\n".join(examples)

def extract_constraints(question_content: str) -> tuple[str, str]:
    """Extract constraints section from question content"""
    constraints_patterns = [
        r'Constraints?:(.*?)(?=Note:|Example:|Input:|$)',
        r'\*\*Constraints?\*\*:(.*?)(?=\*\*|$)',
        r'### Constraints?:?(.*?)(?=###|$)'
    ]
    
    for pattern in constraints_patterns:
        match = re.search(pattern, question_content, re.DOTALL | re.IGNORECASE)
        if match:
            constraints = match.group(1).strip()
            content_without_constraints = question_content.replace(match.group(0), "").strip()
            return content_without_constraints, constraints
    
    return question_content, ""

def stratified_sample_problems(num_problems: int = 50, random_state: int = 42, excluded_indices: List[int] = None) -> tuple[List[Dict], List[int]]:
    """Sample problems using stratified sampling to maintain difficulty distribution"""
    print(f"Loading dataset for stratified sampling of {num_problems} problems...")
    
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5", trust_remote_code=True)
    df = dataset.to_pandas()
    df['original_index'] = df.index
    
    if excluded_indices:
        print(f"Excluding {len(excluded_indices)} indices from previous runs...")
        df = df[~df['original_index'].isin(excluded_indices)]
        print(f"Remaining problems after exclusion: {len(df)}")
    
    difficulty_distribution = df['difficulty'].value_counts(normalize=True)
    print("Original problem distribution by difficulty:")
    for diff, prop in difficulty_distribution.items():
        count = (df['difficulty'] == diff).sum()
        print(f"  {diff}: {count} problems ({prop:.1%})")
    
    if num_problems == -1 or num_problems >= len(df):
        print(f"\nUsing all {len(df)} remaining problems")
        sampled_df = df.sort_values('original_index')
        original_indices = sampled_df['original_index'].tolist()
        problems = sampled_df.drop('original_index', axis=1).to_dict('records')
        print(f"\nSuccessfully loaded {len(problems)} problems.")
        return problems, original_indices
    
    sample_counts = (difficulty_distribution * num_problems).round().astype(int)
    diff = num_problems - sample_counts.sum()
    if diff != 0:
        most_frequent_difficulty = sample_counts.idxmax()
        sample_counts[most_frequent_difficulty] += diff
    
    print(f"\nSampling {num_problems} problems with the following distribution:")
    for diff, count in sample_counts.items():
        print(f"  {diff}: {count} problems ({count/num_problems:.1%})")
    
    sampled_df = df.groupby('difficulty', group_keys=False).apply(
        lambda x: x.sample(
            n=int(sample_counts[x.name]) if x.name in sample_counts else 0, 
            random_state=random_state
        )
    )
    
    sampled_df = sampled_df.sort_values('original_index')
    original_indices = sampled_df['original_index'].tolist()
    problems = sampled_df.drop('original_index', axis=1).to_dict('records')
    
    print(f"\nSuccessfully sampled {len(problems)} problems.")
    return problems, original_indices

def create_enhanced_stub(sample: Dict, out_dir: str) -> tuple[str, str]:
    """Create an enhanced stub"""
    question_title = sample.get("question_title", "")
    question_content = sample.get("question_content", "")
    difficulty = sample.get("difficulty", "unknown")
    starter_code = sample.get("starter_code", "")
    
    function_name = "solve"
    if starter_code:
        func_match = re.search(r'def\s+(\w+)\s*\(', starter_code)
        if func_match:
            function_name = func_match.group(1)

    question_content = clean_html_tags(question_content)
    question_content = extract_code_from_markdown(question_content)
    question_content, constraints = extract_constraints(question_content)
    
    metadata = []
    if "source" in sample:
        metadata.append(f"Source: {sample['source']}")
    if "contest" in sample:
        metadata.append(f"Contest: {sample['contest']}")
    
    stub_content = f'''# EVOLVE-BLOCK-START
"""
========================================
Problem: {question_title}
Difficulty: {difficulty}
{' | '.join(metadata) if metadata else ''}
========================================

PROBLEM DESCRIPTION:
{question_content}

{f'CONSTRAINTS:{chr(10)}{constraints}' if constraints else ''}
CONFIDENCE ASSESSMENT:
After completing your solution, perform a thorough self-review as if you were conducting a code review for a colleague.

"""
import sys
# Your solution implementation here

# EVOLVE-BLOCK-END
# Provide your final confidence assessment below.
# Output only a floating-point value within the <confidence> tags. Do not add extra text or reasoning.

<confidence>
[Confidence score between 0.0 and 10.0]
</confidence>

'''
    os.makedirs(out_dir, exist_ok=True)
    stub_path = os.path.join(out_dir, "prompt.py")
    with open(stub_path, "w") as f:
        f.write(stub_content)
        f.flush()
    
    return stub_path, "solve"


def run_openevolve_sync(evo_instance: OpenEvolve, phase_name: str, problem_id: str):
    """Run OpenEvolve synchronously with LLM semaphore"""
    acquired = False
    try:
        llm_semaphore.acquire()
        acquired = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(evo_instance.run())
            return result
        except Exception as e:
            print(f"Error in {phase_name} for {problem_id}: {str(e)}")
            raise
        finally:
            loop.close()
    finally:
        if acquired:
            llm_semaphore.release()

def update_stats_thread_safe(result: Dict):
    """Update global statistics in a thread-safe manner"""
    with stats_lock:
        processing_stats.problems_completed += 1
        processing_stats.total_score += result["combined_score"]
        
        if result["problem_label"] == "solved_first_iteration":
            processing_stats.solved_first += 1
        elif result["problem_label"] == "solved_later_iteration":
            processing_stats.solved_later += 1
        else:
            processing_stats.unsolved += 1

def increment_model_cost(model_type: str):
    """Thread-safe increment of model usage and cost"""
    with cost_lock:
        if model_type == "small":
            cost_tracking["small_model_calls"] += 1
            cost_tracking["total_flops"] += SMALL_MODEL_COST
        elif model_type == "large":
            cost_tracking["large_model_calls"] += 1
            cost_tracking["total_flops"] += LARGE_MODEL_COST

def print_progress():
    """Print current progress (thread-safe)"""
    with stats_lock:
        if processing_stats.problems_completed == 0:
            return
            
        elapsed = time.time() - processing_stats.start_time
        avg_score = processing_stats.total_score / processing_stats.problems_completed
        eta_minutes = (elapsed/processing_stats.problems_completed) * \
                     (processing_stats.total_problems - processing_stats.problems_completed) / 60
        
        # Get router stats
        router_stats = ROUTER_MODEL.get_stats() if ROUTER_MODEL else {}
        router_acc = router_stats.get('accuracy')
        router_acc_str = f"{router_acc:.3f}" if router_acc is not None else "N/A"
        
        print(f"Progress: {processing_stats.problems_completed}/{processing_stats.total_problems} "
              f"({processing_stats.problems_completed/processing_stats.total_problems*100:.1f}%) | "
              f"Avg Score: {avg_score:.3f} | "
              f"1st: {processing_stats.solved_first} | "
              f"Later: {processing_stats.solved_later} | "
              f"Unsolved: {processing_stats.unsolved} | "
              f"Router Acc: {router_acc_str} | "
              f"Elapsed: {elapsed/60:.1f}min | "
              f"ETA: {eta_minutes:.1f}min")


def determine_actual_outcome(initial_score: float, final_score: float, 
                              used_large_model: bool) -> int:
    """
    Determine the actual outcome for router learning.
    
    Returns:
        0: Small model was sufficient (or large model didn't help much)
        1: Large model was needed (significant improvement)
    """
    improvement = final_score - initial_score
    
    # If problem was solved or nearly solved initially, small model was sufficient
    if initial_score >= 0.95:
        return 0
    
    # If large model was used and provided significant improvement
    if used_large_model and improvement >= 0.2:
        return 1
    
    # If large model was used but didn't help much
    if used_large_model and improvement < 0.1:
        return 0  # Large model wasn't needed
    
    # If small model was used and problem got solved
    if not used_large_model and final_score >= 1.0:
        return 0  # Small model was sufficient
    
    # If small model was used and problem wasn't solved but had low initial score
    if not used_large_model and final_score < 0.5 and initial_score < 0.3:
        return 1  # Large model might have been needed
    
    # Default: if we used small and it worked reasonably, it was sufficient
    return 0 if final_score >= 0.5 else 1


def solve_one_problem_with_online_routing(sample: Dict, idx: int, total: int, original_idx: int) -> Dict:
    """Solve one problem with online routing decision"""
    acquired = False
    try:
        problem_semaphore.acquire()
        acquired = True
        
        question_id = sample.get("question_id", "unknown")
        safe_id = question_id.replace("/", "_").replace(" ", "_")
        task_dir = os.path.join(OUTPUT_ROOT, f"{original_idx:04d}_{safe_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        problem_status = {
            "solved": False,
            "solved_iteration": None,
            "confidence_measurements": [],
            "final_score": 0.0,
            "router_decision": None,
            "router_confidence": None,
            "used_large_model": False,
        }
        
        result = {
            "question_id": question_id,
            "question_title": sample.get("question_title", "N/A"),
            "original_index": original_idx,
            "combined_score": 0.0,
            "pass_rate": 0.0,
            "difficulty": sample.get("difficulty", "N/A"),
            "source": sample.get("source", "N/A"),
            "time_taken": 0.0,
            "solved": False,
            "solved_iteration": None,
            "problem_label": "unsolved",
            "confidence_measurements": [],
            "router_decision": None,
            "router_confidence": None,
            "model_used": "small",
            "error": None
        }
        
        problem_start = time.time()
        
        try:
            # Dump sample.json for the evaluator
            sample_path = os.path.join(task_dir, "sample.json")
            with open(sample_path, "w") as sf:
                json.dump(sample, sf, indent=2)
                sf.flush()
            
            # Create enhanced stub
            stub_path, _ = create_enhanced_stub(sample, task_dir)
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Enhanced stub written")
            
            # Phase 0: Initial generation with small model
            best0 = _run_phase0_initial_generation(
                stub_path, sample_path, task_dir, safe_id, idx, total
            )
            
            if not best0:
                result["error"] = "Phase 0 failed"
                return result
            
            # Get initial metrics
            initial_metrics = best0.metrics
            initial_combined = initial_metrics.get("combined_score", 0.0)
            initial_public = initial_metrics.get("public_pass_rate", 0.0)
            initial_private = initial_metrics.get("private_pass_rate", 0.0)
            
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Initial scores - Combined: {initial_combined:.2%}")
            
            # Check if solved in first iteration
            if initial_combined >= 1.0:
                problem_status["solved"] = True
                problem_status["solved_iteration"] = 1
                problem_status["final_score"] = initial_combined
                result["solved"] = True
                result["solved_iteration"] = 1
                result["problem_label"] = "solved_first_iteration"
                result["combined_score"] = initial_combined
                result["pass_rate"] = initial_combined
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Problem SOLVED in iteration 1!")
                return result
            
            # Extract confidence
            confidence_data = _extract_and_log_confidence(
                best0, task_dir, "phase0", initial_combined, safe_id, idx, total
            )
            problem_status["confidence_measurements"].append(confidence_data)
            
            result["combined_score"] = initial_combined
            result["pass_rate"] = initial_combined
            
            # Check if almost solved
            if initial_combined >= 0.95:
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Almost solved! Skipping refinement.")
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                return result
            
            # Check solution file exists
            full_code_path = os.path.join(task_dir, "phase0", "best", "best_program.py")
            if not os.path.exists(full_code_path):
                result["error"] = "Phase 0 solution file not found"
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                return result
            
            # 🧠 ONLINE ROUTER DECISION
            phase1_config_path, router_decision_label, phase1_model_type = _determine_phase1_config_online(
                confidence_data, result, safe_id, idx, total
            )
            
            problem_status["router_decision"] = router_decision_label
            problem_status["used_large_model"] = (phase1_model_type == "large")
            result["router_decision"] = router_decision_label
            result["model_used"] = phase1_model_type
            
            # Phase 1: Refinement
            best1 = _run_phase1_refinement(
                full_code_path, sample_path, task_dir, phase1_config_path,
                initial_public, initial_private, phase1_model_type,
                safe_id, idx, total
            )
            
            # Process Phase 1 results
            final_combined = initial_combined
            if best1:
                final_metrics = best1.metrics
                final_combined = final_metrics.get("combined_score", 0.0)
                final_public = final_metrics.get("public_pass_rate", 0.0)
                final_private = final_metrics.get("private_pass_rate", 0.0)
                
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Final scores - Combined: {final_combined:.2%}")
                
                if final_combined >= 1.0 and not problem_status["solved"]:
                    problem_status["solved"] = True
                    problem_status["solved_iteration"] = "phase1"
                    result["solved"] = True
                    result["solved_iteration"] = "phase1"
                    result["problem_label"] = "solved_later_iteration"
                    
                    print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Problem SOLVED in later iterations!")
                    
                    # Extract confidence for solved solution
                    phase1_best_path = os.path.join(task_dir, "phase1", "best", "best_program.py")
                    solved_confidence_data = extract_confidence_from_evolution_result(best1, phase1_best_path)
                    solved_confidence_data["iteration"] = "phase1_final"
                    solved_confidence_data["solved"] = True
                    solved_confidence_data["combined_score"] = final_combined
                    problem_status["confidence_measurements"].append(solved_confidence_data)
                
                result["combined_score"] = final_combined
                result["pass_rate"] = final_combined
                problem_status["final_score"] = final_combined
                
                # Save final solution
                _save_final_solution(
                    task_dir, sample, original_idx, final_combined,
                    final_public, final_private, problem_status, router_decision_label
                )
            else:
                problem_status["final_score"] = initial_combined
            
            # 🧠 UPDATE ONLINE ROUTER with actual outcome
            actual_outcome = determine_actual_outcome(
                initial_combined, final_combined, problem_status["used_large_model"]
            )
            
            if ROUTER_MODEL is not None:
                ROUTER_MODEL.learn(
                    confidence_data, 
                    actual_outcome, 
                    problem_id=safe_id
                )
                router_stats = ROUTER_MODEL.get_stats()
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: 🧠 Router updated. "
                      f"Samples: {router_stats['samples_seen']}, "
                      f"Acc: {router_stats['accuracy']:.3f if router_stats['accuracy'] else 'N/A'}")
                
        except Exception as e:
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ❌ Error: {str(e)}")
            result["error"] = str(e)
            traceback.print_exc()
        
        finally:
            result["time_taken"] = time.time() - problem_start
            result["confidence_measurements"] = problem_status["confidence_measurements"]
            
            if result["problem_label"] == "unsolved" and result["combined_score"] > 0:
                result["problem_label"] = "partial"
            
            update_stats_thread_safe(result)
            print_progress()
        
        return result
    finally:
        if acquired:
            problem_semaphore.release()


def _run_phase0_initial_generation(stub_path: str, sample_path: str, task_dir: str,
                                   safe_id: str, idx: int, total: int):
    """Run Phase 0 initial generation with small model"""
    print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Phase 0 - Generating initial solution (Small Model)...")
    
    increment_model_cost("small")
    
    cfg0 = load_config(CONFIG_SMALL_YAML)
    cfg0.max_iterations = 1
    cfg0.database.db_path = os.path.join(task_dir, "evolution_database")
    cfg0.llm.temperature = 0.6
    cfg0.llm.top_p = 0.95
    
    evo0 = OpenEvolve(
        initial_program_path=stub_path,
        evaluation_file=EVAL_SCRIPT,
        config=cfg0,
        output_dir=os.path.join(task_dir, "phase0"),
        test_file_path=sample_path
    )
    
    return run_openevolve_sync(evo0, "phase0", safe_id)


def _extract_and_log_confidence(best_result, task_dir: str, phase: str,
                                combined_score: float, safe_id: str,
                                idx: int, total: int) -> Dict:
    """Extract confidence metrics and log them"""
    best_program_path = os.path.join(task_dir, phase, "best", "best_program.py")
    confidence_data = extract_confidence_from_evolution_result(best_result, best_program_path)
    
    confidence_data["iteration"] = 1 if phase == "phase0" else "phase1_final"
    confidence_data["solved"] = False
    confidence_data["combined_score"] = combined_score
    
    print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Confidence after iter 1 (unsolved) - "
          f"Mean: {confidence_data['mean_confidence']:.3f}, "
          f"Verbalized: {confidence_data.get('verbalized_conf', 'N/A')}")
    
    return confidence_data


def _determine_phase1_config_online(confidence_data: Dict, result: Dict,
                                    safe_id: str, idx: int, total: int) -> Tuple[str, str, str]:
    """Determine which config to use for Phase 1 using ONLINE router"""
    phase1_config_path = CONFIG_SMALL_YAML
    router_decision_label = "Default (Small)"
    phase1_model_type = "small"
    
    if ROUTER_MODEL is not None and not result["solved"]:
        try:
            # Get prediction from online router
            prediction, confidence = ROUTER_MODEL.predict(confidence_data)
            
            result["router_confidence"] = confidence
            
            if prediction == 1:  # Large model recommended
                phase1_config_path = CONFIG_LARGE_YAML
                router_decision_label = f"Online Router: Large (conf={confidence:.3f})"
                phase1_model_type = "large"
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: "
                      f"🧠 Online Router → LARGE model (confidence: {confidence:.3f})")
            else:
                router_decision_label = f"Online Router: Small (conf={confidence:.3f})"
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: "
                      f"🧠 Online Router → SMALL model (confidence: {confidence:.3f})")
                      
        except Exception as e:
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: "
                  f"⚠️ Router error: {e}. Using small model.")
    
    return phase1_config_path, router_decision_label, phase1_model_type


def _run_phase1_refinement(full_code_path: str, sample_path: str, task_dir: str,
                           config_path: str, initial_public: float,
                           initial_private: float, model_type: str,
                           safe_id: str, idx: int, total: int):
    """Run Phase 1 refinement with selected config"""
    increment_model_cost(model_type)
    
    base_cfg = load_config(config_path)
    remaining_iterations = max((base_cfg.max_iterations or 10) - 1, 0)
    base_cfg.max_iterations = remaining_iterations
    base_cfg.database.db_path = os.path.join(task_dir, "evolution_database")
    base_cfg.llm.temperature = 0.6
    base_cfg.llm.top_p = 0.9
    
    if initial_public > 0.8 and initial_private < 0.5:
        print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Focusing on generalization...")
        base_cfg.database.exploration_ratio = 0.7
        base_cfg.database.exploitation_ratio = 0.3
    else:
        base_cfg.database.exploration_ratio = 0.5
        base_cfg.database.exploitation_ratio = 0.5
    
    print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Phase 1 - Refining ({remaining_iterations} iterations) using {os.path.basename(config_path)}...")
    
    evo1 = OpenEvolve(
        initial_program_path=full_code_path,
        evaluation_file=EVAL_SCRIPT,
        config=base_cfg,
        output_dir=os.path.join(task_dir, "phase1"),
        test_file_path=sample_path
    )
    
    return run_openevolve_sync(evo1, "phase1", safe_id)


def _save_final_solution(task_dir: str, sample: Dict, original_idx: int,
                         final_combined: float, final_public: float,
                         final_private: float, problem_status: Dict,
                         router_decision_label: str):
    """Save final solution with metadata"""
    best_path = os.path.join(task_dir, "phase1", "best", "best_program.py")
    if not os.path.exists(best_path):
        return
    
    final_path = os.path.join(task_dir, "final_solution.py")
    with open(best_path, 'r') as src:
        content = src.read()
    
    metadata_comment = f"""# LiveCodeBench Solution
# Problem: {sample.get('question_title', 'Unknown')}
# Difficulty: {sample.get('difficulty', 'Unknown')}
# Original Index: {original_idx}
# Final Score: {final_combined:.2%} (Public: {final_public:.2%}, Private: {final_private:.2%})
# Solved: {problem_status['solved']} (Iteration: {problem_status['solved_iteration']})
# Generated by OpenEvolve with Online Router
# Router Decision: {router_decision_label}

"""
    with open(final_path, 'w') as dst:
        dst.write(metadata_comment + content)
        dst.flush()


def process_problems_parallel(problems: List[Dict], original_indices: List[int]) -> List[Dict]:
    """Process problems in parallel using ThreadPoolExecutor"""
    results = []
    total = len(problems)
    
    with stats_lock:
        processing_stats.total_problems = total
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROBLEMS) as executor:
        future_to_problem = {
            executor.submit(solve_one_problem_with_online_routing, sample, idx, total, original_indices[idx]): (sample, idx)
            for idx, sample in enumerate(problems)
        }
        
        for future in as_completed(future_to_problem):
            sample, idx = future_to_problem[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Problem {idx} failed with error: {e}")
                results.append({
                    "question_id": sample.get("question_id", "unknown"),
                    "question_title": sample.get("question_title", "N/A"),
                    "original_index": original_indices[idx],
                    "combined_score": 0.0,
                    "pass_rate": 0.0,
                    "difficulty": sample.get("difficulty", "N/A"),
                    "source": sample.get("source", "N/A"),
                    "time_taken": 0.0,
                    "solved": False,
                    "solved_iteration": None,
                    "problem_label": "unsolved",
                    "confidence_measurements": [],
                    "error": str(e)
                })
    
    return results


def main():
    """Main entry point with online decision tree routing"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Initialize the online router
    initialize_router()
    
    # Check requirements
    if not os.path.exists(EVAL_SCRIPT):
        print(f"Error: Evaluator script not found at {EVAL_SCRIPT}")
        return
    
    if not os.path.exists(CONFIG_SMALL_YAML):
        print(f"Error: Small Model Config file not found at {CONFIG_SMALL_YAML}")
        return
    if not os.path.exists(CONFIG_LARGE_YAML):
        print(f"Error: Large Model Config file not found at {CONFIG_LARGE_YAML}")
        return
    
    # Load excluded indices from previous runs
    excluded_indices = []
    sampling_info_path = os.path.join(ROOT, "sampling_info.json")
    if os.path.exists(sampling_info_path):
        try:
            with open(sampling_info_path, 'r') as f:
                prev_info = json.load(f)
                excluded_indices = prev_info.get("original_indices", [])
                print(f"Loaded {len(excluded_indices)} previously sampled indices to exclude")
        except Exception as e:
            print(f"Error loading previous sampling info: {e}")
    else:
        print(f"No previous sampling info found at {sampling_info_path}")
        print("Running on full dataset without exclusions...")
    
    # Load dataset
    try:
        problems, original_indices = stratified_sample_problems(
            num_problems=50,
            excluded_indices=excluded_indices
        )
        
        # Save the sampling information
        sampling_info = {
            "num_problems": len(problems),
            "excluded_indices_count": len(excluded_indices),
            "excluded_from_file": sampling_info_path if excluded_indices else None,
            "sampling_method": "stratified_random_sampling",
            "original_indices": original_indices,
            "sampling_timestamp": time.time(),
            "max_concurrent_problems": MAX_CONCURRENT_PROBLEMS,
            "max_concurrent_llm_calls": MAX_CONCURRENT_LLM_CALLS,
            "router_type": "online_hoeffding_tree"
        }
        sampling_file = os.path.join(OUTPUT_ROOT, "sampling_info.json")
        with open(sampling_file, 'w') as f:
            json.dump(sampling_info, f, indent=2)
        print(f"Sampling information saved to {sampling_file}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Process problems in parallel
    start_time = time.time()
    results = process_problems_parallel(problems, original_indices)
    total_time = time.time() - start_time
    
    # Save the router model
    if ROUTER_MODEL is not None:
        ROUTER_MODEL.save(ONLINE_ROUTER_MODEL_PATH)
    
    # Print final statistics
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULTS")
    print(f"{'='*80}")
    
    solved_first = sum(1 for r in results if r["problem_label"] == "solved_first_iteration")
    solved_later = sum(1 for r in results if r["problem_label"] == "solved_later_iteration")
    partial = sum(1 for r in results if r["problem_label"] == "partial")
    unsolved = sum(1 for r in results if r["problem_label"] == "unsolved")
    errors = sum(1 for r in results if r.get("error"))
    avg_score = np.mean([r["combined_score"] for r in results])
    
    print(f"Total problems: {len(results)}")
    print(f"Solved (1st iter): {solved_first} ({solved_first/len(results)*100:.1f}%)")
    print(f"Solved (later): {solved_later} ({solved_later/len(results)*100:.1f}%)")
    print(f"Partial: {partial} ({partial/len(results)*100:.1f}%)")
    print(f"Unsolved: {unsolved} ({unsolved/len(results)*100:.1f}%)")
    print(f"Errors: {errors}")
    print(f"Average score: {avg_score:.3f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    # Print router statistics
    if ROUTER_MODEL is not None:
        router_stats = ROUTER_MODEL.get_stats()
        print(f"\n{'='*80}")
        print(f"🧠 ONLINE ROUTER STATISTICS")
        print(f"{'='*80}")
        print(f"Samples seen: {router_stats['samples_seen']}")
        print(f"Predictions made: {router_stats['predictions_made']}")
        if router_stats['accuracy'] is not None:
            print(f"Accuracy: {router_stats['accuracy']:.3f}")
            print(f"Precision: {router_stats['precision']:.3f}")
            print(f"Recall: {router_stats['recall']:.3f}")
        print(f"Tree height: {router_stats.get('tree_height', 'N/A')}")
        print(f"Number of leaves: {router_stats.get('n_leaves', 'N/A')}")
    
    # Print cost summary
    print(f"\n{'='*80}")
    print(f"💰 COMPUTATIONAL COST SUMMARY (FLOPS)")
    print(f"{'='*80}")
    with cost_lock:
        print(f"Small Model (4B) Calls: {cost_tracking['small_model_calls']}")
        print(f"  Cost per call: {SMALL_MODEL_COST} FLOPS units")
        print(f"  Subtotal: {cost_tracking['small_model_calls'] * SMALL_MODEL_COST:.3f} FLOPS units")
        print(f"\nLarge Model (32B) Calls: {cost_tracking['large_model_calls']}")
        print(f"  Cost per call: {LARGE_MODEL_COST} FLOPS units")
        print(f"  Subtotal: {cost_tracking['large_model_calls'] * LARGE_MODEL_COST:.3f} FLOPS units")
        print(f"\n{'─'*80}")
        print(f"TOTAL FLOPS: {cost_tracking['total_flops']:.3f} units")
        print(f"{'='*80}")
    
    # Results by difficulty
    difficulties = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in difficulties:
            difficulties[diff] = {
                "total": 0, "solved_first": 0, "solved_later": 0,
                "partial": 0, "unsolved": 0, "total_score": 0
            }
        difficulties[diff]["total"] += 1
        difficulties[diff]["total_score"] += r["combined_score"]
        if r["problem_label"] == "solved_first_iteration":
            difficulties[diff]["solved_first"] += 1
        elif r["problem_label"] == "solved_later_iteration":
            difficulties[diff]["solved_later"] += 1
        elif r["problem_label"] == "partial":
            difficulties[diff]["partial"] += 1
        else:
            difficulties[diff]["unsolved"] += 1
    
    print(f"\n📈 Results by difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulties:
            stats = difficulties[diff]
            total_solved = stats['solved_first'] + stats['solved_later']
            avg_diff_score = stats["total_score"] / stats["total"]
            print(f"  {diff.capitalize()}: {total_solved}/{stats['total']} solved "
                  f"({total_solved/stats['total']*100:.1f}%), avg score: {avg_diff_score:.3f}")
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_ROOT, "results_with_online_routing.json")
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total": len(results),
                "solved_first_iteration": solved_first,
                "solved_later_iterations": solved_later,
                "partial": partial,
                "unsolved": unsolved,
                "errors": errors,
                "average_score": avg_score,
                "total_time_minutes": total_time / 60,
                "timestamp": time.time(),
            },
            "router_stats": ROUTER_MODEL.get_stats() if ROUTER_MODEL else None,
            "cost_tracking": dict(cost_tracking),
            "by_difficulty": {
                diff: {
                    "total": stats["total"],
                    "solved_first": stats["solved_first"],
                    "solved_later": stats["solved_later"],
                    "partial": stats["partial"],
                    "unsolved": stats["unsolved"],
                    "average_score": stats["total_score"] / stats["total"]
                }
                for diff, stats in difficulties.items()
            },
            "results": results,
        }, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")


if __name__ == "__main__":
    main()

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
import joblib  # For loading the router model
import atexit
import signal

ROOT = os.path.dirname(__file__)

# -----------------------------------------------------------------------------
# ⚙️ CONFIGURATION PATHS
# -----------------------------------------------------------------------------
# Define the specific config files for your vLLM setups
CONFIG_SMALL_YAML = os.path.join(ROOT, "config_small.yaml") 
CONFIG_LARGE_YAML = os.path.join(ROOT, "config_big.yaml")

EVAL_SCRIPT = os.path.join(ROOT, "evaluator.py")
OUTPUT_ROOT = os.path.join("/", "livecodebench_runs_decision_tree_new_200_400")
ROUTER_MODEL_PATH = os.path.join(ROOT, "llm_router_model_new.pkl")

# -----------------------------------------------------------------------------
# 🧠 GLOBAL ROUTER LOADING
# -----------------------------------------------------------------------------
ROUTER_MODEL = None
try:
    if os.path.exists(ROUTER_MODEL_PATH):
        router_data = joblib.load(ROUTER_MODEL_PATH)
        ROUTER_MODEL = router_data['model']
        print(f"✅ LLM Router Model loaded successfully from {ROUTER_MODEL_PATH}")
    else:
        print(f"⚠️ Router model not found at {ROUTER_MODEL_PATH}. Adaptive switching disabled.")
except Exception as e:
    print(f"⚠️ Failed to load router model: {e}")

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
    # Give threads time to finish
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

def stratified_sample_problems(num_problems: int = 50, random_state: int = 150, excluded_indices: List[int] = None) -> tuple[List[Dict], List[int]]:
    """Sample problems using stratified sampling to maintain difficulty distribution, excluding specified indices"""
    print(f"Loading dataset for stratified sampling of {num_problems} problems...")
    
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5", trust_remote_code=True)
    df = dataset.to_pandas()
    df['original_index'] = df.index
    
    # Filter out excluded indices if provided
    if excluded_indices:
        print(f"Excluding {len(excluded_indices)} indices from previous runs...")
        df = df[~df['original_index'].isin(excluded_indices)]
        print(f"Remaining problems after exclusion: {len(df)}")
    
    difficulty_distribution = df['difficulty'].value_counts(normalize=True)
    print("Original problem distribution by difficulty:")
    for diff, prop in difficulty_distribution.items():
        count = (df['difficulty'] == diff).sum()
        print(f"  {diff}: {count} problems ({prop:.1%})")
    
    # Handle case where we want all remaining problems
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
    """Create an enhanced stub """
    question_title = sample.get("question_title", "")
    question_content = sample.get("question_content", "")
    difficulty = sample.get("difficulty", "unknown")
    starter_code = sample.get("starter_code", "")
    
    # Extract function name from starter code
    function_name = "solve"  # default
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
        
        print(f"Progress: {processing_stats.problems_completed}/{processing_stats.total_problems} "
              f"({processing_stats.problems_completed/processing_stats.total_problems*100:.1f}%) | "
              f"Avg Score: {avg_score:.3f} | "
              f"1st: {processing_stats.solved_first} | "
              f"Later: {processing_stats.solved_later} | "
              f"Unsolved: {processing_stats.unsolved} | "
              f"Elapsed: {elapsed/60:.1f}min | "
              f"ETA: {eta_minutes:.1f}min")

def solve_one_problem_with_confidence(sample: Dict, idx: int, total: int, original_idx: int) -> Dict:
    """Solve one problem with confidence measurement - fully synchronous version for thread pool"""
    acquired = False
    try:
        problem_semaphore.acquire()
        acquired = True
        
        # Prepare directories
        question_id = sample.get("question_id", "unknown")
        safe_id = question_id.replace("/", "_").replace(" ", "_")
        task_dir = os.path.join(OUTPUT_ROOT, f"{original_idx:04d}_{safe_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Initialize problem tracking
        problem_status = {
            "solved": False,
            "solved_iteration": None,
            "confidence_measurements": [],
            "final_score": 0.0
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
                return _handle_first_iteration_solve(
                    result, problem_status, initial_combined, safe_id, idx, total
                )
            
            # Extract and log confidence
            confidence_data = _extract_and_log_confidence(
                best0, task_dir, "phase0", initial_combined, safe_id, idx, total
            )
            problem_status["confidence_measurements"].append(confidence_data)
            
            result["combined_score"] = initial_combined
            result["pass_rate"] = initial_combined
            
            # Check if almost solved
            if initial_combined >= 0.95:
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Almost solved! Skipping refinement.")
                problem_status["final_score"] = 1.0
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                return result
            
            # Check solution file exists
            full_code_path = os.path.join(task_dir, "phase0", "best", "best_program.py")
            if not os.path.exists(full_code_path):
                result["error"] = "Phase 0 solution file not found"
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                return result
            
            # Determine config for Phase 1 using router
            phase1_config_path, router_decision_label, phase1_model_type = _determine_phase1_config(
                confidence_data, result, safe_id, idx, total
            )
            
            # Phase 1: Refinement
            best1 = _run_phase1_refinement(
                full_code_path, sample_path, task_dir, phase1_config_path,
                initial_public, initial_private, phase1_model_type,
                safe_id, idx, total
            )
            
            # Process Phase 1 results
            if best1:
                _process_phase1_results(
                    best1, result, problem_status, sample, task_dir,
                    original_idx, router_decision_label, safe_id, idx, total
                )
            else:
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Phase 1 produced no improvement")
                problem_status["final_score"] = initial_combined
            
        except Exception as e:
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ❌ Error: {str(e)}")
            result["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        finally:
            result["time_taken"] = time.time() - problem_start
            result["confidence_measurements"] = problem_status["confidence_measurements"]
            
            # Record final confidence for unsolved problems
            if not result["solved"] and 'best1' in locals() and best1:
                final_confidence = extract_confidence_from_evolution_result(best1)
                final_confidence["iteration"] = "phase1_final_unsolved"
                final_confidence["solved"] = False
                final_confidence["combined_score"] = result["combined_score"]
                problem_status["confidence_measurements"].append(final_confidence)
            
            # Determine final problem label
            if result["problem_label"] == "unsolved" and result["combined_score"] > 0:
                result["problem_label"] = "partial"
            
            # Update global stats
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


def _handle_first_iteration_solve(result: Dict, problem_status: Dict, 
                                  combined_score: float, safe_id: str, 
                                  idx: int, total: int) -> Dict:
    """Handle problem solved in first iteration"""
    problem_status["solved"] = True
    problem_status["solved_iteration"] = 1
    problem_status["final_score"] = combined_score
    
    result["solved"] = True
    result["solved_iteration"] = 1
    result["problem_label"] = "solved_first_iteration"
    result["combined_score"] = combined_score
    result["pass_rate"] = combined_score
    
    print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Problem SOLVED in iteration 1!")
    return result


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


def _determine_phase1_config(confidence_data: Dict, result: Dict, 
                             safe_id: str, idx: int, total: int) -> Tuple[str, str, str]:
    """Determine which config to use for Phase 1 based on router decision"""
    phase1_config_path = CONFIG_SMALL_YAML
    router_decision_label = "Class 0 (Stay Small)"
    phase1_model_type = "small"
    
    if ROUTER_MODEL and not result["solved"]:
        try:
            # Feature vector: [current_model_size, mean, bottom, tail, least]
            f_mean = float(confidence_data.get("mean_confidence", 0.0))
            f_bottom = float(confidence_data.get("bottom_window_confidence", 0.0))
            f_tail = float(confidence_data.get("tail_confidence", 0.0))
            f_least = float(confidence_data.get("least_grouped_confidence", 0.0))
            
            features = np.array([[4, f_mean, f_bottom, f_tail, f_least]])
            decision = ROUTER_MODEL.predict(features)[0]
            
            if decision == 1:
                print(f"[T{threading.current_thread().ident}] 🔄 Router Decision: SWITCHING to LARGE Config (Class 1)")
                phase1_config_path = CONFIG_LARGE_YAML
                router_decision_label = "Class 1 (Switch Large)"
                phase1_model_type = "large"
            else:
                print(f"[T{threading.current_thread().ident}] ➡️ Router Decision: STAYING with SMALL Config (Class 0)")
        except Exception as e:
            print(f"[T{threading.current_thread().ident}] ⚠️ Router prediction failed, defaulting to small config. Error: {e}")
    
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
    
    # Adjust exploration/exploitation based on initial results
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


def _process_phase1_results(best1, result: Dict, problem_status: Dict, 
                            sample: Dict, task_dir: str, original_idx: int,
                            router_decision_label: str, safe_id: str, 
                            idx: int, total: int):
    """Process Phase 1 results and update problem status"""
    final_metrics = best1.metrics
    final_combined = final_metrics.get("combined_score", 0.0)
    final_public = final_metrics.get("public_pass_rate", 0.0)
    final_private = final_metrics.get("private_pass_rate", 0.0)
    
    print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Final scores - Combined: {final_combined:.2%}")
    
    # Check if problem got solved
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
        
        print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Confidence for solved solution - "
              f"Mean: {solved_confidence_data['mean_confidence']:.3f}, "
              f"Verbalized: {solved_confidence_data.get('verbalized_conf', 'N/A')}")
    
    result["combined_score"] = final_combined
    result["pass_rate"] = final_combined
    problem_status["final_score"] = final_combined
    
    # Save final solution
    _save_final_solution(
        task_dir, sample, original_idx, final_combined, 
        final_public, final_private, problem_status, router_decision_label
    )


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
# Generated by OpenEvolve
# Router Decision: {router_decision_label}

"""
    with open(final_path, 'w') as dst:
        dst.write(metadata_comment + content)
        dst.flush()

def process_problems_parallel(problems: List[Dict], original_indices: List[int]) -> List[Dict]:
    """Process problems in parallel using ThreadPoolExecutor"""
    results = []
    total = len(problems)
    
    # Update global stats
    with stats_lock:
        processing_stats.total_problems = total
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROBLEMS) as executor:
        # Submit all problems
        future_to_problem = {
            executor.submit(solve_one_problem_with_confidence, sample, idx, total, original_indices[idx]): (sample, idx)
            for idx, sample in enumerate(problems)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_problem):
            sample, idx = future_to_problem[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Problem {idx} generated an exception: {e}")
                import traceback
                traceback.print_exc()
                # Add error result
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
                    "problem_label": "error",
                    "confidence_measurements": [],
                    "error": str(e)
                })
    
    return results

def main():
    """Main entry point with ThreadPoolExecutor for parallel processing and confidence measurement"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Check requirements
    if not os.path.exists(EVAL_SCRIPT):
        print(f"Error: Evaluator script not found at {EVAL_SCRIPT}")
        return
    
    # Check config existence
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
                previous_sampling = json.load(f)
                excluded_indices = previous_sampling.get("original_indices", [])
                print(f"Loaded {len(excluded_indices)} indices to exclude from {sampling_info_path}")
        except Exception as e:
            print(f"Warning: Could not load previous sampling info: {e}")
            print("Proceeding without exclusions...")
    else:
        print(f"No previous sampling info found at {sampling_info_path}")
        print("Running on full dataset without exclusions...")
    
    # Load dataset and get first 200 problems (excluding previous indices)
    try:
        print(f"Loading LiveCodeBench dataset...")
        dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5", trust_remote_code=True)
        df = dataset.to_pandas()
        df['original_index'] = df.index
        
        # Filter out excluded indices if provided
        if excluded_indices:
            print(f"Excluding {len(excluded_indices)} indices from previous runs...")
            df = df[~df['original_index'].isin(excluded_indices)]
            print(f"Remaining problems after exclusion: {len(df)}")
        
        # Get first 200 problems
        num_problems = 280
        df = df.iloc[200:400]  # Get problems 200-399 for this run
        # df = stratified_sample_problems(num_problems=50, random_state=150, excluded_indices=excluded_indices)[0]
        original_indices = df['original_index'].tolist()
        problems = df.drop('original_index', axis=1).to_dict('records')
        
        print(f"Loaded first {len(problems)} problems from LiveCodeBench")
        
        # Print distribution by difficulty
        difficulty_counts = df['difficulty'].value_counts()
        print("Problem distribution by difficulty:")
        for diff, count in difficulty_counts.items():
            print(f"  {diff}: {count} problems ({count/len(problems)*100:.1f}%)")
        
        # Save the sampling information
        sampling_info = {
            "num_problems": len(problems),
            "excluded_indices_count": len(excluded_indices),
            "excluded_from_file": sampling_info_path if excluded_indices else None,
            "sampling_method": "sequential_second_200",
            "original_indices": original_indices,
            "sampling_timestamp": time.time(),
            "max_concurrent_problems": MAX_CONCURRENT_PROBLEMS,
            "max_concurrent_llm_calls": MAX_CONCURRENT_LLM_CALLS
        }
        sampling_file = os.path.join(OUTPUT_ROOT, "sampling_info.json")
        with open(sampling_file, 'w') as f:
            json.dump(sampling_info, f, indent=2)
        print(f"Sampling information saved to {sampling_file}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Process problems in parallel with confidence measurement
    start_time = time.time()
    results = process_problems_parallel(problems, original_indices)
    total_time = time.time() - start_time
    
    # Prepare confidence analysis
    confidence_analysis = []
    for result in results:
        if result["confidence_measurements"]:
            for conf_measurement in result["confidence_measurements"]:
                confidence_analysis.append({
                    "question_id": result["question_id"],
                    "original_index": result["original_index"],
                    "difficulty": result["difficulty"],
                    "problem_label": result["problem_label"],
                    "iteration": conf_measurement["iteration"],
                    "solved_at_measurement": conf_measurement["solved"],
                    "combined_score_at_measurement": conf_measurement["combined_score"],
                    "mean_confidence": conf_measurement["mean_confidence"],
                    "least_grouped_confidence": conf_measurement["least_grouped_confidence"],
                    "tail_confidence": conf_measurement["tail_confidence"],
                    "verbalized_conf": conf_measurement.get("verbalized_conf")
                })

    # Update the confidence analysis summary:
    if confidence_analysis:
        print(f"\nCONFIDENCE ANALYSIS:")
        
        # Analyze confidence for unsolved problems after 1st iteration
        unsolved_after_1st = [c for c in confidence_analysis if c["iteration"] == 1 and not c["solved_at_measurement"]]
        if unsolved_after_1st:
            avg_conf_unsolved = np.mean([c["mean_confidence"] for c in unsolved_after_1st])
            avg_least_grouped_unsolved = np.mean([c["least_grouped_confidence"] for c in unsolved_after_1st])
            avg_tail_unsolved = np.mean([c["tail_confidence"] for c in unsolved_after_1st])
            
            verbalized_confs = [c["verbalized_conf"] for c in unsolved_after_1st if c["verbalized_conf"] is not None]
            avg_verbalized_unsolved = np.mean(verbalized_confs) if verbalized_confs else None
            
            print(f"Unsolved after 1st iteration ({len(unsolved_after_1st)} problems):")
            print(f"  Average mean confidence: {avg_conf_unsolved:.3f}")
            print(f"  Average least grouped confidence: {avg_least_grouped_unsolved:.3f}")
            print(f"  Average tail confidence: {avg_tail_unsolved:.3f}")
            if avg_verbalized_unsolved is not None:
                print(f"  Average verbalized confidence: {avg_verbalized_unsolved:.1f}/100 ({len(verbalized_confs)}/{len(unsolved_after_1st)} provided)")

        
        # Analyze confidence for problems solved in later iterations
        solved_later_confs = [c for c in confidence_analysis if c["solved_at_measurement"]]
        if solved_later_confs:
            avg_conf_solved_later = np.mean([c["mean_confidence"] for c in solved_later_confs])
            avg_least_grouped_solved_later = np.mean([c["least_grouped_confidence"] for c in solved_later_confs])
            avg_tail_solved_later = np.mean([c["tail_confidence"] for c in solved_later_confs])
            print(f"Solved in later iterations ({len(solved_later_confs)} problems):")
            print(f"  Average mean confidence: {avg_conf_solved_later:.3f}")
            print(f"  Average least grouped confidence: {avg_least_grouped_solved_later:.3f}")
            print(f"  Average tail confidence: {avg_tail_solved_later:.3f}")
    
    # Results by difficulty
    difficulties = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in difficulties:
            difficulties[diff] = {
                "total": 0,
                "solved_first": 0,
                "solved_later": 0,
                "partial": 0,
                "unsolved": 0,
                "total_score": 0
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
    
    print(f"\nResults by difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulties:
            stats = difficulties[diff]
            total_solved = stats['solved_first'] + stats['solved_later']
            avg_diff_score = stats["total_score"] / stats["total"]
            print(f"  {diff.capitalize()}: {total_solved}/{stats['total']} solved "
                  f"({total_solved/stats['total']*100:.1f}%), "
                  f"avg score: {avg_diff_score:.3f}")
            print(f"    - 1st iter: {stats['solved_first']}, Later: {stats['solved_later']}, "
                  f"Partial: {stats['partial']}, Unsolved: {stats['unsolved']}")
    
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
        
        # Calculate efficiency metrics
        if len(results) > 0:
            avg_flops_per_problem = cost_tracking['total_flops'] / len(results)
            solved_problems = sum(1 for r in results if r['solved'])
            if solved_problems > 0:
                flops_per_solved = cost_tracking['total_flops'] / solved_problems
                print(f"\nEfficiency Metrics:")
                print(f"  Average FLOPS per problem: {avg_flops_per_problem:.3f} units")
                print(f"  Average FLOPS per solved problem: {flops_per_solved:.3f} units")
                print(f"  Model distribution: {cost_tracking['small_model_calls']} small, {cost_tracking['large_model_calls']} large")

if __name__ == "__main__":
    main()
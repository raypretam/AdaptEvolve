"""
Adaptive Multi-LLM System with Per-Iteration Model Switching

This script implements a confidence-based adaptive model selection system that:
1. Starts with a small 4B model for initial solution generation
2. Analyzes confidence metrics after each iteration
3. Switches between 4B and 32B models based on confidence at EVERY iteration
4. Tracks all model decisions and switches for analysis
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
import sys
import argparse

# Add path for rule-based classifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rule_based_classifier import RuleBasedModelSelector, ConfidenceMetrics, ModelChoice

ROOT        = os.path.dirname(__file__)
CONFIG_SMALL = os.path.join(ROOT, "config_small.yaml")  # 4B model
CONFIG_LARGE = os.path.join(ROOT, "config_big.yaml")    # 32B model
EVAL_SCRIPT = os.path.join(ROOT, "evaluator.py")
OUTPUT_ROOT = os.path.join(ROOT, "livecodebench_runs_adaptive_model_selection_run4")

# Define the specific question IDs to process
# TARGET_QUESTION_IDS = [
#     '0087_2999', '0168_3261', '0392_abc342_c', '0481_3354',
#     '0485_3316', '0505_3451', '0555_abc364_f', '0579_3468',
#     '0589_3515', '0645_abc370_f', '0674_arc183_c', '0678_3543',
#     '0689_3528', '0691_3518', '0714_abc374_f', '0785_abc385_f',
#     '0811_arc188_b', '0876_3562'
# ]

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
    model_switches_to_large: int = 0
    model_switches_to_small: int = 0
    total_4b_iterations: int = 0
    total_32b_iterations: int = 0
    total_flops: float = 0.0  # Track FLOPS (32B=1.0, 4B=0.125)

# Global stats for monitoring with thread lock
processing_stats = ProcessingStats(start_time=time.time())
stats_lock = threading.Lock()

MAX_CONCURRENT_PROBLEMS = 32
MAX_CONCURRENT_LLM_CALLS = 256

# Thread-safe semaphores
llm_semaphore = threading.Semaphore(MAX_CONCURRENT_LLM_CALLS)
problem_semaphore = threading.Semaphore(MAX_CONCURRENT_PROBLEMS)

# Initialize rule-based model selector with optimized thresholds
model_selector = RuleBasedModelSelector(
    lgc_threshold=6.65,  # Between unsolved (5.86) and solved (7.98) means
    mc_threshold=8.64,   # Between unsolved (7.65) and solved (9.76) means  
    tc_threshold=9.39,   # Between unsolved (8.51) and solved (10.28) means
    bwc_threshold=7.35,  
    use_ensemble=True   # Use voting across all 4 metrics
)

def extract_confidence_from_evolution_result(best_result, program_path: str = None) -> Dict:
    """Extract logprob-based confidence from OpenEvolve Program metadata
    
    This extracts confidence from the best_result's program object, which contains
    the confidence metrics calculated during the LLM generation (not from the stub).
    The stub code is never stored in the database - only LLM-generated solutions are stored.
    """
    try:
        confidence_data = {"mean_confidence": 0, "least_grouped_confidence": 0, "tail_confidence": 0, "bottom_window_confidence": 0}
        
        # Primary: Extract from the Program object itself (most reliable)
        if hasattr(best_result, "program") and best_result.program is not None:
            program = best_result.program
            
            # Check program's metadata first
            if hasattr(program, "metadata") and isinstance(program.metadata, dict):
                if "confidence" in program.metadata and program.metadata["confidence"]:
                    confidence_data.update(program.metadata["confidence"])
                    return confidence_data
            
            # Check program's confidence attribute
            if hasattr(program, "confidence") and program.confidence:
                confidence_data.update(program.confidence)
                return confidence_data
        
        # Fallback: Extract from best_result metadata
        if hasattr(best_result, "metadata") and isinstance(best_result.metadata, dict):
            if "confidence" in best_result.metadata and best_result.metadata["confidence"]:
                confidence_data.update(best_result.metadata["confidence"])
                return confidence_data

        # Fallback: Extract from best_result confidence attribute
        if hasattr(best_result, "confidence") and best_result.confidence:
            confidence_data.update(best_result.confidence)
            return confidence_data
        
        return confidence_data
        
    except Exception as e:
        print(f"    [ERROR] Exception in confidence extraction: {e}")
        return {"mean_confidence": 0, "least_grouped_confidence": 0, "tail_confidence": 0, "bottom_window_confidence": 0}


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

def filter_target_problems() -> tuple[List[Dict], List[int]]:
    """Load dataset and filter only the target problems"""
    print(f"Loading dataset and filtering for {len(TARGET_QUESTION_IDS)} specific problems...")
    
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5", trust_remote_code=True)
    df = dataset.to_pandas()
    df['original_index'] = df.index
    
    processed_target_ids = [q.split('_', 1)[1] for q in TARGET_QUESTION_IDS]
    # Filter for target question IDs
    filtered_df = df[df['question_id'].isin(processed_target_ids)]
    
    print(f"Found {len(filtered_df)} problems out of {len(processed_target_ids)} requested")
    
    # Show which problems were found and which were missing
    found_ids = set(filtered_df['question_id'].tolist())
    missing_ids = set(processed_target_ids) - found_ids
    
    if missing_ids:
        print(f"Warning: The following question IDs were not found: {missing_ids}")
    
    # Show distribution by difficulty for filtered problems
    difficulty_distribution = filtered_df['difficulty'].value_counts()
    print("\nFiltered problems distribution by difficulty:")
    for diff, count in difficulty_distribution.items():
        print(f"  {diff}: {count} problems")
    
    # Sort by original index to maintain order
    filtered_df = filtered_df.sort_values('original_index')
    original_indices = filtered_df['original_index'].tolist()
    problems = filtered_df.drop('original_index', axis=1).to_dict('records')
    
    print(f"\nSuccessfully loaded {len(problems)} target problems.")
    return problems, original_indices

def create_enhanced_stub(sample: Dict, out_dir: str) -> tuple[str, str]:
    """Create an enhanced stub with confidence assessment prompt"""
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
    
    return stub_path, function_name


def run_openevolve_sync(evo_instance: OpenEvolve, phase_name: str, problem_id: str):
    """Run OpenEvolve synchronously with LLM semaphore"""
    with llm_semaphore:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(evo_instance.run())
                
                # Properly cleanup pending tasks before closing the loop
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                # Give cancelled tasks a chance to finish
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
                return result
            finally:
                # Close the loop after all tasks are done
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
        except Exception as e:
            print(f"Error in {phase_name} for {problem_id}: {str(e)}")
            raise

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

def print_progress():
    """Print current progress (thread-safe)"""
    with stats_lock:
        if processing_stats.problems_completed == 0:
            return
            
        elapsed = time.time() - processing_stats.start_time
        avg_score = processing_stats.total_score / processing_stats.problems_completed
        eta_minutes = (elapsed / processing_stats.problems_completed) * \
                     (processing_stats.total_problems - processing_stats.problems_completed) / 60
        
        print(f"\n{'='*80}")
        print(f"PROGRESS UPDATE")
        print(f"{'='*80}")
        print(f"Completed: {processing_stats.problems_completed}/{processing_stats.total_problems} "
              f"({processing_stats.problems_completed/processing_stats.total_problems*100:.1f}%)")
        print(f"Average Score: {avg_score:.3f}")
        print(f"Solved 1st iter: {processing_stats.solved_first} | "
              f"Later: {processing_stats.solved_later} | "
              f"Unsolved: {processing_stats.unsolved}")
        print(f"Model switches: 4B→32B: {processing_stats.model_switches_to_large} | "
              f"32B→4B: {processing_stats.model_switches_to_small}")
        print(f"Model usage: 4B iterations: {processing_stats.total_4b_iterations} | "
              f"32B iterations: {processing_stats.total_32b_iterations}")
        print(f"Total FLOPS: {processing_stats.total_flops:.2f} units (32B=1.0, 4B=0.125)")
        print(f"Time: Elapsed {elapsed/60:.1f}min | ETA {eta_minutes:.1f}min")
        print(f"{'='*80}\n")

def solve_one_problem_with_confidence(sample: Dict, idx: int, total: int, original_idx: int) -> Dict:
    """
    Solve one problem with adaptive model selection at every iteration.
    
    Flow:
    1. Phase 0: Generate initial solution with 4B model
    2. Extract confidence metrics
    3. For each subsequent iteration:
       - Make model decision based on previous iteration's confidence
       - Run iteration with selected model
       - Extract new confidence metrics
       - Repeat until solved or max iterations
    """
    with problem_semaphore:
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
            "model_used": "4B",
            "model_decisions": [],
            "model_switches": [],
            "model_usage_history": [],
            "flops_used": 0.0,  # Track FLOPS for this problem
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
            print(f"\n[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Starting problem")
            print(f"  Difficulty: {sample.get('difficulty', 'N/A')}")
            print(f"  Source: {sample.get('source', 'N/A')}")
            
            # ========================================
            # PHASE 0: Initial solution with 4B model
            # ========================================
            print(f"\n[Phase 0] Generating initial solution with 4B model...")
            current_model = "4B"
            result["model_usage_history"].append({"iteration": 1, "model": "4B"})
            
            cfg0 = load_config(CONFIG_SMALL)
            cfg0.max_iterations = 1
            
            # Database configuration
            db_path = os.path.join(task_dir, "evolution_database")
            cfg0.database.db_path = db_path
            
            # LLM configuration for initial generation
            cfg0.llm.temperature = 0.6
            cfg0.llm.top_p = 0.95
            
            evo0 = OpenEvolve(
                initial_program_path=stub_path,
                evaluation_file=EVAL_SCRIPT,
                config=cfg0,
                output_dir=os.path.join(task_dir, "phase0"),
                test_file_path=sample_path
            )
            
            best0 = run_openevolve_sync(evo0, "phase0", safe_id)
            
            # Update iteration count and FLOPS
            with stats_lock:
                processing_stats.total_4b_iterations += 1
                processing_stats.total_flops += 0.125
            result["flops_used"] += 0.125
            
            if not best0:
                print(f"[Phase 0] ❌ Failed to generate initial solution")
                result["error"] = "Phase 0 failed"
                return result
            
            # Get metrics
            initial_metrics = best0.metrics
            initial_combined = initial_metrics.get("combined_score", 0.0)
            initial_public = initial_metrics.get("public_pass_rate", 0.0)
            initial_private = initial_metrics.get("private_pass_rate", 0.0)
            
            print(f"[Phase 0] Initial score: {initial_combined:.2%} (public: {initial_public:.2%}, private: {initial_private:.2%})")
            
            # Check if problem is solved after iteration 1
            if initial_combined >= 1.0:
                problem_status["solved"] = True
                problem_status["solved_iteration"] = 1
                problem_status["final_score"] = initial_combined
                result["solved"] = True
                result["solved_iteration"] = 1
                result["problem_label"] = "solved_first_iteration"
                result["combined_score"] = initial_combined
                result["pass_rate"] = initial_combined
                
                # Extract confidence even for solved problems (for analysis)
                confidence_data = extract_confidence_from_evolution_result(best0, best_program_path)
                confidence_data["iteration"] = 1
                confidence_data["solved"] = True
                confidence_data["combined_score"] = initial_combined
                problem_status["confidence_measurements"].append(confidence_data)
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                
                print(f"[Phase 0] ✅ Problem SOLVED in first iteration!")
                
                # Early exit - problem is solved, no need for Phase 1
                result["model_used"] = "4B"
                result["final_iteration"] = 1
                result["time_taken"] = time.time() - problem_start
                
                # Update global stats
                update_stats_thread_safe(result)
                print_progress()
                
                return result
            
            # Extract confidence from Phase 0 (for unsolved problems)
            phase0_output_dir = os.path.join(task_dir, "phase0")
            best_program_path = os.path.join(phase0_output_dir, "best", "best_program.py")

            confidence_data = extract_confidence_from_evolution_result(best0, best_program_path)
            confidence_data["iteration"] = 1
            confidence_data["solved"] = False
            confidence_data["combined_score"] = initial_combined
            problem_status["confidence_measurements"].append(confidence_data)

            print(f"[Phase 0] Confidence: MC={confidence_data['mean_confidence']:.2f}, "
                  f"LGC={confidence_data['least_grouped_confidence']:.2f}")
            
            # ========================================
            # MODEL DECISION FOR ITERATION 2
            # ========================================
            metrics = ConfidenceMetrics(
                least_grouped_confidence=confidence_data['least_grouped_confidence'],
                mean_confidence=confidence_data['mean_confidence'],
                tail_confidence=confidence_data['tail_confidence'],
                bottom_window_confidence=confidence_data.get('bottom_window_confidence', 0)
            )
            
            decision = model_selector.classify(metrics)
            next_model = decision.model_choice.value
            
            decision_record = {
                "iteration": 1,
                "previous_model": "4B",
                "decision": next_model,
                "confidence_score": decision.confidence_score,
                "reasoning": decision.reasoning,
                "metrics": decision.metrics_used,
                "combined_score": initial_combined
            }
            result["model_decisions"].append(decision_record)
            
            print(f"\n[Model Decision] For iteration 2: {next_model}")
            print(f"  Reasoning: {decision.reasoning}")
            
            # Track model switch if needed
            if next_model != "4B":
                switch_info = {
                    "iteration": 1,
                    "from": "4B",
                    "to": next_model,
                    "reason": "confidence-based after Phase 0",
                    "confidence_metrics": {
                        "mean": confidence_data['mean_confidence'],
                        "lgc": confidence_data['least_grouped_confidence'],
                        "tc": confidence_data['tail_confidence'],
                        "bwc": confidence_data.get('bottom_window_confidence', 0)
                    },
                    "combined_score": initial_combined
                }
                result["model_switches"].append(switch_info)
                with stats_lock:
                    processing_stats.model_switches_to_large += 1
                print(f"  🔄 Model switch: 4B → 32B")
            else:
                print(f"  ♻️ Continuing with 4B model")
            
            current_model = next_model
            
            # Update result with initial metrics
            result["combined_score"] = initial_combined
            result["pass_rate"] = initial_combined
            
            # Check if we should continue to phase 1
            if initial_combined >= 0.95:
                print(f"\n[Phase 0] ✅ Almost solved ({initial_combined:.2%})! Skipping refinement.")
                problem_status["final_score"] = initial_combined
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                result["model_used"] = current_model
                result["final_iteration"] = 1
                result["time_taken"] = time.time() - problem_start
                
                # Update global stats
                update_stats_thread_safe(result)
                print_progress()
                
                return result
            
            # Get the generated solution path
            full_code_path = os.path.join(task_dir, "phase0", "best", "best_program.py")
            if not os.path.exists(full_code_path):
                print(f"\n[Phase 0] ❌ Solution file not found")
                result["error"] = "Phase 0 solution file not found"
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                return result
            
            # ========================================
            # PHASE 1: Iterative refinement with adaptive model selection
            # ========================================
            base_cfg = load_config(CONFIG_LARGE if current_model == "32B" else CONFIG_SMALL)
            total_iterations = base_cfg.max_iterations or 10
            remaining_iterations = max(total_iterations - 1, 0)
            
            print(f"\n[Phase 1] Starting iterative refinement (up to {remaining_iterations} more iterations)")
            print(f"  Starting with {current_model} model based on Phase 0 confidence")
            
            current_path = full_code_path
            iteration_num = 1
            best_result = best0
            
            for iter_idx in range(remaining_iterations):
                iteration_num += 1
                
                print(f"\n[Iteration {iteration_num}] Using {current_model} model")
                
                # Track model usage
                result["model_usage_history"].append({"iteration": iteration_num, "model": current_model})
                
                # Select model config based on current_model
                if current_model == "32B":
                    iter_config = load_config(CONFIG_LARGE)
                    flops_cost = 1.0
                    with stats_lock:
                        processing_stats.total_32b_iterations += 1
                        processing_stats.total_flops += flops_cost
                else:
                    iter_config = load_config(CONFIG_SMALL)
                    flops_cost = 0.125
                    with stats_lock:
                        processing_stats.total_4b_iterations += 1
                        processing_stats.total_flops += flops_cost
                
                result["flops_used"] += flops_cost
                
                iter_config.max_iterations = 1
                iter_config.database.db_path = db_path
                iter_config.llm.temperature = 0.6
                iter_config.llm.top_p = 0.9
                
                # Adaptive exploration/exploitation based on performance
                if initial_public > 0.8 and initial_private < 0.5:
                    iter_config.database.exploration_ratio = 0.7
                    iter_config.database.exploitation_ratio = 0.3
                else:
                    iter_config.database.exploration_ratio = 0.5
                    iter_config.database.exploitation_ratio = 0.5
                
                # Run iteration with selected model
                evo_iter = OpenEvolve(
                    initial_program_path=current_path,
                    evaluation_file=EVAL_SCRIPT,
                    config=iter_config,
                    output_dir=os.path.join(task_dir, f"phase1_iter{iteration_num}"),
                    test_file_path=sample_path
                )
                
                best_iter = run_openevolve_sync(evo_iter, f"phase1_iter{iteration_num}", safe_id)
                
                if not best_iter:
                    print(f"[Iteration {iteration_num}] ❌ Failed")
                    break
                
                best_result = best_iter
                iter_metrics = best_iter.metrics
                iter_combined = iter_metrics.get("combined_score", 0.0)
                iter_public = iter_metrics.get("public_pass_rate", 0.0)
                iter_private = iter_metrics.get("private_pass_rate", 0.0)
                
                print(f"[Iteration {iteration_num}] Score: {iter_combined:.2%} (public: {iter_public:.2%}, private: {iter_private:.2%})")
                
                # Check if problem is solved
                if iter_combined >= 1.0 and not problem_status["solved"]:
                    problem_status["solved"] = True
                    problem_status["solved_iteration"] = iteration_num
                    result["solved"] = True
                    result["solved_iteration"] = iteration_num
                    result["problem_label"] = "solved_first_iteration" if iteration_num == 1 else "solved_later_iteration"
                    result["combined_score"] = iter_combined
                    result["pass_rate"] = iter_combined
                    problem_status["final_score"] = iter_combined
                    
                    # Extract confidence for solved problem before exiting
                    iter_confidence = extract_confidence_from_evolution_result(best_iter, current_path)
                    iter_confidence["iteration"] = iteration_num
                    iter_confidence["solved"] = True
                    iter_confidence["combined_score"] = iter_combined
                    problem_status["confidence_measurements"].append(iter_confidence)
                    
                    print(f"[Iteration {iteration_num}] ✅ Problem SOLVED!")
                    
                    # Early exit - store final state and break
                    result["model_used"] = current_model
                    result["final_iteration"] = iteration_num
                    result["confidence_measurements"] = problem_status["confidence_measurements"]
                    result["time_taken"] = time.time() - problem_start
                    
                    # Update global stats
                    update_stats_thread_safe(result)
                    print_progress()
                    
                    # Exit the iteration loop - problem is solved
                    return result
                
                # Update current path for next iteration
                current_path = os.path.join(task_dir, f"phase1_iter{iteration_num}", "best", "best_program.py")
                if not os.path.exists(current_path):
                    print(f"[Iteration {iteration_num}] ❌ No solution file generated")
                    break
                
                # Extract confidence and make model decision for NEXT iteration
                iter_confidence = extract_confidence_from_evolution_result(best_iter, current_path)
                iter_confidence["iteration"] = iteration_num
                iter_confidence["solved"] = problem_status["solved"]
                iter_confidence["combined_score"] = iter_combined
                problem_status["confidence_measurements"].append(iter_confidence)
                
                print(f"[Iteration {iteration_num}] Confidence: MC={iter_confidence['mean_confidence']:.2f}, "
                      f"LGC={iter_confidence['least_grouped_confidence']:.2f}")
                
                # Make model selection decision for NEXT iteration (if not last iteration)
                if iter_idx < remaining_iterations - 1:
                    iter_metrics_obj = ConfidenceMetrics(
                        least_grouped_confidence=iter_confidence['least_grouped_confidence'],
                        mean_confidence=iter_confidence['mean_confidence'],
                        tail_confidence=iter_confidence['tail_confidence'],
                        bottom_window_confidence=iter_confidence.get('bottom_window_confidence', 0)
                    )
                    
                    iter_decision = model_selector.classify(iter_metrics_obj)
                    previous_model = current_model
                    next_model = iter_decision.model_choice.value
                    
                    decision_record = {
                        "iteration": iteration_num,
                        "previous_model": previous_model,
                        "decision": next_model,
                        "confidence_score": iter_decision.confidence_score,
                        "reasoning": iter_decision.reasoning,
                        "metrics": iter_decision.metrics_used,
                        "combined_score": iter_combined
                    }
                    result["model_decisions"].append(decision_record)
                    
                    print(f"\n[Model Decision] For iteration {iteration_num + 1}: {next_model}")
                    
                    # Track model switches
                    if next_model != previous_model:
                        switch_info = {
                            "iteration": iteration_num,
                            "from": previous_model,
                            "to": next_model,
                            "reason": "confidence-based",
                            "confidence_metrics": {
                                "mean": iter_confidence['mean_confidence'],
                                "lgc": iter_confidence['least_grouped_confidence'],
                                "tc": iter_confidence['tail_confidence'],
                                "bwc": iter_confidence.get('bottom_window_confidence', 0)
                            },
                            "combined_score": iter_combined
                        }
                        result["model_switches"].append(switch_info)
                        
                        # Update stats
                        with stats_lock:
                            if next_model == "32B":
                                processing_stats.model_switches_to_large += 1
                                print(f"  🔄 Model switch: {previous_model} → 32B")
                            else:
                                processing_stats.model_switches_to_small += 1
                                print(f"  🔄 Model switch: {previous_model} → 4B")
                    else:
                        print(f"  ♻️ Continuing with {current_model} model")
                    
                    # Update current model for next iteration
                    current_model = next_model
                
                # Update result with current best score
                result["combined_score"] = iter_combined
                result["pass_rate"] = iter_combined
                problem_status["final_score"] = iter_combined
            
            # Store final model used
            result["model_used"] = current_model
            result["final_iteration"] = iteration_num
            
            print(f"\n[Final] Problem completed:")
            print(f"  Status: {result['problem_label']}")
            print(f"  Final score: {result['combined_score']:.2%}")
            print(f"  Iterations: {iteration_num}")
            print(f"  FLOPS used: {result['flops_used']:.2f} units")
            
        except Exception as e:
            print(f"\n[Error] {safe_id}: {str(e)}")
            result["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        finally:
            result["time_taken"] = time.time() - problem_start
            result["confidence_measurements"] = problem_status["confidence_measurements"]
            
            # Update global stats
            update_stats_thread_safe(result)
            print_progress()
        
        return result

def process_problems_parallel(problems: List[Dict], original_indices: List[int]) -> List[Dict]:
    """Process all problems using ThreadPoolExecutor with confidence measurement"""
    processing_stats.total_problems = len(problems)
    results = []
    
    print(f"\n{'='*80}")
    print(f"STARTING PARALLEL PROCESSING")
    print(f"{'='*80}")
    print(f"Max concurrent problems: {MAX_CONCURRENT_PROBLEMS}")
    print(f"Max concurrent LLM calls: {MAX_CONCURRENT_LLM_CALLS}")
    print(f"Total problems: {len(problems)}")
    print(f"{'='*80}\n")
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROBLEMS) as executor:
        # Submit all tasks
        future_to_problem = {}
        for i, (sample, orig_idx) in enumerate(zip(problems, original_indices)):
            future = executor.submit(solve_one_problem_with_confidence, sample, i, len(problems), orig_idx)
            future_to_problem[future] = (i, sample.get("question_id", "unknown"))
        
        # Process results as they complete
        for future in as_completed(future_to_problem):
            idx, question_id = future_to_problem[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n✓ Completed {idx+1}/{len(problems)}: {question_id}")
                print(f"  Score: {result['combined_score']:.2%}")
                print(f"  Label: {result['problem_label']}")
                print(f"  Model switches: {len(result.get('model_switches', []))}")
            except Exception as e:
                print(f"\n✗ Failed {idx+1}/{len(problems)}: {question_id} - Error: {e}")
                # Create a failed result
                results.append({
                    "question_id": question_id,
                    "question_title": "N/A",
                    "original_index": idx,
                    "combined_score": 0.0,
                    "pass_rate": 0.0,
                    "difficulty": "N/A",
                    "source": "N/A",
                    "time_taken": 0.0,
                    "solved": False,
                    "solved_iteration": None,
                    "problem_label": "unsolved",
                    "confidence_measurements": [],
                    "model_decisions": [],
                    "model_switches": [],
                    "model_usage_history": [],
                    "error": str(e)
                })
    
    return results

def stratified_sample_problems(num_problems: int = 20, random_state: int = 42) -> tuple[List[Dict], List[int]]:
    """Sample problems using stratified sampling"""
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", 
                          version_tag="release_v5", trust_remote_code=True)
    df = dataset.to_pandas()
    df['original_index'] = df.index
    
    difficulty_distribution = df['difficulty'].value_counts(normalize=True)
    sample_counts = (difficulty_distribution * num_problems).round().astype(int)
    diff = num_problems - sample_counts.sum()
    if diff != 0:
        sample_counts[sample_counts.idxmax()] += diff
    
    sampled_df = df.groupby('difficulty', group_keys=False).apply(
        lambda x: x.sample(
            n=int(sample_counts[x.name]) if x.name in sample_counts else 0, 
            random_state=random_state
        )
    )
    
    sampled_df = sampled_df.sort_values('original_index')
    return sampled_df.drop('original_index', axis=1).to_dict('records'), sampled_df['original_index'].tolist()

def main():
    """Main entry point with ThreadPoolExecutor for parallel processing and confidence measurement"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Adaptive Multi-LLM System with Per-Iteration Model Switching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_batch_conf_adaptive.py --num_problems 10
  python3 run_batch_conf_adaptive.py -n 50 --random_state 123
  python3 run_batch_conf_adaptive.py --help
        """
    )
    parser.add_argument(
        '-n', '--num_problems',
        type=int,
        default=20,
        help='Number of problems to sample from dataset (default: 20)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )
    parser.add_argument(
        '--max_concurrent_problems',
        type=int,
        default=MAX_CONCURRENT_PROBLEMS,
        help=f'Maximum concurrent problems to process (default: {MAX_CONCURRENT_PROBLEMS})'
    )
    parser.add_argument(
        '--max_concurrent_llm_calls',
        type=int,
        default=MAX_CONCURRENT_LLM_CALLS,
        help=f'Maximum concurrent LLM API calls (default: {MAX_CONCURRENT_LLM_CALLS})'
    )
    
    args = parser.parse_args()
    
    # # Update global settings if provided
    # global MAX_CONCURRENT_PROBLEMS, MAX_CONCURRENT_LLM_CALLS
    # MAX_CONCURRENT_PROBLEMS = args.max_concurrent_problems
    # MAX_CONCURRENT_LLM_CALLS = args.max_concurrent_llm_calls
    
    # # Recreate semaphores with new limits
    # global llm_semaphore, problem_semaphore
    # llm_semaphore = threading.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    # problem_semaphore = threading.Semaphore(MAX_CONCURRENT_PROBLEMS)
    
    global TARGET_QUESTION_IDS
    TARGET_QUESTION_IDS = globals().get("TARGET_QUESTION_IDS", [])
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"ADAPTIVE MULTI-LLM SYSTEM CONFIGURATION")
    print(f"{'='*80}")
    print(f"Number of problems: {args.num_problems}")
    print(f"Random state: {args.random_state}")
    print(f"Max concurrent problems: {MAX_CONCURRENT_PROBLEMS}")
    print(f"Max concurrent LLM calls: {MAX_CONCURRENT_LLM_CALLS}")
    print(f"{'='*80}\n")
    
    # Check requirements
    if not os.path.exists(EVAL_SCRIPT):
        print(f"Error: Evaluator script not found at {EVAL_SCRIPT}")
        return
    
    if not os.path.exists(CONFIG_SMALL):
        print(f"Error: Small model config not found at {CONFIG_SMALL}")
        return
    
    if not os.path.exists(CONFIG_LARGE):
        print(f"Error: Large model config not found at {CONFIG_LARGE}")
        return
    
    # Load dataset
    try:
        if TARGET_QUESTION_IDS:
            problems, original_indices = filter_target_problems()
        else:
            problems, original_indices = stratified_sample_problems(args.num_problems, 42)
        
        print(f"\n✓ Loaded {len(problems)} problems")
        
        # Save the sampling information
        sampling_info = {
            "num_problems": len(problems),
            "original_indices": original_indices,
            "sampling_timestamp": time.time(),
            "max_concurrent_problems": MAX_CONCURRENT_PROBLEMS,
            "max_concurrent_llm_calls": MAX_CONCURRENT_LLM_CALLS,
            "model_selector_config": {
                "lgc_threshold": model_selector.lgc_threshold,
                "mc_threshold": model_selector.mc_threshold,
                "tc_threshold": model_selector.tc_threshold,
                "bwc_threshold": model_selector.bwc_threshold,
                "use_ensemble": model_selector.use_ensemble
            }
        }
        sampling_file = os.path.join(OUTPUT_ROOT, "sampling_info.json")
        with open(sampling_file, 'w') as f:
            json.dump(sampling_info, f, indent=2)
        print(f"Sampling information saved to {sampling_file}\n")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Process problems in parallel with confidence measurement
    start_time = time.time()
    results = process_problems_parallel(problems, original_indices)
    total_time = time.time() - start_time

    # Calculate summary statistics
    solved_first = sum(1 for r in results if r["problem_label"] == "solved_first_iteration")
    solved_later = sum(1 for r in results if r["problem_label"] == "solved_later_iteration")
    unsolved = sum(1 for r in results if r["problem_label"] == "unsolved")
    avg_score = sum(r["combined_score"] for r in results) / len(results) if results else 0
    total_flops = sum(r.get("flops_used", 0) for r in results)
    # FLOPS per problem should be divided by the total number of problems attempted (args.num_problems)
    avg_flops = total_flops / args.num_problems if args.num_problems > 0 else 0
    
    # Model switching statistics
    total_switches = sum(len(r.get("model_switches", [])) for r in results)
    switches_to_32b = sum(len([s for s in r.get("model_switches", []) if s["to"] == "32B"]) for r in results)
    switches_to_4b = sum(len([s for s in r.get("model_switches", []) if s["to"] == "4B"]) for r in results)
    problems_with_switches = len([r for r in results if r.get("model_switches", [])])
    avg_switches_per_problem = total_switches / len(results) if results else 0
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total problems: {len(results)}")
    print(f"Solved first iteration: {solved_first} ({solved_first/len(results)*100:.1f}%)")
    print(f"Solved later iterations: {solved_later} ({solved_later/len(results)*100:.1f}%)")
    print(f"Unsolved: {unsolved} ({unsolved/len(results)*100:.1f}%)")
    print(f"Average score: {avg_score:.3f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nMODEL SWITCHING STATISTICS:")
    print(f"Total switches: {total_switches}")
    print(f"Switches to 32B: {switches_to_32b}")
    print(f"Switches to 4B: {switches_to_4b}")
    print(f"Problems with switches: {problems_with_switches}/{len(results)} ({problems_with_switches/len(results)*100:.1f}%)")
    print(f"Avg switches per problem: {avg_switches_per_problem:.2f}")
    print(f"\nMODEL USAGE STATISTICS:")
    print(f"Total 4B iterations: {processing_stats.total_4b_iterations}")
    print(f"Total 32B iterations: {processing_stats.total_32b_iterations}")
    total_iters = processing_stats.total_4b_iterations + processing_stats.total_32b_iterations
    if total_iters > 0:
        print(f"4B usage: {processing_stats.total_4b_iterations/total_iters*100:.1f}%")
        print(f"32B usage: {processing_stats.total_32b_iterations/total_iters*100:.1f}%")
    print(f"\nFLOPS STATISTICS:")
    print(f"Total FLOPS: {total_flops:.2f} units (32B=1.0, 4B=0.125)")
    print(f"Average FLOPS per problem: {avg_flops:.2f} units")
    print(f"{'='*80}")
    
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
                    "bottom_window_confidence": conf_measurement.get("bottom_window_confidence", 0)
                })

    # Confidence analysis summary
    if confidence_analysis:
        print(f"\nCONFIDENCE ANALYSIS:")
        
        # Analyze confidence for unsolved problems after 1st iteration
        unsolved_after_1st = [c for c in confidence_analysis if c["iteration"] == 1 and not c["solved_at_measurement"]]
        if unsolved_after_1st:
            avg_conf_unsolved = np.mean([c["mean_confidence"] for c in unsolved_after_1st])
            avg_lgc_unsolved = np.mean([c["least_grouped_confidence"] for c in unsolved_after_1st])
            avg_tc_unsolved = np.mean([c["tail_confidence"] for c in unsolved_after_1st])
            avg_bwc_unsolved = np.mean([c["bottom_window_confidence"] for c in unsolved_after_1st])
            
            print(f"Unsolved after 1st iteration ({len(unsolved_after_1st)} problems):")
            print(f"  Average mean confidence: {avg_conf_unsolved:.3f}")
            print(f"  Average LGC: {avg_lgc_unsolved:.3f}")
            print(f"  Average TC: {avg_tc_unsolved:.3f}")
            print(f"  Average BWC: {avg_bwc_unsolved:.3f}")
        
        # Analyze confidence for problems solved in later iterations
        solved_later_confs = [c for c in confidence_analysis if c["solved_at_measurement"]]
        if solved_later_confs:
            avg_conf_solved = np.mean([c["mean_confidence"] for c in solved_later_confs])
            avg_lgc_solved = np.mean([c["least_grouped_confidence"] for c in solved_later_confs])
            avg_tc_solved = np.mean([c["tail_confidence"] for c in solved_later_confs])
            avg_bwc_solved = np.mean([c["bottom_window_confidence"] for c in solved_later_confs])
            
            print(f"Solved in later iterations ({len(solved_later_confs)} measurements):")
            print(f"  Average mean confidence: {avg_conf_solved:.3f}")
            print(f"  Average LGC: {avg_lgc_solved:.3f}")
            print(f"  Average TC: {avg_tc_solved:.3f}")
            print(f"  Average BWC: {avg_bwc_solved:.3f}")
    
    # Results by difficulty
    difficulties = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in difficulties:
            difficulties[diff] = {
                "total": 0,
                "solved_first": 0,
                "solved_later": 0,
                "unsolved": 0,
                "total_score": 0,
                "total_switches": 0,
                "total_flops": 0.0
            }
        difficulties[diff]["total"] += 1
        difficulties[diff]["total_score"] += r["combined_score"]
        difficulties[diff]["total_switches"] += len(r.get("model_switches", []))
        difficulties[diff]["total_flops"] += r.get("flops_used", 0)
        if r["problem_label"] == "solved_first_iteration":
            difficulties[diff]["solved_first"] += 1
        elif r["problem_label"] == "solved_later_iteration":
            difficulties[diff]["solved_later"] += 1
        else:
            difficulties[diff]["unsolved"] += 1
    
    print(f"\nRESULTS BY DIFFICULTY:")
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulties:
            stats = difficulties[diff]
            total_solved = stats['solved_first'] + stats['solved_later']
            avg_diff_score = stats["total_score"] / stats["total"]
            avg_switches = stats["total_switches"] / stats["total"]
            avg_diff_flops = stats["total_flops"] / stats["total"]
            print(f"{diff.capitalize()}: {total_solved}/{stats['total']} solved "
                  f"({total_solved/stats['total']*100:.1f}%), "
                  f"avg score: {avg_diff_score:.3f}, "
                  f"avg switches: {avg_switches:.2f}, "
                  f"avg FLOPS: {avg_diff_flops:.2f}")
            print(f"  - 1st iter: {stats['solved_first']}, Later: {stats['solved_later']}, "
                  f"Unsolved: {stats['unsolved']}")
    
    # Save detailed results with confidence data
    results_file = os.path.join(OUTPUT_ROOT, "results_with_confidence.json")
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total": len(results),
                "solved_first_iteration": solved_first,
                "solved_later_iterations": solved_later,
                "unsolved": unsolved,
                "average_score": avg_score,
                "total_time_minutes": total_time / 60,
                "total_flops": total_flops,
                "average_flops_per_problem": avg_flops,
                "num_problems_argument": args.num_problems,
                "timestamp": time.time(),
                "max_concurrent_problems": MAX_CONCURRENT_PROBLEMS,
                "max_concurrent_llm_calls": MAX_CONCURRENT_LLM_CALLS
            },
            "model_switching_stats": {
                "total_switches": total_switches,
                "switches_to_32B": switches_to_32b,
                "switches_to_4B": switches_to_4b,
                "problems_with_switches": problems_with_switches,
                "avg_switches_per_problem": avg_switches_per_problem,
                "total_4b_iterations": processing_stats.total_4b_iterations,
                "total_32b_iterations": processing_stats.total_32b_iterations,
                "4b_usage_rate": processing_stats.total_4b_iterations / total_iters if total_iters > 0 else 0,
                "32b_usage_rate": processing_stats.total_32b_iterations / total_iters if total_iters > 0 else 0
            },
            "sampling_info": sampling_info,
            "by_difficulty": {
                diff: {
                    "total": stats["total"],
                    "solved_first": stats["solved_first"],
                    "solved_later": stats["solved_later"],
                    "unsolved": stats["unsolved"],
                    "average_score": stats["total_score"] / stats["total"],
                    "avg_switches": stats["total_switches"] / stats["total"],
                    "avg_flops": stats["total_flops"] / stats["total"]
                }
                for diff, stats in difficulties.items()
            },
            "results": results,
            "confidence_analysis": confidence_analysis
        }, f, indent=2)
    print(f"\nResults with confidence analysis saved to {results_file}")
    
    # Save confidence-specific analysis
    confidence_file = os.path.join(OUTPUT_ROOT, "confidence_correlation_analysis.json")
    with open(confidence_file, 'w') as f:
        json.dump({
            "confidence_measurements": confidence_analysis,
            "analysis_summary": {
                "total_confidence_measurements": len(confidence_analysis),
                "measurements_after_first_iteration": len([c for c in confidence_analysis if c["iteration"] == 1]),
                "measurements_for_solved_problems": len([c for c in confidence_analysis if c["solved_at_measurement"]]),
                "measurements_for_unsolved_problems": len([c for c in confidence_analysis if not c["solved_at_measurement"]])
            }
        }, f, indent=2)
    print(f"Confidence correlation analysis saved to {confidence_file}")
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

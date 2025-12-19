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
import signal

ROOT        = os.path.dirname(__file__)
CONFIG_YAML = os.path.join(ROOT, "config_model_sampling.yaml")
EVAL_SCRIPT = os.path.join(ROOT, "evaluator.py")
OUTPUT_ROOT = os.path.join("/", "lcb_43_57_400_880")

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

MAX_CONCURRENT_PROBLEMS = 32
MAX_CONCURRENT_LLM_CALLS = 64

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

def extract_confidence_from_evolution_result(best_result, program_path: str = None) -> Dict:
    """Extract both logprob-based and verbalized confidence from OpenEvolve Program metadata"""
    try:
        confidence_data = {"mean_confidence": 0, "least_grouped_confidence": 0, "tail_confidence": 0, "verbalized_conf": None}
        
        # ✅ Extract logprob-based confidence from metadata
        if hasattr(best_result, "metadata") and isinstance(best_result.metadata, dict):
            if "confidence" in best_result.metadata and best_result.metadata["confidence"]:
                confidence_data.update(best_result.metadata["confidence"])

        # fallback for older results
        elif hasattr(best_result, "confidence") and best_result.confidence:
            confidence_data.update(best_result.confidence)

        # ✅ Extract logprob-based confidence from metadata
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

def load_excluded_indices(sampling_file: str) -> List[int]:
    """Load previously sampled indices from sampling_info.json"""
    if not os.path.exists(sampling_file):
        print(f"No sampling file found at {sampling_file}, proceeding with full dataset")
        return []
    
    try:
        with open(sampling_file, 'r') as f:
            sampling_info = json.load(f)
            excluded = sampling_info.get("original_indices", [])
            print(f"Loaded {len(excluded)} previously sampled indices to exclude")
            return excluded
    except Exception as e:
        print(f"Error loading sampling file: {e}")
        return []

def get_problems_by_index_range(start_idx: int, end_idx: int, excluded_indices: List[int] = None) -> tuple[List[Dict], List[int]]:
    """Get problems from index range, excluding specified indices"""
    print(f"Loading dataset to get problems from index {start_idx} to {end_idx}...")
    
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5", trust_remote_code=True)
    df = dataset.to_pandas()
    df['original_index'] = df.index
    
    # Filter to the specified index range
    df = df[(df['original_index'] >= start_idx) & (df['original_index'] < end_idx)]
    print(f"Problems in range [{start_idx}, {end_idx}): {len(df)}")
    
    # Exclude previously sampled problems
    if excluded_indices:
        excluded_in_range = [idx for idx in excluded_indices if start_idx <= idx < end_idx]
        print(f"Excluding {len(excluded_in_range)} previously sampled problems from this range")
        df = df[~df['original_index'].isin(excluded_indices)]
        print(f"Remaining problems after exclusion: {len(df)}")
    
    # Sort by original index to maintain order
    df = df.sort_values('original_index').reset_index(drop=True)
    
    # Show difficulty distribution of selected problems
    difficulty_distribution = df['difficulty'].value_counts()
    print(f"\nSelected {len(df)} problems with difficulty distribution:")
    for diff, count in difficulty_distribution.items():
        print(f"  {diff}: {count} problems ({count/len(df):.1%})")
    
    original_indices = df['original_index'].tolist()
    problems = df.drop('original_index', axis=1).to_dict('records')
    
    print(f"\nSuccessfully selected {len(problems)} problems.")
    return problems, original_indices

def stratified_sample_problems(num_problems: int = None, random_state: int = 150, excluded_indices: List[int] = None) -> tuple[List[Dict], List[int]]:
    """Sample problems - now just takes first N problems after exclusion"""
    print(f"Loading dataset to get first {num_problems} problems after exclusions...")
    
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5", trust_remote_code=True)
    df = dataset.to_pandas()
    df['original_index'] = df.index
    
    # Exclude previously sampled problems
    if excluded_indices:
        print(f"Excluding {len(excluded_indices)} previously sampled problems")
        df = df[~df['original_index'].isin(excluded_indices)]
        print(f"Remaining problems after exclusion: {len(df)}")
    
    # Sort by original index to maintain order
    df = df.sort_values('original_index').reset_index(drop=True)
    
    # If num_problems is None or >= available problems, use all remaining problems
    if num_problems is None or num_problems >= len(df):
        print(f"\nUsing all {len(df)} remaining problems")
        sampled_df = df
    else:
        print(f"\nTaking first {num_problems} problems from remaining dataset")
        sampled_df = df.head(num_problems)
    
    # Show difficulty distribution of selected problems
    difficulty_distribution = sampled_df['difficulty'].value_counts()
    print(f"\nSelected {len(sampled_df)} problems with difficulty distribution:")
    for diff, count in difficulty_distribution.items():
        print(f"  {diff}: {count} problems ({count/len(sampled_df):.1%})")
    
    original_indices = sampled_df['original_index'].tolist()
    problems = sampled_df.drop('original_index', axis=1).to_dict('records')
    
    print(f"\nSuccessfully selected {len(problems)} problems.")
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
        finally:
            loop.close()
    except Exception as e:
        print(f"Error in {phase_name} for {problem_id}: {str(e)}")
        raise
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
            
            # Phase 0: Initial solution generation
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Phase 0 - Generating initial solution...")
            cfg0 = load_config(CONFIG_YAML)
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
            
            if not best0:
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ❌ Phase 0 failed")
                result["error"] = "Phase 0 failed"
                return result
            
            # Get metrics
            initial_metrics = best0.metrics
            initial_combined = initial_metrics.get("combined_score", 0.0)
            initial_public = initial_metrics.get("public_pass_rate", 0.0)
            initial_private = initial_metrics.get("private_pass_rate", 0.0)
            
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Initial scores - Combined: {initial_combined:.2%}")
            
            # Check if problem is solved after iteration 1
            if initial_combined >= 1.0:  # Completely solved
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
            
            # After Phase 0 (around line where you extract confidence):
            phase0_output_dir = os.path.join(task_dir, "phase0")
            best_program_path = os.path.join(phase0_output_dir, "best", "best_program.py")

            # Try to get confidence from the OpenEvolve result first
            confidence_data = extract_confidence_from_evolution_result(best0, best_program_path)

            # Add metadata to confidence measurement
            confidence_data["iteration"] = 1
            confidence_data["solved"] = False
            confidence_data["combined_score"] = initial_combined
            problem_status["confidence_measurements"].append(confidence_data)

            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Confidence after iter 1 (unsolved) - "
                f"Mean: {confidence_data['mean_confidence']:.3f}, "
                f"Verbalized: {confidence_data.get('verbalized_conf', 'N/A')}")
            
            # Update result with initial metrics
            result["combined_score"] = initial_combined
            result["pass_rate"] = initial_combined
            
            # Check if we should continue to phase 1
            if initial_combined >= 0.95:
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Almost solved! Skipping refinement.")
                problem_status["final_score"] = 1.0
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                return result
            
            # Get the generated solution path
            full_code_path = os.path.join(task_dir, "phase0", "best", "best_program.py")
            if not os.path.exists(full_code_path):
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ❌ Phase 0 solution file not found")
                result["error"] = "Phase 0 solution file not found"
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                return result
            
            # Phase 1: Refinement with different strategy
            base_cfg = load_config(CONFIG_YAML)
            remaining_iterations = max((base_cfg.max_iterations or 10) - 1, 0)
            base_cfg.max_iterations = remaining_iterations
            
            # Use same database
            base_cfg.database.db_path = db_path
            
            # Adjust for exploration
            base_cfg.llm.temperature = 0.6
            base_cfg.llm.top_p = 0.9
            
            if initial_public > 0.8 and initial_private < 0.5:
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Focusing on generalization...")
                base_cfg.database.exploration_ratio = 0.7
                base_cfg.database.exploitation_ratio = 0.3
            else:
                base_cfg.database.exploration_ratio = 0.5
                base_cfg.database.exploitation_ratio = 0.5
            
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Phase 1 - Refining ({remaining_iterations} iterations)...")
            
            evo1 = OpenEvolve(
                initial_program_path=full_code_path,
                evaluation_file=EVAL_SCRIPT,
                config=base_cfg,
                output_dir=os.path.join(task_dir, "phase1"),
                test_file_path=sample_path
            )
            
            best1 = run_openevolve_sync(evo1, "phase1", safe_id)
            
            if best1:
                final_metrics = best1.metrics
                final_combined = final_metrics.get("combined_score", 0.0)
                final_public = final_metrics.get("public_pass_rate", 0.0)
                final_private = final_metrics.get("private_pass_rate", 0.0)
                
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Final scores - Combined: {final_combined:.2%}")
                
                # Check if problem got solved in later iterations
                if final_combined >= 1.0 and not problem_status["solved"]:
                    problem_status["solved"] = True
                    problem_status["solved_iteration"] = "phase1"
                    result["solved"] = True
                    result["solved_iteration"] = "phase1"
                    result["problem_label"] = "solved_later_iteration"
                    
                    print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Problem SOLVED in later iterations!")
                    
                    # Calculate confidence for the solved solution
                    phase1_best_path = os.path.join(task_dir, "phase1", "best", "best_program.py")
    
                    # Try to get confidence from the OpenEvolve result first
                    solved_confidence_data = extract_confidence_from_evolution_result(best1, phase1_best_path)
                    
                    # Add metadata to confidence measurement
                    solved_confidence_data["iteration"] = "phase1_final"
                    solved_confidence_data["solved"] = True
                    solved_confidence_data["combined_score"] = final_combined
                    problem_status["confidence_measurements"].append(solved_confidence_data)
                    
                    print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Confidence for solved solution - "
                        f"Mean: {solved_confidence_data['mean_confidence']:.3f}, "
                        f"Verbalized: {solved_confidence_data.get('verbalized_conf', 'N/A')}")
                
                # Update result with final metrics
                result["combined_score"] = final_combined
                result["pass_rate"] = final_combined
                problem_status["final_score"] = final_combined
                
                # Save final solution with metadata
                best_path = os.path.join(task_dir, "phase1", "best", "best_program.py")
                if os.path.exists(best_path):
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

"""
                    with open(final_path, 'w') as dst:
                        dst.write(metadata_comment + content)
                        dst.flush()
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

            # If problem ended unsolved, still record final confidence
            if not result["solved"]:
                final_output_dir = os.path.join(task_dir, "phase1")
                
                # Try to extract confidence from best1 (if available)
                if 'best1' in locals() and best1:
                    final_confidence = extract_confidence_from_evolution_result(best1)
                    
                    final_confidence["iteration"] = "phase1_final_unsolved"
                    final_confidence["solved"] = False
                    final_confidence["combined_score"] = result["combined_score"]
                    problem_status["confidence_measurements"].append(final_confidence)
                else:
                    # Explicitly mark no code was generated
                    problem_status["confidence_measurements"].append({
                        "iteration": "phase1_final_unsolved",
                        "solved": False,
                        "combined_score": result["combined_score"],
                        "mean_confidence": 0,
                        "least_grouped_confidence": 0,
                        "tail_confidence": 0,
                        "note": "No code generated"
                    })

            # Determine final problem label if not already set
            if result["problem_label"] == "unsolved" and result["combined_score"] > 0:
                result["problem_label"] = "partial"
            
            # Update global stats
            update_stats_thread_safe(result)
            print_progress()
        
        return result
    finally:
        if acquired:
            problem_semaphore.release()

def process_problems_parallel(problems: List[Dict], original_indices: List[int]) -> List[Dict]:
    """Process all problems using ThreadPoolExecutor with confidence measurement"""
    processing_stats.total_problems = len(problems)
    results = []
    
    print(f"\nStarting parallel processing with ThreadPoolExecutor + Confidence Measurement")
    print(f"  Max concurrent problems: {MAX_CONCURRENT_PROBLEMS}")
    print(f"  Max concurrent LLM calls: {MAX_CONCURRENT_LLM_CALLS}")
    print(f"  Total problems: {len(problems)}")
    print("="*80)
    
    executor = None
    try:
        executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROBLEMS)
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
                print(f"\n✓ Completed {idx+1}/{len(problems)}: {question_id} "
                      f"(score: {result['combined_score']:.2%}, label: {result['problem_label']})")
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
                    "error": str(e)
                })
    
    finally:
        if executor:
            print("\n🔄 Shutting down thread pool executor...")
            executor.shutdown(wait=True, cancel_futures=False)
            print("✅ Thread pool executor shut down cleanly")
    
    return results

def main():
    """Main entry point with ThreadPoolExecutor for parallel processing and confidence measurement"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Check requirements
    if not os.path.exists(EVAL_SCRIPT):
        print(f"Error: Evaluator script not found at {EVAL_SCRIPT}")
        return
    
    if not os.path.exists(CONFIG_YAML):
        print(f"Error: Config file not found at {CONFIG_YAML}")
        return
    
    # Load previously sampled indices to exclude
    previous_sampling_file = os.path.join( 
        "/app/examples/livecodebench/livecodebench_runs_conf_qwen4B_it_50problems", 
        "sampling_info.json"
    )
    excluded_indices = load_excluded_indices(previous_sampling_file)
    BATCH_START = 400
    BATCH_END = 880
    # Load dataset - get problems from specified index range, excluding previous samples
    try:
        print(f"\n{'='*60}")
        print(f"RUNNING BATCH: indices {BATCH_START} to {BATCH_END}")
        print(f"{'='*60}\n")
        
        problems, original_indices = get_problems_by_index_range(
            start_idx=BATCH_START,
            end_idx=BATCH_END,
            excluded_indices=excluded_indices
        )
        print(f"Loaded {len(problems)} problems from LiveCodeBench (range {BATCH_START}-{BATCH_END}, excluding {len(excluded_indices)} previously sampled)")
        
        # Save the sampling information
        sampling_info = {
            "num_problems": len(problems),
            "sampling_method": "index_range_with_exclusion",
            "batch_start": BATCH_START,
            "batch_end": BATCH_END,
            "original_indices": original_indices,
            "excluded_indices": excluded_indices,
            "excluded_from_file": previous_sampling_file,
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
                    "verbalized_conf": conf_measurement.get("verbalized_conf")  # ✅ New field
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
            
            # ✅ Add verbalized confidence analysis
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
    
    # # Save detailed results with confidence data
    # results_file = os.path.join(OUTPUT_ROOT, "results_with_confidence.json")
    # with open(results_file, 'w') as f:
    #     json.dump({
    #         "summary": {
    #             "total": len(results),
    #             "solved_first_iteration": solved_first,
    #             "solved_later_iterations": solved_later,
    #             "partial": partial,
    #             "unsolved": unsolved,
    #             "errors": errors,
    #             "average_score": avg_score,
    #             "total_time_minutes": total_time / 60,
    #             "timestamp": time.time(),
    #             "max_concurrent_problems": MAX_CONCURRENT_PROBLEMS,
    #             "max_concurrent_llm_calls": MAX_CONCURRENT_LLM_CALLS
    #         },
    #         "sampling_info": sampling_info,
    #         "by_difficulty": {
    #             diff: {
    #                 "total": stats["total"],
    #                 "solved_first": stats["solved_first"],
    #                 "solved_later": stats["solved_later"],
    #                 "partial": stats["partial"],
    #                 "unsolved": stats["unsolved"],
    #                 "average_score": stats["total_score"] / stats["total"]
    #             }
    #             for diff, stats in difficulties.items()
    #         },
    #         "results": results,
    #         "confidence_analysis": confidence_analysis
    #     }, f, indent=2)
    # print(f"\nResults with confidence analysis saved to {results_file}")
    
    # Save confidence-specific analysis
    # confidence_file = os.path.join(OUTPUT_ROOT, "confidence_correlation_analysis.json")
    # with open(confidence_file, 'w') as f:
    #     json.dump({
    #         "confidence_measurements": confidence_analysis,
    #         "analysis_summary": {
    #             "total_confidence_measurements": len(confidence_analysis),
    #             "measurements_after_first_iteration": len([c for c in confidence_analysis if c["iteration"] == 1]),
    #             "measurements_for_solved_problems": len([c for c in confidence_analysis if c["solved_at_measurement"]]),
    #             "measurements_for_unsolved_problems": len([c for c in confidence_analysis if not c["solved_at_measurement"]])
    #         }
    #     }, f, indent=2)
    # print(f"Confidence correlation analysis saved to {confidence_file}")

if __name__ == "__main__":
    main()

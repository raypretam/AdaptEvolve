import asyncio
import json
import os
import re
import numpy as np
from typing import Dict, List
from datasets import load_dataset
from openevolve import OpenEvolve
from openevolve.config import load_config
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
import threading
import atexit

ROOT = os.path.dirname(__file__)
CONFIG_YAML = os.path.join(ROOT, "config_mbpp_32B.yaml")
EVAL_SCRIPT = os.path.join(ROOT, "evaluator.py")
OUTPUT_ROOT = os.path.join(ROOT, "mbpp_runs_32B_random_50")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, set):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


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
    time.sleep(2)


atexit.register(cleanup_resources)


def extract_confidence_from_evolution_result(best_result, program_path: str = None) -> Dict:
    """Extract both logprob-based and verbalized confidence from OpenEvolve Program metadata"""
    try:
        confidence_data = {
            "mean_confidence": 0,
            "least_grouped_confidence": 0,
            "tail_confidence": 0,
            "verbalized_conf": None
        }
        
        if hasattr(best_result, "metadata") and isinstance(best_result.metadata, dict):
            if "confidence" in best_result.metadata and best_result.metadata["confidence"]:
                confidence_data.update(best_result.metadata["confidence"])
            if "verbalized_conf" in best_result.metadata and best_result.metadata["verbalized_conf"]:
                confidence_data.update(best_result.metadata["verbalized_conf"])

        elif hasattr(best_result, "confidence") and best_result.confidence:
            confidence_data.update(best_result.confidence)

        return confidence_data
    except Exception as e:
        print(f"Error extracting confidence: {e}")
        return {
            "mean_confidence": 0,
            "least_grouped_confidence": 0,
            "tail_confidence": 0,
            "verbalized_conf": None
        }


def format_test_examples(test_list: List[str], max_examples: int = 3) -> str:
    """Format test assertions as examples"""
    examples = []
    for i, test in enumerate(test_list[:max_examples]):
        examples.append(f"Example {i + 1}:\n{test}")
    return "\n\n".join(examples)


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


def get_random_problems(
    num_problems: int,
    excluded_indices: List[int] = None,
    split: str = "test",
    seed: int = 42
) -> tuple[List[Dict], List[int]]:
    """Get random problems from dataset"""
    print(f"Loading MBPP dataset to sample {num_problems} random problems...")
    
    dataset = load_dataset("mbpp", split=split, trust_remote_code=True)
    df = dataset.to_pandas()
    df['original_index'] = df.index
    
    if excluded_indices:
        print(f"Excluding {len(excluded_indices)} previously sampled problems")
        df = df[~df['original_index'].isin(excluded_indices)]
    
    if len(df) > num_problems:
        df = df.sample(n=num_problems, random_state=seed)
    
    df = df.sort_values('original_index').reset_index(drop=True)
    
    original_indices = df['original_index'].tolist()
    problems = df.drop('original_index', axis=1).to_dict('records')
    
    print(f"\nSuccessfully sampled {len(problems)} problems.")
    return problems, original_indices


def get_problems_by_index_range(
    start_idx: int,
    end_idx: int,
    excluded_indices: List[int] = None,
    split: str = "test"
) -> tuple[List[Dict], List[int]]:
    """Get problems from index range, excluding specified indices"""
    print(f"Loading MBPP dataset to get problems from index {start_idx} to {end_idx}...")
    
    # Use the correct dataset name "mbpp" (not "Muennighoff/mbpp")
    dataset = load_dataset("mbpp", split=split, trust_remote_code=True)
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
    
    df = df.sort_values('original_index').reset_index(drop=True)
    
    original_indices = df['original_index'].tolist()
    problems = df.drop('original_index', axis=1).to_dict('records')
    
    print(f"\nSuccessfully selected {len(problems)} problems.")
    return problems, original_indices


def create_enhanced_stub(sample: Dict, out_dir: str) -> tuple[str, str]:
    """Create an enhanced stub for MBPP problem"""
    task_id = sample.get("task_id", "unknown")
    text = sample.get("text", "")  # Problem description
    code = sample.get("code", "")  # Reference solution (we don't show this)
    test_list = sample.get("test_list", [])
    
    # Ensure test_list is a list (handle numpy arrays)
    if isinstance(test_list, np.ndarray):
        test_list = test_list.tolist()
    
    # Extract function name from test assertions if possible
    function_name = "solve"
    if test_list:
        # Try to extract function name from first test assertion
        # e.g., "assert func_name(args) == expected"
        func_match = re.search(r'assert\s+(\w+)\s*\(', test_list[0])
        if func_match:
            function_name = func_match.group(1)
    
    # Format test examples
    test_examples = format_test_examples(test_list)
    
    stub_content = f'''# EVOLVE-BLOCK-START
"""
========================================
MBPP Problem: Task {task_id}
========================================

PROBLEM DESCRIPTION:
{text}

TEST EXAMPLES (your solution must pass these assertions):
{test_examples}

CONFIDENCE ASSESSMENT:
After completing your solution, perform a thorough self-review as if you were conducting a code review for a colleague.

"""
# Your solution implementation here
# Make sure to define the function: {function_name}

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
        eta_minutes = (elapsed / processing_stats.problems_completed) * \
                     (processing_stats.total_problems - processing_stats.problems_completed) / 60
        
        print(f"Progress: {processing_stats.problems_completed}/{processing_stats.total_problems} "
              f"({processing_stats.problems_completed / processing_stats.total_problems * 100:.1f}%) | "
              f"Avg Score: {avg_score:.3f} | "
              f"1st: {processing_stats.solved_first} | "
              f"Later: {processing_stats.solved_later} | "
              f"Unsolved: {processing_stats.unsolved} | "
              f"Elapsed: {elapsed / 60:.1f}min | "
              f"ETA: {eta_minutes:.1f}min")


def solve_one_problem_with_confidence(sample: Dict, idx: int, total: int, original_idx: int) -> Dict:
    """Solve one MBPP problem with confidence measurement"""
    acquired = False
    try:
        problem_semaphore.acquire()
        acquired = True
        
        task_id = sample.get("task_id", "unknown")
        safe_id = f"task_{task_id}"
        task_dir = os.path.join(OUTPUT_ROOT, f"{original_idx:04d}_{safe_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        problem_status = {
            "solved": False,
            "solved_iteration": None,
            "confidence_measurements": [],
            "final_score": 0.0
        }
        
        result = {
            "task_id": task_id,
            "text": sample.get("text", "N/A")[:100],  # Truncate for summary
            "original_index": original_idx,
            "combined_score": 0.0,
            "pass_rate": 0.0,
            "time_taken": 0.0,
            "solved": False,
            "solved_iteration": None,
            "problem_label": "unsolved",
            "confidence_measurements": [],
            "error": None
        }
        
        problem_start = time.time()
        
        try:
            # Convert sample to JSON-serializable format (handle numpy arrays)
            serializable_sample = convert_numpy_types(sample)
            
            # Dump sample.json for the evaluator using atomic write
            sample_path = os.path.join(task_dir, "sample.json")
            temp_sample_path = sample_path + ".tmp"
            with open(temp_sample_path, "w") as sf:
                json.dump(serializable_sample, sf, indent=2, cls=NumpyEncoder)
                sf.flush()
                os.fsync(sf.fileno())
            os.replace(temp_sample_path, sample_path)
            
            # Create enhanced stub using the serializable sample (lists instead of numpy arrays)
            stub_path, _ = create_enhanced_stub(serializable_sample, task_dir)
            print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: Enhanced stub written")
            
            # Phase 0: Initial solution generation
            print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: Phase 0 - Generating initial solution...")
            cfg0 = load_config(CONFIG_YAML)
            cfg0.max_iterations = 1
            
            db_path = os.path.join(task_dir, "evolution_database")
            cfg0.database.db_path = db_path
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
                print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: ❌ Phase 0 failed")
                result["error"] = "Phase 0 failed"
                return result
            
            initial_metrics = best0.metrics
            initial_combined = initial_metrics.get("combined_score", 0.0)
            initial_public = initial_metrics.get("public_pass_rate", 0.0)
            initial_private = initial_metrics.get("private_pass_rate", 0.0)
            
            print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: Initial scores - Combined: {initial_combined:.2%}")
            
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
                print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: ✅ Problem SOLVED in iteration 1!")
                return result
            
            # Extract confidence from Phase 0
            phase0_output_dir = os.path.join(task_dir, "phase0")
            best_program_path = os.path.join(phase0_output_dir, "best", "best_program.py")
            confidence_data = extract_confidence_from_evolution_result(best0, best_program_path)
            confidence_data["iteration"] = 1
            confidence_data["solved"] = False
            confidence_data["combined_score"] = initial_combined
            problem_status["confidence_measurements"].append(confidence_data)
            
            print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: Confidence after iter 1 (unsolved) - "
                  f"Mean: {confidence_data['mean_confidence']:.3f}, "
                  f"Verbalized: {confidence_data.get('verbalized_conf', 'N/A')}")
            
            result["combined_score"] = initial_combined
            result["pass_rate"] = initial_combined
            
            if initial_combined >= 0.95:
                print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: ✅ Almost solved! Skipping refinement.")
                problem_status["final_score"] = 1.0
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                return result
            
            full_code_path = os.path.join(task_dir, "phase0", "best", "best_program.py")
            if not os.path.exists(full_code_path):
                print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: ❌ Phase 0 solution file not found")
                result["error"] = "Phase 0 solution file not found"
                result["confidence_measurements"] = problem_status["confidence_measurements"]
                return result
            
            # Phase 1: Refinement
            base_cfg = load_config(CONFIG_YAML)
            remaining_iterations = max((base_cfg.max_iterations or 10) - 1, 0)
            base_cfg.max_iterations = remaining_iterations
            base_cfg.database.db_path = db_path
            base_cfg.llm.temperature = 0.6
            base_cfg.llm.top_p = 0.9
            
            if initial_public > 0.8 and initial_private < 0.5:
                print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: Focusing on generalization...")
                base_cfg.database.exploration_ratio = 0.7
                base_cfg.database.exploitation_ratio = 0.3
            else:
                base_cfg.database.exploration_ratio = 0.5
                base_cfg.database.exploitation_ratio = 0.5
            
            print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: Phase 1 - Refining ({remaining_iterations} iterations)...")
            
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
                
                print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: ✅ Final scores - Combined: {final_combined:.2%}")
                
                if final_combined >= 1.0 and not problem_status["solved"]:
                    problem_status["solved"] = True
                    problem_status["solved_iteration"] = "phase1"
                    result["solved"] = True
                    result["solved_iteration"] = "phase1"
                    result["problem_label"] = "solved_later_iteration"
                    
                    print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: ✅ Problem SOLVED in later iterations!")
                    
                    phase1_best_path = os.path.join(task_dir, "phase1", "best", "best_program.py")
                    solved_confidence_data = extract_confidence_from_evolution_result(best1, phase1_best_path)
                    solved_confidence_data["iteration"] = "phase1_final"
                    solved_confidence_data["solved"] = True
                    solved_confidence_data["combined_score"] = final_combined
                    problem_status["confidence_measurements"].append(solved_confidence_data)
                    
                    print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: Confidence for solved solution - "
                          f"Mean: {solved_confidence_data['mean_confidence']:.3f}, "
                          f"Verbalized: {solved_confidence_data.get('verbalized_conf', 'N/A')}")
                
                result["combined_score"] = final_combined
                result["pass_rate"] = final_combined
                problem_status["final_score"] = final_combined
                
                best_path = os.path.join(task_dir, "phase1", "best", "best_program.py")
                if os.path.exists(best_path):
                    final_path = os.path.join(task_dir, "final_solution.py")
                    with open(best_path, 'r') as src:
                        content = src.read()
                    
                    metadata_comment = f"""# MBPP Solution
# Task ID: {task_id}
# Original Index: {original_idx}
# Final Score: {final_combined:.2%} (Public: {final_public:.2%}, Private: {final_private:.2%})
# Solved: {problem_status['solved']} (Iteration: {problem_status['solved_iteration']})
# Generated by OpenEvolve

"""
                    with open(final_path, 'w') as dst:
                        dst.write(metadata_comment + content)
                        dst.flush()
            else:
                print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: Phase 1 produced no improvement")
                problem_status["final_score"] = initial_combined
            
        except Exception as e:
            print(f"[T{threading.current_thread().ident}][{idx + 1}/{total}] {safe_id}: ❌ Error: {str(e)}")
            result["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        finally:
            result["time_taken"] = time.time() - problem_start
            result["confidence_measurements"] = problem_status["confidence_measurements"]

            if not result["solved"]:
                if 'best1' in locals() and best1:
                    final_confidence = extract_confidence_from_evolution_result(best1)
                    final_confidence["iteration"] = "phase1_final_unsolved"
                    final_confidence["solved"] = False
                    final_confidence["combined_score"] = result["combined_score"]
                    problem_status["confidence_measurements"].append(final_confidence)
                else:
                    problem_status["confidence_measurements"].append({
                        "iteration": "phase1_final_unsolved",
                        "solved": False,
                        "combined_score": result["combined_score"],
                        "mean_confidence": 0,
                        "least_grouped_confidence": 0,
                        "tail_confidence": 0,
                        "note": "No code generated"
                    })

            if result["problem_label"] == "unsolved" and result["combined_score"] > 0:
                result["problem_label"] = "partial"
            
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
    print("=" * 80)
    
    executor = None
    try:
        executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROBLEMS)
        future_to_problem = {}
        for i, (sample, orig_idx) in enumerate(zip(problems, original_indices)):
            future = executor.submit(solve_one_problem_with_confidence, sample, i, len(problems), orig_idx)
            future_to_problem[future] = (i, sample.get("task_id", "unknown"))
        
        for future in as_completed(future_to_problem):
            idx, task_id = future_to_problem[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n✓ Completed {idx + 1}/{len(problems)}: task_{task_id} "
                      f"(score: {result['combined_score']:.2%}, label: {result['problem_label']})")
            except Exception as e:
                print(f"\n✗ Failed {idx + 1}/{len(problems)}: task_{task_id} - Error: {e}")
                results.append({
                    "task_id": task_id,
                    "text": "N/A",
                    "original_index": idx,
                    "combined_score": 0.0,
                    "pass_rate": 0.0,
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
    """Main entry point for MBPP batch processing"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    if not os.path.exists(EVAL_SCRIPT):
        print(f"Error: Evaluator script not found at {EVAL_SCRIPT}")
        return
    
    if not os.path.exists(CONFIG_YAML):
        print(f"Error: Config file not found at {CONFIG_YAML}")
        print(f"Please copy config_model_sampling.yaml to {CONFIG_YAML}")
        return
    
    # Configuration - MBPP test split has 974 problems
    NUM_PROBLEMS = 50
    
    # Load previously sampled indices to exclude (if any)
    previous_sampling_file = os.path.join(OUTPUT_ROOT, "sampling_info.json")
    excluded_indices = load_excluded_indices(previous_sampling_file)
    
    try:
        print(f"\n{'=' * 60}")
        print(f"RUNNING MBPP BATCH: {NUM_PROBLEMS} random problems")
        print(f"{'=' * 60}\n")
        
        problems, original_indices = get_random_problems(
            num_problems=NUM_PROBLEMS,
            excluded_indices=excluded_indices,
            split="test"
        )
        print(f"Loaded {len(problems)} problems from MBPP")
        
        sampling_info = {
            "num_problems": len(problems),
            "sampling_method": "random_sampling",
            "original_indices": original_indices,
            "excluded_indices": excluded_indices,
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
    
    start_time = time.time()
    results = process_problems_parallel(problems, original_indices)
    total_time = time.time() - start_time
    
    if not results:
        print("\nNo problems were processed.")
        return

    # Prepare confidence analysis
    confidence_analysis = []
    for result in results:
        if result["confidence_measurements"]:
            for conf_measurement in result["confidence_measurements"]:
                confidence_analysis.append({
                    "task_id": result["task_id"],
                    "original_index": result["original_index"],
                    "problem_label": result["problem_label"],
                    "iteration": conf_measurement["iteration"],
                    "solved_at_measurement": conf_measurement["solved"],
                    "combined_score_at_measurement": conf_measurement["combined_score"],
                    "mean_confidence": conf_measurement["mean_confidence"],
                    "least_grouped_confidence": conf_measurement["least_grouped_confidence"],
                    "tail_confidence": conf_measurement["tail_confidence"],
                    "verbalized_conf": conf_measurement.get("verbalized_conf")
                })

    # Print summary
    solved_first = sum(1 for r in results if r["problem_label"] == "solved_first_iteration")
    solved_later = sum(1 for r in results if r["problem_label"] == "solved_later_iteration")
    partial = sum(1 for r in results if r["problem_label"] == "partial")
    unsolved = sum(1 for r in results if r["problem_label"] == "unsolved")
    errors = sum(1 for r in results if r.get("error"))
    avg_score = np.mean([r["combined_score"] for r in results])
    
    print(f"\n{'=' * 60}")
    print(f"MBPP BATCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total problems: {len(results)}")
    print(f"Solved (1st iteration): {solved_first} ({solved_first / len(results) * 100:.1f}%)")
    print(f"Solved (later iterations): {solved_later} ({solved_later / len(results) * 100:.1f}%)")
    print(f"Partial: {partial} ({partial / len(results) * 100:.1f}%)")
    print(f"Unsolved: {unsolved} ({unsolved / len(results) * 100:.1f}%)")
    print(f"Errors: {errors}")
    print(f"Average score: {avg_score:.3f}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    
    # Confidence analysis
    if confidence_analysis:
        print(f"\nCONFIDENCE ANALYSIS:")
        
        unsolved_after_1st = [c for c in confidence_analysis if c["iteration"] == 1 and not c["solved_at_measurement"]]
        if unsolved_after_1st:
            avg_conf_unsolved = np.mean([c["mean_confidence"] for c in unsolved_after_1st])
            verbalized_confs = [c["verbalized_conf"] for c in unsolved_after_1st if c["verbalized_conf"] is not None]
            avg_verbalized_unsolved = np.mean(verbalized_confs) if verbalized_confs else None
            
            print(f"Unsolved after 1st iteration ({len(unsolved_after_1st)} problems):")
            print(f"  Average mean confidence: {avg_conf_unsolved:.3f}")
            if avg_verbalized_unsolved is not None:
                print(f"  Average verbalized confidence: {avg_verbalized_unsolved:.1f}/100")
        
        solved_later_confs = [c for c in confidence_analysis if c["solved_at_measurement"]]
        if solved_later_confs:
            avg_conf_solved_later = np.mean([c["mean_confidence"] for c in solved_later_confs])
            print(f"Solved in later iterations ({len(solved_later_confs)} problems):")
            print(f"  Average mean confidence: {avg_conf_solved_later:.3f}")
    
    # Save results
    results_file = os.path.join(OUTPUT_ROOT, "results_with_confidence.json")
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
                "timestamp": time.time()
            },
            "sampling_info": sampling_info,
            "results": results,
            "confidence_analysis": confidence_analysis
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()

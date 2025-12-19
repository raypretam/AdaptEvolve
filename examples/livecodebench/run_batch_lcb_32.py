import json
import os
import ast
import re
import pandas as pd
import time
import threading
from typing import Dict, List, Optional
from datasets import load_dataset
from openevolve import OpenEvolve
from openevolve.config import load_config
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import atexit
import asyncio

ROOT        = os.path.dirname(__file__)
CONFIG_YAML = os.path.join(ROOT, "config_4B_32B.yaml")
EVAL_SCRIPT = os.path.join(ROOT, "evaluator.py")
OUTPUT_ROOT = os.path.join(ROOT, "livecodebench_runs_4B_32B_46_54_50problems_test")
SAMPLING_INFO_FILE = os.path.join(ROOT, "..", "llm_selection_confidence", "livecodebench_runs_decision_tree", "sampling_info.json")

# Concurrency settings
MAX_CONCURRENT_PROBLEMS = 64
MAX_CONCURRENT_LLM_CALLS = 128

# Thread-safe semaphores
llm_semaphore = threading.Semaphore(MAX_CONCURRENT_LLM_CALLS)
problem_semaphore = threading.Semaphore(MAX_CONCURRENT_PROBLEMS)

@dataclass
class ProcessingStats:
    """Track processing statistics"""
    start_time: float
    problems_completed: int = 0
    total_problems: int = 0
    solved: int = 0
    partial: int = 0
    failed: int = 0
    total_score: float = 0.0

# Global stats for monitoring with thread lock
processing_stats = ProcessingStats(start_time=time.time())
stats_lock = threading.Lock()

def cleanup_resources():
    """Cleanup function to be called on exit"""
    print("\n🧹 Cleaning up resources...")
    time.sleep(2)

# Register cleanup function
atexit.register(cleanup_resources)

def update_stats_thread_safe(result: Dict):
    """Update global statistics in a thread-safe manner"""
    with stats_lock:
        processing_stats.problems_completed += 1
        processing_stats.total_score += result["combined_score"]
        
        if result["combined_score"] == 1.0:
            processing_stats.solved += 1
        elif result["combined_score"] > 0:
            processing_stats.partial += 1
        else:
            processing_stats.failed += 1

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
              f"Solved: {processing_stats.solved} | "
              f"Partial: {processing_stats.partial} | "
              f"Failed: {processing_stats.failed} | "
              f"Elapsed: {elapsed/60:.1f}min | "
              f"ETA: {eta_minutes:.1f}min")

def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown blocks if present"""
    # Look for code blocks
    code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]
    return text

def clean_html_tags(text: str) -> str:
    """Remove HTML tags from text"""
    # Remove common HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Convert HTML entities
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
            # Show up to 3 examples, prioritizing diversity
            for i, test in enumerate(test_cases[:3]):
                input_str = test.get("input", "").strip()
                output_str = test.get("output", "").strip()
                examples.append(format_io_example(input_str, output_str, i + 1))
    except Exception:
        pass
    return "\n\n".join(examples)

def extract_constraints(question_content: str) -> tuple[str, str]:
    """Extract constraints section from question content"""
    # Look for constraints section
    constraints_patterns = [
        r'Constraints?:(.*?)(?=Note:|Example:|Input:|$)',
        r'\*\*Constraints?\*\*:(.*?)(?=\*\*|$)',
        r'### Constraints?:?(.*?)(?=###|$)'
    ]
    
    for pattern in constraints_patterns:
        match = re.search(pattern, question_content, re.DOTALL | re.IGNORECASE)
        if match:
            constraints = match.group(1).strip()
            # Remove constraints from main content
            content_without_constraints = question_content.replace(match.group(0), "").strip()
            return content_without_constraints, constraints
    
    return question_content, ""

def load_problems_from_indices(sampling_info_path: str) -> tuple[List[Dict], List[int]]:
    """
    Load problems using specific indices from sampling_info.json
    Returns: (problems, original_indices)
    """
    print(f"Loading sampling info from {sampling_info_path}...")
    
    if not os.path.exists(sampling_info_path):
        raise FileNotFoundError(f"Sampling info file not found: {sampling_info_path}")
    
    with open(sampling_info_path, 'r') as f:
        sampling_info = json.load(f)
    
    original_indices = sampling_info.get("original_indices", [])
    print(f"Found {len(original_indices)} indices to process: {original_indices[:10]}{'...' if len(original_indices) > 10 else ''}")
    
    # Load the full dataset
    print("Loading LiveCodeBench dataset...")
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5", trust_remote_code=True)
    df = dataset.to_pandas()
    
    # Filter to only the specified indices
    df_filtered = df.iloc[original_indices].copy()
    
    # Show difficulty distribution of selected problems
    difficulty_distribution = df_filtered['difficulty'].value_counts()
    print(f"\nSelected {len(df_filtered)} problems with difficulty distribution:")
    for diff, count in difficulty_distribution.items():
        print(f"  {diff}: {count} problems ({count/len(df_filtered):.1%})")
    
    # Convert to list of dictionaries
    problems = df_filtered.to_dict('records')
    
    print(f"\nSuccessfully loaded {len(problems)} problems.")
    return problems, original_indices

def stratified_sample_problems(num_problems: int = 50, random_state: int = 42) -> tuple[List[Dict], List[int]]:
    """
    Sample problems using stratified sampling to maintain difficulty distribution
    Returns: (sampled_problems, original_indices)
    """
    print(f"Loading dataset for stratified sampling of {num_problems} problems...")
    
    # Load the full dataset
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5", trust_remote_code=True)
    df = dataset.to_pandas()
    
    # Add original index to track which problems we selected
    df['original_index'] = df.index
    
    # Show original distribution
    difficulty_distribution = df['difficulty'].value_counts(normalize=True)
    print("Original problem distribution by difficulty:")
    for diff, prop in difficulty_distribution.items():
        count = (df['difficulty'] == diff).sum()
        print(f"  {diff}: {count} problems ({prop:.1%})")
    
    # Calculate sample counts maintaining the same distribution
    sample_counts = (difficulty_distribution * num_problems).round().astype(int)
    
    # Adjust for rounding differences
    diff = num_problems - sample_counts.sum()
    if diff != 0:
        most_frequent_difficulty = sample_counts.idxmax()
        sample_counts[most_frequent_difficulty] += diff
    
    print(f"\nSampling {num_problems} problems with the following distribution:")
    for diff, count in sample_counts.items():
        print(f"  {diff}: {count} problems ({count/num_problems:.1%})")
    
    # Perform stratified sampling
    sampled_df = df.groupby('difficulty', group_keys=False).apply(
        lambda x: x.sample(
            n=int(sample_counts[x.name]) if x.name in sample_counts else 0, 
            random_state=random_state
        )
    )
    
    # Sort by original index to maintain some order
    sampled_df = sampled_df.sort_values('original_index')
    
    # Extract the original indices
    original_indices = sampled_df['original_index'].tolist()
    
    # Convert to list of dictionaries (removing the added index column)
    problems = sampled_df.drop('original_index', axis=1).to_dict('records')
    
    print(f"\nSuccessfully sampled {len(problems)} problems.")
    print(f"Original indices: {original_indices[:10]}{'...' if len(original_indices) > 10 else ''}")
    
    return problems, original_indices

def create_enhanced_stub(sample: Dict, out_dir: str) -> tuple[str, str]:
    """
    Create an enhanced stub following Qwen3-Coder's approach
    """
    # Extract problem information
    question_title = sample.get("question_title", "")
    question_content = sample.get("question_content", "")
    difficulty = sample.get("difficulty", "unknown")
    starter_code = sample.get("starter_code", "")
    
    # Extract function name from starter code
    function_name = "solve"  # default
    if starter_code:
        # Look for function definition pattern: def function_name(
        import re
        func_match = re.search(r'def\s+(\w+)\s*\(', starter_code)
        if func_match:
            function_name = func_match.group(1)
    
    # Clean HTML and markdown
    question_content = clean_html_tags(question_content)
    question_content = extract_code_from_markdown(question_content)
    
    # Extract constraints
    question_content, constraints = extract_constraints(question_content)
    
    # Parse metadata for additional context
    metadata = []
    if "source" in sample:
        metadata.append(f"Source: {sample['source']}")
    if "contest" in sample:
        metadata.append(f"Contest: {sample['contest']}")
    
    # Build the enhanced stub content
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
    
    # Write the stub file
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

def solve_one_enhanced_sync(sample: Dict, idx: int, total: int, original_idx: int) -> Dict:
    """Enhanced solving with better configuration - synchronous version for thread pool"""
    acquired = False
    try:
        problem_semaphore.acquire()
        acquired = True
        
        # Prepare directories
        question_id = sample.get("question_id", "unknown")
        safe_id = question_id.replace("/", "_").replace(" ", "_")
        task_dir = os.path.join(OUTPUT_ROOT, f"{original_idx:04d}_{safe_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        result = {
            "question_id": question_id,
            "question_title": sample.get("question_title", "N/A"),
            "original_index": original_idx,
            "combined_score": 0.0,
            "pass_rate": 0.0,
            "difficulty": sample.get("difficulty", "N/A"),
            "source": sample.get("source", "N/A"),
            "time_taken": 0.0,
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
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id} (orig_idx: {original_idx}): Enhanced stub written")
            
            # Phase 0: Initial solution generation
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Phase 0 - Generating initial solution...")
            cfg0 = load_config(CONFIG_YAML)
            cfg0.max_iterations = 1
            
            # Database configuration
            db_path = os.path.join(task_dir, "evolution_database")
            cfg0.database.db_path = db_path
            
            # LLM configuration for initial generation
            cfg0.llm.temperature = 0.3
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
            
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Initial scores - Combined: {initial_combined:.2%}, "
                  f"Public: {initial_public:.2%}, Private: {initial_private:.2%}")
            
            # Check if we should continue to phase 1
            if initial_combined >= 0.95:
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Already solved well! Skipping refinement.")
                result["combined_score"] = initial_combined
                result["pass_rate"] = initial_combined
                return result
            
            # Get the generated solution path
            full_code_path = os.path.join(task_dir, "phase0", "best", "best_program.py")
            if not os.path.exists(full_code_path):
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ❌ Phase 0 solution file not found")
                result["combined_score"] = initial_combined
                result["pass_rate"] = initial_combined
                result["error"] = "Phase 0 solution file not found"
                return result
            
            # Phase 1: Refinement with different strategy
            base_cfg = load_config(CONFIG_YAML)
            remaining_iterations = max((base_cfg.max_iterations or 10) - 1, 0)
            base_cfg.max_iterations = remaining_iterations
            
            # Use same database
            base_cfg.database.db_path = db_path
            
            # Adjust for exploration
            base_cfg.llm.temperature = 0.5
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
                
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ✅ Final scores - Combined: {final_combined:.2%}, "
                      f"Public: {final_public:.2%}, Private: {final_private:.2%}")
                
                result["combined_score"] = final_combined
                result["pass_rate"] = final_combined
                
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
# Generated by OpenEvolve

"""
                    with open(final_path, 'w') as dst:
                        dst.write(metadata_comment + content)
                        dst.flush()
            else:
                print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: Phase 1 produced no improvement")
                result["combined_score"] = initial_combined
                result["pass_rate"] = initial_combined
            
        except Exception as e:
            print(f"[T{threading.current_thread().ident}][{idx+1}/{total}] {safe_id}: ❌ Error: {str(e)}")
            result["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        finally:
            result["time_taken"] = time.time() - problem_start
            update_stats_thread_safe(result)
            print_progress()
        
        return result
    
    finally:
        if acquired:
            problem_semaphore.release()

def process_problems_parallel(problems: List[Dict], original_indices: List[int]) -> List[Dict]:
    """Process all problems using ThreadPoolExecutor"""
    processing_stats.total_problems = len(problems)
    processing_stats.start_time = time.time()
    results = []
    
    print(f"\nStarting parallel processing with ThreadPoolExecutor")
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
            future = executor.submit(solve_one_enhanced_sync, sample, i, len(problems), orig_idx)
            future_to_problem[future] = (i, sample.get("question_id", "unknown"))
        
        # Process results as they complete
        for future in as_completed(future_to_problem):
            idx, question_id = future_to_problem[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n✓ Completed {idx+1}/{len(problems)}: {question_id} "
                      f"(score: {result['combined_score']:.2%})")
            except Exception as e:
                print(f"\n✗ Failed {idx+1}/{len(problems)}: {question_id} - Error: {e}")
                results.append({
                    "question_id": question_id,
                    "question_title": "N/A",
                    "original_index": idx,
                    "combined_score": 0.0,
                    "pass_rate": 0.0,
                    "difficulty": "N/A",
                    "source": "N/A",
                    "time_taken": 0.0,
                    "error": str(e)
                })
    
    finally:
        if executor:
            print("\n🔄 Shutting down thread pool executor...")
            executor.shutdown(wait=True, cancel_futures=False)
            print("✅ Thread pool executor shut down cleanly")
    
    return results

def main():
    """Main entry point with parallel processing using ThreadPoolExecutor"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Check requirements
    if not os.path.exists(EVAL_SCRIPT):
        print(f"Error: Evaluator script not found at {EVAL_SCRIPT}")
        return
    
    if not os.path.exists(CONFIG_YAML):
        print(f"Error: Config file not found at {CONFIG_YAML}")
        return
    
    # Load dataset using indices from sampling_info.json
    try:
        problems, original_indices = load_problems_from_indices(SAMPLING_INFO_FILE)
        print(f"Loaded {len(problems)} problems from LiveCodeBench using indices from sampling_info.json")
        
        # Save the sampling information (copy from source + add our settings)
        with open(SAMPLING_INFO_FILE, 'r') as f:
            original_sampling_info = json.load(f)
        
        sampling_info = {
            **original_sampling_info,
            "source_sampling_file": SAMPLING_INFO_FILE,
            "run_timestamp": time.time(),
            "max_concurrent_problems": MAX_CONCURRENT_PROBLEMS,
            "max_concurrent_llm_calls": MAX_CONCURRENT_LLM_CALLS
        }
        sampling_file = os.path.join(OUTPUT_ROOT, "sampling_info.json")
        with open(sampling_file, 'w') as f:
            json.dump(sampling_info, f, indent=2)
        print(f"Sampling information saved to {sampling_file}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process problems in parallel
    start_time = time.time()
    results = process_problems_parallel(problems, original_indices)
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r["combined_score"] == 1)
    partial = sum(1 for r in results if 0.0 < r["combined_score"] < 1.00)
    failed = sum(1 for r in results if r["combined_score"] == 0)
    errors = sum(1 for r in results if r.get("error"))
    avg_score = sum(r["combined_score"] for r in results) / len(results) if results else 0
    
    print(f"Total problems: {len(results)}")
    print(f"Solved (==100%): {successful} ({successful/len(results)*100:.1f}%)")
    print(f"Partial (0-100%): {partial} ({partial/len(results)*100:.1f}%)")
    print(f"Failed (==0%): {failed} ({failed/len(results)*100:.1f}%)")
    print(f"Errors: {errors}")
    print(f"Average score: {avg_score:.3f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per problem: {total_time/len(results):.1f} seconds")
    
    # Results by difficulty
    difficulties = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in difficulties:
            difficulties[diff] = {
                "total": 0,
                "solved": 0,
                "partial": 0,
                "failed": 0,
                "total_score": 0
            }
        difficulties[diff]["total"] += 1
        difficulties[diff]["total_score"] += r["combined_score"]
        if r["combined_score"] == 1.00:
            difficulties[diff]["solved"] += 1
        elif r["combined_score"] >= 0.5 and r["combined_score"] < 1.00:
            difficulties[diff]["partial"] += 1
        else:
            difficulties[diff]["failed"] += 1
    
    print(f"\nResults by difficulty:")
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulties:
            stats = difficulties[diff]
            avg_diff_score = stats["total_score"] / stats["total"]
            print(f"  {diff.capitalize()}: {stats['solved']}/{stats['total']} solved "
                  f"({stats['solved']/stats['total']*100:.1f}%), "
                  f"avg score: {avg_diff_score:.3f}")
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_ROOT, "results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total": len(results),
                "solved": successful,
                "partial": partial,
                "failed": failed,
                "errors": errors,
                "average_score": avg_score,
                "total_time_minutes": total_time / 60,
                "timestamp": time.time(),
                "max_concurrent_problems": MAX_CONCURRENT_PROBLEMS,
                "max_concurrent_llm_calls": MAX_CONCURRENT_LLM_CALLS
            },
            "sampling_info": sampling_info,
            "by_difficulty": {
                diff: {
                    "total": stats["total"],
                    "solved": stats["solved"],
                    "partial": stats["partial"],
                    "failed": stats["failed"],
                    "average_score": stats["total_score"] / stats["total"]
                }
                for diff, stats in difficulties.items()
            },
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Save problems that need improvement
    need_improvement = [r for r in results if r["combined_score"] < 0.95]
    if need_improvement:
        improvement_file = os.path.join(OUTPUT_ROOT, "needs_improvement.json")
        with open(improvement_file, 'w') as f:
            json.dump(sorted(need_improvement, key=lambda x: x["combined_score"]), f, indent=2)
        print(f"Problems needing improvement saved to {improvement_file}")

if __name__ == "__main__":
    main()

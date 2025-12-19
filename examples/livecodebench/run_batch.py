import asyncio
import json
import os
import ast
import re
import pandas as pd
from typing import Dict, List, Optional
from datasets import load_dataset
from openevolve import OpenEvolve
from openevolve.config import load_config

ROOT        = os.path.dirname(__file__)
CONFIG_YAML = os.path.join(ROOT, "config_model_sampling.yaml")
EVAL_SCRIPT = os.path.join(ROOT, "evaluator.py")
OUTPUT_ROOT = os.path.join(ROOT, "livecodebench_runs_gbfn")

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
    # df = df[df['difficulty'] != 'easy']
    
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

INSTRUCTIONS:
1. Read ALL input from standard input (stdin)
2. Write ALL output to standard output (stdout)
3. Do NOT add any extra print statements or prompts
4. Handle edge cases according to the constraints
5. Ensure your solution handles multiple test cases if required
6. Pay attention to the exact output format shown in examples

SOLUTION APPROACH:
- First, understand the input format from the examples
- Identify the algorithm or approach needed
- Consider edge cases mentioned in constraints
- Implement a clean, efficient solution
"""

import sys
# Your implementation here
    
# EVOLVE-BLOCK-END
'''
    
    # Write the stub file
    os.makedirs(out_dir, exist_ok=True)
    stub_path = os.path.join(out_dir, "prompt.py")
    with open(stub_path, "w") as f:
        f.write(stub_content)
        f.flush()
    
    return stub_path, "solve"

async def solve_one_enhanced(sample: Dict, idx: int, total: int, original_idx: int) -> float:
    """Enhanced solving with better configuration"""
    # Prepare directories
    question_id = sample.get("question_id", "unknown")
    safe_id = question_id.replace("/", "_").replace(" ", "_")
    task_dir = os.path.join(OUTPUT_ROOT, f"{original_idx:04d}_{safe_id}")
    os.makedirs(task_dir, exist_ok=True)
    
    # Dump sample.json for the evaluator
    sample_path = os.path.join(task_dir, "sample.json")
    with open(sample_path, "w") as sf:
        json.dump(sample, sf, indent=2)
        sf.flush()
    
    # Create enhanced stub
    stub_path, _ = create_enhanced_stub(sample, task_dir)
    print(f"[{idx+1}/{total}] {safe_id} (orig_idx: {original_idx}): Enhanced stub written")
    
    # Phase 0: Initial solution generation
    print(f"[{idx+1}/{total}] {safe_id}: Phase 0 - Generating initial solution...")
    cfg0 = load_config(CONFIG_YAML)
    cfg0.max_iterations = 1
    
    # Database configuration
    db_path = os.path.join(task_dir, "evolution_database")
    cfg0.database.db_path = db_path
    
    # LLM configuration for initial generation
    cfg0.llm.temperature = 0.3
    cfg0.llm.top_p = 0.95
    
    try:
        evo0 = OpenEvolve(
            initial_program_path=stub_path,
            evaluation_file=EVAL_SCRIPT,
            config=cfg0,
            output_dir=os.path.join(task_dir, "phase0"),
            test_file_path=sample_path
        )
        
        best0 = await evo0.run()
        
        if not best0:
            print(f"[{idx+1}/{total}] {safe_id}: ❌ Phase 0 failed")
            return 0.0
        
        # Get metrics
        initial_metrics = best0.metrics
        initial_combined = initial_metrics.get("combined_score", 0.0)
        initial_public = initial_metrics.get("public_pass_rate", 0.0)
        initial_private = initial_metrics.get("private_pass_rate", 0.0)
        
        print(f"[{idx+1}/{total}] {safe_id}: Initial scores - Combined: {initial_combined:.2%}, "
              f"Public: {initial_public:.2%}, Private: {initial_private:.2%}")
        
        # Check if we should continue to phase 1
        if initial_combined >= 0.95:  # High enough score
            print(f"[{idx+1}/{total}] {safe_id}: ✅ Already solved well! Skipping refinement.")
            return initial_combined
        
        # Get the generated solution path
        full_code_path = os.path.join(task_dir, "phase0", "best", "best_program.py")
        if not os.path.exists(full_code_path):
            print(f"[{idx+1}/{total}] {safe_id}: ❌ Phase 0 solution file not found")
            return initial_combined
        
        # Phase 1: Refinement with different strategy
        base_cfg = load_config(CONFIG_YAML)
        remaining_iterations = max((base_cfg.max_iterations or 10) - 1, 0)
        base_cfg.max_iterations = remaining_iterations
        
        # Use same database
        base_cfg.database.db_path = db_path
        
        # Adjust for exploration
        base_cfg.llm.temperature = 0.5  # Moderate temperature for refinement
        base_cfg.llm.top_p = 0.9
        
        # Enable different evolution strategies based on current performance
        if initial_public > 0.8 and initial_private < 0.5:
            # Good on public, bad on private - need to generalize
            print(f"[{idx+1}/{total}] {safe_id}: Focusing on generalization...")
            base_cfg.database.exploration_ratio = 0.7  # More exploration
            base_cfg.database.exploitation_ratio = 0.3
        else:
            # Standard refinement
            base_cfg.database.exploration_ratio = 0.5
            base_cfg.database.exploitation_ratio = 0.5
        
        print(f"[{idx+1}/{total}] {safe_id}: Phase 1 - Refining ({remaining_iterations} iterations)...")
        
        evo1 = OpenEvolve(
            initial_program_path=full_code_path,
            evaluation_file=EVAL_SCRIPT,
            config=base_cfg,
            output_dir=os.path.join(task_dir, "phase1"),
            test_file_path=sample_path
        )
        
        best1 = await evo1.run()
        
        if best1:
            final_metrics = best1.metrics
            final_combined = final_metrics.get("combined_score", 0.0)
            final_public = final_metrics.get("public_pass_rate", 0.0)
            final_private = final_metrics.get("private_pass_rate", 0.0)
            
            print(f"[{idx+1}/{total}] {safe_id}: ✅ Final scores - Combined: {final_combined:.2%}, "
                  f"Public: {final_public:.2%}, Private: {final_private:.2%}")
            
            # Save final solution with metadata
            best_path = os.path.join(task_dir, "phase1", "best", "best_program.py")
            if os.path.exists(best_path):
                final_path = os.path.join(task_dir, "final_solution.py")
                with open(best_path, 'r') as src:
                    content = src.read()
                
                # Add metadata as comments
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
            
            return final_combined
        else:
            print(f"[{idx+1}/{total}] {safe_id}: Phase 1 produced no improvement")
            return initial_combined
        
    except Exception as e:
        print(f"[{idx+1}/{total}] {safe_id}: ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0

async def main():
    """Main entry point with enhanced features and stratified sampling"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Check requirements
    if not os.path.exists(EVAL_SCRIPT):
        print(f"Error: Evaluator script not found at {EVAL_SCRIPT}")
        return
    
    if not os.path.exists(CONFIG_YAML):
        print(f"Error: Config file not found at {CONFIG_YAML}")
        return
    
    # Load dataset with stratified sampling
    try:
        num_problems = 50
        problems, original_indices = stratified_sample_problems(num_problems=num_problems, random_state=42)
        print(f"Loaded {len(problems)} problems from LiveCodeBench using stratified sampling")
        
        # Save the sampling information
        sampling_info = {
            "num_problems": num_problems,
            "random_state": 42,
            "original_indices": original_indices,
            "sampling_timestamp": asyncio.get_event_loop().time()
        }
        sampling_file = os.path.join(OUTPUT_ROOT, "sampling_info.json")
        with open(sampling_file, 'w') as f:
            json.dump(sampling_info, f, indent=2)
        print(f"Sampling information saved to {sampling_file}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Process each problem
    results = []
    start_time = asyncio.get_event_loop().time()
    
    for i, (sample, original_idx) in enumerate(zip(problems, original_indices)):
        print(f"\n{'='*80}")
        print(f"Problem {i+1}/{len(problems)}: {sample.get('question_id', 'unknown')}")
        print(f"Original Index: {original_idx}")
        print(f"Title: {sample.get('question_title', 'N/A')}")
        print(f"Difficulty: {sample.get('difficulty', 'N/A')}")
        if "source" in sample:
            print(f"Source: {sample['source']}")
        print(f"{'='*80}")
        
        problem_start = asyncio.get_event_loop().time()
        combined_score = await solve_one_enhanced(sample, i, len(problems), original_idx)
        problem_time = asyncio.get_event_loop().time() - problem_start
        
        results.append({
            "question_id": sample.get("question_id"),
            "question_title": sample.get("question_title", "N/A"),
            "original_index": original_idx,
            "combined_score": combined_score,
            "pass_rate": combined_score,  # For compatibility
            "difficulty": sample.get("difficulty", "N/A"),
            "source": sample.get("source", "N/A"),
            "time_taken": problem_time
        })
    
    # Summary
    total_time = asyncio.get_event_loop().time() - start_time
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r["combined_score"] == 1)
    partial = sum(1 for r in results if 0.0 < r["combined_score"] < 1.00)
    failed = sum(1 for r in results if r["combined_score"] == 0)
    avg_score = sum(r["combined_score"] for r in results) / len(results)
    
    print(f"Total problems: {len(results)}")
    print(f"Solved (==100%): {successful} ({successful/len(results)*100:.1f}%)")
    print(f"Partial (0-100%): {partial} ({partial/len(results)*100:.1f}%)")
    print(f"Failed (==0%): {failed} ({failed/len(results)*100:.1f}%)")
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
    for diff in ["easy", "medium", "hard"]:  # Order by expected difficulty
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
                "average_score": avg_score,
                "total_time_minutes": total_time / 60,
                "timestamp": asyncio.get_event_loop().time()
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
    asyncio.run(main())

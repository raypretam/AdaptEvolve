#!/usr/bin/env python3
"""
MBPP Evaluator for OpenEvolve

This file contains the evaluate() function that OpenEvolve calls to evaluate code.
It evaluates on both public (test_list) and private (challenge_test_list) test cases.
"""

import json
import sys
import os
import subprocess
import tempfile
import time
import traceback


def find_sample_json(code_file_path):
    """Find sample.json by walking up the directory tree"""
    cur = os.path.dirname(os.path.abspath(code_file_path))
    while cur != os.path.dirname(cur):
        candidate = os.path.join(cur, "sample.json")
        if os.path.exists(candidate):
            return candidate
        cur = os.path.dirname(cur)
    raise FileNotFoundError("Could not locate sample.json above " + code_file_path)


def run_test_case(code, test_assertion, test_setup_code="", timeout=10):
    """
    Run a single test case (assert statement) and return the result
    
    Args:
        code: The solution code to test
        test_assertion: An assert statement string to execute
        test_setup_code: Optional setup code (imports, etc.) to run before the solution
        timeout: Maximum execution time in seconds
    
    Returns:
        tuple: (passed: bool, execution_time: float, error: str or None)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Write setup code first (if any)
        if test_setup_code and test_setup_code.strip():
            f.write("# Test setup code\n")
            f.write(test_setup_code)
            f.write("\n\n")
        
        # Write the solution code
        f.write("# Solution code\n")
        f.write(code)
        f.write("\n\n# Test assertion\n")
        f.write(test_assertion)
        f.write("\n")
        temp_file = f.name
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        execution_time = time.time() - start_time
        
        # If return code is 0, the assertion passed
        if result.returncode == 0:
            return True, execution_time, None
        else:
            error_msg = result.stderr.strip()
            # Check if it's an assertion error
            if "AssertionError" in error_msg:
                return False, execution_time, "AssertionError"
            return False, execution_time, f"Runtime error: {error_msg}"
        
    except subprocess.TimeoutExpired:
        return False, timeout, "Timeout"
    except Exception as e:
        return False, 0, str(e)
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass


def evaluate(file_path, test_file_path=None):
    """
    Evaluate function called by OpenEvolve
    
    Args:
        file_path: Path to the code file to evaluate
        test_file_path: Path to the test data (sample.json)
    
    Returns:
        dict: Evaluation metrics including pass_rate and combined_score
    """
    try:
        # If no test_file_path provided, find it by walking up
        if not test_file_path:
            test_file_path = find_sample_json(file_path)
        
        # Load test data
        if not os.path.exists(test_file_path):
            return {
                "pass_rate": 0.0,
                "public_pass_rate": 0.0,
                "private_pass_rate": 0.0,
                "combined_score": 0.0,
                "error": "No test file found"
            }
        
        with open(test_file_path, 'r') as f:
            sample_data = json.load(f)
        
        # Read the code to evaluate
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Extract code if it's between EVOLVE-BLOCK markers
        if "# EVOLVE-BLOCK-START" in code and "# EVOLVE-BLOCK-END" in code:
            start_idx = code.find("# EVOLVE-BLOCK-START")
            end_idx = code.find("# EVOLVE-BLOCK-END")
            if start_idx != -1 and end_idx != -1:
                code = code[start_idx + len("# EVOLVE-BLOCK-START"):end_idx].strip()
        
        # Get test setup code (imports, etc.)
        test_setup_code = sample_data.get("test_setup_code", "") or ""
        
        # Get test cases from MBPP format
        # test_list: public test cases (assert statements)
        # challenge_test_list: private/challenge test cases (assert statements) - may be empty
        public_test_cases = sample_data.get("test_list", [])
        private_test_cases = sample_data.get("challenge_test_list", [])
        
        # Ensure they are lists and filter out empty strings
        if isinstance(public_test_cases, str):
            public_test_cases = [public_test_cases] if public_test_cases.strip() else []
        public_test_cases = [t for t in public_test_cases if t and t.strip()]
        
        if isinstance(private_test_cases, str):
            private_test_cases = [private_test_cases] if private_test_cases.strip() else []
        private_test_cases = [t for t in private_test_cases if t and t.strip()]
        
        # Run all test cases
        public_passed = 0
        public_total = len(public_test_cases)
        private_passed = 0
        private_total = len(private_test_cases)
        
        all_results = []
        total_execution_time = 0
        
        # Run public tests
        for i, test_assertion in enumerate(public_test_cases):
            passed, exec_time, error = run_test_case(code, test_assertion, test_setup_code)
            total_execution_time += exec_time
            
            if passed:
                public_passed += 1
                
            all_results.append({
                "test": f"public_{i}",
                "type": "public",
                "status": "passed" if passed else "failed",
                "assertion": test_assertion,
                "time": exec_time,
                "error": error
            })
        
        # Run private/challenge tests (if any exist)
        for i, test_assertion in enumerate(private_test_cases):
            passed, exec_time, error = run_test_case(code, test_assertion, test_setup_code)
            total_execution_time += exec_time
            
            if passed:
                private_passed += 1
                
            all_results.append({
                "test": f"private_{i}",
                "type": "private",
                "status": "passed" if passed else "failed",
                "time": exec_time,
                "error": error
            })
        
        # Calculate metrics
        # If no private tests exist, use only public tests for scoring
        total_tests = public_total + private_total
        total_passed = public_passed + private_passed
        
        # Calculate pass rates
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        public_pass_rate = public_passed / public_total if public_total > 0 else 0.0
        private_pass_rate = private_passed / private_total if private_total > 0 else 1.0  # Default to 1.0 if no private tests
        
        # Calculate execution time score
        avg_time = total_execution_time / total_tests if total_tests > 0 else 10.0
        
        # Combined score: if no private tests, use public pass rate
        # Otherwise use overall pass rate
        if private_total == 0:
            combined_score = public_pass_rate
        else:
            combined_score = overall_pass_rate
        
        # Print summary
        print(f"\n=== Test Summary ===")
        print(f"Public: {public_passed}/{public_total} passed ({public_pass_rate:.1%})")
        if private_total > 0:
            print(f"Private: {private_passed}/{private_total} passed ({private_pass_rate:.1%})")
        else:
            print(f"Private: No challenge tests available")
        print(f"Overall: {total_passed}/{total_tests} passed ({overall_pass_rate:.1%})")
        print(f"Combined Score: {combined_score:.3f}")
        print("=" * 20)
        
        return {
            "pass_rate": overall_pass_rate,
            "combined_score": combined_score,
            "public_pass_rate": public_pass_rate,
            "private_pass_rate": private_pass_rate,
            "public_passed": public_passed,
            "public_total": public_total,
            "private_passed": private_passed,
            "private_total": private_total,
            "total_passed": total_passed,
            "total_tests": total_tests,
            "avg_execution_time": avg_time,
            "test_results": all_results
        }
        
    except Exception as e:
        return {
            "pass_rate": 0.0,
            "combined_score": 0.0,
            "public_pass_rate": 0.0,
            "private_pass_rate": 0.0,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

#!/usr/bin/env python3
"""
LiveCodeBench Evaluator for OpenEvolve

This file contains the evaluate() function that OpenEvolve calls to evaluate code.
It evaluates on both public and private test cases, showing details for debugging.
"""

import json
import sys
import os
import ast
import subprocess
import tempfile
import time
import traceback
import base64
import zlib
import pickle


def find_sample_json(code_file_path):
    """Find sample.json by walking up the directory tree"""
    cur = os.path.dirname(os.path.abspath(code_file_path))
    while cur != os.path.dirname(cur):
        candidate = os.path.join(cur, "sample.json")
        if os.path.exists(candidate):
            return candidate
        cur = os.path.dirname(cur)
    raise FileNotFoundError("Could not locate sample.json above " + code_file_path)


def decode_private_test_cases(encoded_string):
    """
    Decode private test cases from base64 -> zlib -> pickle format
    
    Args:
        encoded_string: Base64 encoded string containing compressed test cases
        
    Returns:
        List of test case dictionaries
    """
    try:
        # Step 1: Base64 decode
        decoded_bytes = base64.b64decode(encoded_string.encode('utf-8'))
        
        # Step 2: Zlib decompress
        decompressed_bytes = zlib.decompress(decoded_bytes)
        
        # Step 3: Pickle loads
        test_cases = pickle.loads(decompressed_bytes)
        
        # The result should be a list of test cases
        if isinstance(test_cases, list):
            return test_cases
        elif isinstance(test_cases, str):
            # Sometimes it might be a JSON string
            return json.loads(test_cases)
        else:
            return []
            
    except Exception as e:
        print(f"Error decoding private test cases: {e}")
        traceback.print_exc()
        return []


def run_test_case(code, test_input, expected_output, timeout=10):
    """
    Run a single test case and return the result
    
    Returns:
        tuple: (passed: bool, actual_output: str, execution_time: float, error: str or None)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Write the test harness
        f.write("import sys\n")
        f.write("from io import StringIO\n\n")
        f.write("# Redirect stdin\n")
        f.write(f"sys.stdin = StringIO({repr(test_input)})\n\n")
        f.write("# User code\n")
        f.write(code)
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
        
        actual_output = result.stdout.strip()
        
        # Check both stdout and stderr for common errors
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            return False, actual_output, execution_time, f"Runtime error: {error_msg}"
        
        passed = actual_output == expected_output.strip()
        return passed, actual_output, execution_time, None
        
    except subprocess.TimeoutExpired:
        return False, "", timeout, "Timeout"
    except Exception as e:
        return False, "", 0, str(e)
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
        
        # Parse public test cases
        public_test_cases = []
        if "public_test_cases" in sample_data and sample_data["public_test_cases"]:
            try:
                # Public test cases are usually provided as a JSON string
                public_test_cases = json.loads(sample_data["public_test_cases"])
            except:
                try:
                    # Fallback to ast.literal_eval
                    public_test_cases = ast.literal_eval(sample_data["public_test_cases"])
                except:
                    pass
        
        # private test cases (compressed format)
        private_test_cases = []
        if "private_test_cases" in sample_data and sample_data["private_test_cases"]:
            if isinstance(sample_data["private_test_cases"], str):
                private_test_cases = decode_private_test_cases(sample_data["private_test_cases"])
            elif isinstance(sample_data["private_test_cases"], list):
                # Already decoded
                private_test_cases = sample_data["private_test_cases"]
        
        # Run all test cases
        public_passed = 0
        public_total = len(public_test_cases)
        private_passed = 0
        private_total = len(private_test_cases)
        
        all_results = []
        total_execution_time = 0
        
        # Run public tests
        for i, test_case in enumerate(public_test_cases):
            test_input = test_case.get("input", "").strip()
            expected_output = test_case.get("output", "")
            
            passed, actual_output, exec_time, error = run_test_case(code, test_input, expected_output)
            total_execution_time += exec_time
            
            if passed:
                public_passed += 1
                
            all_results.append({
                "test": f"public_{i}",
                "type": "public",
                "status": "passed" if passed else "failed",
                "expected": expected_output.strip(),
                "actual": actual_output,
                "time": exec_time,
                "error": error
            })
        
        # Run private tests - NOW WITH FULL VISIBILITY
        for i, test_case in enumerate(private_test_cases):
            # Handle both dict and object formats
            if isinstance(test_case, dict):
                test_input = test_case.get("input", "").strip()
                expected_output = test_case.get("output", "")
            else:
                # Handle object with attributes
                test_input = getattr(test_case, "input", "").strip()
                expected_output = getattr(test_case, "output", "")
            
            passed, actual_output, exec_time, error = run_test_case(code, test_input, expected_output)
            total_execution_time += exec_time
            
            if passed:
                private_passed += 1
                
            # SHOW FULL DETAILS FOR PRIVATE TESTS
            all_results.append({
                "test": f"private_{i}",
                "type": "private",
                "status": "passed" if passed else "failed",
                "time": exec_time,
                "error": error
            })
            
            # # Print debug info for failed private tests
            # if not passed:
            #     print(f"\n=== Failed Private Test {i} ===")
            #     print(f"Input: {repr(test_input)}")
            #     print(f"Expected: {repr(expected_output.strip())}")
            #     print(f"Actual: {repr(actual_output)}")
            #     if error:
            #         print(f"Error: {error}")
            #     print("=" * 30)
        
        # Calculate metrics
        total_tests = public_total + private_total
        total_passed = public_passed + private_passed
        
        # Calculate pass rates
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
        public_pass_rate = public_passed / public_total if public_total > 0 else 0.0
        private_pass_rate = private_passed / private_total if private_total > 0 else 0.0
        
        # Calculate execution time score
        avg_time = total_execution_time / total_tests if total_tests > 0 else 10.0
        
        # Calculate combined score with emphasis on private tests
        combined_score = overall_pass_rate
        
        # Print summary
        print(f"\n=== Test Summary ===")
        print(f"Public: {public_passed}/{public_total} passed ({public_pass_rate:.1%})")
        print(f"Private: {private_passed}/{private_total} passed ({private_pass_rate:.1%})")
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

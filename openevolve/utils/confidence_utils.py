import numpy as np
import re

def compute_confidence(logprobs_data):
    confs = []
    for token_data in logprobs_data:
        all_logprobs = []
        if getattr(token_data, "logprob", None) is not None:
            all_logprobs.append(token_data.logprob)
        if getattr(token_data, "top_logprobs", None):
            for top_lp in token_data.top_logprobs:
                if getattr(top_lp, "logprob", None) is not None:
                    all_logprobs.append(top_lp.logprob)
        if all_logprobs:
            mean_logprob = np.mean(all_logprobs)
            confs.append(round(-mean_logprob, 3))
        else:
            confs.append(0.0)
    return confs

def compute_least_grouped(confs, group_size=1024):
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]
    sliding = [
        round(sum(confs[i:i+group_size]) / group_size, 3)
        for i in range(len(confs) - group_size + 1)
    ]
    return sliding

def compute_tail_confidence(confs, tail_tokens=2048):
    if not confs:
        return 0.0
    tail = confs[-tail_tokens:] if len(confs) > tail_tokens else confs
    return float(np.mean(tail))

def compute_bottom_window_confidence(confs, window_size: int = 2048, bottom_percent: float = 0.1) -> float:
    """Calculate mean confidence from sliding windows, return average of bottom percentile"""
    try:
        if len(confs) < window_size:
            return np.mean(confs)
            
        window_means = []
        current_sum = sum(confs[:window_size])
        window_means.append(current_sum / window_size)
        
        for i in range(1, len(confs) - window_size + 1):
            current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
            window_means.append(current_sum / window_size)
        
        if not window_means:
            return 0.0
        
        if bottom_percent == -1:  # Min window
            return min(window_means)
        
        num_bottom = max(1, int(len(window_means) * bottom_percent))
        if num_bottom == 1:
            return min(window_means)
        else:
            bottom_means = np.partition(window_means, num_bottom-1)[:num_bottom]
            return np.mean(bottom_means)
        
    except Exception:
        return 0.0

def get_generation_confidence(choice, group_size=1024):
    try:
        if not getattr(choice, "logprobs", None) or not getattr(choice.logprobs, "content", None):
            print("Logprobs not found")
            return {"mean_confidence": 0, "least_grouped_confidence": 0, "tail_confidence": 0, "bottom_window_confidence": 0}

        logprobs_data = choice.logprobs.content
        confs = compute_confidence(logprobs_data)
        if not confs:
            return {"mean_confidence": 0, "least_grouped_confidence": 0, "tail_confidence": 0, "bottom_window_confidence": 0}

        grouped = compute_least_grouped(confs, group_size=group_size)
        tail = compute_tail_confidence(confs, tail_tokens=2048)
        bottom_10 = compute_bottom_window_confidence(confs, window_size=2048)

        return {
            "mean_confidence": round(sum(confs) / len(confs), 3),
            "least_grouped_confidence": min(grouped) if grouped else 0,
            "tail_confidence": tail,
            "bottom_window_confidence": bottom_10,
        }
    except Exception as e:
        print(f"[confidence_utils] Error computing confidence: {e}")
        return {"mean_confidence": 0, "least_grouped_confidence": 0, "tail_confidence": 0, "bottom_window_confidence": 0}

        
def extract_verbalized_confidence(response: str):
    """Extract verbalized confidence score from generated code using <confidence> tags"""
    try:
        # First, try to extract content within <confidence> tags
        confidence_tag_pattern = r'<confidence>(.*?)</confidence>'
        confidence_match = re.search(confidence_tag_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if confidence_match:
            confidence_content = confidence_match.group(1)
            
            # Look for decimal confidence scores (0.0-10.0 scale)
            decimal_patterns = [
                r'(\d+\.\d+)',  # Match any decimal number
                r'confidence.*?(\d+\.\d+)',
                r'score.*?(\d+\.\d+)',
                r'rating.*?(\d+\.\d+)',
            ]
            
            for pattern in decimal_patterns:
                matches = re.findall(pattern, confidence_content, re.IGNORECASE)
                if matches:
                    # Take the first valid decimal score
                    for match in matches:
                        score = float(match)
                        # Ensure it's within the 0.0-10.0 range
                        if 0.0 <= score <= 10.0:
                            return score
        
        # Fallback: Look for confidence patterns anywhere in the response
        fallback_patterns = [
            r'<confidence>\s*(\d+\.\d+)',
            r'confidence.*?(\d+\.\d+)',
            r'Confidence:\s*(\d+\.\d+)',
            r'My confidence.*?(\d+\.\d+)',
            r'I rate.*?(\d+\.\d+)',
            r'score.*?(\d+\.\d+)',
            # Also handle integer scores that might be on 0-10 scale
            r'<confidence>\s*(\d+)',
            r'Confidence:\s*(\d+)',
        ]
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Take the last match (most recent confidence statement)
                try:
                    confidence_score = float(matches[-1])
                    # If it's an integer between 0-10, treat as decimal scale
                    if confidence_score <= 10:
                        return max(0.0, min(10.0, confidence_score))
                    # If it's a larger number, assume it's on 0-100 scale and convert
                    elif confidence_score <= 100:
                        return max(0.0, min(10.0, confidence_score / 10.0))
                except ValueError:
                    continue
        
        # Legacy patterns for backward compatibility (0-100 scale)
        legacy_patterns = [
            r'Confidence:\s*(\d+)/100',
            r'Confidence:\s*(\d+)%',
            r'confidence:\s*(\d+)/100',
            r'confidence:\s*(\d+)%',
            r'My confidence.*?(\d+)/100',
            r'My confidence.*?(\d+)%',
            r'I rate.*?confidence.*?(\d+)',
            r'confidence.*?(\d+)\s*out\s*of\s*100',
        ]
        
        for pattern in legacy_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                confidence_score = int(matches[-1])
                # Convert from 0-100 scale to 0-10 scale
                return max(0.0, min(10.0, confidence_score / 10.0))
        
        return None
        
    except Exception as e:
        print(f"Error extracting verbalized confidence: {e}")
        return None


# def extract_confidence_details(response: str):
#     """Extract detailed confidence information including reasoning"""
#     try:
#         confidence_tag_pattern = r'<confidence>(.*?)</confidence>'
#         confidence_match = re.search(confidence_tag_pattern, response, re.DOTALL | re.IGNORECASE)
        
#         if not confidence_match:
#             return None
            
#         confidence_content = confidence_match.group(1)
        
#         # Extract the numerical score
#         score = extract_verbalized_confidence(response)
        
#         # Extract reasoning components
#         reasoning_patterns = {
#             'correctness': r'Correctness:\s*(.*?)(?=\n|$)',
#             'efficiency': r'Efficiency:\s*(.*?)(?=\n|$)',
#             'edge_cases': r'Edge Cases:\s*(.*?)(?=\n|$)',
#             'code_quality': r'Code Quality:\s*(.*?)(?=\n|$)',
#             'testing': r'Testing:\s*(.*?)(?=\n|$)',
#         }
        
#         reasoning = {}
#         for key, pattern in reasoning_patterns.items():
#             match = re.search(pattern, confidence_content, re.IGNORECASE)
#             if match:
#                 reasoning[key] = match.group(1).strip()
        
#         # Extract strengths and concerns
#         strengths_match = re.search(r'Key strengths:\s*(.*?)(?=Potential concerns:|</confidence>|$)', 
#                                   confidence_content, re.DOTALL | re.IGNORECASE)
#         concerns_match = re.search(r'Potential concerns:\s*(.*?)(?=</confidence>|$)', 
#                                  confidence_content, re.DOTALL | re.IGNORECASE)
        
#         return {
#             'score': score,
#             'reasoning': reasoning,
#             'strengths': strengths_match.group(1).strip() if strengths_match else None,
#             'concerns': concerns_match.group(1).strip() if concerns_match else None,
#             'raw_content': confidence_content.strip()
#         }
        
#     except Exception as e:
#         print(f"Error extracting confidence details: {e}")
#         return None


def extract_confidence_from_program_file(program_path: str):
    """Extract verbalized confidence from a program file"""
    try:
        if not os.path.exists(program_path):
            return None
            
        with open(program_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return extract_verbalized_confidence(content)
    except Exception as e:
        print(f"Error reading program file for confidence extraction: {e}")
        return None
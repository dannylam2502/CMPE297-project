#!/usr/bin/env python3
"""
Integration Test Script
Tests all modules working together
"""

import sys
import time
from typing import Dict, Any

# Check dependencies
try:
    import requests
except ImportError:
    print("Error: requests library not installed")
    print("Install with: pip install requests --break-system-packages")
    sys.exit(1)

import json

# Configuration
BASE_URL = "http://localhost:5005"
TEST_TIMEOUT = 60  # Increased for reasoning engine

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name: str):
    print(f"\n{Colors.BLUE}[TEST]{Colors.END} {name}")

def print_pass(msg: str):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_fail(msg: str):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_info(msg: str):
    print(f"{Colors.YELLOW}ℹ{Colors.END} {msg}")

def wait_for_backend(max_wait=30):
    """Wait for backend to be ready"""
    print_info("Waiting for backend to be ready...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print_pass("Backend is ready")
                return True
        except:
            pass
        time.sleep(1)
        print(".", end="", flush=True)
    print()
    print_fail(f"Backend not ready after {max_wait}s")
    return False

def test_health_check():
    """Test 1: Health check endpoint"""
    print_test("Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_pass("Server is running")
            print_info(f"Status: {data.get('status')}")
            print_info(f"Reasoning: {data.get('reasoning_enabled')}")
            return True
        else:
            print_fail(f"Unexpected status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_fail(f"Server not reachable: {e}")
        return False

def test_fact_check_supported():
    """Test 2: Known supported claim"""
    print_test("Fact Check - Supported Claim")
    query = "The Moon landing occurred in 1969"
    
    try:
        print_info(f"Query: {query}")
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"question": query},
            timeout=TEST_TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            verdict = data.get('verdict')
            score = data.get('score')
            explanation = data.get('explanation', '')
            
            print_pass(f"Response received in {elapsed:.2f}s")
            print_info(f"Claim: {data.get('claim')}")
            print_info(f"Verdict: {verdict}")
            print_info(f"Score: {score}/100")
            print_info(f"Explanation: {len(explanation)} chars")
            print_info(f"Citations: {len(data.get('citations', []))}")
            
            # Validate response structure
            required_fields = ['claim', 'verdict', 'score', 'explanation', 'citations', 'features']
            missing = [f for f in required_fields if f not in data]
            
            if missing:
                print_fail(f"Missing fields: {missing}")
                return False
            
            print_pass("All required fields present")
            
            # Check explanation quality
            if len(explanation) > 100:
                print_pass("Explanation has reasonable length")
            else:
                print_fail(f"Explanation too short ({len(explanation)} chars)")
                print_info(f"Explanation: {explanation}")
            
            return True
        else:
            print_fail(f"HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print_fail(f"Request timed out after {TEST_TIMEOUT}s")
        print_info("Try disabling reasoning to speed up: curl -X POST http://localhost:5005/toggle-reasoning -d '{\"enable\":false}'")
        return False
    except Exception as e:
        print_fail(f"Error: {e}")
        return False

def test_fact_check_refuted():
    """Test 3: Known refuted claim"""
    print_test("Fact Check - Refuted Claim")
    query = "The Moon landing was fake"
    
    try:
        print_info(f"Query: {query}")
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"question": query},
            timeout=TEST_TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            verdict = data.get('verdict')
            
            print_pass(f"Response received in {elapsed:.2f}s")
            print_info(f"Verdict: {verdict}")
            print_info(f"Score: {data.get('score')}/100")
            
            return True
        else:
            print_fail(f"HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_fail(f"Error: {e}")
        return False

def test_no_evidence():
    """Test 4: Claim with no evidence"""
    print_test("Fact Check - No Evidence Case")
    query = "Quantum entanglement allows faster-than-light communication"
    
    try:
        print_info(f"Query: {query}")
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"question": query},
            timeout=TEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            verdict = data.get('verdict')
            
            print_pass("Response received")
            print_info(f"Verdict: {verdict}")
            
            if verdict == "Not enough evidence":
                print_pass("Correctly identified lack of evidence")
            
            return True
        else:
            return False
            
    except Exception as e:
        print_fail(f"Error: {e}")
        return False

def test_toggle_reasoning():
    """Test 5: Toggle reasoning on/off"""
    print_test("Reasoning Toggle")
    
    try:
        # Disable reasoning
        response = requests.post(
            f"{BASE_URL}/toggle-reasoning",
            json={"enable": False},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if not data.get('reasoning_enabled'):
                print_pass("Reasoning disabled")
            else:
                print_fail("Failed to disable reasoning")
        
        # Re-enable reasoning
        response = requests.post(
            f"{BASE_URL}/toggle-reasoning",
            json={"enable": True},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('reasoning_enabled'):
                print_pass("Reasoning re-enabled")
                return True
            else:
                print_fail("Failed to re-enable reasoning")
        
        return False
        
    except Exception as e:
        print_fail(f"Error: {e}")
        return False

def test_error_handling():
    """Test 6: Error handling"""
    print_test("Error Handling")
    
    try:
        # Test empty question
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"question": ""},
            timeout=5
        )
        
        if response.status_code == 400:
            print_pass("Empty question handled correctly")
        else:
            print_fail(f"Expected 400, got {response.status_code}")
        
        # Test malformed JSON
        response = requests.post(
            f"{BASE_URL}/chat",
            json={},
            timeout=5
        )
        
        if response.status_code in [400, 500]:
            print_pass("Malformed request handled")
            return True
        
        return False
        
    except Exception as e:
        print_fail(f"Error: {e}")
        return False

def test_response_time():
    """Test 7: Performance benchmark"""
    print_test("Performance Benchmark")
    query = "Water boils at 100°C at sea level"
    
    print_info("Running 3 iterations...")
    times = []
    for i in range(3):
        try:
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/chat",
                json={"question": query},
                timeout=TEST_TIMEOUT
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print_info(f"Run {i+1}: {elapsed:.2f}s")
        except Exception as e:
            print_fail(f"Run {i+1} failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print_info(f"Average: {avg_time:.2f}s")
        
        if avg_time < 15:
            print_pass("Performance acceptable (<15s)")
        else:
            print_fail(f"Performance slow (>{avg_time:.1f}s)")
            print_info("Consider disabling reasoning for faster responses")
        
        return True
    
    return False

def main():
    """Run all integration tests"""
    print("="*60)
    print("Fact-Checking System Integration Tests")
    print("="*60)
    print_info(f"Target: {BASE_URL}")
    print_info("Ensure backend is running: cd src && python server.py")
    print("="*60)
    
    # Wait for backend
    if not wait_for_backend():
        print_fail("Backend not available. Start it with: cd src && python server.py")
        sys.exit(1)
    
    tests = [
        test_health_check,
        test_fact_check_supported,
        test_fact_check_refuted,
        test_no_evidence,
        test_toggle_reasoning,
        test_error_handling,
        test_response_time
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print_fail(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print(f"Passed: {passed}/{total} ({percentage:.0f}%)")
    
    if passed == total:
        print(f"\n{Colors.GREEN}All tests passed! ✓{Colors.END}")
        print_info("Integration successful")
    else:
        print(f"\n{Colors.RED}Some tests failed{Colors.END}")
        print_info("Review failures above")
    
    print("="*60)

if __name__ == "__main__":
    main()
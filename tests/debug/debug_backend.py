#!/usr/bin/env python3
"""
Debug script to test backend response and diagnose issues
"""

import sys
import time

try:
    import requests
except ImportError:
    print("Error: requests library not installed")
    print("Install with: pip install requests --break-system-packages")
    sys.exit(1)

import json

BASE_URL = "http://localhost:5005"

def check_health():
    """Check if backend is running"""
    print("="*60)
    print("1. HEALTH CHECK")
    print("="*60)
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        data = resp.json()
        print(f"✓ Health check passed")
        print(f"  Status: {data.get('status')}")
        print(f"  Service: {data.get('service')}")
        print(f"  Reasoning: {data.get('reasoning_enabled')}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to {BASE_URL}")
        print("\nBackend not running. Start it with:")
        print("  cd src && python server.py")
        return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_factcheck():
    """Test fact-check endpoint"""
    print("\n" + "="*60)
    print("2. FACT-CHECK TEST")
    print("="*60)
    
    query = "The Moon landing happened in 1969"
    print(f"Query: {query}\n")
    
    start = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"question": query},
            timeout=60
        )
        elapsed = time.time() - start
        
        print(f"✓ Response received in {elapsed:.2f}s")
        print(f"Status code: {resp.status_code}\n")
        
        if resp.status_code != 200:
            print("✗ Error response:")
            print(resp.text)
            return False
        
        data = resp.json()
        
        # Response structure
        print("-" * 60)
        print("RESPONSE STRUCTURE")
        print("-" * 60)
        print(json.dumps(data, indent=2, default=str))
        
        # Field validation
        print("\n" + "-" * 60)
        print("FIELD VALIDATION")
        print("-" * 60)
        
        fields = {
            'claim': (str, None),
            'verdict': (str, ['Supported', 'Refuted', 'Contested', 'Not enough evidence']),
            'score': (int, range(0, 101)),
            'explanation': (str, None),
            'citations': (list, None),
            'features': (dict, None)
        }
        
        all_valid = True
        for field, (expected_type, valid_values) in fields.items():
            value = data.get(field)
            type_ok = isinstance(value, expected_type)
            
            status = "✓" if type_ok else "✗"
            print(f"{status} {field}: {type(value).__name__}", end="")
            
            if not type_ok:
                print(f" (expected {expected_type.__name__})")
                all_valid = False
                continue
            
            if field == 'claim':
                print(f" ({len(value)} chars)")
            elif field == 'verdict':
                valid = value in valid_values if valid_values else True
                if not valid:
                    print(f" - Invalid value: {value}")
                    all_valid = False
                else:
                    print(f" = '{value}'")
            elif field == 'score':
                valid = value in valid_values if valid_values else True
                if not valid:
                    print(f" - Out of range: {value}")
                    all_valid = False
                else:
                    print(f" = {value}/100")
            elif field == 'explanation':
                length = len(value)
                print(f" ({length} chars)")
                if length < 50:
                    print(f"  ⚠ Explanation is very short")
                    print(f"  Content: {value}")
            elif field == 'citations':
                print(f" ({len(value)} items)")
                if value:
                    print(f"  First citation: {value[0].get('title', 'N/A')[:50]}...")
            elif field == 'features':
                print(f" (keys: {list(value.keys())})")
        
        # Explanation preview
        if data.get('explanation') and len(data['explanation']) >= 50:
            print("\n" + "-" * 60)
            print("EXPLANATION PREVIEW")
            print("-" * 60)
            print(data['explanation'][:300])
            if len(data['explanation']) > 300:
                print("...")
        
        # Citations
        if data.get('citations'):
            print("\n" + "-" * 60)
            print(f"CITATIONS ({len(data['citations'])})")
            print("-" * 60)
            for i, cite in enumerate(data['citations'], 1):
                print(f"{i}. {cite.get('title', 'N/A')}")
                print(f"   {cite.get('url', 'N/A')}")
                print(f"   {cite.get('snippet', 'N/A')[:100]}...")
                print()
        
        return all_valid
        
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"✗ Request timed out after {elapsed:.2f}s")
        print("\nDiagnostic suggestions:")
        print("1. Check if reasoning is enabled (makes responses slower)")
        print("2. Disable reasoning: curl -X POST http://localhost:5005/toggle-reasoning \\")
        print("   -H 'Content-Type: application/json' -d '{\"enable\":false}'")
        print("3. Check server logs for errors")
        print("4. Verify OpenAI API key in .env file")
        return False
    except Exception as e:
        print(f"✗ Request failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_knowledge_base():
    """Check if knowledge base has data"""
    print("\n" + "="*60)
    print("3. KNOWLEDGE BASE CHECK")
    print("="*60)
    
    import os
    data_paths = [
        "data/mock.json",
        "data/fever.json",
        "../data/mock.json",
        "../data/fever.json"
    ]
    
    found = False
    for path in data_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                    print(f"✓ Found: {path}")
                    print(f"  Entries: {len(data)}")
                    if data:
                        print(f"  Sample claim: {data[0].get('claim', 'N/A')[:60]}...")
                    found = True
                    break
            except Exception as e:
                print(f"✗ Error reading {path}: {e}")
    
    if not found:
        print("✗ No knowledge base found")
        print("\nTo load data:")
        print("  python load_fever.py")
        print("or check data/mock.json exists")

def main():
    print("Backend Debug Script")
    print("="*60)
    print(f"Target: {BASE_URL}")
    print("="*60)
    
    # Run checks
    health_ok = check_health()
    if not health_ok:
        return
    
    factcheck_ok = test_factcheck()
    diagnose_knowledge_base()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if health_ok and factcheck_ok:
        print("✓ Backend is working correctly")
    else:
        print("✗ Issues detected - see above")
    print("="*60)

if __name__ == "__main__":
    main()
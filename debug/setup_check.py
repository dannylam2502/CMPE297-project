#!/usr/bin/env python3
"""
Setup script for CMPE297 Fact-Checking System Integration
Checks dependencies and creates necessary files
"""

import os
import sys
import json
from pathlib import Path

def check_python_version():
    """Ensure Python 3.10+"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        'qdrant_client',
        'sentence_transformers',
        'torch',
        'openai',
        'flask',
        'flask_cors',
        'ollama',
        'pytest'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg}")
            missing.append(pkg)
    
    if missing:
        print("\n⚠️  Missing packages. Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    return True

def check_ollama():
    """Check if Ollama is available"""
    try:
        import ollama
        models = ollama.list()
        print(f"✅ Ollama running with {len(models)} models")
        
        # Check for llama3.1
        has_llama = any('llama3.1' in str(m) for m in models)
        if not has_llama:
            print("⚠️  llama3.1 not found. Run: ollama pull llama3.1")
        return True
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("   Install from: https://ollama.ai/")
        return False

def check_openai_key():
    """Check if OpenAI API key is set"""
    key = os.environ.get('OPENAI_API_KEY')
    if key:
        print(f"✅ OPENAI_API_KEY set ({key[:10]}...)")
        return True
    else:
        print("❌ OPENAI_API_KEY not set")
        print("   export OPENAI_API_KEY='sk-...'")
        return False

def create_directory_structure():
    """Create necessary directories"""
    dirs = [
        'data',
        'src/modules/input_extraction',
        'src/modules/misinformation_module',
        'src/modules/claim_extraction',
        'src/modules/llm'
    ]
    
    for d in dirs:
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        init_file = path / '__init__.py'
        if not init_file.exists():
            init_file.touch()
        print(f"✅ {d}")
    
    return True

def create_mock_data():
    """Create minimal mock.json if it doesn't exist"""
    data_file = Path('data/mock.json')
    
    if data_file.exists():
        print(f"✅ data/mock.json exists")
        return True
    
    mock_data = [
        {
            "id": 1,
            "claim": "The Moon landing occurred in 1969",
            "source": "https://nasa.gov",
            "confidence": 1.0
        },
        {
            "id": 2,
            "claim": "Water boils at 100 degrees Celsius at sea level",
            "source": "https://physics.edu",
            "confidence": 1.0
        },
        {
            "id": 3,
            "claim": "The Earth is approximately 4.5 billion years old",
            "source": "https://science.org",
            "confidence": 1.0
        },
        {
            "id": 4,
            "claim": "The speed of light is approximately 299,792 kilometers per second",
            "source": "https://physics.org",
            "confidence": 1.0
        },
        {
            "id": 5,
            "claim": "Humans have 46 chromosomes in most cells",
            "source": "https://biology.edu",
            "confidence": 1.0
        }
    ]
    
    data_file.write_text(json.dumps(mock_data, indent=2))
    print(f"✅ Created data/mock.json with {len(mock_data)} entries")
    return True

def check_module_files():
    """Check if all module files exist"""
    required_files = {
        'src/pipeline.py': 'Integration pipeline',
        'src/server.py': 'Flask backend server',
        'src/modules/input_extraction/input_extractor.py': 'Danny\'s claim extractor',
        'src/modules/misinformation_module/src/qdrant_db.py': 'Adam\'s vector DB',
        'src/modules/misinformation_module/src/embedder.py': 'Adam\'s embedder',
        'src/modules/claim_extraction/fact_validator.py': 'Sam\'s fact validator',
        'src/modules/claim_extraction/fact_validator_interface.py': 'Sam\'s interfaces',
        'src/modules/llm/llm_ollama.py': 'Ollama LLM implementation',
        'src/modules/llm/llm_engine_interface.py': 'LLM interface',
    }
    
    all_exist = True
    for file, desc in required_files.items():
        path = Path(file)
        if path.exists():
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc} ({file})")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 60)
    print("CMPE297 Fact-Checking System - Setup Check")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Ollama", check_ollama),
        ("OpenAI API Key", check_openai_key),
        ("Directory Structure", create_directory_structure),
        ("Mock Data", create_mock_data),
        ("Module Files", check_module_files),
    ]
    
    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Setup Check: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All checks passed! Ready to run.")
        print("\nNext steps:")
        print("  1. python src/pipeline.py          # Test pipeline")
        print("  2. python src/server.py            # Start backend")
        print("  3. cd src/frontEnd && npm start    # Start frontend")
    else:
        print("⚠️  Some checks failed. Fix issues above before running.")
        sys.exit(1)

if __name__ == '__main__':
    main()

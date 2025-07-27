"""
Test file to verify all imports work correctly
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    results = []
    
    try:
        # Test ML config
        from ml.config.settings import MLConfig
        print("✅ MLConfig imported successfully")
        results.append(("MLConfig", True))
    except ImportError as e:
        print(f"❌ MLConfig import failed: {e}")
        results.append(("MLConfig", False))
    
    try:
        # Test models
        from ml.models.compliance_checker import AIComplianceChecker
        print("✅ ComplianceChecker imported successfully")
        results.append(("ComplianceChecker", True))
    except ImportError as e:
        print(f"❌ ComplianceChecker import failed: {e}")
        results.append(("ComplianceChecker", False))
    
    try:
        from ml.models.policy_comparator import AIPolicyComparator
        print("✅ PolicyComparator imported successfully")
        results.append(("PolicyComparator", True))
    except ImportError as e:
        print(f"❌ PolicyComparator import failed: {e}")
        results.append(("PolicyComparator", False))
    
    try:
        from ml.models.principle_assessor import PrincipleBasedAIAssessment
        print("✅ PrincipleAssessor imported successfully")
        results.append(("PrincipleAssessor", True))
    except ImportError as e:
        print(f"❌ PrincipleAssessor import failed: {e}")
        results.append(("PrincipleAssessor", False))
    
    try:
        # Test embeddings
        from ml.embeddings.embedding_system import RAGEmbeddingSystem
        print("✅ EmbeddingSystem imported successfully")
        results.append(("EmbeddingSystem", True))
    except ImportError as e:
        print(f"❌ EmbeddingSystem import failed: {e}")
        results.append(("EmbeddingSystem", False))
    
    try:
        from ml.embeddings.text_extractor import AdvancedTextExtractor
        print("✅ TextExtractor imported successfully")
        results.append(("TextExtractor", True))
    except ImportError as e:
        print(f"❌ TextExtractor import failed: {e}")
        results.append(("TextExtractor", False))
    
    try:
        # Test utils
        from ml.utils.db_connection import search_global_chunks
        print("✅ DB connection utilities imported successfully")
        results.append(("DBConnection", True))
    except ImportError as e:
        print(f"❌ DB connection utilities import failed: {e}")
        results.append(("DBConnection", False))
    
    # Summary
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n📊 Import Test Summary:")
    print(f"   ✅ Successful: {successful}/{total}")
    print(f"   ❌ Failed: {total - successful}/{total}")
    
    if successful == total:
        print("🎉 All imports successful!")
        return True
    else:
        print("⚠️ Some imports failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    test_imports()

#!/usr/bin/env python3
"""
GovernAIce - AI Policy Compliance Assessment Tool
================================================

Main entry point for the AI policy compliance assessment system.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point for the application"""
    print("GovernAIce - AI Policy Compliance Assessment Tool")
    print("=" * 50)
    print("Available modules:")
    print("1. Compliance Checker - MongoDB-based compliance analysis")
    print("2. Policy Comparator - Framework comparison tool")
    print("3. Principle Assessor - Principle-based assessment")
    
    choice = input("\nSelect module (1-3): ").strip()
    
    try:
        if choice == "1":
            from ml.models.compliance_checker import main as compliance_main
            compliance_main()
        elif choice == "2":
            from ml.models.policy_comparator import main as comparator_main
            comparator_main()
        elif choice == "3":
            from ml.models.principle_assessor import main as assessor_main
            assessor_main()
        else:
            print("Invalid choice")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

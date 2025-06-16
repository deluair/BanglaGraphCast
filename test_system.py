#!/usr/bin/env python3
"""
Simple test script to verify the BanglaGraphCast system works
"""

import sys
import traceback

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy not available: {e}")
        return False
    
    # Test our modules
    try:
        from configs.bangladesh_config import BangladeshConfig
        print("✓ Configuration module")
    except ImportError as e:
        print(f"✗ Configuration module failed: {e}")
        return False
    
    try:
        from models.core.graphcast_bangladesh import BangladeshGraphCast
        print("✓ Core GraphCast model")
    except ImportError as e:
        print(f"✗ Core GraphCast model failed: {e}")
        return False
    
    try:
        from training.losses.bangladesh_loss import BangladeshLoss
        print("✓ Loss functions")
    except ImportError as e:
        print(f"✗ Loss functions failed: {e}")
        return False
    
    return True

def test_training_system():
    """Test the training system initialization"""
    print("\nTesting training system...")
    
    try:
        from train import IntegratedTrainingSystem
        from configs.bangladesh_config import BangladeshConfig
        
        # Create a minimal config for testing
        config = BangladeshConfig()
        
        # Initialize training system
        training_system = IntegratedTrainingSystem(config)
        print("✓ Training system initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Training system failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("BanglaGraphCast System Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return 1
    
    # Test training system
    if not test_training_system():
        print("\n❌ Training system test failed")
        return 1
    
    print("\n✅ All tests passed! System is ready.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

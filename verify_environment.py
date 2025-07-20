#!/usr/bin/env python3
"""
Environment verification script for OCEAN implementation.
Checks if all required dependencies are properly installed and working.
"""

import sys
import importlib
import subprocess
from typing import List, Tuple, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.9+) ✗"


def check_package_import(package_name: str, import_name: str = None) -> Tuple[bool, str, str]:
    """Check if a package can be imported and get its version."""
    if import_name is None:
        import_name = package_name
        
    try:
        module = importlib.import_module(import_name)
        
        # Try to get version
        version = "unknown"
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                break
                
        return True, f"{package_name} {version} ✓", ""
    except ImportError as e:
        return False, f"{package_name} ✗", str(e)


def check_torch_features() -> Dict[str, Tuple[bool, str]]:
    """Check PyTorch specific features."""
    results = {}
    
    try:
        import torch
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            results["CUDA"] = (True, f"CUDA available with {device_count} device(s): {device_name} ✓")
        else:
            results["CUDA"] = (False, "CUDA not available ✗")
        
        # Check MPS availability (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            results["MPS"] = (mps_available, f"MPS {'available' if mps_available else 'not available'} {'✓' if mps_available else '✗'}")
        else:
            results["MPS"] = (False, "MPS not supported ✗")
        
        # Test basic tensor operations
        try:
            x = torch.randn(2, 3)
            y = torch.mm(x, x.t())
            results["Basic Operations"] = (True, "Basic tensor operations working ✓")
        except Exception as e:
            results["Basic Operations"] = (False, f"Basic tensor operations failed: {e} ✗")
            
    except ImportError:
        results["PyTorch"] = (False, "PyTorch not installed ✗")
    
    return results


def check_torch_geometric() -> Tuple[bool, str]:
    """Check PyTorch Geometric installation and features."""
    try:
        import torch_geometric
        from torch_geometric.nn import GATConv
        from torch_geometric.data import Data
        import torch
        
        # Test basic graph operation
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 16)
        data = Data(x=x, edge_index=edge_index)
        
        # Test GAT layer
        conv = GATConv(16, 32, heads=2)
        out = conv(data.x, data.edge_index)
        
        return True, f"torch-geometric {torch_geometric.__version__} with GAT support ✓"
    except ImportError as e:
        return False, f"torch-geometric import failed: {e} ✗"
    except Exception as e:
        return False, f"torch-geometric functionality test failed: {e} ✗"


def check_transformers() -> Tuple[bool, str]:
    """Check transformers library and BERT model access."""
    try:
        from transformers import BertTokenizer, BertModel
        import torch
        
        # Test BERT model loading
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        # Test tokenization and encoding
        text = "Test log message for tokenization"
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**tokens)
        
        return True, "transformers with BERT model access ✓"
    except Exception as e:
        return False, f"transformers/BERT test failed: {e} ✗"


def run_verification() -> None:
    """Run complete environment verification."""
    print("OCEAN Environment Verification")
    print("=" * 50)
    
    # Check Python version
    python_ok, python_msg = check_python_version()
    print(f"Python Version: {python_msg}")
    
    if not python_ok:
        print("\nERROR: Python 3.9+ is required!")
        sys.exit(1)
    
    print("\nCore Dependencies:")
    print("-" * 20)
    
    # List of packages to check
    packages = [
        ("torch", "torch"),
        ("torch-geometric", "torch_geometric"),
        ("transformers", "transformers"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("networkx", "networkx"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    all_packages_ok = True
    for package_name, import_name in packages:
        success, msg, error = check_package_import(package_name, import_name)
        print(f"  {msg}")
        if not success:
            print(f"    Error: {error}")
            all_packages_ok = False
    
    print("\nPyTorch Features:")
    print("-" * 17)
    torch_features = check_torch_features()
    for feature, (success, msg) in torch_features.items():
        print(f"  {msg}")
    
    print("\nSpecialized Libraries:")
    print("-" * 21)
    
    # Check PyTorch Geometric
    pyg_ok, pyg_msg = check_torch_geometric()
    print(f"  {pyg_msg}")
    if not pyg_ok:
        all_packages_ok = False
    
    # Check transformers
    transformers_ok, transformers_msg = check_transformers()
    print(f"  {transformers_msg}")
    if not transformers_ok:
        all_packages_ok = False
    
    print("\nVerification Results:")
    print("-" * 20)
    
    if all_packages_ok and python_ok:
        print("✓ All dependencies are properly installed and working!")
        print("✓ Environment is ready for OCEAN implementation.")
    else:
        print("✗ Some dependencies are missing or not working properly.")
        print("Please install missing packages using: pip install -r requirements.txt")
        sys.exit(1)
    
    print(f"\nRecommended device: {get_recommended_device()}")


def get_recommended_device() -> str:
    """Get recommended device for computation."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "MPS (Apple Silicon GPU)"
        elif torch.cuda.is_available():
            return f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
        else:
            return "CPU"
    except:
        return "CPU (fallback)"


if __name__ == "__main__":
    run_verification()
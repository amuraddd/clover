"""Unified entry point for all Clover baselines.

This runs all baselines sequentially with default arguments.
Simply run: python main.py

Available baselines:
    - b2diffurl: B2-DiffuRL with backward-progressive, branch-based sampling
    - ddpo: Denoising Diffusion Policy Optimization
    - dpok: Diffusion Policy Optimization with online KL regularization
"""

import sys

# Import all baseline main functions
from clover.baselines.b2diffurl import main as b2diffurl_main
from clover.baselines.ddpo import main as ddpo_main
from clover.baselines.dpok import main as dpok_main


BASELINES = [
    ("ddpo", ddpo_main),
    ("dpok", dpok_main),
    ("b2diffurl", b2diffurl_main),
]


def main():
    """Run all baseline experiments sequentially with default arguments."""
    print("=" * 80)
    print("Running all Clover baselines with default arguments")
    print("=" * 80)
    
    # Clear sys.argv to ensure each baseline uses its default config
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    
    for i, (name, baseline_fn) in enumerate(BASELINES, 1):
        print(f"\n[{i}/{len(BASELINES)}] Starting {name.upper()} baseline...")
        print("-" * 80)
        
        try:
            baseline_fn()
            print(f"\n✓ {name.upper()} completed successfully")
        except Exception as e:
            print(f"\n✗ {name.upper()} failed with error: {e}")
            # Continue with other baselines even if one fails
            continue
        
        print("-" * 80)
    
    # Restore original argv
    sys.argv = original_argv
    
    print("\n" + "=" * 80)
    print("All baselines completed")
    print("=" * 80)


if __name__ == "__main__":
    main()

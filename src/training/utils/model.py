"""
Model utilities - backward compatibility layer.

This module provides backward compatibility by re-exporting classes
from their new modular locations. Use the new imports directly:
- from src.training.utils.optimizer import ModelOptimizer
- from src.training.utils.evaluator import ModelEvaluator
- from src.training.utils.champion import ModelChampionManager
"""

# Re-export for backward compatibility
from src.training.utils.champion import ModelChampionManager
from src.training.utils.evaluator import ModelEvaluator
from src.training.utils.optimizer import ModelOptimizer

__all__ = ["ModelOptimizer", "ModelEvaluator", "ModelChampionManager"]

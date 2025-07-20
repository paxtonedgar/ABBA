"""
ABBA - Advanced Baseball Betting Analytics

A comprehensive sports betting analytics platform focused on MLB and NHL.
"""

__version__ = "1.0.0"
__author__ = "ABBA Team"
__email__ = "support@abba.com"

# Core imports
from .core.config import Config
from .core.logging import setup_logging

# Analytics imports
try:
    from .analytics.advanced_analytics import AdvancedAnalyticsManager
    from .analytics.manager import AnalyticsManager
except ImportError:
    pass  # Analytics module not available

# API imports
try:
    from .api.real_time_connector import RealTimeConnector
except ImportError:
    pass  # API module not available

# Agents modules imports
try:
    from .agents_modules.dynamic_orchestrator import DynamicOrchestrator
    from .agents_modules.guardrail_agent import GuardrailAgent
    from .agents_modules.reflection_agent import ReflectionAgent
except ImportError:
    pass  # Agents modules not available

__all__ = [
    "Config",
    "setup_logging",
    "AnalyticsManager",
    "AdvancedAnalyticsManager",
    "RealTimeConnector",
    "DynamicOrchestrator",
    "GuardrailAgent",
    "ReflectionAgent",
]

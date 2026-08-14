from .base import BaseAgent

try:
    from .math import MathAgent
except ModuleNotFoundError:
    MathAgent = None

try:
    from .travel_planner import TravelPlannerAgent
except ModuleNotFoundError:
    TravelPlannerAgent = None

try:
    from .webshop import WebShopAgent
except ModuleNotFoundError:
    WebShopAgent = None
# from .search import SearchAgent

__all__ = [
    "BaseAgent",
    "MathAgent",
    "TravelPlannerAgent",
    "WebShopAgent",
    "SearchAgent",
]

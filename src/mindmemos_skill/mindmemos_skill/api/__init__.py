"""Public service interface and construction entry point."""

from .base import MindMemOSSkill
from .config import ImplementationConfig, PersistenceConfig, ServiceConfig
from .factory import create_service

__all__ = ["MindMemOSSkill", "ImplementationConfig", "PersistenceConfig", "ServiceConfig", "create_service"]

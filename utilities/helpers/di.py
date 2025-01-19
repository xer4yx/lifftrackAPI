from typing import TypeVar, Type, Dict, Any, Optional
from functools import wraps
import inspect

T = TypeVar('T')

class DependencyContainer:
    _instance = None
    _dependencies: Dict[Type, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, interface: Type[T], implementation: Type[T]):
        """Register an implementation for an interface"""
        cls._dependencies[interface] = implementation
    
    @classmethod
    def resolve(cls, interface: Type[T]) -> T:
        """Resolve an implementation for an interface"""
        if interface not in cls._dependencies:
            raise KeyError(f"No implementation registered for {interface}")
        return cls._dependencies[interface]

def inject(*dependencies):
    """Decorator to inject dependencies into a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            container = DependencyContainer()
            resolved_deps = [container.resolve(dep) for dep in dependencies]
            return func(*args, *resolved_deps, **kwargs)
        return wrapper
    return decorator

def singleton(cls):
    """Decorator to make a class a singleton"""
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance 
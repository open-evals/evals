from stevedore import DriverManager, _cache, exception

from evals.base import ModelSpec

from .base import _ModelRunner
from . import openai, llama
from importlib.metadata import EntryPoint


def load_runner(model_spec: ModelSpec) -> _ModelRunner:
    manager = DriverManager("open-evals", model_spec.runner, invoke_on_load=True)
    return manager.driver


def get_model_spec(name: str) -> ModelSpec:
    # FIXME: use yaml or other configuable way to setup model's spec

    # NOTE: load all plugins
    entry_point: EntryPoint
    for entry_point in _cache.get_group_all("open-evals"):
        manager = DriverManager("open-evals", entry_point.name, invoke_on_load=False)
        resolver: _ModelRunner = manager.driver

        try:
            import pdb

            pdb.set_trace()
            model_spec = resolver.resolve(name)
            return model_spec
        except ValueError:
            pass

    raise ValueError(f"Model {name} not found")

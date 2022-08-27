import re
from contextlib import contextmanager
from typing import Iterator, Dict, Iterable, Tuple, List

from allennlp.models import Model

from qdecomp_with_dependency_graphs.utils.helpers import rgetattr, rsetattr


@contextmanager
def capture_model_internals(model: Model, module_regex: str = ".*") -> Iterator[dict]:
    """
    Based on allennlp/predictors/predictor.py, tag: v1.1.0

    Context manager that captures the internal-module outputs of
    this model. The idea is that you could use it as follows:
    ```
        with capture_model_internals(model) as internals:
            outputs = model.forward(*inputs, **kwargs)
        return {**outputs, "model_internals": internals}
    ```
    """
    results = {}
    hooks = []

    # First we'll register hooks to add the outputs of each module to the results dict.
    def add_output(name: str):
        def _add_output(mod, _, outputs):
            results[name] = outputs

        return _add_output

    regex = re.compile(module_regex)
    for idx, (name, module) in enumerate(model.named_modules()):
        if regex.fullmatch(name) and module != model:
            hook = module.register_forward_hook(add_output(name))
            hooks.append(hook)

    # If you capture the return value of the context manager, you get the results dict.
    yield results

    # And then when you exit the context we remove all the hooks.
    for hook in hooks:
        hook.remove()


def tie_models_modules(models: Dict[str, Model], tie_modules_groups: Iterable[List[str]]):
    def get_module(path: str) -> Tuple[Model, str]:
        model_name, module_path = path.split('.', maxsplit=1)
        model = models[model_name]
        return model, module_path

    for group in tie_modules_groups:
        if len(group) > 1:
            main_model, main_module_path = get_module(group[0])
            main_module = rgetattr(main_model, main_module_path)
            for g in group[1:]:
                model, module_path = get_module(g)
                rsetattr(model, module_path, main_module)
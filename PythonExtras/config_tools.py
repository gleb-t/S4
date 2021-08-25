import copy
import inspect
import itertools
import os
from typing import *

from deprecation import deprecated


T = TypeVar('T')


@deprecated("Only to be used by 'normal' configs. PyPlant configs should use pyplant_extras.")
def expand_config_placeholders(config: Dict[str, Any], placeholderMap: Dict[str, str]):

    def _expand_in_string(string: str):
        for placeholder, value in placeholderMap.items():
            placeholderString = '${}$'.format(placeholder)
            if placeholderString in string:
                string = string.replace(placeholderString, value)
        return string

    config = config.copy()
    for key in list(config.keys()):  # use a copy of keys, since we're modifying the dict on the fly.
        if isinstance(config[key], str):
            expandedString = _expand_in_string(config[key])
            if expandedString != config[key]:
                config[key + '.orig'] = config[key]  # Save the original value.
                config[key] = expandedString
        elif isinstance(config[key], list):
            expandedList = config[key].copy()
            listChanged = False

            for i, element in enumerate(expandedList):
                if isinstance(element, str):
                    expandedString = _expand_in_string(element)
                    if expandedString != element:
                        expandedList[i] = expandedString
                        listChanged = True

            if listChanged:
                config[key + '.orig'] = config[key]  # Save the original value.
                config[key] = expandedList

    return config


def kebab_case_to_camel(string: str):
    if '-' not in string:
        return string

    chunks = string.split('-')
    return ''.join([chunks[0]] + [c.capitalize() for c in chunks[1:]])


def merge_dicts(dictUnder: Dict[str, Any], dictOver: Dict[str, Any], throwIfMissing=False):
    result = dictUnder.copy()
    for k, v in dictOver.items():
        if throwIfMissing and k not in result:
            raise RuntimeError("Key '{}' doesn't exist.".format(k))
        result[k] = v

    return result


def parse_and_apply_config(baseConfig: object, configPath: str):
    """
    Parse a python-code config. Find 'configure' function with a list of assignments,
    and replay them on the provided object.
    This is safer and cleaner than just eval-ing the function.
    """
    import ast
    import asteval

    with open(configPath, 'r') as file:
        source = ''.join(file.readlines())

    foundConfig = False
    configAssignments = []

    def visit_ast_node(node, level: int = 0):
        if isinstance(node, ast.FunctionDef):
            fields = list(ast.iter_fields(node))
            funcName = next((f for f in fields if f[0] == 'name'))[1]
            bodyStatements = next((f for f in fields if f[0] == 'body'))[1]

            if funcName == 'configure':
                nonlocal foundConfig
                foundConfig = True
            else:
                return

            for assignment in bodyStatements:
                if not isinstance(assignment, ast.Assign):
                    raise RuntimeError("Only assignments are allowed within config. Got '{}' instead."
                                       .format(type(assignment).__name__))

                if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Attribute):
                    raise RuntimeError("Invalid assignment target at {}:{}".format(configPath, assignment.lineno))

                # Get the config field name.
                attrNode = assignment.targets[0]  # Assignment target, attribute.
                objectName = attrNode.value.id
                configFieldName = attrNode.attr

                if objectName != 'config':
                    raise RuntimeError("Expected assignment to 'config' object at {}:{}"
                                       .format(configPath, assignment.lineno))

                # Get the config field value.
                configFieldValue = asteval.Interpreter().eval(assignment.value)
                configAssignments.append((configFieldName, configFieldValue))

            # Parse the 'configure' function, we're done.
            return

        # Iterate through the tree.
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        visit_ast_node(item, level=level + 1)
            elif isinstance(value, ast.AST):
                visit_ast_node(value, level=level + 1)

    visit_ast_node(ast.parse(source))

    if not foundConfig:
        raise RuntimeError("Failed to find a 'configure' function in the config file '{}'".format(configPath))

    for name, value in configAssignments:
        if not hasattr(baseConfig, name):
            raise RuntimeError("Config parameter '{}' doesn't exist.".format(name))

        setattr(baseConfig, name, value)

    return baseConfig


def parse_and_apply_config_eval(baseConfig: T, configPath: str) -> T:
    """
    Parse a python-code config.
    Import the source code as a module and run the 'configure' function.
    """

    import copy

    module = load_module_from_file(configPath)
    configureFunc = module.configure

    config = copy.deepcopy(baseConfig)
    configureFunc(config)

    return config


def load_subconfig_func(configPath: str, relSubconfigPath: str, funcName: str) -> Callable:
    """
    This function is meant to be used within a python-code config to load another config function
    using the same method (loading a python module from file).

    :param configPath:
    :param relSubconfigPath:
    :param funcName:
    :return:
    """
    subconfigPath = os.path.abspath(os.path.join(os.path.dirname(configPath), relSubconfigPath))
    subconfigModule = load_module_from_file(subconfigPath)

    return getattr(subconfigModule, funcName)


def load_module_from_file(configPath):
    import importlib.util
    import random
    import string

    if not os.path.exists(configPath):
        raise ValueError("Module file not found: '{}'".format(configPath))

    randomString = ''.join(random.choices(string.ascii_uppercase, k=10))
    spec = importlib.util.spec_from_file_location("_custom_config." + randomString, configPath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def is_parameter_sweep_config(config: object) -> bool:

    for name, value in config.__dict__.items():
        if isinstance(value, SweepParameter):
            return True

    return False


def expand_parameter_sweep_config(config: T) -> List[T]:
    # Find and expand SweepParameter objects into a list of values.
    sweepParams = []
    for name, value in inspect.getmembers(config):
        if isinstance(value, SweepParameter):
            sweepParams.append(zip(itertools.repeat(name, len(value)), value))

    # Generate config instances by inserting all possible parameter combinations.
    configs = []
    for paramItems in itertools.product(*sweepParams):  # type: Tuple[Tuple[str, Any]]
        configInstance = copy.deepcopy(config)
        for name, value in paramItems:
            setattr(configInstance, name, value)

        configs.append(configInstance)

    return configs


class SweepParameter:
    """
    A simple wrapper/marker class to define parameter sweeps inline in the config objects.
    This class is used by functions in this module to expand a config object into a set of
    concrete configs defining the parameter sweep.
    """

    def __init__(self, values: List[Any]):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def __iter__(self):
        return iter(self.values)

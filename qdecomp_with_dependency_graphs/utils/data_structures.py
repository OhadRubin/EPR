"""
Utils functions for data structures
"""
import collections
from typing import Any, Callable, Dict, List, Iterable, Set
from collections import defaultdict
import json
import dataclasses
import more_itertools as mit


###################################################
#           Iterables                             #
###################################################

def find_ranges(iterable):
    """Yield range of consecutive numbers.
    Example:
        input: [2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 20]
        output: [(2, 5), (12, 17), (20,20)]
    """
    for group in mit.consecutive_groups(sorted(iterable)):
        group = list(group)
        yield group[0], group[-1]


###################################################
#           Dictionaries                          #
###################################################

def list_to_index_dict(seq: list) -> Dict[Any, List[int]]:
    """
    Given a list of items, return a dictionary of indexes for each item in the list
    :param seq: list of hashable items
    :return: dictionary of indexes for each item in seq
    """
    dict = defaultdict(list)
    for i, item in enumerate(seq):
        dict[item].append(i)
    return dict


def list_to_multivalue_dict(seq: Iterable, key: Callable[[Any], Any]) -> Dict[Any, List[Any]]:
    """
    Given a list of items, return a dictionary where each item is mapped by
    :param seq: list of hashable items
    :return: dictionary of <key, items-list>
    """
    dict = defaultdict(list)
    for item in seq:
        dict[key(item)].append(item)
    return dict


def swap_keys_and_values_dict(dictionary: Dict):
    new_dict = {}
    for key, value in dictionary.items():
        if value in new_dict:
            new_dict[value].append(key)
        else:
            new_dict[value] = [key]
    return new_dict


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Json encoder for dataclasses
    usage: json.dumps(foo, cls=EnhancedJSONEncoder)
    """
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def merge_equivalent_classes(classes: Iterable[Iterable[str]]) -> Dict[str, Iterable[str]]:
    """
    Get classes of equivalent tokens, and return for each token its class.
    In case of overlap classes, merge them.
    e.g: [[a,b], [b,c], [d]] => {a:[a,b,c], b:[a,b,c], c:[a,b,c], d:[d]}
    :param classes:
    :return:
    """
    tokens_to_sets: Dict[str, Iterable[str]] = {}
    for cls in classes:
        sets = [tokens_to_sets.get(x, {x}) for x in cls]
        new_set = set().union(*sets)
        for x in new_set:
            tokens_to_sets[x] = new_set
    return tokens_to_sets


def flatten_dict(dictionary: Dict[str, Any], separator='.', prefix='') -> Dict[str, Any]:
    """
    Flatten a dictionary of dictionaries, where the keys are joined py separator.
    {"a": {"b": {"c": v}}} => {"a.b.c":v}
    :param dictionary: dictionary to flatten
    :param separator: separator between nested keys
    :param prefix: prefix of keys. uses for recursion
    :return: a flatten dictionary
    """
    items = []
    for k, v in dictionary.items():
        new_key = prefix + separator + k if prefix else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, separator, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def nest_flatten_dict(dictionary: Dict[str, Any], separator='.') -> Dict[str, Any]:
    """
    Re-nest a flatten a dictionary, assuming the keys are joined py separator.
    {"a.b.c":v} => {"a": {"b": {"c": v}}}
    :param dictionary: dictionary to flatten
    :param separator: separator between nested keys
    :return: a nested dictionary
    """
    new_keys = set([])
    output = defaultdict(dict)
    for k, v in list(dictionary.items()):
        if separator in k:
            new_k, suffix = k.split(separator, maxsplit=1)
            output[new_k][suffix] = v
            new_keys.add(new_k)
        else:
            output[k] = v
    for k in new_keys:
        output[k] = nest_flatten_dict(output[k], separator=separator)
    return output


###################################################
#           DataClasses                           #
###################################################

def merge_dataclasses(items: List[Any], ctor):
    """
    Merge a list of dataclass instances to a single instance, while ignoring None.
    Note: in case of conflicts, the latest instance of no-None value is assigned
    e.g:
        merge_dataclasses([Token(text='text1'), Token(tag_='tag2)], Token) -> Token(text='text1',tag_='tag2)
        merge_dataclasses([Token(text='text1', tag_='tag1'), Token(tag_='tag2)], Token) -> Token(text='text1',tag_='tag2)
    :param items: dataclass instances to merge
    :param ctor: dataclass class
    :return: merged dataclass of type 'ctor'
    """
    assert len(items) >= 1, f'got an empty list of items {items}'
    dict_list = [dataclasses.asdict(x) for x in items]
    pivot = dict_list[0].copy()
    for x in dict_list[1:]:
        pivot.update({k: v for k, v in x.items() if v is not None})
    return ctor(**pivot)
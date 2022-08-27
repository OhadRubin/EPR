from enum import Enum


class QDMROperation(str, Enum):
    FIND, SELECT, FILTER, PROJECT, AGGREGATE, GROUP, SUPERLATIVE, COMPARATIVE, UNION, INTERSECTION, DISCARD, SORT, \
    BOOLEAN, ARITHMETIC, COMPARISON, NONE = \
    'find', 'select', 'filter', 'project', 'aggregate', 'group', 'superlative', 'comparative', 'union', \
    'intersection', 'discard', 'sort', 'boolean', 'arithmetic', 'comparison', 'None'
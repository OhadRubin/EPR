from enum import Enum
from typing import Dict, Iterable, List, Tuple, Union, Generator
from collections import OrderedDict
import itertools
from overrides import overrides
from collections import defaultdict

from qdecomp_with_dependency_graphs.scripts.data_processing.break_app_store_generation import token_variations
from .utils_ import *


class QDMROperator(str, Enum):
    AGGREGATE = 'aggregate'
    ARITHMETIC = 'arithmetic'
    BOOLEAN = 'boolean'
    COMPARATIVE = 'comparative'
    COMPARISON = 'comparison'
    DISCARD = 'discard'
    FILTER = 'filter'
    GROUP = 'group'
    INTERSECTION = 'intersection'
    PROJECT = 'project'
    SELECT = 'select'
    SORT = 'sort'
    SUPERLATIVE = 'superlative'
    UNION = 'union'

    NONE = 'none'

    def __str__(self):
        return self.value


class ArgumentType(str, Enum):
    AGGREGATE_ARG = 'aggregate-arg'
    ARITHMETIC_ARG = 'arithmetic-arg'
    ARITHMETIC_LEFT = 'arithmetic-left'
    ARITHMETIC_RIGHT = 'arithmetic-right'
    # BOOLEAN_ARG = 'boolean-arg'
    BOOLEAN_SUB = 'boolean-sub'
    BOOLEAN_CONDITION = 'boolean-condition'
    COMPARATIVE_SUB = 'comparative-sub'
    # COMPARATIVE_ARG = 'comparative-arg'
    COMPARATIVE_ATTRIBUTE = 'comparative-attribute'
    COMPARATIVE_CONDITION = 'comparative-condition'
    COMPARISON_ARG = 'comparison-arg'
    DISCARD_SUB = 'discard-sub'
    DISCARD_EXCLUDE = 'discard-exclude'
    FILTER_SUB = 'filter-sub'
    FILTER_CONDITION = 'filter-condition'
    GROUP_VALUE = 'group-value'
    GROUP_KEY = 'group-key'
    INTERSECTION_PROJECTION = 'intersection-projection'
    INTERSECTION_INTERSECTION = 'intersection-intersection'
    PROJECT_PROJECTION = 'project-projection'
    PROJECT_SUB = 'project-sub'
    SELECT_SUB = 'select-sub'
    SORT_SUB = 'sort-sub'
    SORT_ORDER = 'sort-order'
    SUPERLATIVE_SUB = 'superlative-sub'
    SUPERLATIVE_ATTRIBUTE = 'superlative-attribute'
    UNION_SUB = 'union-sub'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    def __str__(self):
        return self.value

    def get_operator(self) -> QDMROperator:
        return QDMROperator(self._get_parts()[0])

    def get_arg_name(self):
        return self._get_parts()[1]

    def _get_parts(self):
        return self.value.split('-')


def _get_variations(tokens: List[str]) -> List[str]:
    return list(set(tokens + [x for t in tokens for x in token_variations(t)]))


def _is_in(part: str, text: str) -> bool:
    num_pref = part and part[0].isdigit() and '0-9' or ''
    num_suf = part and part[-1].isdigit() and '0-9' or ''
    return re.search(fr'(^|[^a-zA-Z{num_pref}]){part}([^a-zA-Z{num_suf}]|$)', text) is not None


class IdentifyOperator(object):
    def __init__(self, properties_tokens: Dict[str, List[str]] = {}, properties_order: List[List[str]] = None):
        self._properties_tokens: Dict[str, List[str]] = properties_tokens
        self._terms_to_properties: List[Tuple[str, List[str]]] = list(
            self._generate_term_to_properties_order(properties_tokens=properties_tokens,
                                                    properties_order=properties_order))

    def extract_references(self, step: str):
        """Extracts a list of references to previous steps"""
        # make sure decomposition does not contain a mere '# ' other than a reference.
        step = step.replace("# ", "hashtag ")
        references = []
        l = step.split(REF)
        for chunk in l[1:]:
            if len(chunk) > 1:
                ref = chunk.split()[0]
                ref = int(ref)
                references += [ref]
            if len(chunk) == 1:
                ref = int(chunk)
                references += [ref]
        return references

    def extract_properties(self, step: str, references: List[int] = None) -> List[str]:
        """Extract properties expression from QDMR step
        e.g string of the aggregate: max/min/count/sum/avg

        Returns
        -------
        List[str]
            string of the properties
        """
        references = references or self.extract_references(step)
        return self._extract_properties(step=step, references=references)

    def operator_indicators(self) -> List[str]:
        return []

    def properties_indicators(self, properties: List[str] = None) -> Union[List[str], Dict[str, Iterable[str]]]:
        if properties:
            return [x for p in properties for x in self._properties_tokens[p]]
        return self._properties_tokens

    def identify_op(self, step: str):
        references = self.extract_references(step)
        operator = self._identify_op(step, references)
        return operator

    def extract_args(self, step: str) -> Dict[ArgumentType, Iterable[str]]:
        references = self.extract_references(step)
        args = self._extract_args(step, references)
        assert args is not None
        return {k: ([x.strip() for x in v] if isinstance(v,list) else (v and [v.strip()])) for k, v in args.items()}

    def _identify_op(self, step: str, references: List[int]):
        raise NotImplementedError

    def _extract_args(self, step: str, references: List[int]) -> Dict[ArgumentType, Union[str,Iterable[str]]]:
        raise NotImplementedError

    def _extract_properties(self, step: str, references: List[int]) -> List[str]:
        step = step.lower()
        for term, properties in self._terms_to_properties:
            if _is_in(term, step):
                return properties
        return None

    @staticmethod
    def _generate_term_to_properties_order(properties_tokens: Dict[str, List[str]] = {},
                                           properties_order: List[List[str]] = None) -> Generator[Tuple[str, List[str]], None, None]:
        properties_terms_list = [properties_tokens] if not properties_order else [
            {k: v for k, v in properties_tokens.items() if k in x} for x in properties_order]
        for properties_terms in properties_terms_list:
            term_to_property = defaultdict(list)
            for k, v in properties_terms.items():
                for x in v:
                    term_to_property[x].append(k)
            for term, properties in sorted(term_to_property.items(), key=lambda x: len(x[0]), reverse=True):
                yield term, properties


class IdentifyOperatorAggregate(IdentifyOperator):
    """
    Example: "lowest of #2"
    Example: "the number of #1"
    """
    properties_tokens = {
        "max": _get_variations(['max',  'most', 'more', 'last',
                                 'bigger', 'biggest', 'larger', 'largest', 'higher', 'highest', 'longer', 'longest']),
        "min": _get_variations(['min', 'least', 'less', 'first', 'fewer',
                                 'smaller', 'smallest', 'lower', 'lowest',  'shortest', 'shorter', 'earlier']),
        "count": _get_variations(['number of', 'total number of']),
        "sum": _get_variations(['sum', 'total']),
        "avg": _get_variations(['avg', 'average', 'mean']),
    }
    properties_order = [["max", "min"], ["avg"], ["count", "sum"]]

    def __init__(self):
        super().__init__(
            properties_tokens=IdentifyOperatorAggregate.properties_tokens,
            properties_order=IdentifyOperatorAggregate.properties_order
        )

    def _identify_op(self, step: str, references: List[int]):
        # AGGREGATION step - aggregation applied to one reference
        if len(references) != 1:
            return False
        aggregators = [x for v in self._properties_tokens.values() for x in v]
        for aggr in aggregators:
            aggr_ref = aggr + ' #'
            aggr_of_ref = aggr + ' of #'
            if _is_in(aggr_ref, step) or _is_in(aggr_of_ref, step):
                return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        # extract the referenced objects
        ref = "#%s" % references[0]
        return {ArgumentType.AGGREGATE_ARG: ref}


class IdentifyOperatorArithmetic(IdentifyOperator):
    """
    Example: "difference of #3 and #5"
    """

    def __init__(self):
        super().__init__(
            properties_tokens={
                'sum': _get_variations(['sum']),
                'difference': _get_variations(['difference']),
                'multiplication': _get_variations(['multiplication']),
                'division': _get_variations(['division']),
            })

    def _identify_op(self, step: str, references: List[int]):
        # ARITHMETIC step - starts with arithmetic operation
        arithmetics = [x for v in self._properties_tokens.values() for x in v]
        for a in arithmetics:
            if (step.startswith(a) or step.startswith('the ' + a)) \
                    and (len(references) > 1 or
                         (len(references) == 1 and _is_in('and', step))  # the difference of 100 and #1
            ):
                return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        properties = self.extract_properties(step, references)
        parts = re.split('and|,', step)
        if len(parts) != 2 or 'sum' in properties or 'multiplication' in properties:
            refs = ['#%d' % ref for ref in references]
            return {ArgumentType.ARITHMETIC_ARG: refs}

        prefix, suffix = parts
        prefix = prefix.split()[-1]

        def get_part(text: str):
            refs = self.extract_references(text)
            if refs:
                return ['#%d' % ref for ref in refs]
            return text

        left, right = get_part(prefix), get_part(suffix)
        return {ArgumentType.ARITHMETIC_LEFT: left, ArgumentType.ARITHMETIC_RIGHT: right}


class IdentifyOperatorBoolean(IdentifyOperator):
    """
    Example: "if both #2 and #3 are true"
    Example: "is #2 more than #3"
    Example: "if #1 is american"
    """
    conditional_properties = {
        'equals': _get_variations(['equal', 'equals', 'same as']),
        **{f'equals_{num}': [' '.join(x) for x in itertools.product(_get_variations(['equals', 'is']), ['', 'to'], _get_variations([num]))]
           for num in ['0', '1', '2']},
        **{f'more_than_{num}': [' '.join(x) for x in itertools.product(_get_variations(['bigger', 'more']), ['', 'than'], _get_variations([num]))]
           for num in ['0', '1', '2']},
        **{f'less_than_{num}': [' '.join(x) for x in itertools.product(_get_variations(['lower', 'less']), ['', 'than'],_get_variations([num]))]
           for num in ['0', '1', '2']},
    }

    def __init__(self):
        super().__init__(
            properties_tokens={
                'logical_and': ['both', 'and'],
                'logical_or': ['either', 'or'],
                'true': ['true'],
                'false': ['false'],
                'if_exist': ['any', 'there'],
                **self.conditional_properties
            },
            properties_order=[list(self.conditional_properties.keys())]
        )

    @overrides
    def operator_indicators(self) -> List[str]:
        return ['#REF']

    def _identify_op(self, step: str, references: List[int]):
        # BOOLEAN step - starts with either 'if', 'is' or 'are'
        if step.lower().startswith('if ') or step.lower().startswith('is ') or \
                step.lower().startswith('are ') or step.lower().startswith('did '):
            return True
        return False

    def _extract_properties(self, step: str, references: List[int]) -> List[str]:
        # logical or/and boolean steps, e.g., "if either #1 or #2 are true"
        logical_op = None
        if len(references) == 2 and _is_in("both", step) and _is_in("and", step):
            logical_op = "logical_and"
        elif len(references) == 2 and _is_in("either", step) and _is_in("or", step):
            logical_op = "logical_or"
        if logical_op is not None:
            bool_expr = "false" if _is_in("false", step) else "true"
            return [logical_op, bool_expr]

        props = []
        if len(references) == 2:
            objects = "#%s" % references[0]
            prefix = step.split(objects)[0].lower()
            if "any" in prefix or "is there" in prefix \
                    or "there is" in prefix or "there are" in prefix:
                # exists boolean "if any #2 are the same as #3"
                props.append("if_exist")
        if props: return props

        return super()._extract_properties(step=step, references=references)

    def _extract_args(self, step: str, references: List[int]):
        # _, args = self._extract_properties_and_args(step, references)
        props = self._extract_properties(step, references)
        if props:
            # logical or/and boolean steps, e.g., "if either #1 or #2 are true"
            if "logical_and" in props or "logical_and" in props:
                sub_expressions = ["#%s" % ref for ref in references]
                return {ArgumentType.BOOLEAN_SUB: sub_expressions}

            # exists boolean "if any #2 are the same as #3"
            if "if_exist" in props:
                objects = "#%s" % references[0]
                condition = step.split(objects)[1]
                return {ArgumentType.BOOLEAN_SUB: objects, ArgumentType.BOOLEAN_CONDITION: condition}

        if step.split()[1].startswith("#"):
            # filter boolean, e.g., "if #1 is american"
            objects = "#%s" % references[0]
            condition = step.split(objects)[1]
            return {ArgumentType.BOOLEAN_SUB: objects, ArgumentType.BOOLEAN_CONDITION: condition}
        if len(references) == 1 \
                and not step.split()[1].startswith("#"):
            # projection boolean "if dinner is served on #1"
            objects = "#%s" % references[0]
            condition = step.replace(objects, "#REF")
            return {ArgumentType.BOOLEAN_SUB: objects, ArgumentType.BOOLEAN_CONDITION: condition}
        return {ArgumentType.BOOLEAN_CONDITION: step}


class IdentifyOperatorComparative(IdentifyOperator):
    """
    Example: "#1 where #2 is at most three"
    Example: "#3 where #4 is higher than #2"
    """

    def __init__(self):
        super().__init__(
            properties_tokens={
                'equals': _get_variations(['equal', 'equals', 'same as']),
                'more': _get_variations(['more', 'at least']) + [f'{x} than' for x in _get_variations(['higher', 'larger', 'bigger'])],
                'less': _get_variations(['less', 'at most']) + [f'{x} than' for x in _get_variations(['smaller', 'lower'])],
                # 'contain': _get_variations(['contain', 'include']),
                # 'has': _get_variations(['has', 'have']),
                # 'starts': [f'{x} with' for x in _get_variations(['start', 'starts', 'begin', 'begins'])],
                # 'ends': [f'{x} with' for x in _get_variations(['end', 'ends'])],
                **IdentifyOperatorBoolean.conditional_properties
            }
        )

    def _identify_op(self, step: str, references: List[int]):
        comparatives = [x for v in self._properties_tokens.values() for x in v] + \
                       _get_variations(['contain', 'include', 'has', 'have']) + \
                       [f'{x} with' for x in _get_variations(['start', 'starts', 'begin', 'begins', 'end', 'ends'])] + \
                       ['is', 'are', 'was']

        if len(references) >= 2 and len(references) <= 3 \
                and _is_in('where', step) and (step.startswith('#') or step.startswith('the #')):
            for comp in comparatives:
                if _is_in(comp, step):
                    return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        to_filter = "#%s" % references[0]
        attribute = "#%s" % references[1]
        condition = step.split(attribute)[1]
        return {ArgumentType.COMPARATIVE_SUB: to_filter,
                ArgumentType.COMPARATIVE_ATTRIBUTE: attribute,
                ArgumentType.COMPARATIVE_CONDITION: condition}


class IdentifyOperatorComparison(IdentifyOperator):
    """
    Example: "which is highest of #1, #2"
    """

    def __init__(self):
        super().__init__(
            properties_tokens={
                **IdentifyOperatorAggregate.properties_tokens,
                'true': ['true'],
                'false': ['false'],
            },
            properties_order=IdentifyOperatorAggregate.properties_order + [['true', 'false']]
        )

    def _identify_op(self, step: str, references: List[int]):
        # COMPARISON step - 'which is better A or B or C'
        if step.lower().startswith('which') and len(references) > 1:
            return True
        return False

    @overrides
    def _extract_properties(self, *args, **kwargs) -> List[str]:
        comp = super()._extract_properties(*args, **kwargs)
        assert comp is not None
        return comp

    def _extract_args(self, step: str, references: List[int]):
        args = ["#%s" % ref for ref in references]
        return {ArgumentType.COMPARISON_ARG: args}


class IdentifyOperatorDiscard(IdentifyOperator):
    """
    Example: "#2 besides #3"
    Exmple: "#1 besides cats"
    """

    def __init__(self):
        super().__init__()

    @overrides
    def operator_indicators(self):
        return _get_variations(['besides', 'not in'])

    def _identify_op(self, step: str, references: List[int]):
        if (len(references) >= 1) and (len(references) <= 2) and \
                (re.search("^[#]+[0-9]+[\s]+", step) or re.search("[#]+[0-9]+$", step)) and \
                any(_is_in(x, step) for x in _get_variations(['besides', 'not in'])):
            return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        discard_expr = None
        for expr in _get_variations(['besides', 'not in']):
            if _is_in(expr, step):
                discard_expr = expr
        set_1, set_2 = step.split(discard_expr)
        return {ArgumentType.DISCARD_SUB: set_1, ArgumentType.DISCARD_EXCLUDE: set_2}


class IdentifyOperatorFilter(IdentifyOperator):
    """
    Example: "#2 that is wearing #3"
    Example: "#1 from Canada"
    """

    def __init__(self):
        super().__init__()

    def _identify_op(self, step: str, references: List[int]):
        # FILTER starts with '#'
        refs = len(references)
        if refs > 0 and refs <= 3 and step.startswith("#"):
            return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        # extract the reference to be filtered
        to_filter = "#%s" % references[0]
        # extract the filter condition
        filter_condition = step.split(to_filter)[1]
        return {ArgumentType.FILTER_SUB: to_filter, ArgumentType.FILTER_CONDITION: filter_condition}


class IdentifyOperatorGroup(IdentifyOperator):
    """
    Example: "number of #3 for each #2"
    Example: "average of #1 for each #2"
    """

    def __init__(self):
        super().__init__(
            properties_tokens=IdentifyOperatorAggregate.properties_tokens,
            properties_order=IdentifyOperatorAggregate.properties_order
        )

    @overrides
    def operator_indicators(self) -> List[str]:
        return ['for each']

    def _identify_op(self, step: str, references: List[int]):
        # GROUP step - contains the phrase 'for each'
        if _is_in('for each', step) and len(references) > 0:
            return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        # need to extract the group values and keys
        # split the step to the aggregated values (prefix) and keys (suffix)
        value, key = step.split('for each')
        val_refs = self.extract_references(value)
        key_refs = self.extract_references(key)
        # check if both parts actually contained references
        arg_value = value.split()[-1] if len(val_refs) == 0 else "#%s" % val_refs[0]
        arg_key = key.split()[-1] if len(key_refs) == 0 else "#%s" % key_refs[0]
        return {ArgumentType.GROUP_VALUE: arg_value, ArgumentType.GROUP_KEY: arg_key}


class IdentifyOperatorIntersect(IdentifyOperator):
    """
    Example: "countries in both #1 and #2"
    Example: "#3 of both #4 and #5"
    """

    def __init__(self):
        super().__init__()

    def operator_indicators(self) -> List[str]:
        return ['of both', 'in both', 'at both', 'by both', 'between both', \
                     'for both', 'are both', 'both of', 'both']+['and']

    def _identify_op(self, step: str, references: List[int]):
        if len(references) >= 2 and _is_in('both', step) and _is_in('and', step):
            return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        interesect_expr = None
        for expr in ['of both', 'in both', 'at both', 'by both', 'between both', \
                     'for both', 'are both', 'both of', 'both']:
            if _is_in(expr, step):
                interesect_expr = expr
                break
        projection, intersection = step.split(interesect_expr)
        # add all previous references as the intersection arguments
        refs = self.extract_references(intersection)
        intersection_refs = ["#%s" % ref for ref in refs]
        return {ArgumentType.INTERSECTION_PROJECTION: projection,
                ArgumentType.INTERSECTION_INTERSECTION: intersection_refs}


class IdentifyOperatorProject(IdentifyOperator):
    """
    Example: "first name of #2"
    Example: "who was #1 married to"
    """

    def __init__(self):
        super().__init__()

    @overrides
    def operator_indicators(self) -> List[str]:
        return ['#REF']

    def _identify_op(self, step: str, references: List[int]):
        if len(references) == 1 and \
                re.search("[\s]+[#]+[0-9\s]+", step):
            return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        # extract the referenced objects
        ref = "#%s" % references[0]
        # extract the projected relation phrase, anonymized
        projection = step.replace(ref, "#REF")
        return {ArgumentType.PROJECT_PROJECTION: projection, ArgumentType.PROJECT_SUB: ref}


class IdentifyOperatorSelect(IdentifyOperator):
    """
    Example: "countries"
    """

    def __init__(self):
        super().__init__()

    def _identify_op(self, step: str, references: List[int]):
        # SELECT step has no references to previous steps.
        if len(references) == 0:
            return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        return {ArgumentType.SELECT_SUB: step}


class IdentifyOperatorSort(IdentifyOperator):
    """
    Example: "#1 sorted by #2"
    Example: "#1 ordered by #2"
    """

    def __init__(self):
        super().__init__()

    @overrides
    def operator_indicators(self) -> List[str]:
        return ['sort by', 'sorted by', 'order by', 'ordered by']

    def _identify_op(self, step: str, references: List[int]):
        for expr in self.operator_indicators():
            if _is_in(expr, step):
                return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        sort_expr = None
        for expr in self.operator_indicators():
            if _is_in(expr, step):
                sort_expr = expr
        objects, order = [frag.strip() for frag in step.split(sort_expr)]
        return {ArgumentType.SORT_SUB: objects, ArgumentType.SORT_ORDER: order}


class IdentifyOperatorSuperlative(IdentifyOperator):
    """
    Example: "#1 where #2 is highest"
    Example: "#1 where #2 is smallest"
    """

    def __init__(self):
        super().__init__(
            properties_tokens={
                "max": _get_variations(['most', 'biggest', 'largest', 'highest', 'longest']),
                "min": _get_variations(['least', 'fewest', 'smallest', 'lowest',  'shortest', 'earliest']),
            }
        )

    def _identify_op(self, step: str, references: List[int]):
        superlatives = [x for v in self._properties_tokens.values() for x in v]
        superlatives = [f"{pref} {sup}" for pref in ["is", "is the", "are", "are the"] for sup in superlatives]
        if step.startswith('#') and len(references) == 2 \
                and _is_in('where', step) and (step.startswith('#') or step.startswith('the #')):
            for s in superlatives:
                if _is_in(s, step):
                    return True
        return False

    def _extract_args(self, step: str, references: List[int]):
        entity_ref, attribute_ref = references
        return {ArgumentType.SUPERLATIVE_SUB: "#%s" % entity_ref,
                ArgumentType.SUPERLATIVE_ATTRIBUTE: "#%s" % attribute_ref}


class IdentifyOperatorUnion(IdentifyOperator):
    """
    Example: "#1 or #2"
    Example: "#1, #2, #3, #4"
    Example: "#1 and #2"
    """

    def __init__(self):
        super().__init__()

    def _identify_op(self, step: str, references: List[int]):
        if len(references) > 1:
            substitute_step = step.replace('and', ',').replace('or', ',')
            is_union = re.search("^[#0-9,\s]+$", substitute_step)
            return is_union
        return False

    def _extract_args(self, step: str, references: List[int]):
        args = []
        for ref in references:
            args += ["#%s" % ref]
        return {ArgumentType.UNION_SUB: args}


_operator_to_identifier = {
    QDMROperator.SELECT: IdentifyOperatorSelect(),
    QDMROperator.FILTER: IdentifyOperatorFilter(),
    QDMROperator.PROJECT: IdentifyOperatorProject(),
    QDMROperator.AGGREGATE: IdentifyOperatorAggregate(),
    QDMROperator.GROUP: IdentifyOperatorGroup(),
    QDMROperator.SUPERLATIVE: IdentifyOperatorSuperlative(),
    QDMROperator.COMPARATIVE: IdentifyOperatorComparative(),
    QDMROperator.UNION: IdentifyOperatorUnion(),
    QDMROperator.INTERSECTION: IdentifyOperatorIntersect(),
    QDMROperator.DISCARD: IdentifyOperatorDiscard(),
    QDMROperator.SORT: IdentifyOperatorSort(),
    QDMROperator.BOOLEAN: IdentifyOperatorBoolean(),
    QDMROperator.ARITHMETIC: IdentifyOperatorArithmetic(),
    QDMROperator.COMPARISON: IdentifyOperatorComparison()
}


def get_identifier(operator: QDMROperator) -> IdentifyOperator:
    return _operator_to_identifier[operator]


def get_operators() -> Iterable[QDMROperator]:
    return _operator_to_identifier.keys()

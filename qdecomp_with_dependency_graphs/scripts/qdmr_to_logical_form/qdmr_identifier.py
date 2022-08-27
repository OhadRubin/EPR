from typing import Dict, Iterable
import traceback

from .operator_identifier import *
from .utils_ import *
import logging
_logger = logging.getLogger(__name__)

class QDMRStep:
    def __init__(self, step_text: str, operator: QDMROperator, properties: Iterable[str], arguments: Dict[ArgumentType, Iterable[str]]):
        self.step: str = step_text
        self.operator: QDMROperator = operator
        self.properties: Iterable[str] = properties
        self.arguments: Dict[ArgumentType, Iterable[str]] = arguments

    def __str__(self):
        properties = f"[{','.join(self.properties)}]" if self.properties else ""
        arguments = self.arguments and ";".join(sorted(f"{k.get_arg_name()}: {v_itm}" for k, v in self.arguments.items() for v_itm in v))
        return f"{self.operator.upper()}{properties}({arguments})"

    def get_references(self, arg_type: ArgumentType = None) -> Union[Dict[ArgumentType, Iterable[int]], Iterable[int]]:
        res = {}
        arg_groups = [arg_type] if arg_type else self.arguments.keys()
        for group in arg_groups:
            refs = []
            for arg in self.arguments[group]:
                refs.extend(int(x) for x in re.findall(r"#(\d+)", arg))
            res[group] = refs
        if arg_type:
            return res[arg_type]
        return res


class StepIdentifier(object):

    def step_type(self, step_text) -> QDMROperator:
        potential_operators = set()
        for op in get_operators():
            identifier = get_identifier(op)
            if identifier.identify_op(step_text):
                potential_operators.add(op)
        # no matching operator found
        if len(potential_operators) == 0:
            return None
        operators = potential_operators.copy()
        # duplicate candidates
        if len(operators) > 1:
            # avoid project duplicity with aggregate
            if QDMROperator.PROJECT in operators:
                operators.remove(QDMROperator.PROJECT)
            # avoid filter duplcitiy with comparative, superlative, sort, discard
            if QDMROperator.FILTER in operators:
                operators.remove(QDMROperator.FILTER)
            # return boolean (instead of intersect)
            if QDMROperator.BOOLEAN in operators:
                operators = {QDMROperator.BOOLEAN}
            # return intersect (instead of filter)
            # if "intersect" in operators:
            #     operators = {"intersect"}
            # return superlative (instead of comparative)
            if QDMROperator.SUPERLATIVE in operators:
                operators = {QDMROperator.SUPERLATIVE}
            # return group (instead of arithmetic)
            if QDMROperator.GROUP in operators:
                operators = {QDMROperator.GROUP}
            # return comparative (instead of discard)
            if QDMROperator.COMPARATIVE in operators:
                operators = {QDMROperator.COMPARATIVE}
            # return intersection (instead of comparison)
            if QDMROperator.INTERSECTION in operators:
                operators = {QDMROperator.INTERSECTION}
            # return arithmetic (instead of aggregation)
            if QDMROperator.ARITHMETIC in operators:
                operators = {QDMROperator.ARITHMETIC}
        assert (len(operators) == 1)
        return list(operators)[0]

    def identify(self, step_text) -> QDMRStep:
        operator = self.step_type(step_text)
        identifier = get_identifier(operator)
        props = identifier.extract_properties(step_text)
        args = identifier.extract_args(step_text)
        return QDMRStep(step_text, operator, props, args)


class QDMRProgramBuilder(object):
    def __init__(self, qdmr_text):
        self.qdmr_text: str = qdmr_text
        self.steps: List[QDMRStep] = None
        self.operators: List[QDMROperator] = None
        self.program = None

    def build(self):
        try:
            self.get_operators()
            self.build_steps()
        except:
            _logger.debug("Unable to identify all steps: %s" % self.qdmr_text)
        return True

    def build_steps(self):
        self.steps = []
        steps = parse_decomposition(self.qdmr_text)
        step_identifier = StepIdentifier()
        for step_text in steps:
            try:
                step = step_identifier.identify(step_text)
            except:
                _logger.debug("Unable to identify step: %s" % step_text)
                # traceback.print_exc()
                step = None
            self.steps += [step]
        return self.steps

    def get_operators(self):
        self.operators = []
        steps = parse_decomposition(self.qdmr_text)
        step_identifier = StepIdentifier()
        for step_text in steps:
            try:
                op = step_identifier.step_type(step_text)
            except:
                _logger.debug("Unable to identify operator: %s" % step_text)
                op = None
            self.operators += [op]
        return self.operators

    def build_program(self):
        raise NotImplementedError
        return True

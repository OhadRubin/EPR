import re
from typing import List, Dict, Tuple, Any, Iterable

import abc
from itertools import product
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from ortools.sat.python import cp_model

from overrides import overrides

from qdecomp_with_dependency_graphs.dependencies_graph.check_frequency import BaseMatcher, DataStructuresParam
from qdecomp_with_dependency_graphs.dependencies_graph.check_frequency__structural_constraints import get_operator, SingleTagViolationMatcher, \
    SpanSingleInOutViolationMatcher, DuplicateSingleOutViolationMatcher, InconsistencyDependenciesMatcher, \
    SpanSingleRepresentativeViolationMatcher, ConnectivityViolationMatcher, BoundAllowedCombinationsViolationMatcher, \
    AllowedCombinationsViolationMatcher, DuplicateFromUsedDUPViolationMatcher
from qdecomp_with_dependency_graphs.dependencies_graph.data_types import TokensDependencies
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.logical_form_matcher import CleanNormalizationRule


#############################
#       Constraints         #
#############################

@dataclass
class ILPConstraintParam:
    model: cp_model.CpModel
    x: Dict[Tuple[int, int, str], Any]
    N: Iterable[int]
    T: Iterable[str] = None
    Operators: Iterable[str] = None
    tokens: List[str] = None
    vars: Dict[str, Dict[str, Any]] = None
    curr_arc_tags: List[Tuple[int, int, str]] = None

    x_original: Dict[Tuple[int, int, str], Any] = None
    T_original: Iterable[str] = None


class ILPConstraint(abc.ABC):
    matcher: BaseMatcher = None

    def __init__(self, explicit: bool = True):
        self.explicit = explicit

    def add(self, p: ILPConstraintParam):
        raise NotImplementedError()

    def is_violate(self, question_id:str, tokens: List[str], arcs: List[Tuple[int, int]], arcs_tags: List[str]):
        if self.matcher is None:
            return False
        tokens_deps = TokensDependencies.from_tokens(tokens=tokens, token_tags=[None]*len(tokens),
                                                     dependencies=[(i,j,d) for (i,j), d in zip(arcs, arcs_tags)])
        param = DataStructuresParam(question_id=question_id, tok_dep=tokens_deps)
        return self.matcher.is_match(param)


class SingleTagPerArcConstraint(ILPConstraint):
    matcher: BaseMatcher = SingleTagViolationMatcher()

    @overrides
    def add(self, p: ILPConstraintParam):
        # single tag per arc
        for i, j in product(p.N, p.N):
            # p.model.Add(sum(p.x[i, j, t] for t in p.T) <= 1)
            p.model.Add(sum(p.x_original[i, j, t] for t in p.T_original) <= 1)


class SpanSingleInOutConstraint(ILPConstraint):
    matcher: BaseMatcher = SpanSingleInOutViolationMatcher()

    @overrides
    def add(self, p: ILPConstraintParam):
        # span - single out/in
        for i in p.N:
            p.model.Add(sum(p.x[i, j, 'span'] for j in p.N) <= 1)
            p.model.Add(sum(p.x[k, i, 'span'] for k in p.N) <= 1)


class DuplicateSingleOutConstraint(ILPConstraint):
    matcher: BaseMatcher = DuplicateSingleOutViolationMatcher()

    @overrides
    def add(self, p: ILPConstraintParam):
        # duplicate - single out
        for i in p.N:
            p.model.Add(sum(p.x[i, j, 'duplicate'] for j in p.N) <= 1)


class DuplicateDependencyForDUPConstraint(ILPConstraint):
    matcher: BaseMatcher = DuplicateFromUsedDUPViolationMatcher()

    @overrides
    def add(self, p: ILPConstraintParam):
        # sum_jt(x_ijt + xjit)>=1 => sum_j(x_i,j,duplicate)>=1
        for i in p.N:
            if p.tokens[i].upper() == '[DUP]':
                p.model.Add(-sum(p.x[i,j,t]+p.x[j,i,t] for j,t in product(p.N, p.T)) +
                            2*len(p.N)*len(p.T)*sum(p.x[i, j, 'duplicate'] for j in p.N) >= 0)


class OperatorsConsistencyConstraint(ILPConstraint):
    matcher: BaseMatcher = InconsistencyDependenciesMatcher()

    @overrides
    def add(self, p: ILPConstraintParam):
        self.verify_auxiliaries(p, self.explicit)
        y_out = p.vars['y_out']
        # operators consistency - single operator for all out arcs
        for i in p.N:
            p.model.Add(sum(y_out[i, o] for o in p.Operators if o != 'duplicate') <= 1)

    @staticmethod
    def verify_auxiliaries(p: ILPConstraintParam, explicit: bool):
        # operators auxiliary variables
        # y_in[j,o] - indicates that there is an arc (i,j) with tag t where operator(t)=o
        # y_in = {(j, o): model.NewBoolVar(f'y_in[{j},{o}]') for j, o in product(N, Operators)}
        # for j, o in product(N, Operators):
        #     model.Add(-sum(x[i, j, t] for t in T if get_operator(t) == o for i in N) + len(N)*y_in[j, o] >= 0)
        #     model.Add(-y_in[j, o] + sum(x[i, j, t] for t in T if get_operator(t) == o for i in N) <= 0)
        # y_out[i,o] - indicates that there is an arc (i,j) with tag t where operator(t)=o
        if 'y_out' not in p.vars:
            y_out = {(i, o): p.model.NewBoolVar(f'y_out[{i},{o}]') for i, o in product(p.N, p.Operators)}
            for i, o in product(p.N, p.Operators):
                if explicit:
                    p.model.Add(-sum(p.x[i, j, t] for j, t in product(p.N, p.T) if get_operator(t) == o) + len(p.N)*len(p.T) * y_out[i, o] >= 0)
                    p.model.Add(-y_out[i, o] + sum(p.x[i, j, t] for j, t in product(p.N, p.T) if get_operator(t) == o) >= 0)
                else:
                    p.model.Add(sum(p.x[i, j, t] for j, t in product(p.N, p.T) if get_operator(t) == o) > 0).OnlyEnforceIf(y_out[i,o])
                    p.model.Add(sum(p.x[i, j, t] for j, t in product(p.N, p.T) if get_operator(t) == o) == 0).OnlyEnforceIf(y_out[i,o].Not())
            p.vars['y_out'] = y_out


class SpanRepresentativeRightmostConstraint(ILPConstraint):
    matcher: BaseMatcher = SpanSingleRepresentativeViolationMatcher()

    @overrides
    def add(self, p: ILPConstraintParam):
        OperatorsConsistencyConstraint.verify_auxiliaries(p, self.explicit)
        y_out = p.vars['y_out']
        # span representative - the rightmost token in the span
        # y_out[j, span] = 1 => (1) sum_{i,t!=span}x_{i,j,t} = 0 and (2) sum_{i,span}x_{i,j,t} <= 1
        for j in p.N:
            if self.explicit:
                # model.Add((len(Operators)-1)*y_out[j, 'span']-sum((1-y_in[j, o]) for o in Operators if o != 'span') <= 0)
                # (1)
                p.model.Add(sum(p.x[i, j, t] for i, t in product(p.N, p.T) if t != 'span') + len(p.N) * (len(p.T) - 1) * y_out[j, 'span'] <= len(p.N) * (len(p.T) - 1))
                # (2)
                p.model.Add(sum(p.x[i, j, 'span'] for i in p.N) + len(p.N) * y_out[j, 'span'] <= len(p.N) + 1)
            else:
                p.model.Add(sum(p.x[i, j, t] for i, t in product(p.N, p.T) if t != 'span') == 0).OnlyEnforceIf(y_out[j, 'span'])
                p.model.Add(sum(p.x[i, j, 'span'] for i in p.N) <= 1).OnlyEnforceIf(y_out[j, 'span'])


class ConnectivityConstraint(ILPConstraint):
    matcher: BaseMatcher = ConnectivityViolationMatcher()

    @overrides
    def add(self, p: ILPConstraintParam):
        # connectivity
        r = {i: p.model.NewBoolVar(f'r[{i}]') for i in p.N}
        r1 = {i: p.model.NewBoolVar(f'r1[{i}]') for i in p.N}
        r2 = {i: p.model.NewBoolVar(f'r2[{i}]') for i in p.N}
        # at most a single root
        p.model.Add(sum(r.values()) == 1)
        for i in p.N:
            if self.explicit:
                # r1[i]: at least one outgoing dependency
                p.model.Add(-r1[i] + sum(p.x[i, j, t] for j, t in product(p.N, p.T)) + sum(p.x[k, i, 'span'] for k in p.N) >= 0)
                p.model.Add(len(p.N)*(len(p.T)+1)*r1[i] - sum(p.x[i, j, t] for j, t in product(p.N, p.T)) - sum(p.x[k, i, 'span'] for k in p.N) >= 0)

                # r2[i]: no outgoing span and no incoming dependency
                p.model.Add(-(1-r2[i]) + sum(p.x[k, i, t] for k, t in product(p.N, p.T) if t != 'span') + sum(p.x[i, j, 'span'] for j in p.N) >= 0)
                p.model.Add(len(p.N)*len(p.T)*(1-r2[i]) - sum(p.x[k, i, t] for k, t in product(p.N, p.T) if t != 'span') - sum(p.x[i, j, 'span'] for j in p.N) >= 0)

                # r[i] <=> r1[i] and r2[i]
                p.model.Add(-2*r[i] + r1[i] + r2[i] >= 0)
                p.model.Add(r[i] - r1[i] - r2[i] >= -1)
            else:
                p.model.Add(sum(p.x[i, j, t] for j, t in product(p.N, p.T)) + sum(p.x[k, i, 'span'] for k in p.N) > 0).OnlyEnforceIf(r1[i])
                p.model.Add(sum(p.x[i, j, t] for j, t in product(p.N, p.T)) + sum(p.x[k, i, 'span'] for k in p.N) <= 0).OnlyEnforceIf(r1[i].Not())

                p.model.Add(sum(p.x[k, i, t] for k, t in product(p.N, p.T) if t != 'span') + sum(p.x[i, j, 'span'] for j in p.N) <= 0).OnlyEnforceIf(r2[i])
                p.model.Add(sum(p.x[k, i, t] for k, t in product(p.N, p.T) if t != 'span') + sum(p.x[i, j, 'span'] for j in p.N) > 0).OnlyEnforceIf(r2[i].Not())

                p.model.Add(r1[i] + r2[i] == 2).OnlyEnforceIf(r[i])
                p.model.Add(r1[i] + r2[i] < 2).OnlyEnforceIf(r[i].Not())


class ValidDependenciesCountConstraint(ILPConstraint):
    matcher: BoundAllowedCombinationsViolationMatcher = BoundAllowedCombinationsViolationMatcher()

    @overrides
    def add(self, p: ILPConstraintParam):
        for c in self.matcher.combinations:
            for i in p.N:
                p.model.Add(c.count_condition(sum(p.x[i,j,t] for j,t in product(p.N, p.T)
                                                  if re.match(c.patten, t))))


class ValidDependenciesCombinationConstraint(ILPConstraint):
    matcher = AllowedCombinationsViolationMatcher = AllowedCombinationsViolationMatcher()

    @overrides
    def add(self, p: ILPConstraintParam):
        # validate args
        mask = self._get_mask(tokens=p.tokens)
        # c_i = 1 iff the span of i is not empty
        c = {0: mask[0]}
        c.update({i: p.model.NewBoolVar(f'c_comb[{i}]') for i in p.N if i > 0})
        # c_i := sum_k<i(x_k,i,span * c_k) + mask_i
        # c_i <=> or_k<i(x_k,i,span and c_k) or mask_i
        k_i = [(k,i) for k,i in product(p.N, p.N) if k<i]
        c_pair = {(k,i): p.model.NewBoolVar(f'c_comb[{k},{i}]') for k,i in k_i}

        # c_k,i = 1 <=> x_k,i,span and c_k
        for k,i in k_i:
            if self.explicit:
                p.model.Add(2*c_pair[k, i] - (p.x[k, i, 'span']+c[k]) <= 0)
                p.model.Add(-c_pair[k, i] + (p.x[k, i, 'span']+c[k]) <= 1)
            else:
                p.model.Add(p.x[k, i, 'span'] + c[k] == 2).OnlyEnforceIf(c_pair[k,i])
                p.model.Add(p.x[k, i, 'span'] + c[k] < 2).OnlyEnforceIf(c_pair[k,i].Not())

        for i in p.N:
            if i == 0: continue
            if p.tokens[i].upper() != '[DUP]':
                # c_i = m_i or Or_k<i(c_k,i)
                if self.explicit:
                    p.model.Add(sum(c_pair[k,i] for k in p.N if k<i)+mask[i]+(1-c[i])>=1)
                    p.model.Add(-(sum(c_pair[k,i] for k in p.N if k<i)+mask[i])+(i+1)*c[i]>=0)  # i is 0-based, so we need i+1
                else:
                    p.model.Add(sum(c_pair[k,i] for k in p.N if k<i)+mask[i] >= 1).OnlyEnforceIf(c[i])
                    p.model.Add(sum(c_pair[k,i] for k in p.N if k<i)+mask[i] == 0).OnlyEnforceIf(c[i].Not())
            else:
                # c_i = m_i or Or_k<i(c_k,i) or Or_k<i(x_i,k,dup*m_k)
                if self.explicit:
                    p.model.Add(sum(c_pair[k,i] for k in p.N if k<i) + sum(p.x[i,k,'duplicate']*mask[k] for k in p.N if k<i)+mask[i]+(1-c[i])>=1)
                    p.model.Add(-(sum(c_pair[k,i] for k in p.N if k<i) + sum(p.x[i,k,'duplicate']*mask[k] for k in p.N if k<i) +mask[i])+(i+1)*c[i]>=0) # i is 0-based, so we need i+1
                else:
                    p.model.Add(sum(c_pair[k,i] for k in p.N if k<i) + sum(p.x[i,k,'duplicate']*mask[k] for k in p.N if k<i) + mask[i] >= 1).OnlyEnforceIf(c[i])
                    p.model.Add(sum(c_pair[k,i] for k in p.N if k<i)+ sum(p.x[i,k,'duplicate']*mask[k] for k in p.N if k<i) + mask[i] == 0).OnlyEnforceIf(c[i].Not())

        triggers_and_groups = self._get_triggers_and_groups(p)
        tags = set([x for t, g in triggers_and_groups for x in [t]+g])
        y_out = {(i, t): p.model.NewBoolVar(f'y_allow[{i},{t}]') for i,t in product(p.N, tags)}
        for i,t in y_out.keys():
            p.model.Add(sum(p.x[i,j,t] for j in p.N)>0).OnlyEnforceIf(y_out[i,t])
            p.model.Add(sum(p.x[i,j,t] for j in p.N)==0).OnlyEnforceIf(y_out[i,t].Not())

        for i in p.N:
            for trig, group in triggers_and_groups:
                p.model.Add(sum(y_out[i,t] for t in group)+c[i]>=len(group)).OnlyEnforceIf(y_out[i,trig])


    _masked_tokens = CleanNormalizationRule()._tokens
    @staticmethod
    def _get_mask(tokens: List[str]):
        def mask(token: str):
            if '[' in token:
                return False
            if token in ValidDependenciesCombinationConstraint._masked_tokens:
                return False
            return True

        return [int(mask(t)) for t in tokens]

    def _get_triggers_and_groups(self, p: ILPConstraintParam):
        res = []
        for t in p.T:
            for c in self.matcher.combinations:
                group = self.matcher.get_variations(t, c)
                if group:
                    res.append((t, group))
        return res


###### sanity check constraints

class LimitArcsNumberConstraint(ILPConstraint):
    @overrides
    def add(self, p: ILPConstraintParam):
        p.model.Add(sum(p.x[i, j, t] for i, j, t in product(p.N, p.N, p.T)) <= len(p.curr_arc_tags))


class DropArcsOnlyConstraint(ILPConstraint):
    @overrides
    def add(self, p: ILPConstraintParam):
        allowed_arcs = set(p.curr_arc_tags)
        for i, j, t in product(p.N, p.N, p.T):
            if (i, j, t) not in allowed_arcs:
                p.model.Add(p.x[i, j, t] == 0)

class KeepArcsConstraint(ILPConstraint):
    @overrides
    def add(self, p: ILPConstraintParam):
        allowed_arcs = set((i,j) for i,j,_ in p.curr_arc_tags)
        for i, j, t in product(p.N, p.N, p.T):
            if (i, j) not in allowed_arcs:
                p.model.Add(p.x[i, j, t] == 0)


#############################
#       Decoder             #
#############################

@dataclass
class ILPResult:
    arcs: List[Tuple[int, int]]
    arc_tags: List[str]
    objective: int
    satisfy_all: bool


class ILPDecoder:
    def __init__(self, tags_by_index: Dict[int, str], max_time_in_seconds: float = None,
                 ignore_concatenated_tags: bool = True,
                 limit_arcs_number: bool = False, drop_arcs_only: bool = False, keep_arcs: bool = False,
                 sanity_check: bool = False, explicit: bool = False, skip_violation_check: bool = False):
        self.max_time_in_seconds = max_time_in_seconds
        self.tags_by_index: Dict[int, str] = tags_by_index
        self.index_by_tag: Dict[str, int] = {v: k for k, v in self.tags_by_index.items()}
        self.special_tokens = ['[RSC]', '[DUP]', '[DUM]']
        self.constraints: List[ILPConstraint] = [
            SingleTagPerArcConstraint(explicit=explicit),
            SpanSingleInOutConstraint(explicit=explicit),
            DuplicateSingleOutConstraint(explicit=explicit),
            DuplicateDependencyForDUPConstraint(explicit=explicit),
            OperatorsConsistencyConstraint(explicit=explicit),
            SpanRepresentativeRightmostConstraint(explicit=explicit),
            ConnectivityConstraint(explicit=explicit),
            ValidDependenciesCountConstraint(explicit=explicit),
            ValidDependenciesCombinationConstraint(explicit=explicit),
        ]

        if limit_arcs_number:
            self.constraints.append(LimitArcsNumberConstraint(explicit=explicit))
        if drop_arcs_only:
            self.constraints.append(DropArcsOnlyConstraint(explicit=explicit))
        if keep_arcs:
            self.constraints.append(KeepArcsConstraint(explicit=explicit))
        self.sanity_check = sanity_check
        self.skip_violation_check = skip_violation_check
        self.ignore_concatenated_tags = ignore_concatenated_tags

    def _get_log_probabilities(self, arc_prob: np.ndarray, arc_tag_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # return list(self.tags_by_index.values())

        # n, _, t = arc_tag_prob.shape
        # # split multilabel
        # for ind, tag in self.tags_by_index.items():
        #     if '&' in tag:
        #         parts = tag.split('&')
        #         parts_inds = [self.index_by_tag[x] for x in parts]
        #         arc_tag_prob[:, :, parts_inds] += arc_tag_prob[:, :, ind].reshape(n, n, 1)/len(parts)
        #         arc_tag_prob[:, :, ind] = 0

        # ignore NONE
        none_ind = self.index_by_tag.get('NONE')
        if none_ind is None:
            none_log_prob = np.log(1-arc_prob)
            arc_tag_log_prob = np.log(np.expand_dims(arc_prob, axis=-1)) + np.log(arc_tag_prob)
        else:
            # make sure valid distribution (in terms of none)
            arc_tag_prob[:, :, none_ind] = 0
            arc_tag_log_prob = np.log(arc_tag_prob)
            none_log_prob = np.log(1-arc_tag_prob.sum(-1))

        return arc_tag_log_prob, none_log_prob

    def decode(self, question_id: str, tokens: List[str], arc_prob: np.ndarray, arc_tag_prob: np.ndarray,
               curr_arcs: List[Tuple[int, int]], curr_arc_tags: List[str], precision: int = 1) -> ILPResult:

        satisfy_all = False
        if not any(c.is_violate(question_id, tokens, curr_arcs, curr_arc_tags) for c in self.constraints):
            # current solution satisfies all constraints
            if self.sanity_check or self.skip_violation_check:
                satisfy_all = True
            else:
                return None

        model = cp_model.CpModel()
        objective_terms = []

        arc_and_tag_log_prob, none_log_prob = self._get_log_probabilities(arc_prob=arc_prob, arc_tag_prob=arc_tag_prob)
        N = range(len(tokens))  # question=q1,...,qn
        T_original = [t for t in self.tags_by_index.values() if t != 'NONE']  # available tags
        T = [t for t in T_original if '&' not in t]  # available pure (native) tags
        Operators = set([get_operator(t) for t in T])

        # declare x_ijt - indicates whether the tag of arc (i,j) is t
        x = defaultdict(
            lambda: 0,
            {(i, j, t): model.NewBoolVar(f'x[{i},{j},{t}]') for i, j, t in product(N, N, T)
             if not
             (
                 # spans are left to right
                 (t == 'span' and i >= j) or
                 # duplicate is from [DUP] to no-special token
                 (t == 'duplicate' and (tokens[i] != '[DUP]' or tokens[j] in self.special_tokens)) or
                 # zero prob
                 not np.isfinite(arc_and_tag_log_prob[i, j, self.index_by_tag[t]])
             )
             }
        )

        if self.ignore_concatenated_tags:
            T_original = T
            x_original = x
        else:
            # model tags, including concatenations (&)
            x_original = defaultdict(
                lambda: 0,
                {(i, j, t): model.NewBoolVar(f'x_original[{i},{j},{t}]') for i, j, t in product(N, N, T_original)
                 if all((not isinstance(x[i,j,p], int)) for p in t.split('&'))}
            )
            #make sure that if x_ijt was chose there is some original tag that includes it
            tag_to_original_tags = defaultdict(set)
            for t in T_original:
                for p in t.split('&'):
                    tag_to_original_tags[p].add(t)
            for i, j, t in product(N, N, T):
                if isinstance(x[i,j,t], int): continue
                model.Add(sum(x_original[i,j,p] for p in tag_to_original_tags[t]) >= 1).OnlyEnforceIf(x[i,j,t])
            # make sure that if x_original_ijt was chose, all it's partial tags are chosen
            for i, j, t in product(N, N, T_original):
                if isinstance(x_original[i,j,t], int): continue
                if '&' not in t:
                    model.AddImplication(x_original[i,j,t], x[i,j,t])
                else:
                    t_parts = t.split('&')
                    for p in t_parts:
                        model.AddImplication(x_original[i,j,t], x[i,j,p])
                        model.AddImplication(x_original[i,j,p], x_original[i,j,t].Not())

        # no need due to single tag constraint
        # none indicators
        # x_none = {(i, j): model.NewBoolVar(f'x_none[{i},{j}]') for i, j in product(N, N)}
        # for i, j in product(N, N):
        #     model.Add(sum(x[i,j,t] for i, j, t in product(N, N, T)) == 0).OnlyEnforceIf(x_none[i,j])
        #     model.Add(sum(x[i,j,t] for i, j, t in product(N, N, T)) > 0).OnlyEnforceIf(x_none[i,j].Not())


        # objective maximal prob
        def format_coeff(probs: np.ndarray):
            # -inf => proper number
            probs = np.nan_to_num(probs)
            # avoid overflow
            probs = np.clip(probs * pow(10, precision), a_min=np.iinfo(np.int32).min, a_max=np.iinfo(np.int32).max)
            return probs.astype(np.int32)

        arc_and_tag_log_prob = format_coeff(arc_and_tag_log_prob)
        none_log_prob = format_coeff(none_log_prob)
        # objective_terms.append(sum(x[i, j, t]*arc_and_tag_log_prob[i, j, self.index_by_tag[t]] for i, j, t in product(N, N, T)) +
        #                        sum((1-sum(x[i, j, t] for t in T))*none_log_prob[i, j] for i, j in product(N, N)))
        objective_terms.append(sum(x_original[i, j, t]*arc_and_tag_log_prob[i, j, self.index_by_tag[t]] for i, j, t in product(N, N, T_original)) +
                               sum((1-sum(x_original[i, j, t] for t in T_original))*none_log_prob[i, j] for i, j in product(N, N)))


        # constraints
        params = ILPConstraintParam(model=model, x=x, N=N, T=T, Operators=Operators, tokens=tokens, vars={},
                                    x_original=x_original, T_original=T_original,
                                    curr_arc_tags=[(i, j, t) for (i, j), t in zip(curr_arcs, curr_arc_tags)])
        for c in self.constraints:
            c.add(params)

        # objective
        objective = sum(objective_terms)
        model.Maximize(objective)

        # add independent solution as a hint
        if curr_arcs:
            current = defaultdict(lambda: 0, {(i, j, t): 1 for (i, j), t in zip(curr_arcs, curr_arc_tags)})
            for i, j, t in product(N, N, T):
                if self.sanity_check:
                    model.Add(x[i, j, t] == current[i, j, t])  # for debugging
                else:
                    model.AddHint(x[i, j, t], current[i, j, t])

        # solve
        solver = cp_model.CpSolver()
        if self.max_time_in_seconds:
            solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            arcs = []
            arcs_tags = []
            for i, j, t in product(N, N, T):
                if solver.BooleanValue(x[i, j, t]):
                    arcs.append((i, j))
                    arcs_tags.append(t)
            return ILPResult(arcs=arcs, arc_tags=arcs_tags, objective=int(solver.Value(objective)), satisfy_all=satisfy_all)
        else:
            # todo: independent decode
            # assumption = [model.VarIndexToVarProto(varIndex) for varIndex in solver.ResponseProto().sufficient_assumptions_for_infeasibility]
            raise Exception(f'No solution found, status: {solver.StatusName(status)}'+
                            (f', satisfy_all: {satisfy_all}' if self.sanity_check else ''))

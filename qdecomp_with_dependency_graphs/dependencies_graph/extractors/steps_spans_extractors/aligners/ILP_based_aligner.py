from collections import defaultdict
from typing import List, Set, Tuple

from spacy.tokens.doc import Doc
import numpy as np
from ortools.sat.python import cp_model

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import QDMROperation
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_spans_extractors.aligners.base_aligner import BaseAligner
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition

class ILPAligner(BaseAligner):
    def __init__(self, max_time_in_seconds: float = None):
        super().__init__()
        self.max_time_in_seconds = max_time_in_seconds

    def align(self, question: Doc, steps: List[Doc], steps_operators: List[QDMROperation],
              index_to_steps: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
        n = len(question)  # question=q1,...,qn
        n_k = [len(x) for x in steps]  # sk =sk_1,...,sk_nk
        m = len(steps)  # steps: s1,...,sm

        # a^k_ij - potential alignment indicator
        a = np.zeros((m, n, max(n_k)), dtype=np.int)
        for i, steps_list in enumerate(index_to_steps):
            for k, j in steps_list:
                a[k, i, j] = 1

        # init b^k_ij - exact match indicator
        b = np.zeros((m, n, max(n_k)), dtype=np.int)
        for i, steps_list in enumerate(index_to_steps):
            for k, j in steps_list:
                b[k, i, j] = question[i].text.lower() == steps[k][j].text.lower()

        model = cp_model.CpModel()
        objective_terms = []
        c = defaultdict(lambda: 0)
        c['min'] = 1000
        c['unique'] = 100
        c['seq'] = 10
        c['exact'] = 1
        c['ref'] = 1

        # create x^k_ij - indicates whether t_i, s^k_j are aligned
        x = {(k, i, j): model.NewBoolVar(f'x[{k},{i},{j}]') for k in range(m) for i in range(n) for j in range(n_k[k])}

        # minimalism
        objective_terms.append(c['min'] * sum(x.values()))

        # validity constraints
        for k in range(m):
            for i in range(n):
                for j in range(n_k[k]):
                    model.Add(x[k, i, j] <= a[k, i, j])

        # step coverage constraints
        for k in range(m):
            for j in range(n_k[k]):
                model.Add(-sum(a[k, i, j] for i in range(n)) + n * sum(x[k, i, j] for i in range(n)) >= 0)

        # question coverage constraints - ? we prefer but it is not must... (see uniqueness)
        # for i in range(n):
        #     model.Add(-sum(a[k, i, j] for k in range(m) for j in range(n_k[k])) +
        #               n * sum(x[k, i, j] for i in range(n) for k in range(m) for j in range(n_k[k])) >= 0)

        # sequences preference
        y = {(d, k, i, j): model.NewBoolVar(f'y[{d},{k},{i},{j}]') for k in range(m) for i in range(n) for j in
             range(n_k[k])
             for d in range(1, min(n - i, n_k[k] - j))}
        for (d, k, i, j), y_var in y.items():
            model.Add(-(d + 1) * y_var + sum(x[k, i + p, j + p] for p in range(d + 1)) >= 0)
            model.Add(y_var - sum(x[k, i + p, j + p] for p in range(d + 1)) >= -d)
        objective_terms.append(-c['seq'] * sum(y.values()))

        # references preferences
        r = np.zeros((m, m), dtype=np.int)
        for k1, step_doc in enumerate(steps):
            references = Decomposition._get_references_ids(str(step_doc))
            for k2 in references:
                r[k1, k2 - 1] = 1
        x_step = {(k, i): model.NewBoolVar(f'x[{k},{i}]') for k in range(m) for i in range(n)}
        for k in range(m):
            for i in range(n):
                model.Add(-x_step[k, i] + sum(x[k, i, j] for j in range(n_k[k])) >= 0)
                model.Add(n_k[k] * x_step[k, i] - sum(x[k, i, j] for j in range(n_k[k])) >= 0)
        z_plus = {(k1, k2, i): model.NewBoolVar(f'z+[{k1},{k2},{i}]') for k1 in range(m) for k2 in range(m) for i in
                  range(n - 1)}
        z_minus = {(k1, k2, i): model.NewBoolVar(f'z-[{k1},{k2},{i}]') for k1 in range(m) for k2 in range(m) for i in
                   range(1, n)}
        for k1, k2, i in z_plus:
            model.Add(-3 * z_plus[k1, k2, i] + x_step[k1, i] + r[k1, k2] + x_step[k2, i + 1] >= 0)
        for k1, k2, i in z_minus:
            model.Add(-3 * z_minus[k1, k2, i] + x_step[k1, i] + r[k1, k2] + x_step[k2, i - 1] >= 0)
        objective_terms.append(-c['ref'] * sum([*z_plus.values(), *z_minus.values()]))

        # uniqueness - prefer single step for a question token
        u = {(d, i): model.NewBoolVar(f'u[{d},{i}]') for i in range(n) for d in range(m)}
        for d, i in u:
            model.Add(-d * u[d, i] + sum(x_step[k, i] for k in range(m)) >= 0)
            model.Add(-m * u[d, i] + sum(x_step[k, i] for k in range(m)) <= d - 1)
        objective_terms.append(-c['unique'] * sum(u.values()))

        # exact match preference
        objective_terms.append(-c['exact'] * sum(b[ind] * x[ind] for ind in x))

        # objective
        model.Minimize(sum(objective_terms))

        # solve
        solver = cp_model.CpSolver()
        if self.max_time_in_seconds:
            solver.parameters.max_time_in_seconds = self.max_time_in_seconds
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            res = np.zeros((m, n, max(n_k)))
            res_ = [set([]) for _ in range(n)]
            for k in range(m):
                for i in range(n):
                    for j in range(n_k[k]):
                        if solver.BooleanValue(x[k, i, j]):
                            res[k, i, j] = 1
                            res_[i].add((k, j))
            return res_
        else:
            print('No solution found.')
            return index_to_steps
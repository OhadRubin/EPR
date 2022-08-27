import pandas as pd
import os
from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.qdmr_identifier import *


class AnnotationEvaluator(object):
    def __init__(self, log_path):
        self.log_path = log_path
        self.operator_stats = None
        self.step_lengths = None

    def evaluate(self, output_path=None):
        log_name = os.path.basename(self.log_path).split('.')[0]
        output = 'annotation_eval_%s.csv' % log_name if output_path is None else output_path
        # read from annotation log
        df = pd.read_csv(self.log_path)
        decompositions = df['decomposition']
        dec_col = []
        qid_col = []
        qtext_col = []
        program_col = []
        operators_col = []
        split_col = []
        
        for i in range(len(decompositions)):
            question_id = df.loc[i, 'question_id']
            question_text = df.loc[i, 'question_text']
            dec = df.loc[i, 'decomposition']
            split = df.loc[i, 'split']
            builder = QDMRProgramBuilder(dec)
            try:
                builder.build()
                program = "\n".join(f"{i+1}. {str(step)}" for i, step in enumerate(builder.steps))
                operators = [str(op) for op in builder.operators]
            except:
                dec_col += [dec]
                qid_col += [question_id]
                qtext_col += [question_text]
                program_col += ['ERROR']
                operators_col += ['ERROR']
                split_col += [split]
            else: 
                dec_col += [dec]
                qid_col += [question_id]
                qtext_col += [question_text]
                program_col += [program]
                operators_col += [operators]
                split_col += [split]
    
        d = {'question_id': qid_col, 'question_text': qtext_col, 'decomposition': dec_col,\
             'program': program_col, 'operators': operators_col, 'split': split_col}
        programs_df = pd.DataFrame(data=d)
        # randomized_df = programs_df.sample(frac = 1)
        randomized_df = programs_df ###### delete
        randomized_df.to_csv('%s' % output, encoding='utf-8')
        print('Annotation evaluation done. Updating annotations statistics.')
        self.operator_stats = {}
        self.step_lengths = {}
        for operators in operators_col:
            num_steps = len(operators)
            self.step_lengths[num_steps] = self.step_lengths.get(num_steps, 0) + 1
            unique_ops = list(set(operators))
            for op in unique_ops:
                self.operator_stats[op] = self.operator_stats.get(op, 0) + 1
        print('Done updating annotations satistics.')
        return True
    
        
    def print_stats(self):
        assert(self.operator_stats is not None)
        assert(self.step_lengths is not None)
        operator_names = list(self.operator_stats.keys())
        operator_names.sort()
        print("* QDMR operator Prevalence:")
        for op in operator_names:
            print("-- Prevalence of operator %s : %d" % (op, self.operator_stats[op]))
        lens = list(self.step_lengths.keys())
        lens.sort()
        print("* QDMR step lengths:")
        for l in lens:
            print("-- Steps of length %d : %d" % (l, self.step_lengths[l]))
        print(self.operator_stats)
        print(self.step_lengths)

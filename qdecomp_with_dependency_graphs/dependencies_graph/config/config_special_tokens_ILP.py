from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.spans_dependencies_to_logical_form_tokens import SpansDepToQDMRStepTokensConverter
from qdecomp_with_dependency_graphs.dependencies_graph.extractors import *


###################
# spans_extractor #
###################
spans_extractor = VariationsBasedSpansExtractor(allow_conflicts=True, include_gaps=False, aligner=ILPAligner(max_time_in_seconds=10))


#################################
# tokens_dependencies_extractor #
#################################
# steps_dependencies_extractor = PatternBasedStepsDependenciesExtractor()
steps_dependencies_extractor = LogicalFormBasedStepsDependenciesExtractor()
spans_dependencies_extractor = MergeSpansDependenciesExtractor(spans_extractor, steps_dependencies_extractor)
spans_dependencies_collapsers = [
    # make sure steps ids are sequential (in cases of removed nodes)
    ToSequentialIdsCollapser(),
    # make sure we backed to proper dependencies type on unwind
    ToDependencyTypeCollapser(),

    MissingResourcesCollapser(),
    # LastStepCollapser(create_separate_span=True),

    # single to dup
    # PreSingleToMultipleStepsCollapser(),  # works only with NotAlignedCollapser
    DupSingleToMultipleStepsCollapser(count=5),

    # JoinCollapser(),
    # ConcatCollapser(),
    NotAlignedDumCollapser(count=5),

    AddOperatorsPropertiesCollapser(),
]
tokens_dependencies_extractor = TokensDependenciesExtractor(spans_dependencies_extractor, spans_dependencies_collapsers)

###############################
# tokens dependencies to QDMR #
###############################
spans_to_qdmr_converter = RuleBasedSpansDepToQdmrConverter()
tokens_dependencies_to_qdmr_extractor = SpansBasedTokensDependenciesToQDMRExtractor(spans_to_qdmr_converter,
                                                                                    tokens_dependencies_extractor)


########
# Eval #
########
spans_dependencies_to_logical_form_converter = SpansDepToQDMRStepTokensConverter(infer_properties=False)
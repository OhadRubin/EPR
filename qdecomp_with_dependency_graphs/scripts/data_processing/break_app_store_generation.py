"""
A variation of Break/annotation_pipeline/utils/app_store_generation.py
(https://github.com/tomerwolgithub/Break.git)
"""
from typing import List, Union, Iterable, Dict
from enum import Enum
from spacy.tokens import Token

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import inflect

import logging

from qdecomp_with_dependency_graphs.utils.data_structures import merge_equivalent_classes

_logger = logging.getLogger(__name__)

p = inflect.engine()
wordnet_lemmatizer = WordNetLemmatizer()


class WNPOS(Enum):
    NOUN = 'n'
    VERB = 'v'
    ADJECTIVE = 'a'
    ADJECTIVE_SATELLITE = 's'
    ADVERB = 'r'


def convert(word, from_pos, to_pos):
    """ Transform words given from/to POS tags """

    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return []

    # Get all lemmas of the word (consider 'a' and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in (WNPOS.ADJECTIVE.value, WNPOS.ADJECTIVE_SATELLITE.value) and s.name().split('.')[1] in (WNPOS.ADJECTIVE.value, WNPOS.ADJECTIVE_SATELLITE.value):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or to_pos in (WNPOS.ADJECTIVE.value, WNPOS.ADJECTIVE_SATELLITE.value) and l.synset().name().split('.')[1] in (WNPOS.ADJECTIVE.value, WNPOS.ADJECTIVE_SATELLITE.value):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w:-w[1])

    # return all the possibilities sorted by probability
    return result


def noun_inflections(token: Token):
    if token.tag_ in ['NN', 'NNP']:
        return p.plural_noun(token.text) or None
    elif token.tag_ in ['NNS', 'NNPS']:
        return p.singular_noun(token.text) or None
    else:
        # workaround nltk wrong detection VB->NN
        # e.g: GEO_dev_10: which states do colorado river flow through
        # flow: nltk=NN (so we have flows), spacy: VB
        return p.plural_noun(token.text) or None
    return None


def nounify_adjectives(token: Token):
    """
    input: token
    output: list of its lemmatized comparatives/superlatives
    example: taller-->talleness, oldest-->oldness
    """
    if token.tag_ in ['RB', 'JJS', 'JJR']:
        noun_list = convert(token.text, WNPOS.ADJECTIVE.value, WNPOS.NOUN.value)
        if len(noun_list) > 0:
            noun = noun_list[0][0]
            return noun
    return None


def lemmatize_adjectives(token: Token):
    """
    input: token
    output: list of its lemmatized comparatives/superlatives
    example: taller-->tall, oldest-->old
    """
    if token.tag_ in ['RB', 'JJS', 'JJR']:
        lemma = wordnet_lemmatizer.lemmatize(token.text, pos=WNPOS.ADJECTIVE.value)
        return lemma
    return None


def lemmatize_verbs(token: Token):
    """
    input: token
    output: list of its lemmatized verbs
    example: uses-->use, died-->die, working --> work
    """
    if token.tag_.startswith('VB'):  # in ['VBZ', 'VBD', 'VBG']:
        lemma = wordnet_lemmatizer.lemmatize(token.text, pos=WNPOS.VERB.value)
        return lemma
    return None


def _init_variations_map():
    groups = [
        # question
        # ['if', 'is'],
        ['who', 'which'],

        # aggregate
        # max
        ['max', 'maximum'],
        ['most'],
        ['more', 'at least', 'no less'],
        ['last'],
        ['highest', 'largest', 'longest', 'biggest'],
        ['higher', 'larger', 'longer', 'bigger'],
        # min
        ['min', 'minimum'],
        ['least'],
        ['less', 'at most', 'no more'],
        ['first'],
        ['lowest', 'smallest', 'shortest', 'fewest'],
        ['lower', 'smaller', 'shorter', 'fewer'],
        ['earlier'],
        # count
        ['count', 'number of', 'total number of'],
        # sum
        ['sum', 'total'],
        # average
        ['avg', 'average', 'mean'],

        # arithmetic
        ['multiply', 'multiplication'],
        #['difference', 'many'],  # how many => difference
        ['difference', 'decline'],
        ['division', 'divide'],
        ['100', 'hundred'],
        ['0', 'zero'],
        ['1', 'one'],
        ['2', 'two'],

        # boolean
        ['same', 'same as', 'equal', 'equals'],

        # comparative
        ['contain', 'contains', 'include', 'includes'],
        ['start', 'starts', 'begin', 'begins'],
        ['end', 'ends'],

        # discard
        ['besides', 'not in', 'discard', 'not'],

        # sort
        ['sort', 'sorted', 'order', 'ordered'],

        # union
        ['and', ',', 'both'],

        # extra
        ['height', 'elevation'],

    ]
    return {k:g for g in groups for k in g}


_tokens_variations_map: Dict[str, List[str]] = _init_variations_map()
_tokens_operational_variations_map: Dict[str, List[str]] = None
def token_operational_variations_map() -> Dict[str, List[str]]:
    from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.operator_identifier import get_operators, get_identifier

    global _tokens_operational_variations_map
    if _tokens_operational_variations_map is None:
        variations_groups = _tokens_variations_map.values()
        op_variations_groups = [x for op in get_operators()
                                for x in get_identifier(op).properties_indicators().values()]
        _tokens_operational_variations_map = merge_equivalent_classes(list(variations_groups) + op_variations_groups)
    return _tokens_operational_variations_map

def token_variations(token: str, is_operational:bool = False) -> List[str]:
    mapping = token_operational_variations_map() if is_operational else _tokens_variations_map
    return mapping.get(token, [token])

def get_token_variations_groups(is_operational:bool = False) -> Iterable[List[str]]:
    mapping = token_operational_variations_map() if is_operational else _tokens_variations_map
    return mapping.values()

def get_lexicon_tokens(token: Token) -> List[str]:
    singleton_rules = [
        lambda x: x.text,
        lambda x: x.text.lower(),
        #lambda x: x.lemma_,
        noun_inflections,
        nounify_adjectives,
        lemmatize_adjectives,
        lemmatize_verbs
    ]

    def except_wrapper(func):
        try:
            return func()
        except:
            _logger.exception(f'get_lexicon_tokens error for token {token.text}')
    variations = [except_wrapper(lambda:r(token)) for r in singleton_rules]
    variations.extend(token_variations(token.text, is_operational=True))
    variations = [x.lower() for x in variations if isinstance(x, str)]
    return variations


def get_additional() -> List[str]:
    words = ['a', 'if', 'how', 'where', 'when', 'which', 'who', 'what', 'with', 'was', 'were', 'did', 'to', 'from', 'both', 'and', 'or', 'the', 'of', 'is', 'are', 'besides', 'that', 'have', 'has', 'for each', 'number of', 'not', 'than', 'those', 'on', 'in', 'any', 'there', 'distinct', ',', ', ']
    return words


def get_additional_resources() -> List[str]:
    words = ['height', 'population', 'size', 'elevation', 'flights', 'objects', 'price', 'date', 'true', 'false']
    return words
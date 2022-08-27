import re
from dataclasses import dataclass
from typing import List, Iterator, Tuple, Iterable, Any


@dataclass
class Span:
    start: int
    end: int


@dataclass
class TokenData:
    bio_tag: str
    text: str


class StepsSpans:
    def __init__(self,
                 question_tokens: List[str],
                 steps: List[str],
                 steps_spans: List[List[Span]],
                 steps_tags: List[str]):
        self.question_tokens: List[str] = question_tokens
        self._steps_spans: List[List[Span]] = steps_spans
        self._steps: List[str] = steps
        self.steps_tags = steps_tags
        self._bio_tags = get_bio_tags(len(self.question_tokens), self._steps_spans, steps_tags)

    @staticmethod
    def from_dict(dct: dict):
        return StepsSpans(
            question_tokens=dct['question_tokens'],
            steps=dct['steps'],
            steps_spans=[[Span(y['start'], y['end']) for y in x] for x in dct['steps_spans']],
            steps_tags=dct['steps_tags'],
        )

    def to_dict(self) -> dict:
        return {
            'question_tokens': self.question_tokens,
            'steps': self._steps,
            'steps_spans': [[{'start':y.start, 'end':y.end} for y in x] for x in self._steps_spans],
            'steps_tags': self.steps_tags,
        }

    def copy(self):
        return self.from_dict(self.to_dict())

    def __iter__(self) -> Iterator[Tuple[int, List[Span]]]:
        return iter((i+1, x) for i, x in enumerate(self._steps_spans))

    def tokens(self) -> Iterable[TokenData]:
        return iter(TokenData(text=x, bio_tag=y) for x,y in zip(self.question_tokens, self._bio_tags))

    def step_tokens_indexes(self) -> Iterator[Tuple[int, List[int]]]:
        return iter((i, [x for span in step_spans_list for x in range(span.start, span.end+1)])
                    for i, step_spans_list in self.__iter__())

    def step_spans_text(self) -> Iterator[Tuple[int, List[str]]]:
        return iter((i, [' '.join(self.question_tokens[span.start:span.end+1]) for span in step_spans_list])
                    for i, step_spans_list in self.__iter__())

    def get_alignments_str(self, steps_context: Iterable[Any] = None):
        strs = [f"question:{' '.join(self.question_tokens)}"]
        steps_context = steps_context or ['']*len(self._steps)
        for i, (step, context) in enumerate(zip(self._steps, steps_context)):
            step = re.sub(r"@@(\d+)@@", r"#\g<1>", step)
            strs.append(
                f'\t{i + 1}.{step} ({self.steps_tags[i]}{context})\t==>\t{"|".join([" ".join(self.question_tokens[s.start:s.end + 1])+f" {s.start, s.end}" for s in self._steps_spans[i]])}')
        return '\n'.join(strs)


def get_bio_tags(size: int, spans_list:List[List[Span]], steps_tags: List[str]):
    tags = ['O'] * size
    for spans, tag in zip(spans_list, steps_tags):
        for s in spans:
            tags[s.start] = f'B-{tag}'
            for i in range(s.start, s.end):
                tags[i + 1] = f'I-{tag}'
    return tags
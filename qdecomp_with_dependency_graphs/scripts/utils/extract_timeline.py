from typing import List
from dataclasses import dataclass

import re

import argparse

import os


@dataclass
class Event:
    description: str
    time: str

    def __str__(self):
        return f'{self.time} {self.description}'

    @staticmethod
    def from_log_line(line: str):
        time, desc = re.match(r'([0-9\-\:\,\s]+)-(.*)', line).groups()
        return Event(desc, time)


def extract_timeline_for_allennlp_log(log_path: str) -> List[Event]:
    events_regexes = [
        '.*'+re.escape('Reading training data from')+'.*',
        '.*'+re.escape('Reading validation data from')+'.*',
        '.*'+re.escape('allennlp.training.trainer - Epoch ')+'[0-9]+/[0-9]+.*',
        '.*'+re.escape('allennlp.training.trainer - Training')+'.*',
        '.*'+re.escape('allennlp.training.trainer - Validating')+'.*',
        '.*'+re.escape('allennlp.training.trainer - Epoch duration:')+'.*',
    ]

    events: List[Event] = []
    with open(log_path, 'rt') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                ev = Event.from_log_line(line)
                ev.description = 'init'
                events.append(ev)
                continue
            if any(re.match(x, line) for x in events_regexes):
                ev = Event.from_log_line(line)
                events.append(ev)
    return events


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='allennlp log timeline')
    parser.add_argument('input_dir', type=str, help='model dir')
    args = parser.parse_args()

    events = extract_timeline_for_allennlp_log(os.path.join(args.input_dir, 'out.log'))
    for x in events:
        print(x)
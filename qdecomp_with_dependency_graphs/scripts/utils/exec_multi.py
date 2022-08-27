from typing import Callable, Generator, Tuple
import argparse
import os
import subprocess
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import traceback
import sys
available_gpus = {}
gpus_usage = defaultdict(int)
errors_dir = None


def get_next_gpu():
    free_slots = {gpu: available - gpus_usage[gpu] for gpu, available in available_gpus.items()}
    return max(free_slots, key=free_slots.get)


def run(cmd):
    with open(os.devnull, 'wb') as devnull:
        gpu = get_next_gpu()
        gpus_usage[gpu] += 1

        try:
            print(f'Starting command: {cmd}, gpu: {gpu} ')

            cmd = cmd.replace('"', r'\"')
            cmd = f'bash -c "source ~/anaconda3/etc/profile.d/conda.sh; conda activate qdecomp; {cmd}"'

            #os.putenv('CUDA_VISIBLE_DEVICES', str(gpu))  #  parallelism issue
            #cmd = f'CUDA_VISIBLE_DEVICES={str(gpu)}; {cmd}'  #  doesnt cover python scripts

            cmd_parts = cmd.split(';')
            for i, part in enumerate(cmd_parts):
                if i==0 or "python " in part:  # for multiple bash and python commands
                    cmd_parts[i] = f'CUDA_VISIBLE_DEVICES={str(gpu)} {part}'
            cmd = ';'.join(cmd_parts)

            #subprocess.check_call(cmd, stdout=devnull, stderr=devnull, shell=True)
            if errors_dir:
                subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
            else:
                subprocess.check_output(cmd, shell=True)   # writes to current stderr
        except subprocess.CalledProcessError as e:
            if errors_dir:
                error_file = os.path.join(errors_dir, f'{datetime.now().strftime("%Y-%m-%d--%H-%M-%S.%f")}.txt')
                print('error: ', e, f'see errors log: {error_file}')
                with open(error_file, 'wt') as f:
                    f.write(e.output)
            else:
                print('error: ', e)
                # print(traceback.format_exc())

        except Exception as e:
            print('error: ', e)
            print(traceback.format_exc())

        finally:
            gpus_usage[gpu] -= 1

        print(f'Finished command: {cmd}')


def refresh(executor, queue_file_path: str, available_gpu_generator: Callable[[],Generator[Tuple[int,int], None, None]]):
    for gpu, available in available_gpu_generator():
        available_gpus[gpu] = available
    executor._max_workers = sum(available_gpus.values())

    path, ext = os.path.splitext(queue_file_path)
    processed_file_path = path+'-processed'+ext

    queue_file = open(queue_file_path, 'rt')
    processed_file = open(processed_file_path, 'a')
    for line in queue_file:
        executor.submit(run, line.strip())
        processed_file.write(line.strip() + '\n')
    queue_file.close()
    queue_file = open(queue_file_path, "wt")
    queue_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple commands on cuda pool.'
                                                 'example: PYTHONPATH=. nohup python scripts/utils/exec_multi.py --cuda 2 3 > exec/log.txt &')
    parser.add_argument('-q', type=str, default='exec/queue.txt',
                        help='txt file with new-line separated commands to execute on gpus (default: exec/queue.txt)')
    parser.add_argument('-c', type=str, default='exec/available_gpus.txt',
                         help='txt file with new-line separated available gpus in format: <gpu>:<available-slots:int>. (default: exec/available_gpus.txt)')
    parser.add_argument('--cuda', nargs='+',
                        help='constant available gpus list (instead of -c arg), one slot per gpu')
    parser.add_argument('-e', type=str, default='exec/errors',
                        help='directory for errors logs. If None, writes the errors to current stderr (default: exec/errors)')

    args = parser.parse_args()
    assert args.q
    assert args.cuda or args.c

    if args.e:
        errors_dir = args.e
        os.makedirs(errors_dir, exist_ok=True)

    def available_gpus_generator() -> Generator[Tuple[int, int], None, None]:
        if args.cuda:
            for x in args.cuda:
                yield x, 1
        else:
            with open(args.c) as f:
                for line in f.readlines():
                    gpu, available = line.split(':')
                    gpu = int(gpu)
                    available = int(available)
                    yield gpu, available

    with ThreadPoolExecutor(1) as executor:
        while True:
            refresh(executor, queue_file_path=args.q, available_gpu_generator=available_gpus_generator)
            time.sleep(10)
            sys.stdout.flush()

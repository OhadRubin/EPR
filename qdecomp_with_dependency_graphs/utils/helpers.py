import time
import functools
import inspect
import json
import logging


def print_func(func, fp):
    lines = inspect.getsource(func)
    if isinstance(fp, str):
        with open(fp, 'wt') as fp:
            fp.writelines(lines)
    else:
        fp.writelines(lines)


def print_object(obj, fp):
    def get_file(x):
        try:
            return inspect.getfile(x)
        except:
            return ''

    def get_vars(x):
        if not (inspect.isclass(type(x))
            and 'qdecomp' in get_file(x.__class__)
            and 'lib' not in get_file(x.__class__)
        ):
            return x if isinstance(x,(int,float,str)) else str(type(x))
        return {x.__class__.__name__: {k: get_vars(v) for k,v in vars(x).items()}}

    if isinstance(fp, str):
        with open(fp, 'wt') as fp:
            json.dump(get_vars(obj), fp, indent=2, sort_keys=True)
    else:
        json.dump(get_vars(obj), fp, indent=2, sort_keys=True)


def rsetattr(obj, attr, val):
    """
    Set a (nested, delimited by '.') named attribute from an object
    e.g: rsetattr(x, 'y.z', v) is equivalent to x.y.z=v
    :param obj: object
    :param attr: attribute path, delimited by '.'
    :param val: the new value to set
    :return:
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    Get a (nested, delimited by '.') named attribute from an object
    e.g: rgetattr(x, 'y.z') is equivalent to x.y.z
    :param obj: object
    :param attr: attribute path, delimited by '.'
    :param args: getattr() args
    :return: the value of the attribute
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def silent_logs(func):
    @functools.wraps(func)
    def wrapped_fnc(*args, **kwargs):
        logging.disable()
        try:
            res = func(*args, **kwargs)
            return res
        finally:
            logging.disable(logging.NOTSET)
    return wrapped_fnc


class Timer:
    def __init__(self):
        self.start = time.time()

    def get_time_diff(self) -> float:
        return time.time() - self.start

    def get_time_diff_str(self) -> str:
        diff = self.get_time_diff()
        hours, rem = divmod(diff, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
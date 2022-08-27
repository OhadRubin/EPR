from typing import TypeVar, Generic, Dict, Any, Union
import os
import pickle as pkl
from copy import copy
from functools import wraps


T = TypeVar('T')


class SavableCache:
    def save_cache(self, dest_path: str = None):
        pass


class CacheExtractor(Generic[T]):
    """
    Wrap an extractor, so extract() method uses cached data by question_id
    Example:
         wr = CacheExtractor(file_path='_cache/cache.pkl', extractor=...)
         res = wr.extract(...)
         wr.save_cache()
    """
    def __new__(cls, file_path: str, extractor: T) -> Union[T, SavableCache]:
        # copy extractor in case of multiple use (for the added attributes)
        extractor = copy(extractor)

        if os.path.exists(file_path):
            with open(file_path, 'rb') as fp:
                cache = pkl.load(fp)
        else:
            cache: Dict[str, Dict[Any, Any]] = {}

        def wrapper(method):
            @wraps(method)
            def wrapped_extract(question_id, *args, **kwargs):
                res = cache.get(question_id, None)
                if res:
                    return res
                res = method(question_id, *args, **kwargs)
                cache[question_id] = res
                return res
            return wrapped_extract

        def save(dest_path: str = None):
            dest_path = dest_path or file_path
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, 'wb') as fp:
                pkl.dump(cache, fp)

        extractor.extract = wrapper(extractor.extract)
        extractor.save_cache = save
        return extractor


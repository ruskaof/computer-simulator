from typing import Optional, TypeVar

T = TypeVar('T')


def get_or_none(lst: list[T], idx: int) -> Optional[T]:
    if idx < len(lst):
        return lst[idx]
    return None

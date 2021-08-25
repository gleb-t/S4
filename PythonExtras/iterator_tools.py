import random
from collections import defaultdict
from typing import TypeVar, Iterator, List, Sequence, Callable, Any, Generator, Tuple, Iterable

TItem = TypeVar('TItem')


def reservoir_sample(iterator: Iterator[TItem], sampleSize: int) -> List[TItem]:
    """
    Perform reservoir sampling of an iterator. Useful, when the length of a sequence is not known.
    """
    sample = []
    i = 0
    try:
        for i in range(sampleSize):
            sample.append(next(iterator))
    except StopIteration:
        raise ValueError("Provided iterator is shorter than the sample size: {} < {}".format(i, sampleSize))

    random.shuffle(sample)
    for i, value in enumerate(iterator, sampleSize):
        r = random.randint(0, i)
        if r < sampleSize:
            sample[r] = value

    return sample


def group_by(seq: Sequence[TItem], key: Callable[[TItem], Any]) -> Generator[Tuple[Any, List[TItem]], None, None]:
    groups = defaultdict(list)
    for item in seq:
        groups[key(item)].append(item)

    for key, groupedItems in groups.items():
        yield key, groupedItems


def decorate_iter(iterable: Iterable[TItem], pre: Callable = None, post: Callable = None) -> Generator[TItem, None, None]:
    """
    Call callbacks before and after advancing an iterator. Useful for timing the iteration itself.
    """
    it = iter(iterable)
    while True:
        if pre is not None:
            pre()

        try:
            yield next(it)
        except StopIteration:
            break
        else:
            if post is not None:
                post()

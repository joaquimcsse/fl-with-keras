from itertools import tee, islice, chain

def previous_and_current(some_iterable):
    """
        Easy iteration tool for use in the inputs/models dicts
    """
    prevs, items = tee(some_iterable, 2)
    prevs = chain([None], prevs)
    return zip(prevs, items)

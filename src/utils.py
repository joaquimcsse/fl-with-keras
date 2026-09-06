from itertools import tee, islice, chain

def previous_and_current(some_iterable):
    """
        Tool for training and testing models with data from different cicles
            in arrays stored in a dictionary
    """
    prevs, items = tee(some_iterable, 2)
    prevs = chain([None], prevs)
    return zip(prevs, items)

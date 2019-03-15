from datetime import datetime
from os import path

RANDOM_STATE = 100


def pretty_print(text):
    """Prints the given text with an underscore line below it."""
    print('\n\n' + str(text))
    print('_' * len(text))


def get_abspath(module__file__, file_path):
    """
    Returns the absolute path to the relative path specified in a module.
    :param module__file__: __file__ variable of the module that calls this function.
    :param file_path: Relative path from the module that calls this function, pointing to
    the resource that module wants to access.
    :return: The absolute path to the relative path specified in a module.
    """
    return path.abspath(path.join(path.dirname(module__file__), file_path))


def now_as_str():
    return str(datetime.now())

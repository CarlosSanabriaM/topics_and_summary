import pickle
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
    """
    :return: The current time as str.
    """
    return str(datetime.now())


def save_obj_to_disk(obj, name):
    """
    Stores an object in disk, using pickle.
    :param obj: Object to be stored in disk.
    :param name: Name of the pickle file to be created in saved-models/objects/ and
    where the object info will be stored.
    """
    obj_path = get_abspath(__file__, 'saved-models/objects/{}.pickle'.format(name))
    with open(obj_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj_from_disk(name):
    """
    Loads a saved object from disk, using pickle.
    :param name: Name of the pickle file stores in saved-models/objects/.
    :return: The object load from disk.
    """
    obj_path = get_abspath(__file__, 'saved-models/objects/{}.pickle'.format(name))
    with open(obj_path, 'rb') as f:
        return pickle.load(f)

import pickle
from datetime import datetime
from os import path

RANDOM_STATE = 100
# This python module (utils.py) must be in the root folder of the project. If not, the following PATH won't be OK.
PROJECT_ROOT_PATH = path.dirname(path.abspath(__file__))


def pretty_print(text):
    """Prints the given text with an underscore line below it."""
    print('\n\n' + str(text))
    print('_' * len(text))


def join_paths(path1, *paths):
    # If paths contains '/', transform to '\\' if os is Windows.
    path1 = path.normcase(path1)
    paths = map(lambda p: path.normcase(p), paths)

    # Join all the paths
    return path.join(path1, *paths)


def get_abspath_from_project_root(_path):
    return path.abspath(join_paths(PROJECT_ROOT_PATH, _path))


def now_as_str():
    """
    :return: The current time as str.
    """
    return str(datetime.now().strftime('%Y-%m-%d_%H-%M'))


def save_obj_to_disk(obj, name):
    """
    Stores an object in disk, using pickle.
    :param obj: Object to be stored in disk.
    :param name: Name of the pickle file to be created in saved-models/objects/ and
    where the object info will be stored.
    """
    obj_path = get_abspath_from_project_root('saved-models/objects/{}.pickle'.format(name))
    with open(obj_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj_from_disk(name):
    """
    Loads a saved object from disk, using pickle.
    :param name: Name of the pickle file stores in saved-models/objects/.
    :return: The object load from disk.
    """
    obj_path = get_abspath_from_project_root('saved-models/objects/{}.pickle'.format(name))
    with open(obj_path, 'rb') as f:
        return pickle.load(f)

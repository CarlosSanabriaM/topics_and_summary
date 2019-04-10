import pickle
from datetime import datetime
from os import path
from typing import Callable

import dill

RANDOM_STATE = 100
# This python module (utils.py) must be in the root folder of the project. If not, the following PATH won't be OK.
PROJECT_ROOT_PATH = path.dirname(path.abspath(__file__))


def pretty_print(text: str):
    """Prints the given text with an underscore line below it."""
    print('\n\n' + str(text))
    print('_' * len(text))


def join_paths(path1: str, *paths: str) -> str:
    # If paths contains '/', transform to '\\' if os is Windows.
    path1 = path.normcase(path1)
    paths = map(lambda p: path.normcase(p), paths)

    # Join all the paths
    return path.join(path1, *paths)


def get_abspath_from_project_root(_path: str) -> str:
    return path.abspath(join_paths(PROJECT_ROOT_PATH, _path))


def now_as_str() -> str:
    """
    :return: The current time as str.
    """
    return str(datetime.now().strftime('%Y-%m-%d_%H-%M'))


def save_obj_to_disk(obj: object, name: str):
    """
    Stores an object on disk, using pickle.
    :param obj: Object to be stored in disk.
    :param name: Name of the pickle file to be created in saved-models/objects/ and
    where the object info will be stored.
    """
    obj_path = get_abspath_from_project_root('saved-models/objects/{}.pickle'.format(name))
    with open(obj_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj_from_disk(name: str) -> object:
    """
    Loads a saved object from disk, using pickle.
    :param name: Name of the pickle file stored in saved-models/objects/.
    :return: The object load from disk.
    """
    obj_path = get_abspath_from_project_root('saved-models/objects/{}.pickle'.format(name))
    with open(obj_path, 'rb') as f:
        return pickle.load(f)


def save_func_to_disk(func: Callable, name: str):
    """
    Stores a function on disk, using dill.
    :param func: Function to be stored in disk.
    :param name: Name of the dill file to be created in saved-models/funcs/ and
    where the object info will be stored.
    """
    obj_path = get_abspath_from_project_root('saved-models/funcs/{}.dill'.format(name))
    with open(obj_path, 'wb') as f:
        dill.dump(func, f, pickle.HIGHEST_PROTOCOL)


def load_func_from_disk(name: str) -> Callable:
    """
    Loads a saved function from disk, using dill.
    :param name: Name of the dill file stored in saved-models/funcs/.
    :return: The function load from disk.
    """
    func_path = get_abspath_from_project_root('saved-models/funcs/{}.dill'.format(name))
    with open(func_path, 'rb') as f:
        return dill.load(f)

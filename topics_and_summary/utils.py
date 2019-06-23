import configparser
import pickle
from datetime import datetime
from os import path
from typing import Callable

import dill

RANDOM_STATE = 100
# This python module (utils.py) must be in the root folder of the python package project.
PROJECT_SOURCE_ROOT_PATH = path.dirname(path.abspath(__file__))


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


def get_abspath_from_project_source_root(_path: str) -> str:
    """
    Returns the absolute path of the relative path passed as a parameter.

    :param _path: Path relative from the project source root (folder that contains an __init__.py file \
    and the rest of the Python packages and modules).
    """
    return path.abspath(join_paths(PROJECT_SOURCE_ROOT_PATH, _path))


def now_as_str() -> str:
    """:return: The current time as str."""
    return str(datetime.now().strftime('%Y-%m-%d_%H-%M'))


def save_obj_to_disk(obj: object, name: str, folder_path: str = None):
    """
    Stores an object on disk, using pickle.

    :param obj: Object to be stored in disk.
    :param name: Name of the pickle file to be stored on disk.
    :param folder_path: Path of the folder where the object will be stored on disk.
    """
    if folder_path is None:
        obj_path = get_abspath_from_project_source_root('saved-elements/objects/{}.pickle'.format(name))
    else:
        obj_path = join_paths(folder_path, '{}.pickle'.format(name))

    with open(obj_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj_from_disk(name: str, folder_path: str = None) -> object:
    """
    Loads a saved object from disk, using pickle.

    :param name: Name of the pickle file stored on disk.
    :param folder_path: Path of the folder where the object is stored on disk.
    :return: The object loaded from disk.
    """
    if folder_path is None:
        obj_path = get_abspath_from_project_source_root('saved-elements/objects/{}.pickle'.format(name))
    else:
        obj_path = join_paths(folder_path, '{}.pickle'.format(name))

    with open(obj_path, 'rb') as f:
        return pickle.load(f)


def save_func_to_disk(func: Callable, name: str, folder_path: str = None):
    """
    Stores a function on disk, using dill.

    :param func: Function to be stored in disk.
    :param name: Name of the dill file to be stored on disk.
    :param folder_path: Path of the folder where the function will be stored on disk.
    """
    if folder_path is None:
        func_path = get_abspath_from_project_source_root('saved-elements/funcs/{}.dill'.format(name))
    else:
        func_path = join_paths(folder_path, '{}.dill'.format(name))

    with open(func_path, 'wb') as f:
        dill.dump(func, f, pickle.HIGHEST_PROTOCOL)


def load_func_from_disk(name: str, folder_path: str = None) -> Callable:
    """
    Loads a saved function from disk, using dill.

    :param name: Name of the dill file stored on disk.
    :param folder_path: Path of the folder where the function is stored on disk.
    :return: The function loaded from disk.
    """
    if folder_path is None:
        func_path = get_abspath_from_project_source_root('saved-elements/funcs/{}.dill'.format(name))
    else:
        func_path = join_paths(folder_path, '{}.dill'.format(name))

    with open(func_path, 'rb') as f:
        return dill.load(f)


def get_param_value_from_conf_ini_file(conf_ini_file_path: str, section: str, param: str) -> str:
    """
    Returns the value of the specified param from the specified .ini configuration file.

    :param conf_ini_file_path: Path to the .ini configuration file.
    :param section: Name of the section in the .ini file. For example: '[MALLET]'.
    :param param: Name of the param inside that section. For example: 'SOURCE_CODE_PATH'.
    :return: A str with the value specified in the .ini file for that param.

    Example:

    ; demo-conf.ini

    [MALLET]

    SOURCE_CODE_PATH = /path/to/mallet

    To access that value, execute:

    >>> get_param_value_from_conf_ini_file('MALLET', 'SOURCE_CODE_PATH')

    """

    config = configparser.ConfigParser()
    config.read(conf_ini_file_path)

    return config[section][param]

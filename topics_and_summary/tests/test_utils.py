import os
import platform
import unittest

from topics_and_summary.tests.paths import SAVED_OBJECTS_PATH, SAVED_FUNCS_PATH
from topics_and_summary.utils import join_paths, save_obj_to_disk, load_obj_from_disk, save_func_to_disk, \
    load_func_from_disk


class TestUtils(unittest.TestCase):

    def test_join_paths(self):
        path = join_paths('Users/name/', 'Desktop', 'class/', 'files')

        if platform.system() in ['Linux', 'Darwin']:
            self.assertEqual('Users/name/Desktop/class/files', path)
        elif platform.system() == 'Windows':
            self.assertEqual('Users\\name\\Desktop\\class\\files', path)
        else:
            raise Exception('OS not found!')

    def test_save_and_load_obj_on_disk(self):
        test_list = [1, 2, 3, 4]
        save_obj_to_disk(test_list, 'test_list', SAVED_OBJECTS_PATH)
        test_list_from_disk = load_obj_from_disk('test_list', SAVED_OBJECTS_PATH)

        os.remove(join_paths(SAVED_OBJECTS_PATH, 'test_list.pickle'))

        self.assertEqual(test_list, test_list_from_disk)

    def test_save_and_load_func_on_disk(self):
        def test_func(x):
            return x ** 3

        save_func_to_disk(test_func, 'test_func', SAVED_FUNCS_PATH)
        test_func_from_disk = load_func_from_disk('test_func', SAVED_FUNCS_PATH)

        os.remove(join_paths(SAVED_FUNCS_PATH, 'test_func.dill'))

        # To compare the functions, we have to use them
        test_func_result_list = [test_func(1), test_func(2), test_func(3), test_func(4)]
        test_func_from_disk_result_list = [test_func_from_disk(1), test_func_from_disk(2),
                                           test_func_from_disk(3), test_func_from_disk(4)]

        self.assertEqual(test_func_result_list, test_func_from_disk_result_list)


if __name__ == '__main__':
    unittest.main()

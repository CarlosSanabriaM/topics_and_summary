import os
import platform
import unittest

from utils import get_abspath_from_project_root, join_paths, save_obj_to_disk, load_obj_from_disk, save_func_to_disk, \
    load_func_from_disk


class TestUtils(unittest.TestCase):
    TESTS_BASE_PATH = get_abspath_from_project_root('tests')

    def test_join_paths(self):
        path = join_paths('Users/name/', 'Desktop', 'class/', 'files')

        if platform.system() in ['Linux', 'Darwin']:
            self.assertEqual('Users/name/Desktop/class/files', path)
        elif platform.system() == 'Windows':
            self.assertEqual('Users\\name\\Desktop\\class\\files', path)
        else:
            raise Exception('OS not found!')

    def test_save_and_load_obj_on_disk(self):
        dir_path = join_paths(self.TESTS_BASE_PATH, 'saved-elements/objects')

        test_list = [1, 2, 3, 4]
        save_obj_to_disk(test_list, 'test_list', dir_path)
        test_list_from_disk = load_obj_from_disk('test_list', dir_path)

        os.remove(join_paths(dir_path, 'test_list.pickle'))

        self.assertEqual(test_list, test_list_from_disk)

    def test_save_and_load_func_on_disk(self):
        dir_path = join_paths(self.TESTS_BASE_PATH, 'saved-elements/funcs')

        def test_func(x):
            return x ** 3

        save_func_to_disk(test_func, 'test_func', dir_path)
        test_func_from_disk = load_func_from_disk('test_func', dir_path)

        os.remove(join_paths(dir_path, 'test_func.dill'))

        # To compare the functions, we have to use them
        test_func_result_list = [test_func(1), test_func(2), test_func(3), test_func(4)]
        test_func_from_disk_result_list = [test_func_from_disk(1), test_func_from_disk(2),
                                           test_func_from_disk(3), test_func_from_disk(4)]

        self.assertEqual(test_func_result_list, test_func_from_disk_result_list)


if __name__ == '__main__':
    unittest.main()

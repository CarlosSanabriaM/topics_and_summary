import unittest
from shutil import rmtree

from models.topics import LdaGensimModel, LdaMalletModel, LsaGensimModel
from utils import get_abspath_from_project_root, join_paths, load_obj_from_disk


class TestTopicsModel(unittest.TestCase):
    TESTS_BASE_PATH = get_abspath_from_project_root('tests')

    @classmethod
    def setUpClass(cls) -> None:
        cls.models_dir_path = join_paths(cls.TESTS_BASE_PATH, 'saved-elements/models/topics')
        cls.dataset = load_obj_from_disk('trigrams_dataset', join_paths(cls.TESTS_BASE_PATH, 'saved-elements/objects'))

    # noinspection PyTypeChecker
    def test_save_and_load_lda_gensim_model_on_disk(self):
        # Instead of creating a new model, we load a pre-created model from disk
        model = LdaGensimModel.load('lda-gensim-model', self.dataset, self.models_dir_path)

        # Here we really test the save and load methods
        model_name = 'test-lda-gensim-model'
        model.save(model_name, self.models_dir_path)
        test_model_from_disk = LdaGensimModel.load(model_name, self.dataset, self.models_dir_path)

        # Remove the created model (it's directory and it's files inside that directory)
        rmtree(join_paths(self.models_dir_path, model_name))

        self.assertEqual(model, test_model_from_disk)

    # noinspection PyTypeChecker
    def test_save_and_load_lda_mallet_model_on_disk(self):
        # Lda mallet models cant' be stored in a different path than the original one
        # To test correctly this 2 methods, we need to create a new model
        model_name = 'test-lda-mallet-model'
        test_model = LdaMalletModel(self.dataset, num_topics=17,
                                    model_name=model_name,
                                    model_path=self.models_dir_path,
                                    iterations=10)  # 10 iterations to make it too much faster (default is 1000)

        # Here we really test the save and load methods
        test_model.save()
        test_model_from_disk = LdaMalletModel.load(model_name, self.dataset, self.models_dir_path)

        # Remove the created model (it's directory and it's files inside that directory)
        rmtree(join_paths(self.models_dir_path, model_name))

        self.assertEqual(test_model, test_model_from_disk)

    # noinspection PyTypeChecker
    def test_save_and_load_lsa_gensim_model_on_disk(self):
        # Instead of creating a new model, we load a pre-created model from disk
        model = LsaGensimModel.load('lsa-gensim-model', self.dataset, self.models_dir_path)

        # Here we really test the save and load methods
        model_name = 'test-lsa-gensim-model'
        model.save(model_name, self.models_dir_path)
        test_model_from_disk = LsaGensimModel.load(model_name, self.dataset, self.models_dir_path)

        # Remove the created model (it's directory and it's files inside that directory)
        rmtree(join_paths(self.models_dir_path, model_name))

        self.assertEqual(model, test_model_from_disk)


if __name__ == '__main__':
    unittest.main()

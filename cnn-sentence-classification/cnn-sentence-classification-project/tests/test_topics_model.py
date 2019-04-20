import unittest
from shutil import rmtree

from models.topics import LdaGensimModel, LdaMalletModel, LsaGensimModel
from tests.paths import SAVED_OBJECTS_PATH, SAVED_TOPICS_MODELS_PATH, SAVED_FUNCS_PATH
from utils import join_paths, load_obj_from_disk, load_func_from_disk


class TestTopicsModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_obj_from_disk('trigrams_dataset', SAVED_OBJECTS_PATH)

    # noinspection PyTypeChecker
    def test_save_and_load_lda_gensim_model_on_disk(self):
        # Instead of creating a new model, we load a pre-created model from disk
        model = LdaGensimModel.load('lda-gensim-model', self.dataset, SAVED_TOPICS_MODELS_PATH)

        # Here we really test the save and load methods
        model_name = 'test-lda-gensim-model'
        model.save(model_name, SAVED_TOPICS_MODELS_PATH)
        test_model_from_disk = LdaGensimModel.load(model_name, self.dataset, SAVED_TOPICS_MODELS_PATH)

        # Remove the created model (it's directory and it's files inside that directory)
        rmtree(join_paths(SAVED_TOPICS_MODELS_PATH, model_name))

        self.assertEqual(model, test_model_from_disk)

    # noinspection PyTypeChecker
    def test_save_and_load_lda_mallet_model_on_disk(self):
        # Lda mallet models cant' be stored in a different path than the original one
        # To test correctly this 2 methods, we need to create a new model
        model_name = 'test-lda-mallet-model'
        test_model = LdaMalletModel(self.dataset, num_topics=17,
                                    model_name=model_name,
                                    model_path=SAVED_TOPICS_MODELS_PATH,
                                    iterations=10)  # 10 iterations to make it too much faster (default is 1000)

        # Here we really test the save and load methods
        test_model.save()
        test_model_from_disk = LdaMalletModel.load(model_name, self.dataset, SAVED_TOPICS_MODELS_PATH)

        # Remove the created model (it's directory and it's files inside that directory)
        rmtree(join_paths(SAVED_TOPICS_MODELS_PATH, model_name))

        self.assertEqual(test_model, test_model_from_disk)

    # noinspection PyTypeChecker
    def test_save_and_load_lsa_gensim_model_on_disk(self):
        # Instead of creating a new model, we load a pre-created model from disk
        model = LsaGensimModel.load('lsa-gensim-model', self.dataset, SAVED_TOPICS_MODELS_PATH)

        # Here we really test the save and load methods
        model_name = 'test-lsa-gensim-model'
        model.save(model_name, SAVED_TOPICS_MODELS_PATH)
        test_model_from_disk = LsaGensimModel.load(model_name, self.dataset, SAVED_TOPICS_MODELS_PATH)

        # Remove the created model (it's directory and it's files inside that directory)
        rmtree(join_paths(SAVED_TOPICS_MODELS_PATH, model_name))

        self.assertEqual(model, test_model_from_disk)

    # noinspection PyTypeChecker
    def test_get_topics(self):
        # For testing this method, an LdaMalletModel loaded from disk will be used
        model = LdaMalletModel.load('lda-mallet-model', self.dataset, SAVED_TOPICS_MODELS_PATH)

        # Expected result was previously calculated and stored in disk
        expected_result = load_obj_from_disk('test_get_topics_expected_result', SAVED_OBJECTS_PATH)

        result = model.get_topics()

        self.assertEqual(expected_result, result)

    # noinspection PyTypeChecker
    def test_predict_topic_prob_on_text(self):
        # For testing this method, an LdaMalletModel loaded from disk will be used
        model = LdaMalletModel.load('lda-mallet-model', self.dataset, SAVED_TOPICS_MODELS_PATH)
        trigrams_func = load_func_from_disk('trigrams_func', SAVED_FUNCS_PATH)

        text = """The baptism of Jesus is described in the gospels of Matthew, Mark and Luke. John's gospel does not
        directly describe Jesus' baptism. Most modern theologians view the baptism of Jesus by John the Baptist as a
        historical event to which a high degree of certainty can be assigned.[1][2][3][4][5] Along with the crucifixion
        of Jesus, most biblical scholars view it as one of the two historically certain facts about him, and often use 
        it as the starting point for the study of the historical Jesus.[6]"""

        # Expected result was previously calculated and stored in disk
        expected_result = load_obj_from_disk('test_predict_topic_prob_on_text_expected_result',
                                             SAVED_OBJECTS_PATH)

        result = model.predict_topic_prob_on_text(text, ngrams='tri', ngrams_model_func=trigrams_func,
                                                  print_table=False)

        self.assertEqual(expected_result, result)

    # noinspection PyTypeChecker
    def test_get_dominant_topic_of_each_doc_as_df(self):
        # For testing this method, an LdaGensimModel loaded from disk will be used,
        # because LdaMallet is extremely slow to generate the docs_topics_df
        model = LdaGensimModel.load('lda-gensim-model', self.dataset, SAVED_TOPICS_MODELS_PATH)

        # Expected result was previously calculated and stored in disk
        expected_result = load_obj_from_disk('test_get_dominant_topic_of_each_doc_as_df_expected_result',
                                             SAVED_OBJECTS_PATH)

        result = model.get_dominant_topic_of_each_doc_as_df()

        # noinspection PyUnresolvedReferences
        self.assertTrue(expected_result.equals(result))

    # noinspection PyTypeChecker
    def test_get_related_docs_as_df(self):
        # TODO: Change this test to use LdaMallet with docs_topics_df loaded from disk
        # # For testing this method, an LdaMalletModel loaded from disk will be used
        # docs_topics_df = load_obj_from_disk('lda_mallet_docs_topics_df', SAVED_OBJECTS_PATH)
        # model = LdaMalletModel.load('lda-mallet-model', self.dataset, SAVED_TOPICS_MODELS_PATH,
        #                             docs_topics_df=docs_topics_df)
        # trigrams_func = load_func_from_disk('trigrams_func', SAVED_FUNCS_PATH)

        docs_topics_df = load_obj_from_disk('test_get_dominant_topic_of_each_doc_as_df_expected_result',
                                            SAVED_OBJECTS_PATH)
        model = LdaGensimModel.load('lda-gensim-model', self.dataset, SAVED_TOPICS_MODELS_PATH, docs_topics_df)
        trigrams_func = load_func_from_disk('trigrams_func', SAVED_FUNCS_PATH)

        text = """The baptism of Jesus is described in the gospels of Matthew, Mark and Luke. John's gospel does not
        directly describe Jesus' baptism. Most modern theologians view the baptism of Jesus by John the Baptist as a
        historical event to which a high degree of certainty can be assigned.[1][2][3][4][5] Along with the crucifixion
        of Jesus, most biblical scholars view it as one of the two historically certain facts about him, and often use 
        it as the starting point for the study of the historical Jesus.[6]"""

        # Expected result was previously calculated and stored in disk
        expected_result = load_obj_from_disk('test_get_related_docs_as_df_expected_result', SAVED_OBJECTS_PATH)

        result = model.get_related_docs_as_df(text, ngrams='tri', ngrams_model_func=trigrams_func)

        # noinspection PyUnresolvedReferences
        self.assertTrue(expected_result.equals(result))

    # noinspection PyTypeChecker
    def test_get_k_most_representative_docs_per_topic_as_df(self):
        # TODO: Change this test to use LdaMallet with docs_topics_df loaded from disk
        # # For testing this method, an LdaMalletModel loaded from disk will be used
        # docs_topics_df = load_obj_from_disk('lda_mallet_docs_topics_df', SAVED_OBJECTS_PATH)
        # model = LdaMalletModel.load('lda-mallet-model', self.dataset, SAVED_TOPICS_MODELS_PATH,
        #                             docs_topics_df=docs_topics_df)
        # trigrams_func = load_func_from_disk('trigrams_func', SAVED_FUNCS_PATH)

        docs_topics_df = load_obj_from_disk('test_get_dominant_topic_of_each_doc_as_df_expected_result',
                                            SAVED_OBJECTS_PATH)
        model = LdaGensimModel.load('lda-gensim-model', self.dataset, SAVED_TOPICS_MODELS_PATH, docs_topics_df)

        # Expected result was previously calculated and stored in disk
        expected_result = load_obj_from_disk('test_get_k_most_representative_docs_per_topic_as_df_expected_result',
                                             SAVED_OBJECTS_PATH)

        result = model.get_k_most_repr_docs_per_topic_as_df(k=5)

        # noinspection PyUnresolvedReferences
        self.assertTrue(expected_result.equals(result))

    # noinspection PyTypeChecker
    def test_get_k_most_representative_docs_of_topic_as_df(self):
        # TODO: Change this test to use LdaMallet with docs_topics_df loaded from disk
        # # For testing this method, an LdaMalletModel loaded from disk will be used
        # docs_topics_df = load_obj_from_disk('lda_mallet_docs_topics_df', SAVED_OBJECTS_PATH)
        # model = LdaMalletModel.load('lda-mallet-model', self.dataset, SAVED_TOPICS_MODELS_PATH,
        #                             docs_topics_df=docs_topics_df)
        # trigrams_func = load_func_from_disk('trigrams_func', SAVED_FUNCS_PATH)

        docs_topics_df = load_obj_from_disk('test_get_dominant_topic_of_each_doc_as_df_expected_result',
                                            SAVED_OBJECTS_PATH)
        model = LdaGensimModel.load('lda-gensim-model', self.dataset, SAVED_TOPICS_MODELS_PATH, docs_topics_df)

        # Expected result was previously calculated and stored in disk
        expected_result = load_obj_from_disk('test_get_k_most_representative_docs_of_topic_as_df_expected_result',
                                             SAVED_OBJECTS_PATH)

        result = model.get_k_most_repr_docs_of_topic_as_df(topic=0, k=5)

        # noinspection PyUnresolvedReferences
        self.assertTrue(expected_result.equals(result))


if __name__ == '__main__':
    unittest.main()

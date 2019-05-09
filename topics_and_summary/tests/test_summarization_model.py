import unittest

from topics_and_summary.models.summarization import TextRank
from topics_and_summary.tests.paths import SAVED_OBJECTS_PATH
from topics_and_summary.utils import load_obj_from_disk


class TestSummarizationModel(unittest.TestCase):

    def test_get_k_best_sentences_of_text(self):
        text = """The baptism of Jesus is described in the gospels of Matthew, Mark and Luke. 
        John's gospel does not directly describe Jesus' baptism.
        Most modern theologians view the baptism of Jesus by John the Baptist as a historical event 
        to which a high degree of certainty can be assigned.[1][2][3][4][5] Along with the crucifixion 
        of Jesus, most biblical scholars view it as one of the two historically certain facts about him, 
        and often use it as the starting point for the study of the historical Jesus.[6]
        The baptism is one of the five major milestones in the gospel narrative of the life of Jesus, 
        the others being the Transfiguration, Crucifixion, Resurrection, and Ascension.[7][8] 
        Most Christian denominations view the baptism of Jesus as an important event and a basis for the 
        Christian rite of baptism (see also Acts 19:1–7). In Eastern Christianity, Jesus' baptism is commemorated 
        on 6 January (the Julian calendar date of which corresponds to 19 January on the Gregorian calendar), 
        the feast of Epiphany.[9] In the Roman Catholic Church, the Anglican Communion, the Lutheran Churches and 
        some other Western denominations, it is recalled on a day within the following week, the feast of the 
        baptism of the Lord. In Roman Catholicism, the baptism of Jesus is one of the Luminous Mysteries sometimes 
        added to the Rosary. It is a Trinitarian feast in the Eastern Orthodox Churches."""

        # Expected result was previously calculated and stored in disk
        expected_result = load_obj_from_disk('test_get_k_best_sentences_of_text_expected_result', SAVED_OBJECTS_PATH)

        tr_model = TextRank()
        result = tr_model.get_k_best_sentences_of_text(text)

        self.assertEqual(expected_result, result)


if __name__ == '__main__':
    unittest.main()

from topics_and_summary.models.summarization import TextRank
from topics_and_summary.utils import pretty_print


def print_sentences(sentences):
    for i, sent in enumerate(sentences):
        print()
        print('Sentence {}:'.format(i), sent)


if __name__ == '__main__':
    """
    This Python module compares the use of different word-embeddings in the TextRank summarization algorithm.
    """

    text_rank_word2vec_300 = TextRank(embedding_model='word2vec')
    text_rank_glove_100 = TextRank(embedding_model='glove', glove_embedding_dim=100)
    text_rank_glove_300 = TextRank(embedding_model='glove', glove_embedding_dim=300)

    text = """
    Harry Potter is a series of fantasy novels written by British author J. K. Rowling. The novels chronicle the lives 
    of a young wizard, Harry Potter, and his friends Hermione Granger and Ron Weasley, all of whom are students at 
    Hogwarts School of Witchcraft and Wizardry. The main story arc concerns Harry's struggle against Lord Voldemort, 
    a dark wizard who intends to become immortal, overthrow the wizard governing body known as the Ministry of Magic, 
    and subjugate all wizards and Muggles (non-magical people).
    
    Since the release of the first novel, Harry Potter and the Philosopher's Stone, on 26 June 1997, the books have 
    found immense popularity, critical acclaim and commercial success worldwide. They have attracted a wide adult 
    audience as well as younger readers and are often considered cornerstones of modern young adult literature.[2] 
    The series has also had its share of criticism, including concern about the increasingly dark tone as the series 
    progressed, as well as the often gruesome and graphic violence it depicts.[citation needed] As of February 2018, 
    the books have sold more than 500 million copies worldwide, making them the best-selling book series in history, 
    and have been translated into eighty languages.[3] The last four books consecutively set records as the 
    fastest-selling books in history, with the final instalment selling roughly eleven million copies in the 
    United States within twenty-four hours of its release.
    
    The series was originally published in English by two major publishers, Bloomsbury in the United Kingdom and 
    Scholastic Press in the United States. A play, Harry Potter and the Cursed Child, based on a story co-written by 
    Rowling, premiered in London on 30 July 2016 at the Palace Theatre, and its script was published by Little, Brown. 
    The original seven books were adapted into an eight-part namesake film series by Warner Bros. Pictures, which is 
    the third highest-grossing film series of all time as of February 2018. In 2016, the total value of the Harry 
    Potter franchise was estimated at $25 billion,[4] making Harry Potter one of the highest-grossing media franchises 
    of all time.
    
    A series of many genres, including fantasy, drama, coming of age, and the British school story (which includes 
    elements of mystery, thriller, adventure, horror, and romance), the world of Harry Potter explores numerous themes 
    and includes many cultural meanings and references.[5] According to Rowling, the main theme is death.[6] Other major 
    themes in the series include prejudice, corruption, and madness.[7]
    
    The success of the books and films has allowed the Harry Potter franchise to expand with numerous derivative works, 
    a travelling exhibition that premiered in Chicago in 2009, a studio tour in London that opened in 2012, a digital 
    platform on which J.K. Rowling updates the series with new information and insight, and a pentalogy of spin-off 
    films premiering in November 2016 with Fantastic Beasts and Where to Find Them, among many other developments. 
    Most recently, themed attractions, collectively known as The Wizarding World of Harry Potter, have been built at 
    several Universal Parks & Resorts amusement parks around the world.
    """

    pretty_print('TextRank Word2Vec 300 results')
    print_sentences(text_rank_word2vec_300.get_k_best_sentences_of_text(text))

    pretty_print('TextRank Glove 100 results')
    print_sentences(text_rank_glove_100.get_k_best_sentences_of_text(text))

    pretty_print('TextRank Glove 300 results')
    print_sentences(text_rank_glove_300.get_k_best_sentences_of_text(text))

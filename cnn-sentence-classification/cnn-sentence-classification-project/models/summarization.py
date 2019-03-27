class TextRank:

    def __init__(self, embedding_model='glove', glove_embedding_dim=100):
        self.embedding_model = embedding_model
        self.glove_embedding_dim = glove_embedding_dim

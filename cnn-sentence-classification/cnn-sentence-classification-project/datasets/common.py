def get_file_content(file_path, encoding):
    """
    Returns the content os the specified file as a string.
    :param file_path: Path of the file to be converted into a string (String).
    :param encoding: Encoding of the file (String).
    :return: The content of the file as a string.
    """
    with open(file_path, encoding=encoding) as f:
        return f.read()


class Document:
    """
    Simple class to store name and content of a document.
    """

    def __init__(self, name, content):
        self.name = name
        self.content = content

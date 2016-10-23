"""Abstraction to read and write documents with NER and NEU predictions."""


class PredictionDocument(object):
    """Represents a document with predictions in column format."""
    def __init__(self):
        # List of sentences. Each sentence is a list of pairs (word, postag)
        self.sentences = []
        # List with original labels. One label per word in sentences.
        self.tags = []
        # Representation of document as continous text.
        self.text = u''
        # First rows of document (without processing) that are tagged as title.
        self.title = []
        # Map from indices in self.raw text to indices in self.tags
        self.char_map = []
        self.start_sentence()

    def is_document_start(self):
        return self.sentences == [[]]

    def loads(self, text):
        """Construct document from raw text in column format."""
        lines = text.split('\n')
        # Read title
        for line in lines:
            if len(line.split('\t')) == 5 and line.endswith('DOCUMENT START'):
                self.title.append(line)
            else:
                break
        # Read words
        for line in lines[len(self.title) + 1:]:
            if line == '\n' or line == '':
                self.start_sentence()
            else:
                word, pos_tag, tag = line.split('\t')[1:4]
                if (word not in ['.', '?', '!', ',', '"', '\''] and not
                        self.is_document_start()):
                    self.text += u' '
                self.add_word(word, pos_tag, tag)
                self.char_map.append(
                    (len(self.text), len(self.text) + len(word) - 1))
                self.text += word.decode('utf-8')

    def dump(self, output_file, new_tags=None):
        """Write document in original format to output_file.

        Appends document to output_file."""
        # Write title
        for title_line in self.title:
            output_file.write(title_line + '\n')
        output_file.write('\n')
        # Write content
        if not new_tags or len(new_tags) != len(self.tags):
            new_tags = self.tags
        tag_index = 0
        for sentence in self.sentences:
            for word_index, word in enumerate(sentence):
                try:
                    output_file.write('{}\t{}\t{}\t{}\t\n'.format(
                        word_index, word[0], word[1],
                        new_tags[tag_index]))
                except (UnicodeEncodeError, UnicodeDecodeError):
                    output_file.write('{}\t{}\t{}\t{}\t\n'.format(
                        word_index, word[0], word[1],
                        'I-ENCODE_ERROR'))
                tag_index += 1
            if len(sentence) > 0:
                output_file.write('\n')

    def start_sentence(self):
        """Records the start of a new sentence in document"""
        self.sentences.append([])

    def add_word(self, word, pos_tag, tag):
        """Add word to document."""
        self.sentences[-1].append((word, pos_tag))
        self.tags.append(tag)

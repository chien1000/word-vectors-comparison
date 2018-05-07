import re
import html
import os
import json
from smart_open import smart_open
import itertools

class LineCorpus(object):
    """ one line for one doc or one sentence """
    def __init__(self, corpus_path, limit=None):

        self.corpus_path = corpus_path
        self.limit = limit

    def __iter__(self):
        """Iterate through the lines in the source."""
        with smart_open(self.corpus_path) as fin:
            for line in itertools.islice(fin, self.limit):
                yield line.decode('utf-8')


class Reuters(object):
    """docstring for Reuters"""
    def __init__(self, data_dir):
        # super(Reuters, self).__init__()
        self.data_dir = data_dir

    def read_all_data(self):
        data_dir = self.data_dir
        file_ids = list(range(0,22))
        data_paths = list(map(lambda x: '{}/reut2-{:03d}.sgm'.format(data_dir, x), file_ids))

        all_data = []
        for data_path in data_paths:
        #     print(data_path)
            articles = self.read_data(data_path)
            all_data.extend(articles)
            
        return all_data

    def iter_all_data(self):
        data_dir = self.data_dir
        file_ids = list(range(0,22))
        data_paths = list(map(lambda x: '{}/reut2-{:03d}.sgm'.format(data_dir, x), file_ids))

        for data_path in data_paths:
            articles = self.read_data(data_path)
            for a in articles:
                yield a

    def read_data(self, data_path):
        with open(data_path, 'r') as file:
            raw = file.read()
            
        pattern_reu = re.compile(r'<REUTERS .*>([\s\S]+?)<\/REUTERS>')
        pattern_title = re.compile(r'<TITLE>([\s\S]+?)<\/TITLE>')
        pattern_body = re.compile(r'<BODY>([\s\S]+?)<\/BODY>')
        pattern_unproc = re.compile(r'<TEXT TYPE="UNPROC">')
        pattern_brief = re.compile(r'<TEXT TYPE="BRIEF">')
        
        reuters = pattern_reu.findall(raw)
        
        articles = []
        for reu in reuters:
            if pattern_unproc.search(reu) is None and pattern_brief.search(reu) is None:
        #         m = pattern_title.search(reu) 
        #         title = m.group(1)
        #         print(title)
        #         print(html.unescape(title))

                m = pattern_body.search(reu) 
                body = m.group(1)
                body = html.unescape(body).strip().strip('Reuter')
            
                body = body.replace('\n',' ')
                
                articles.append(body)
            
        return articles
        
        #iterator

from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.textcorpus import lower_to_unicode, strip_multiple_whitespaces, remove_stopwords, remove_short
from gensim.utils import deaccent, simple_tokenize

def remove_numbers(tokens):
    return [token for token in tokens if not str.isdigit(token)]
        
class MyTextCorpus(TextCorpus):
    #add remove_numbers

    def __init__(self, input=None, dictionary=None, metadata=False, character_filters=None,
                 tokenizer=None, token_filters=None):

        # super(MyTextCorpus, self).__init__(input=input, dictionary=dictionary, metadata=metadata,
        #     character_filters=character_filters, tokenizer=tokenizer, token_filters=token_filters)
        # self.token_filters.append(remove_numbers)

        self.input = input
        self.metadata = metadata

        self.character_filters = character_filters
        if self.character_filters is None:
            self.character_filters = [lower_to_unicode, deaccent, strip_multiple_whitespaces]

        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = simple_tokenize

        self.token_filters = token_filters
        if self.token_filters is None:
            self.token_filters = [remove_short, remove_stopwords, remove_numbers]
        
        self.length = None
        self.dictionary = None
        self.init_dictionary(dictionary)


    def get_text_str(self):
        lines = self.getstream()
        if self.metadata:
            for lineno, line in enumerate(lines):
                yield ' '.join(self.preprocess_text(line)), (lineno,)
        else:
            for line in lines:
                yield ' '.join(self.preprocess_text(line))

    def get_texts(self, max_sentence_length=None):
        """Generate documents from corpus.
        Yields
        ------
        list of str
            Document as sequence of tokens (+ lineno if self.metadata)
        """
        lines = self.getstream()
        
        for lineno, line in enumerate(lines):
            tokens = self.preprocess_text(line)

            if max_sentence_length is None:
                max_sentence_length = len(tokens) 
            i = 0
            while i < len(tokens):
                if self.metadata:
                    yield tokens[i: i + max_sentence_length], (lineno,)
                else:
                    yield tokens[i: i + max_sentence_length]

                i += max_sentence_length

    # def __iter__(self):
    #     for text in self.get_texts():
    #         yield text

    def __getstate__(self):
        attr = {}
        attr['input'] = self.input
        attr['character_filters'] = [func.__name__ for func in self.character_filters]
        attr['token_filters'] = [func.__name__ for func in self.token_filters]
        attr['tokenizer'] = self.tokenizer.__name__
        attr['length'] = self.length

        return attr

    def write_file(self, name):
        output_path = os.path.join('data/preprocessed', name+'.txt')
        setting_path = os.path.join('data/preprocessed', name+'_setting.txt')

        output = open(output_path, 'w')

        for tokens in self.get_texts():
            output.write(' '.join(tokens))
            output.write('\n')
        output.close()

        with open(setting_path, 'w') as out:
            json.dump(self.__getstate__(), out)


if __name__ == '__main__':

    docs = MyTextCorpus(input = 'data/wikipedia/enwiki-20180101-p30304p88444-processed.txt')
    # docs = MyTextCorpus(input = 'data/preprocessed/reuters_docperline.txt')
    docs.write_file('wiki_part')
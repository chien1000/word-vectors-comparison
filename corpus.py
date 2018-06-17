import re
import html
import os
import json
import numbers
from smart_open import smart_open
import itertools
import random
from collections import Counter

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

from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus
from gensim.corpora.textcorpus import lower_to_unicode, strip_multiple_whitespaces, remove_stopwords, remove_short
from gensim.utils import deaccent, simple_tokenize

import types
import subprocess

import logging
logger = logging.getLogger(__name__)

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

class MyDictionary(Dictionary):
    """add min_count filtering to Dictionary"""
    def __init__(self, documents=None, min_count=0):

        self.token2id = {}
        self.id2token = {}
        self.dfs = {} #Document frequencies: token_id -> in how many documents contain this token.

        self.num_docs = 0
        self.num_pos = 0 # Total number of corpus positions (number of processed words).
        self.num_nnz = 0 # Total number of non-zeroes in the BOW matrix.

        self.min_count = min_count

        if documents is not None:
            self.add_documents(documents)

    def add_documents(self, documents):
        min_count = self.min_count
        wcounter = Counter()
        
        for docno, document in enumerate(documents):
            # log progress every 10k docs
            if docno % 10000 == 0:
                logger.info("adding document #%i to %s", docno, self)

            # update Dictionary with the document
            id2freq = self.doc2bow(document, allow_update=True) 
            wcounter.update(dict(id2freq))

        #filter tokens less than min_count
        good_ids = [wid for wid,c in wcounter.items() if c>=min_count]
        self.filter_tokens(good_ids=good_ids)

        logger.info(
            "built %s from %i documents (total %i corpus positions)",
            self, self.num_docs, self.num_pos
        )     

def remove_numbers(tokens):
    return [token for token in tokens if not str.isdigit(token)]
        
class MyTextCorpus(TextCorpus):
    #add remove_numbers

    def __init__(self, input=None, dictionary=None, min_count=0, metadata=False, character_filters=None,
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
            # self.token_filters = [remove_short, remove_stopwords, remove_numbers]
            self.token_filters = [remove_short, remove_stopwords]
        
        self.length = None
        self.dictionary = None
        self.init_dictionary(dictionary, min_count=min_count)

    def init_dictionary(self, dictionary, min_count=0):
        """Initialize/update dictionary.
        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            If a dictionary is provided, it will not be updated with the given corpus on initialization.
            If None - new dictionary will be built for the given corpus.
        Notes
        -----
        If self.input is None - make nothing.
        """
        self.dictionary = dictionary if dictionary is not None else MyDictionary(min_count=min_count)
        if self.input is not None:
            if dictionary is None:
                logger.info("Initializing dictionary")
                metadata_setting = self.metadata
                self.metadata = False
                self.dictionary.add_documents(self.get_texts())
                self.metadata = metadata_setting
            else:
                logger.info("Input stream provided but dictionary already initialized")
        else:
            logger.warning("No input document stream provided; assuming dictionary will be initialized some other way.")

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
        
        if self.metadata:
            for lineno, line in enumerate(lines):
                tokens = self.preprocess_text(line)
                if isinstance(tokens, types.GeneratorType):
                    tokens = list(tokens)

                yield tokens, (lineno,)
 
        else:
            for line in lines:
                tokens = self.preprocess_text(line)
                if isinstance(tokens, types.GeneratorType):
                    tokens = list(tokens)

                if max_sentence_length is None:
                    max_sentence_length = len(tokens) 
                i = 0
                while i < len(tokens):
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

    def write_file(self, name, rand_sample=None):
        output_path = os.path.join('data/preprocessed', name+'.txt')
        setting_path = os.path.join('data/preprocessed', name+'_setting.txt')

        # n_lines = self.__len__()
        # n_lines = len(self)
        n_lines = file_len(self.input)
        print('{} lines in thie file'.format(n_lines))

        if rand_sample is None:
            take = set(range(n_lines))
        elif isinstance(rand_sample, numbers.Integral):
            take = random.sample(range(n_lines), rand_sample)
            take = set(take)
        elif isinstance(rand_sample, float):
            k = int(rand_sample * n_lines)
            take = random.sample(range(n_lines), k)
            take = set(take)
        else:
            raise(ValueError('rand_sample must be int or float'))

        output = open(output_path, 'w')

        i = 0
        for tokens in self.get_texts():
            if i in take:
                output.write(' '.join(tokens))
                output.write('\n')
            i+=1
            
        output.close()

        with open(setting_path, 'w') as out:
            info = self.__getstate__()
            info['rand_sample'] = rand_sample 
            json.dump(info, out)


if __name__ == '__main__':

    random.seed(2018)
    # docs = MyTextCorpus(input = 'data/wikipedia/enwiki-20180101-p30304p88444-processed.txt')
    # docs = MyTextCorpus(input = 'data/preprocessed/reuters_docperline.txt')
    docs = MyTextCorpus(input = 'data/wikipedia/enwiki-20180120-processed.txt', dictionary={})
    # docs.write_file('test', 0.1)
    docs.write_file('wiki_10percent', 0.1)
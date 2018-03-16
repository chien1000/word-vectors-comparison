import re
import pickle
import six
from collections import Mapping, defaultdict
import numbers
import numpy as np
import scipy.sparse as sp
import array
from operator import itemgetter
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances

from stop_words import ENGLISH_STOP_WORDS
from base import BaseWordVectorizer
from exceptions import *

def _check_stop_list(stop):
    if stop == "english":
        return ENGLISH_STOP_WORDS
    elif isinstance(stop, six.string_types):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    else:               # assume it's a collection
        return frozenset(stop)

def _document_frequency(X): ## TODO doc-term matrix沒惹..
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)
    
def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))

class BaseEstimator(object):
    """Base class for all estimators in scikit-learn
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

#     @classmethod
#     def _get_param_names(cls):

#     def get_params(self, deep=True):
       
#     def set_params(self, **params):
       
    # def __repr__(self):
    #     class_name = self.__class__.__name__
    #     return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
    #                                            offset=len(class_name),),)

    def __getstate__(self):
        try:
            state = super(BaseEstimator, self).__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        if type(self).__module__.startswith('sklearn.'):
            return dict(state.items(), _sklearn_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        if type(self).__module__.startswith('sklearn.'):
            pickle_version = state.pop("_sklearn_version", "pre-0.18")
            if pickle_version != __version__:
                warnings.warn(
                    "Trying to unpickle estimator {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        self.__class__.__name__, pickle_version, __version__),
                    UserWarning)
        try:
            super(BaseEstimator, self).__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)

class VectorizerMixin(object):
    """Provides common code for text vectorizers (tokenization logic)."""

    _white_spaces = re.compile(r"\s\s+")

    def decode(self, doc):
        """Decode the input into a string of unicode symbols
        The decoding strategy depends on the vectorizer parameters.
        """
        if self.input == 'filename':
            with open(doc, 'rb') as fh:
                doc = fh.read()

        elif self.input == 'file':
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)

        if doc is np.nan:
            raise ValueError("np.nan is an invalid document, expected byte or "
                             "unicode string.")

        return doc

    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        min_n, max_n = self.ngram_range
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the string
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngrams_append(text_document[i: i + n])
        return ngrams

    def _char_wb_ngrams(self, text_document):
        """Whitespace sensitive char-n-gram tokenization.
        Tokenize text_document into a sequence of character n-grams
        operating only inside word boundaries. n-grams at the edges
        of words are padded with space."""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        min_n, max_n = self.ngram_range
        ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for w in text_document.split():
            w = ' ' + w + ' '
            w_len = len(w)
            for n in range(min_n, max_n + 1):
                offset = 0
                ngrams_append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams_append(w[offset:offset + n])
                if offset == 0:   # count a short word (w_len < n) only once
                    break
        return ngrams

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization"""
        if self.preprocessor is not None:
            return self.preprocessor

        # unfortunately python functools package does not have an efficient
        # `compose` function that would have allowed us to chain a dynamic
        # number of functions. However the cost of a lambda call is a few
        # hundreds of nanoseconds which is negligible when compared to the
        # cost of tokenizing a string of 1000 chars for instance.
        noop = lambda x: x

        # accent stripping
        if not self.strip_accents:
            strip_accents = noop
        elif callable(self.strip_accents):
            strip_accents = self.strip_accents
        elif self.strip_accents == 'ascii':
            strip_accents = strip_accents_ascii
        elif self.strip_accents == 'unicode':
            strip_accents = strip_accents_unicode
        else:
            raise ValueError('Invalid value for "strip_accents": %s' %
                             self.strip_accents)

        if self.lowercase:
            return lambda x: strip_accents(x.lower())
        else:
            return strip_accents

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens"""
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        return lambda doc: token_pattern.findall(doc)

    def get_stop_words(self):
        """Build or fetch the effective stop words list"""
        return _check_stop_list(self.stop_words)

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif self.analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(self.decode(doc)))

        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()

            return lambda doc: self._word_ngrams(
                tokenize(preprocess(self.decode(doc))), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)

    def _validate_vocabulary(self): #使用使用者傳入的字典需要先檢查
        vocabulary = self.vocabulary 
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(six.itervalues(vocabulary))
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in range(len(vocabulary)):
                    if i not in indices: #index編號不連續
                        msg = ("Vocabulary of size %d doesn't contain index "
                               "%d." % (len(vocabulary), i))
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary) #vocabulary_為調整過的
        else:
            self.fixed_vocabulary_ = False

    def _check_vocabulary(self):
        """Check if vocabulary is empty or missing (not fit-ed)"""
#         msg = "%(name)s - Vocabulary wasn't fitted."
#         check_is_fitted(self, 'vocabulary_', msg=msg),

        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary is empty")

class HalWordVectorizer(BaseEstimator, VectorizerMixin, BaseWordVectorizer):
    """Convert a collection of text documents to a matrix of token counts
    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.
    If you do not provide an a-priori dictionary and you do not use an analyzer
    that does some kind of feature selection then the number of features will
    be equal to the vocabulary size found by analyzing the data.
    Read more in the :ref:`User Guide <text_feature_extraction>`.
    Parameters
    ----------
    window_size : context window size
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.
        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.
    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    max_features : int or None, default=None
        If not None, conserving k context words(columns) with the 
        highest variance.
    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().
    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.
    See also
    --------
    HashingVectorizer, TfidfVectorizer
    Notes
    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.
    """

    def __init__(self, window_size = 10, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', max_features=None,
                 vocabulary=None, dtype=np.int64):
        
        # super(HalWordVectorizer, self).__init__()

        self.window_size = window_size
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        self.dtype = dtype


    def _count_cooccurence(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__ #自動幫新字產生index

        context_vocabulary = defaultdict()
        context_vocabulary.default_factory = context_vocabulary.__len__ #自動幫新字產生index

        #
        analyze = self.build_analyzer() #如何斷字 by word, char or ngram?

        row = _make_int_array()
        col = _make_int_array()
        values = _make_int_array()

        window_size = self.window_size
        for doc in raw_documents:
            tokens = analyze(doc)
            doc_length = len(tokens)

            for i, feature in enumerate(tokens):
                try:
                    #左右window要分開計算！！！！！
                    feature_idx = vocabulary[feature]
                    for j in range(max(i - window_size, 0), i) :
                        context_word = (tokens[j], 'l')
                        context_idx = context_vocabulary[context_word]
                        row.append(feature_idx)
                        col.append(context_idx)
                        diff = i-j-1
                        values.append(window_size-diff)

                    for j in range(i+1, min(i + window_size, doc_length-1)+1):
                        context_word = (tokens[j], 'r')
                        context_idx = context_vocabulary[context_word]
                        row.append(feature_idx)
                        col.append(context_idx)
                        diff = j-i-1
                        values.append(window_size-diff)

                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary) #不要自動幫新字產生index！！
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")
            context_vocabulary = dict(context_vocabulary)

            
        ###sort by alphebetic order
        sorted_features = sorted(six.iteritems(vocabulary))
        sorted_context = sorted(six.iteritems(context_vocabulary), key= lambda x: (x[0][1], x[0][0]))

        map_index_v = np.empty(len(sorted_features), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index_v[old_val] = new_val

        map_index_c = np.empty(len(sorted_context), dtype=np.int32)
        for new_val, (term, old_val) in enumerate(sorted_context):
            context_vocabulary[term] = new_val
            map_index_c[old_val] = new_val

        row = map_index_v.take(row, mode='clip')
        col = map_index_c.take(col, mode='clip')

        values = np.frombuffer(values, dtype=np.intc)
        cooccurence_matrix = sp.coo_matrix((values, (row, col)), shape=(len(vocabulary), 
                                                                         len(context_vocabulary))
                                           ,dtype=self.dtype)
        cooccurence_matrix = cooccurence_matrix.tocsc()
        # cooccurence_matrix.sort_indices()

#         print(cooccurence_matrix.toarray())
        return vocabulary, context_vocabulary, cooccurence_matrix

    def fit_word_vectors(self, raw_documents):

        self._validate_vocabulary()

        vocabulary, context_vocabulary, cooccurence_matrix = self._count_cooccurence(raw_documents, False)
        self.vocabulary_ = vocabulary

        if self.max_features: #conserve top k cols with highest variance
            # compute variance 
            # E[X^2] - (E[X])^2 or np.var?
            squared = cooccurence_matrix.copy() 
            squared.data = np.power(squared.data, 2)
            mean_of_squared = squared.mean(0)
            squared_of_mean = np.power(cooccurence_matrix.mean(0), 2)
            variance = (mean_of_squared - squared_of_mean).A
            variance = np.squeeze(variance, axis = 0)
            del squared

            # conserve top k cols
            k = self.max_features
            topk_ind = np.sort(np.argsort(-variance)[:k])
            cooccurence_matrix = cooccurence_matrix[:, topk_ind]

            #update context vobabulary
            terms = list(context_vocabulary.keys())
            indices = np.array(list(context_vocabulary.values()))
            sort_ind = np.argsort(indices)
            inverse_context_vocabulary = [terms[ind] for ind in sort_ind]
            new_context_vocabulary = {inverse_context_vocabulary[ind]:new_ind for new_ind, ind in enumerate(topk_ind)}
            context_vocabulary = new_context_vocabulary


        #normalize
        cooccurence_matrix = normalize(cooccurence_matrix, norm='l2', axis=1, copy=True)

        self.context_vocabulary = context_vocabulary
        self.cooccurence_matrix = cooccurence_matrix

        return self

    def get_feature_names(self):
        """Array mapping from feature integer indices to feature name"""
        self._check_vocabulary()

        return [t for t, i in sorted(six.iteritems(self.context_vocabulary),
                                     key=itemgetter(1))] #iteritems: (key,value), 因此是根據value做排序

    def get_similarity(self, term1, term2):


        v1 = self.__getitem__(term1)
        v2 = self.__getitem__(term2)

        distance = pairwise_distances(np.vstack((v1, v2)), metric='euclidean')[0,1]

        sim = 1 / (distance + 1)
        return sim

    def __getitem__(self, key):

        if not hasattr(self, 'cooccurence_matrix'):
            raise NotFittedError('Raw documented needed be fed first to estimate word vectors before\
             acquiring specific word vector. Call fit_word_vectors(raw_documents)')

        ind = self.vocabulary_.get(key)
        if not ind:
            raise KeyError('term {} is not in the vocabulary'.format(key))

        word_vec = self.cooccurence_matrix[ind, :].toarray().squeeze()
        return word_vec


#normalization???

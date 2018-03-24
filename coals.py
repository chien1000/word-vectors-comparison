from collections import defaultdict
import six
from hal import HalWordVectorizer, _make_int_array
from stop_words import ENGLISH_CLOSED_CLASS_WORDS
import numbers
import numpy as np 
import scipy.sparse as sp

class CoalsWordVectorizer(HalWordVectorizer):
    """docstring for CoalsWordVectorizer"""
    def __init__(self, window_size = 10, max_features=None,
                 input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b", 
                 ngram_range=(1, 1), analyzer='word', vocabulary=None, dtype=np.int64):

        self.window_size = window_size
        self.max_features = max_features
        if max_features is not None:
            if (not isinstance(max_features, numbers.Integral) or
                    max_features <= 0):
                raise ValueError(
                    "max_features=%r, neither a positive integer nor None"
                    % max_features)
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer

        self.stop_words = stop_words
        stop_words_set = set(ENGLISH_CLOSED_CLASS_WORDS)
        stop_words_user = self.get_stop_words()
        if stop_words_user is not None:
            stop_words_set.update(stop_words_user)
        self.stop_words = stop_words_set
        # self.stop_words = None
        self.token_pattern = token_pattern
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
        # analyze = lambda doc: list(map(str.lower, doc.split()))

        row = _make_int_array()
        col = _make_int_array()
        values = _make_int_array()

        window_size = self.window_size
        for doc in raw_documents:
            tokens = analyze(doc)
            doc_length = len(tokens)

            for i, feature in enumerate(tokens):
                try:
                    feature_idx = vocabulary[feature]
                    for j in range(max(i - window_size, 0), min(i + window_size, doc_length-1)+1):
                        if j == i:
                            continue
                        context_word = tokens[j]
                        context_idx = context_vocabulary[context_word]
                        row.append(feature_idx)
                        col.append(context_idx)
                        diff = abs(j-i)-1
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
        sorted_context = sorted(six.iteritems(context_vocabulary))

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

        vocabulary, context_vocabulary, cooccurence_matrix = self._count_cooccurence(raw_documents,  self.fixed_vocabulary_)
        self.vocabulary_ = vocabulary

        if self.max_features: #discard all but the k columns reflecting the most common open-class words
            freqs = np.sum(cooccurence_matrix, axis=0).A[0]
            k = self.max_features
            topk_ind = np.sort(np.argsort(-freqs)[:k])
            cooccurence_matrix = cooccurence_matrix[:, topk_ind]

            #update context vobabulary
            terms = list(context_vocabulary.keys())
            indices = np.array(list(context_vocabulary.values()))
            sort_ind = np.argsort(indices)
            inverse_context_vocabulary = [terms[ind] for ind in sort_ind]
            new_context_vocabulary = {inverse_context_vocabulary[ind]:new_ind for new_ind, ind in enumerate(topk_ind)}
            context_vocabulary = new_context_vocabulary

        #normalize
        ##convert counts to word pair correlations
        t_sum = cooccurence_matrix.sum()
        row_sum = cooccurence_matrix.sum(axis = 1)
        col_sum = cooccurence_matrix.sum(axis = 0)

        cooccurence_matrix = cooccurence_matrix.tocoo()

        multi_rsum_csum_value = np.multiply(col_sum.take(cooccurence_matrix.col), 
                                                            row_sum.take(cooccurence_matrix.row)).A.squeeze()
        assert (multi_rsum_csum_value >=0).all() #check overflow
        multi_rsum_csum = sp.coo_matrix((multi_rsum_csum_value, 
                                                        (cooccurence_matrix.row, cooccurence_matrix.col)))
    
        deno = t_sum*cooccurence_matrix.tocsr() - multi_rsum_csum.tocsr()

        row_d = np.multiply(row_sum , (t_sum - row_sum))
        col_d = np.multiply(col_sum , (t_sum - col_sum))
        assert (row_d >=0).all() #check overflow
        assert (col_d >=0).all() #check overflow
      
        col_d_target_value = col_d.take(cooccurence_matrix.col).A.squeeze()
        col_d_target = sp.coo_matrix((col_d_target_value, 
                                                    (cooccurence_matrix.row, cooccurence_matrix.col)))
        col_d_target.data = 1 / np.sqrt(col_d_target.data)

        row_d_target_value = row_d.take(cooccurence_matrix.row).A.squeeze()
        row_d_target = sp.coo_matrix((row_d_target_value, 
                                                    (cooccurence_matrix.row, cooccurence_matrix.col)))
        row_d_target.data = 1 / np.sqrt(row_d_target.data)

        cooccurence_matrix = deno.multiply(col_d_target.tocsr()).multiply(row_d_target.tocsr())
        
        ##set negative values to 0
        cooccurence_matrix[cooccurence_matrix < 0] = 0

        ##take square roots
        cooccurence_matrix = np.sqrt(cooccurence_matrix)

        self.context_vocabulary = context_vocabulary
        self.cooccurence_matrix = cooccurence_matrix

        return self

    def get_similarity(self, term1, term2):


        v1 = self.__getitem__(term1)
        v2 = self.__getitem__(term2)

        sim =  np.corrcoef(v1, v2)[0, 1]
    
        return sim
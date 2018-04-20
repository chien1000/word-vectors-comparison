import gensim.utils
import gensim.matutils 
from scipy import stats
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def log_evaluate_word_sims(pearson, spearman, oov, name, pairs):
    # print('--------------------{}--------------------'.format(name))
    pairs = os.path.basename(pairs)
    print('{} Pearson correlation coefficient against {}: {:.4f}'.format(name, pairs, pearson[0]))
    print('{} Spearman rank-order correlation coefficient against {}: {:.4f}'.format(name, pairs, spearman[0]))
    print('{} Pairs with unknown words ratio: {:.1f}%'.format(name, oov))

def evaluate_word_sims(model, name, pairs, delimiter='\t', restrict_vocab=300000,
                            case_insensitive=True, dummy4unknown=False):

    """
    Compute correlation of the model with human similarity judgments. `pairs` is a filename of a dataset where
    lines are 3-tuples, each consisting of a word pair and a similarity value, separated by `delimiter`.
    An example dataset is included in Gensim (test/test_data/wordsim353.tsv). More datasets can be found at
    http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html or https://www.cl.cam.ac.uk/~fh295/simlex.html.
    The model is evaluated using Pearson correlation coefficient and Spearman rank-order correlation coefficient
    between the similarities from the dataset and the similarities produced by the model itself.

    The results are printed to log and returned as a triple (pearson, spearman, ratio of pairs with unknown words).
    Use `restrict_vocab` to ignore all word pairs containing a word not in the first `restrict_vocab`
    words (default 300,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
    If `case_insensitive` is True, the first `restrict_vocab` words are taken, and then case normalization
    is performed.
    Use `case_insensitive` to convert all words in the pairs and vocab to their uppercase form before
    evaluating the model (default True). Useful when you expect case-mismatch between training tokens
    and words pairs in the dataset. If there are multiple case variants of a single word, the vector for the first
    occurrence (also the most frequent if vocabulary is sorted) is taken.
    Use `dummy4unknown=True` to produce zero-valued similarities for pairs with out-of-vocabulary words.
    Otherwise (default False), these pairs are skipped entirely.
    """
    ok_vocab = list(model.vocabulary.items()) #restrict_vocab 
    ok_vocab = {w.lower(): v for w, v in ok_vocab} if case_insensitive else dict(ok_vocab)

    similarity_gold = []
    similarity_model = []
    oov = 0

    # original_vocab = model.vocabulary
    # model.vocabulary = ok_vocab

    for line_no, line in enumerate(gensim.utils.smart_open(pairs)):
        line = gensim.utils.to_unicode(line)
        if line.startswith('#'):
            # May be a comment
            continue
        else:
            try:
                if case_insensitive:
                    a, b, sim = [word.lower() for word in line.split(delimiter)]
                else:
                    a, b, sim = [word for word in line.split(delimiter)]
                sim = float(sim)
            except (ValueError, TypeError):
                logger.debug('Skipping invalid line #%d in %s', line_no, pairs)
                continue
            if a not in ok_vocab or b not in ok_vocab:
                oov += 1
                if dummy4unknown:
                    logger.debug('Zero similarity for line #%d with OOV words: %s', line_no, line.strip())
                    similarity_model.append(0.0)
                    similarity_gold.append(sim)
                    continue
                else:
                    logger.debug('Skipping line #%d with OOV words: %s', line_no, line.strip())
                    continue
            similarity_gold.append(sim)  # Similarity from the dataset
            similarity_model.append(model.get_similarity(a, b))  # Similarity from the model
    # self.vocab = original_vocab
    spearman = stats.spearmanr(similarity_gold, similarity_model)
    pearson = stats.pearsonr(similarity_gold, similarity_model)
    if dummy4unknown:
        oov_ratio = float(oov) / len(similarity_gold) * 100
    else:
        oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

    logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
    logger.debug(
        'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
        pairs, spearman[0], spearman[1]
    )
    logger.debug('Pairs with unknown words: %d', oov)

    log_evaluate_word_sims(pearson, spearman, oov_ratio, name, pairs)

    return pearson, spearman, oov_ratio

def log_evaluate_word_analogies(name, section):
        """Calculate score by section, helper for
        :meth:`evaluate_word_analogies`.
        Parameters
        ----------
        section : dict of (str, (str, str, str, str))
            Section given from evaluation.
        Returns
        -------
        float
            Accuracy score.
        """
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            score = correct / (correct + incorrect)
            print("{} {}: {:.1f}% ({}/{})".format(name, section['section'], 100.0 * score, correct, correct + incorrect))
            return score

def evaluate_word_analogies(model, name, analogies, restrict_vocab=300000, case_insensitive=True, dummy4unknown=False):
    """Compute performance of the model on an analogy test set.
    This is modern variant of :meth:`~gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.accuracy`, see
    `discussion on GitHub #1935 <https://github.com/RaRe-Technologies/gensim/pull/1935>`_.
    The accuracy is reported (printed to log and returned as a score) for each section separately,
    plus there's one aggregate summary at the end.
    This method corresponds to the `compute-accuracy` script of the original C word2vec.
    See also `Analogy (State of the art) <https://aclweb.org/aclwiki/Analogy_(State_of_the_art)>`_.
    Parameters
    ----------
    analogies : str
        Path to file, where lines are 4-tuples of words, split into sections by ": SECTION NAME" lines.
        See `gensim/test/test_data/questions-words.txt` as example.
    restrict_vocab : int, optional
        Ignore all 4-tuples containing a word not in the first `restrict_vocab` words.
        This may be meaningful if you've sorted the model vocabulary by descending frequency (which is standard
        in modern word embedding models).
    case_insensitive : bool, optional
        If True - convert all words to their uppercase form before evaluating the performance.
        Useful to handle case-mismatch between training tokens and words in the test set.
        In case of multiple case variants of a single word, the vector for the first occurrence
        (also the most frequent if vocabulary is sorted) is taken.
    dummy4unknown : bool, optional
        If True - produce zero accuracies for 4-tuples with out-of-vocabulary words.
        Otherwise, these tuples are skipped entirely and not used in the evaluation.
    Returns
    -------
    (float, list of dict of (str, (str, str, str))
        Overall evaluation score and full lists of correct and incorrect predictions divided by sections.
    """

    ok_vocab = list(model.vocabulary.items()) #restrict_vocab 
    ok_vocab = {w.lower(): v for w, v in ok_vocab} if case_insensitive else dict(ok_vocab)
    oov = 0

    # print("Evaluating word analogies for top %i words in the model on %s", restrict_vocab, analogies)

    sections, section = [], None
    quadruplets_no = 0
    for line_no, line in enumerate(gensim.utils.smart_open(analogies)):
        line = gensim.utils.to_unicode(line)
        if line.startswith(': '):
            # a new section starts => store the old section
            if section:
                sections.append(section)
                log_evaluate_word_analogies(name, section)
            section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
        else:
            if not section:
                raise ValueError("Missing section header before line #%i in %s" % (line_no, analogies))
            try:
                if case_insensitive:
                    a, b, c, expected = [word.lower() for word in line.split()]
                else:
                    a, b, c, expected = [word for word in line.split()]
            except ValueError:
                print("Skipping invalid line #%i in %s", line_no, analogies)
                continue
            quadruplets_no += 1
            if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                oov += 1
                if dummy4unknown:
                    logger.debug('Zero accuracy for line #%d with OOV words: %s', line_no, line.strip())
                    section['incorrect'].append((a, b, c, expected))
                else:
                    logger.debug("Skipping line #%i with OOV words: %s", line_no, line.strip())
                continue
    #         original_vocab = model.vocabulary
    #         model.vocabulary = ok_vocab
            ignore = {a, b, c}  # input words to be ignored
            predicted = None
            # find the most likely prediction using 3CosAdd (vector offset) method
            # TODO: implement 3CosMul and set-based methods for solving analogies
            sims = model.most_similar(positive=[b, c], negative=[a], topn=5, restrict_vocab=restrict_vocab)
    #         model.vocabulary = original_vocab
            for element in sims:
                predicted = element[0].lower() if case_insensitive else element[0]
                if predicted in ok_vocab and predicted not in ignore:
                    if predicted != expected:
                        logger.debug("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                    break
            if predicted == expected:
                section['correct'].append((a, b, c, expected))
            else:
                section['incorrect'].append((a, b, c, expected))
    if section:
        # store the last section, too
        sections.append(section)
        log_evaluate_word_analogies(name, section)

    total = {
        'section': 'Total accuracy',
        'correct': sum((s['correct'] for s in sections), []),
        'incorrect': sum((s['incorrect'] for s in sections), []),
    }

    oov_ratio = float(oov) / quadruplets_no * 100
    print('{} Quadruplets with out-of-vocabulary words: {:.1f}%'.format(name, oov_ratio))
    # if not dummy4unknown:
    #     print(
    #         'NB: analogies containing OOV words were skipped from evaluation! '
    #         'To change this behavior, use "dummy4unknown=True"'
    #     )
    analogies_score = log_evaluate_word_analogies(name, total)
    sections.append(total)
    # Return the overall score and the full lists of correct and incorrect analogies
    return analogies_score, sections

#TODO logger.info
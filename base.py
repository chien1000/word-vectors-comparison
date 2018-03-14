from stop_words import ENGLISH_STOP_WORDS
import six

class BaseWordVectorizer(object):
    """docstring for BaseWordVectorizer"""
    def __init__(self):
        super(BaseWordVectorizer, self).__init__()

    def _check_stop_list(self, stop):
        if stop == "english":
            return ENGLISH_STOP_WORDS
        elif isinstance(stop, six.string_types):
            raise ValueError("not a built-in stop list: %s" % stop)
        elif stop is None:
            return None
        else:               # assume it's a collection
            return frozenset(stop)
        
try:
    from .autogoal import ClassifierWrapper
except ImportError:
    print("Warning: autogoal is not installed. ClassifierWrapper will not be available.")
from .features import encode_features
from .md2text import md2text

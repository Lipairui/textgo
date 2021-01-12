from .embeddings import Embeddings
from .preprocess import Preprocess
from .textsim import TextSim
from .metrics import Metrics
from .utils import transformGlove
from .classifier import Classifier

from pbr.version import VersionInfo
all = ('__version__')
__version__ = VersionInfo('textgo').release_string()

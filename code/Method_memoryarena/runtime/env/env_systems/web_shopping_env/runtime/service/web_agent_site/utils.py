import bisect
import hashlib
import logging
import random
from os.path import dirname, abspath, join

from ...runtime_paths import (
    default_attr_file,
    default_feat_conv_file,
    default_feat_ids_file,
    default_human_attr_file,
    default_items_file,
    default_review_file,
    default_search_engine_root,
)

BASE_DIR = dirname(abspath(__file__))
DEBUG_PROD_SIZE = None  # set to `None` to disable

DEFAULT_ATTR_PATH = str(default_attr_file())
DEFAULT_FILE_PATH = str(default_items_file())
DEFAULT_REVIEW_PATH = str(default_review_file())
FEAT_CONV = str(default_feat_conv_file())
FEAT_IDS = str(default_feat_ids_file())
HUMAN_ATTR_PATH = str(default_human_attr_file())
SEARCH_ENGINE_ROOT = str(default_search_engine_root())

def random_idx(cum_weights):
    """Generate random index by sampling uniformly from sum of all weights, then
    selecting the `min` between the position to keep the list sorted (via bisect)
    and the value of the second to last index
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx

def setup_logger(session_id, user_log_dir):
    """Creates a log file and logging object for the corresponding session ID"""
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(
        user_log_dir / f'{session_id}.jsonl',
        mode='w'
    )
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

def generate_mturk_code(session_id: str) -> str:
    """Generates a redeem code corresponding to the session ID for an MTurk
    worker once the session is completed
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()

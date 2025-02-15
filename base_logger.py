import logging


logger = logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s -> %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S')

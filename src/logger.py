import logging

def get_logger() -> logging.Logger:

    '''
    Returns logger description
    '''

    logger = logging.getLogger('dataflow')
    logger.setLevel(logging.INFO)
    
    return logger
import re

DELIMITER = ';'
REF = '#'


def parse_decomposition(qdmr):
    """Parses the decomposition into an ordered list of steps
    
    Parameters
    ----------
    qdmr : str
        String representation of the QDMR
    
    Returns
    -------
    list
        returns ordered list of qdmr steps
    """
    # parse commas as separate tokens
    qdmr = qdmr.replace(",", " , ")
    crude_steps = qdmr.split(DELIMITER)
    steps = []
    for i in range(len(crude_steps)):
        step = crude_steps[i]
        tokens = step.split()
        step = ""
        # remove 'return' prefix
        for tok in tokens[1:]:
            step += tok.strip() + " "
        step = step.strip()
        steps += [step]
    return steps
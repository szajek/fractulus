import logging


solver = logging.getLogger('solver')
solver.setLevel(logging.DEBUG)
solver.addHandler(
    logging.StreamHandler()
)

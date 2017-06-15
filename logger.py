import logging


solver = logging.getLogger('solver')
solver.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
solver.addHandler(ch)

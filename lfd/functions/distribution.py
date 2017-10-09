class Distribution(object):
    def __init__(self, target_distribution):
        self._cache = {}
        self.target_distribution = target_distribution

    def __call__(self, *args):
        pass

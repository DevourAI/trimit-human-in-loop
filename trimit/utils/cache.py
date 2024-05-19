class CacheMixin:
    def __init__(self, cache_prefix, cache=None):
        import diskcache as dc

        self.cache_prefix = cache_prefix
        self.cache = cache or dc.Cache(".cache")

    def cache_get(self, key, *default):
        use_default = len(default) > 0
        if use_default:
            return self.cache.get(self.cache_prefix + key, default[0])
        return self.cache.get(self.cache_prefix + key)

    def cache_set(self, key, value):
        self.cache[self.cache_prefix + key] = value

    def in_cache(self, key):
        return self.cache_prefix + key in self.cache

async def async_passthrough(x):
    return x


async def async_passthrough_gen(x):
    yield x, True

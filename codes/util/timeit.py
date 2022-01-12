import time


def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r  took: %2.4f sec' %
              (f.__name__, te-ts))
        return result

    return timed
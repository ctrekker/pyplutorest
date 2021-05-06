import requests
from requests.utils import quote
import msgpack
import array


VERSION = 'v1'
DEFAULT_HOST = 'localhost:1234'


def default(obj):
    if isinstance(obj, array.array) and obj.typecode == 'b':
        return msgpack.ExtType(0x11, obj.tostring())
    elif isinstance(obj, array.array) and obj.typecode == 'B':
        return msgpack.ExtType(0x12, obj.tostring())
    elif isinstance(obj, array.array) and obj.typecode == 'h':
        return msgpack.ExtType(0x13, obj.tostring())
    elif isinstance(obj, array.array) and obj.typecode == 'H':
        return msgpack.ExtType(0x14, obj.tostring())
    elif isinstance(obj, array.array) and obj.typecode == 'i':
        return msgpack.ExtType(0x15, obj.tostring())
    elif isinstance(obj, array.array) and obj.typecode == 'I':
        return msgpack.ExtType(0x16, obj.tostring())
    elif isinstance(obj, array.array) and obj.typecode == 'f':
        return msgpack.ExtType(0x17, obj.tostring())
    elif isinstance(obj, array.array) and obj.typecode == 'd':
        return msgpack.ExtType(0x18, obj.tostring())
    raise TypeError("Unknown type: %r" % (obj,))

def ext_hook(code, data):
    if code == 0x11:
        return array.array('b', data)
    elif code == 0x12:
        return array.array('B', data)
    elif code == 0x13:
        return array.array('h', data)
    elif code == 0x14:
        return array.array('H', data)
    elif code == 0x15:
        return array.array('i', data)
    elif code == 0x16:
        return array.array('I', data)
    elif code == 0x17:
        return array.array('f', data)
    elif code == 0x18:
        return array.array('d', data)
    return msgpack.ExtType(code, data)


def evaluate(output, inputs, filename, host=DEFAULT_HOST):
    body = msgpack.packb({
        'inputs': inputs,
        'outputs': [output]
    }, default=default, use_bin_type=True)
    headers = {
        'Content-Type': 'application/x-msgpack',
        'Accept': 'application/x-msgpack'
    }
    req = requests.post(f'''http://{host}/{VERSION}/notebook/{quote(filename, safe='')}/eval''', data=body, headers=headers)
    
    if req.status_code >= 300:
        raise Exception(req.content)
    return msgpack.unpackb(req.content, ext_hook=ext_hook)[output]

def call(symbol, args, kwargs, filename, host=DEFAULT_HOST):
    body = msgpack.packb({
        'function': symbol,
        'args': args,
        'kwargs': kwargs
    }, default=default, use_bin_type=True)
    headers = {
        'Content-Type': 'application/x-msgpack',
        'Accept': 'application/x-msgpack'
    }
    req = requests.post(f'''http://{host}/{VERSION}/notebook/{quote(filename, safe='')}/call''', data=body, headers=headers)
    
    return msgpack.unpackb(req.content, ext_hook=ext_hook)


class PlutoNotebook:
    def __init__(self, filename, host=DEFAULT_HOST):
        self.filename = filename
        self.host = host

    def __getattr__(self, attr):
        try:
            return evaluate(attr, {}, self.filename, self.host)
        except Exception as e:
            if 'function' in str(e):
                return PlutoCallable(attr, self)
            else:
                raise e

    def __call__(self, **kwargs):
        return PlutoNotebookWithArgs(kwargs, self)

class PlutoNotebookWithArgs:
    def __init__(self, args, notebook):
        self.notebook = notebook
        self.args = args

    def __getattr__(self, attr):
        return evaluate(attr, self.args, self.notebook.filename, self.notebook.host)

class PlutoCallable:
    def __init__(self, symbol, notebook):
        self.symbol = symbol
        self.notebook = notebook

    def __call__(self, *args, **kwargs):
        return call(self.symbol, args, kwargs, self.notebook.filename, self.notebook.host)


# Some test cases...

# print(evaluate('dist', {
#     'z': [1., 1., 2.]
# }, 'Softmax.jl'))
# print(call('σ', [[1., 1., 3.]], {}, 'Softmax.jl'))

# nb = PlutoNotebook('Softmax.jl')
# print(nb(z=[1., 1., 2.]).dist)
# print(nb.σ([1., 1., 3.]))

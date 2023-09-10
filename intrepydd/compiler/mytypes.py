import abc


class MyType(abc.ABC):
    @abc.abstractmethod
    def get_cpp_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_mangled_name(self) -> str:
        pass

    def get_declare_str(self):
        if pass_by_ref(self):
            return self.get_cpp_name() + '*'
        else:
            return self.get_cpp_name()

    def is_subtype_of(self, other):
        if type(self) == NpArray and type(other) == NpArray:
            return self.etype == other.etype \
                and (other.ndim == -1 or self.ndim == other.ndim) \
                and self.layout == other.layout
        return self == other

    # return true if types can be broadcasted
    # ex: (array<int>, int) -> true
    # ex: (array<double>, int) -> false
    # ex: (tuple<int>, int) -> false
    def can_broadcast_with(self, ty):
        if not (is_basic_type(self) or is_array(self)):
            return False
        if not (is_basic_type(ty) or is_array(ty)):
            return False
        e_type_1 = self if is_basic_type(self) else self.etype
        e_type_2 = ty if is_basic_type(ty) else ty.etype
        return e_type_1 == e_type_2


class IterableType(MyType):
    etype: MyType

    def get_elt_type(self):
        '''
        Returns the element type in a `for...in` clause
        '''
        return self.etype


class ListType(IterableType):
    def __init__(self, etype):
        self.etype = etype
        self.initsize = 0

    def set_init_size(self, n):
        self.initsize = n

    def get_init_size(self):
        return self.initsize

    def get_cpp_name(self):
        return 'std::vector<%s>' % str(self.etype)

    def get_mangled_name(self):
        return f'l{self.etype.get_mangled_name()}'

    def __str__(self):
        return 'std::vector<%s>' % str(self.etype)

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return other.etype == self.etype
        return False


class TupleType(MyType):
    def __init__(self, types):
        self.etypes = tuple(types)

    def set_init_size(self, n):
        self.initsize = n

    def get_init_size(self):
        return self.initsize

    def get_cpp_name(self):
        if self.etypes:
            es = self.etypes[0].get_cpp_name()
        else:
            es = 'int64_t'
        return f'std::vector<{es}>'

    def get_mangled_name(self):
        name_segs = [f't{len(self.etypes)}']
        for t in self.etypes:
            name_segs.append(t.get_mangled_name())
        return ''.join(name_segs)

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return other.etypes == self.etypes
        return False


class DictType(IterableType):
    def __init__(self, key_ty, value_ty):
        self.key = key_ty
        self.value = value_ty

    def get_elt_type(self):
        return self.key

    def get_cpp_name(self):
        return 'std::unordered_map<%s, %s>' % (self.key, self.value)

    def get_mangled_name(self):
        return f'd{self.key.get_mangled_name()}{self.value.get_mangled_name()}'

    def __str__(self):
        # return 'std::map<%s, %s>' % (self.key, self.value)
        return 'std::unordered_map<%s, %s>' % (self.key, self.value)

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return other.key == self.key and other.value == self.value
        return False


class BoolType(MyType):
    def __init__(self):
        self.by_ref = False

    def get_cpp_name(self):
        return 'bool'

    def get_mangled_name(self):
        return 'b'

    def __str__(self):
        return 'bool'

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)


class IntType(MyType):
    def __init__(self, w=64):
        self.by_ref = False
        self.width = w

    def get_width(self):
        return self.width

    def get_cpp_name(self):
        if self.width == 32:
            return 'int'
        elif self.width == 64:
            return 'int64_t'
        else:
            raise Exception()

    def get_mangled_name(self):
        return f'i{self.width}'

    def __str__(self):
        if self.width == 32:
            return 'int'
        elif self.width == 64:
            return 'int64_t'
        else:
            raise Exception()

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return other.width == self.width
        return False


class FloatType(MyType):
    def __init__(self, w=64):
        self.by_ref = False
        self.width = w

    def get_width(self):
        return self.width

    def get_cpp_name(self):
        if self.width == 64:
            return 'double'
        elif self.width == 32:
            return 'float'
        else:
            assert False

    def get_mangled_name(self):
        return f'f{self.width}'

    def __str__(self):
        if self.width == 64:
            return 'double'
        elif self.width == 32:
            return 'float'
        else:
            assert False

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return other.width == self.width
        return False


class StrType(MyType):
    def __str__(self):
        return "std::string"

    def get_cpp_name(self):
        return "std::string"

    def get_elt_type(self):
        return IntType()

    def get_mangled_name(self):
        return 's'

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)


class VoidType(MyType):
    def __init__(self):
        self.by_ref = False
        pass

    def get_cpp_name(self):
        return 'void *'

    def get_mangled_name(self):
        return 'v'

    def __str__(self):
        return 'void *'

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)


class NpArray(IterableType):
    def __init__(self, dtype, ndim=-1, layout='K'):
        self.etype = dtype
        self.ndim = ndim
        self.layout = layout

    def get_dimension(self):
        return self.ndim

    def __str__(self):
        return "py::array_t<%s>" % self.etype
        # return "pydd::NpArray<%s>*" % self.etype

    def get_cpp_name(self):
        return "py::array_t<%s>" % self.etype.get_cpp_name()

    def get_new_str(self):
        return "py::array_t<%s>" % self.etype
        # return "new pydd::NpArray<%s>" % self.etype

    def get_mangled_name(self):
        nd = self.ndim
        if nd == -1:
            nd = 'u'
        return f'a{self.etype.get_mangled_name()}{nd}{self.layout}'

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return other.etype == self.etype \
                and other.ndim == self.ndim \
                and other.layout == self.layout
        return False


class SparseMat(IterableType):
    def __init__(self, dtype, ndim=2, layout='K'):
        self.etype = dtype
        self.ndim = ndim
        self.layout = layout

    def get_cpp_name(self):
        return 'pydd::Sparse_matrix<%s>' % self.etype.get_cpp_name()

    def get_mangled_name(self):
        return f'm{self.etype.get_mangled_name()}{self.ndim}{self.layout}'

    def __str__(self):
        return 'pydd::Sparse_matrix<%s>' % self.etype

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return other.etype == self.etype \
                and other.ndim == self.ndim \
                and other.layout == self.layout
        return False


class Heap(IterableType):
    def __init__(self, key_ty, val_ty):
        self.key = key_ty
        self.val = val_ty
        # self.etype = dtype

    def get_elt_type(self):
        return self.key

    def get_cpp_name(self):
        return 'pydd::Heap<%s, %s>' % \
               (self.key.get_cpp_name(), self.val.get_cpp_name())

    def get_mangled_name(self):
        return f'h{self.key.get_mangled_name()}{self.val.get_mangled_name()}'

    def __str__(self):
        return 'pydd::Heap<%s, %s>' % (self.key, self.val)

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return other.key == self.key and other.val == self.val
        return False


def pass_by_ref(ty):
    # if isinstance(ty, VoidType):
        # return True
    return False
    if isinstance(ty, IntType) or \
            isinstance(ty, FloatType) or \
            isinstance(ty, NpArray) or \
            isinstance(ty, SparseMat) or \
            isinstance(ty, Heap) or \
            isinstance(ty, VoidType) or \
            isinstance(ty, StrType) or \
            isinstance(ty, BoolType) or \
            isinstance(ty, DictType):
        return False
    else:
        return True


def is_float(ty):
    return isinstance(ty, FloatType)


def is_all_float(t1, t2):
    return is_float(t1) and is_float(t2)


def is_int(ty):
    return isinstance(ty, IntType)


def is_all_int(t1, t2):
    return is_int(t1) and is_int(t2)


def is_array(ty):
    # return isinstance(ty, ListType) or isinstance(ty, NpArray)
    return isinstance(ty, NpArray)


def is_list(ty):
    return isinstance(ty, ListType)


def is_tuple(ty):
    return isinstance(ty, TupleType)


def is_basic_type(ty):
    return not hasattr(ty, 'etype')


bool = BoolType()
int32 = IntType(32)
int64 = IntType(64)
float32 = FloatType(32)
double = float64 = FloatType(64)
float32_list = ListType(float32)
float64_list = ListType(float64)
int32_list = ListType(int32)
int64_list = ListType(int64)
void = VoidType()
string = StrType()
float64_ndarray = NpArray(float64)
float64_1darray = NpArray(float64, 1)
float64_2darray = NpArray(float64, 2)
float32_ndarray = NpArray(float32)
int32_ndarray = NpArray(int32)
int32_1darray = NpArray(int32, 1)
int64_ndarray = NpArray(int64)
bool_ndarray = NpArray(bool)
float64_sparray = SparseMat(float64)
float32_heap = Heap(float32, int32)
int32_int32_heap = Heap(int32, int32)

# Special types for return vars
arg_dependent_type = None

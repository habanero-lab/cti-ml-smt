from typing import cast, Callable, Union, Optional
from functools import partial
import mytypes
from . import solver
from .solver import z3


class TypeManager:

    def __init__(self, max_tuple_size: int):
        self.max_tuple_size = max(max_tuple_size, 2)

        dtype = z3.Datatype('DType')
        dtype.declare('bool')
        dtype.declare('int32')
        dtype.declare('int64')
        dtype.declare('float32')
        dtype.declare('float64')

        type = z3.Datatype('Type')
        type.declare('none')
        type.declare('array', ('array_dtype', dtype),
                     ('array_ndim', z3.IntSort()))
        type.declare('sparsemat', ('sparsemat_dtype', dtype))
        type.declare('list', ('list_etype', type))
        type.declare('dict', ('dict_ktype', type), ('dict_vtype', type))
        type.declare('heap', ('heap_ktype', type), ('heap_vtype', type))
        for n in range(self.max_tuple_size + 1):
            params = [(f'tuple{n}_e{i}', type) for i in range(n)]
            type.declare(f'tuple{n}', *params)

        self.dtype, self.type = z3.CreateDatatypes(dtype, type)

        self.is_none = self.type.is_none
        self.is_array = self.type.is_array
        self.is_sparsemat = self.type.is_sparsemat
        self.is_list = self.type.is_list
        self.is_dict = self.type.is_dict
        self.is_heap = self.type.is_heap

        self.is_valid_type = self._create_is_valid_type()

        self.solver = solver.Solver()
        self.num_consts = 0

    def _create_is_valid_type(self):
        is_valid_type = solver.RecFunction('is_valid_type',
                                           self.type, z3.BoolSort())
        t = solver.RecVar('t', self.type)
        return solver.RecAddDefinition(
            is_valid_type,
            [t],
            z3.And(
                z3.Implies(
                    self.is_array(t),
                    self.type.array_ndim(t) >= 0
                ),
                z3.Implies(
                    self.is_list(t),
                    is_valid_type(self.type.list_etype(t))
                ),
                z3.Implies(
                    self.is_dict(t),
                    z3.And(
                        self.is_int(self.type.dict_ktype(t)),
                        is_valid_type(self.type.dict_vtype(t)),
                    )
                ),
                z3.Implies(
                    self.is_heap(t),
                    z3.And(
                        self.is_scalar(self.type.heap_ktype(t)),
                        is_valid_type(self.type.heap_vtype(t)),
                    )
                ),
                z3.Implies(
                    self.is_tuple(t),
                    z3.And(
                        self.tuple_all(t, is_valid_type),
                        self.tuple_is_homogenous(t)
                    )
                )
            )
        )

    def add_constraint(self, constraint):
        self.solver.add(constraint)

    def create_const(self, sort: z3.SortRef):
        const = z3.Const(f'c{self.num_consts}', sort)
        self.num_consts += 1
        return const

    def create_size(self):
        size = cast(z3.IntNumRef, self.create_const(z3.IntSort()))
        self.add_constraint(size >= 0)
        return size

    def create_dtype(self):
        return self.create_const(self.dtype)

    def create_tuple(self,
                     etypes: Union[z3.ExprRef, list[z3.ExprRef]],
                     size: Union[int, z3.ExprRef, None] = None):
        ty = self.create_const(self.type)

        if isinstance(etypes, list):
            if size is None:
                size = len(etypes)
            else:
                assert isinstance(size, int)
            assert size <= self.max_tuple_size
            self.add_constraint(
                ty == getattr(self.type, f'tuple{size}')(*etypes)
            )
        else:
            etype = etypes
            if isinstance(size, int):
                assert size <= self.max_tuple_size
                etypes = [etype] * size
                self.add_constraint(
                    ty == getattr(self.type, f'tuple{size}')(etypes)
                )
            elif size is None:
                self.add_constraint(
                    z3.And(self.is_tuple(ty),
                           self.tuple_all(ty, lambda t: t == etype)))
            else:
                self.add_constraint(
                    z3.And(self.is_tuple(ty),
                           self.tuple_all(ty, lambda t: t == etype),
                           self.get_tuple_size(ty) == size))

        return ty

    def is_tuple(self, ty: z3.ExprRef):
        return z3.Or(*[
            self.is_tuple_of_size(ty, i)
            for i in range(self.max_tuple_size + 1)
        ])

    def tuple_is_homogenous(self, ty: z3.ExprRef):
        exprs = []
        for i in range(2, self.max_tuple_size + 1):
            exprs.append(
                z3.Implies(
                    self.is_tuple_of_size(ty, i),
                    z3.And(*[
                        getattr(self.type, f'tuple{i}_e{j}')(ty) == getattr(
                            self.type, f'tuple{i}_e0')(ty)
                        for j in range(1, i)
                    ])))
        return z3.And(*exprs)

    def is_tuple_of_size(self, ty: z3.ExprRef, size: int):
        assert 0 <= size <= self.max_tuple_size
        return getattr(self.type, f'is_tuple{size}')(ty)

    def get_tuple_size(self, ty: z3.ExprRef):
        size = self.create_size()
        for i in range(self.max_tuple_size + 1):
            self.add_constraint(
                z3.Implies(self.is_tuple_of_size(ty, i), size == i))
        return size

    def tuple_all(self, ty: z3.ExprRef, cond: Callable[[z3.ExprRef],
                                                       z3.BoolRef]):
        exprs = []
        for i in range(1, self.max_tuple_size + 1):
            exprs.append(
                z3.Implies(
                    self.is_tuple_of_size(ty, i),
                    z3.And(*[
                        cond(getattr(self.type, f'tuple{i}_e{j}')(ty))
                        for j in range(i)
                    ])))
        return z3.And(*exprs)

    def is_scalar(self, ty: z3.ExprRef):
        return z3.And(self.type.is_array(ty), self.type.array_ndim(ty) == 0)

    def is_bool(self, ty: z3.ExprRef):
        return z3.And(self.is_scalar(ty),
                      self.type.array_dtype(ty) == self.dtype.bool)

    def is_int(self, ty: z3.ExprRef):
        return z3.And(
            self.is_scalar(ty),
            z3.Or(
                self.type.array_dtype(ty) == self.dtype.int32,
                self.type.array_dtype(ty) == self.dtype.int64))

    def is_float(self, ty: z3.ExprRef):
        return z3.And(
            self.is_scalar(ty),
            z3.Or(
                self.type.array_dtype(ty) == self.dtype.float32,
                self.type.array_dtype(ty) == self.dtype.float64))

    def create_type(self):
        ty = self.create_const(self.type)
        self.add_constraint(self.is_valid_type(ty))
        return ty

    def create_none(self):
        return self.type.none

    def create_array(self, dtype=None, ndim=None):
        ty = self.create_const(self.type)
        self.add_constraint(self.type.is_array(ty))
        if dtype is not None:
            self.add_constraint(self.type.array_dtype(ty) == dtype)
        if ndim is not None:
            self.add_constraint(self.type.array_ndim(ty) == ndim)
        return ty

    def create_sparsemat(self, dtype=None):
        ty = self.create_const(self.type)
        self.add_constraint(self.type.is_sparsemat(ty))
        if dtype is not None:
            self.add_constraint(self.type.sparsemat_dtype(ty) == dtype)
        return ty

    def create_list(self, etype=None):
        ty = self.create_const(self.type)
        self.add_constraint(self.type.is_list(ty))
        if etype is not None:
            self.add_constraint(self.type.list_etype(ty) == etype)
        return ty

    def create_dict(self, ktype=None, vtype=None):
        ty = self.create_const(self.type)
        self.add_constraint(self.type.is_dict(ty))
        if ktype is not None:
            self.add_constraint(self.type.dict_ktype(ty) == ktype)
        if vtype is not None:
            self.add_constraint(self.type.dict_vtype(ty) == vtype)
        return ty

    def create_heap(self, ktype=None, vtype=None):
        ty = self.create_const(self.type)
        self.add_constraint(self.type.is_heap(ty))
        if ktype is not None:
            self.add_constraint(self.type.heap_ktype(ty) == ktype)
        if vtype is not None:
            self.add_constraint(self.type.heap_vtype(ty) == vtype)
        return ty

    def to_mytype(self, ty: z3.ExprRef) -> mytypes.MyType:
        name = solver.get_constructor_name(ty)
        if name == 'none':
            return mytypes.VoidType()
        if name == 'array':
            dtype = self._dtype_to_mytype(solver.get_arg(ty, 0))
            assert dtype is not None
            ndim = solver.as_long(solver.get_arg(ty, 1))
            if ndim == 0:
                return dtype
            return mytypes.NpArray(dtype, ndim)
        if name == 'sparsemat':
            dtype = self._dtype_to_mytype(solver.get_arg(ty, 0))
            assert dtype is not None
            return mytypes.SparseMat(dtype)
        if name == 'list':
            etype = self.to_mytype(solver.get_arg(ty, 0))
            return mytypes.ListType(etype)
        if name == 'dict':
            ktype = self.to_mytype(solver.get_arg(ty, 0))
            vtype = self.to_mytype(solver.get_arg(ty, 1))
            return mytypes.DictType(ktype, vtype)
        if name == 'heap':
            ktype = self.to_mytype(solver.get_arg(ty, 0))
            vtype = self.to_mytype(solver.get_arg(ty, 1))
            return mytypes.Heap(ktype, vtype)
        if name.startswith('tuple'):
            size = int(name[5:])
            etypes = [self.to_mytype(solver.get_arg(ty, i))
                      for i in range(size)]
            return mytypes.TupleType(etypes)
        assert False

    def _dtype_to_mytype(self, dtype: z3.ExprRef) -> Optional[mytypes.MyType]:
        dtype_name = solver.get_constructor_name(dtype)
        if dtype_name == 'bool':
            return mytypes.bool
        if dtype_name == 'int32':
            return mytypes.int32
        if dtype_name == 'int64':
            return mytypes.int64
        if dtype_name == 'float32':
            return mytypes.float32
        if dtype_name == 'float64':
            return mytypes.float64
        return None

    def from_mytype(self, t: mytypes.MyType) -> z3.ExprRef:
        dtype = self._mytype_to_dtype(t)
        if dtype is not None:
            return self.type.array(dtype, 0)
        if isinstance(t, mytypes.VoidType):
            return self.type.none
        if isinstance(t, mytypes.NpArray):
            dtype = self._mytype_to_dtype(t.etype)
            assert dtype is not None
            ndim = t.ndim
            assert ndim > 0
            return self.type.array(dtype, ndim)
        if isinstance(t, mytypes.SparseMat):
            dtype = self._mytype_to_dtype(t.etype)
            assert dtype is not None
            assert t.ndim == 2
            return self.type.sparsemat(dtype)
        if isinstance(t, mytypes.ListType):
            etype = self.from_mytype(t.etype)
            return self.type.list(etype)
        if isinstance(t, mytypes.DictType):
            ktype = self.from_mytype(t.key)
            vtype = self.from_mytype(t.value)
            return self.type.dict(ktype, vtype)
        if isinstance(t, mytypes.Heap):
            ktype = self.from_mytype(t.key)
            vtype = self.from_mytype(t.val)
            return self.type.heap(ktype, vtype)
        if isinstance(t, mytypes.TupleType):
            etypes = [self._mytype_to_dtype(et) for et in t.etypes]
            size = len(etypes)
            return getattr(self.type, f'tuple{size}')(*etypes)
        assert False

    def _mytype_to_dtype(self, t: mytypes.MyType) -> Optional[z3.ExprRef]:
        if t == mytypes.bool:
            return self.dtype.bool
        if t == mytypes.int32:
            return self.dtype.int32
        if t == mytypes.int64:
            return self.dtype.int64
        if t == mytypes.float32:
            return self.dtype.float32
        if t == mytypes.float64:
            return self.dtype.float64
        return None

    def is_valid_param_type(self, ty: z3.ExprRef, max_ndim: int = 3, max_container_depth: int = 2,
                            dtypes: Optional[list[z3.ExprRef]] = None):
        assert max_container_depth >= 0

        def is_valid_array(t: z3.ExprRef):
            constraints = [self.is_array(t),
                           self.type.array_ndim(t) >= 0,
                           self.type.array_ndim(t) <= max_ndim]
            if dtypes is not None:
                dtype_constraints = [self.type.array_dtype(t) == dtype
                                     for dtype in dtypes]
                constraints.append(z3.Or(*dtype_constraints))
            return z3.And(*constraints)

        if max_container_depth == 0:
            return z3.Or(
                self.is_none(ty),
                is_valid_array(ty),
                self.is_sparsemat(ty)
            )
        else:
            return z3.Or(
                self.is_none(ty),
                is_valid_array(ty),
                self.is_sparsemat(ty),
                z3.And(self.is_list(ty),
                       self.is_valid_param_type(self.type.list_etype(ty),
                                                max_container_depth=max_container_depth-1)),
                z3.And(self.is_dict(ty),
                       self.is_int(self.type.dict_ktype(ty)),
                       self.is_valid_param_type(self.type.dict_vtype(ty),
                                                max_container_depth=max_container_depth-1)),
                z3.And(self.is_heap(ty),
                       self.is_scalar(self.type.heap_ktype(ty)),
                       self.is_valid_param_type(self.type.heap_vtype(ty),
                                                max_container_depth=max_container_depth-1)),
                z3.And(self.is_tuple(ty),
                       self.tuple_all(ty, partial(self.is_valid_param_type,
                                                  max_container_depth=max_container_depth-1)))
            )

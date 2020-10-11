#  libfive-python: libfive bindings for Python
#
#  Copyright (C) 2020  Sem Mulder
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this file,
#  You can obtain one at http://mozilla.org/MPL/2.0/.
import numbers
from collections import abc

from cpython cimport buffer
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.stdint cimport int32_t, uint32_t, uintptr_t
from libcpp cimport bool as cpp_bool


cdef extern from "<libfive.h>" nogil:
    ctypedef const void*_opaque_ptr

    struct libfive_interval:
        float lower
        float upper

    struct libfive_region2:
        libfive_interval X
        libfive_interval Y

    struct libfive_region3:
        libfive_interval X
        libfive_interval Y
        libfive_interval Z

    struct libfive_vec2:
        float x
        float y

    struct libfive_vec3:
        float x
        float y
        float z

    struct libfive_vec4:
        float x
        float y
        float z
        float w

    struct libfive_tri:
        uint32_t a
        uint32_t b
        uint32_t c

    struct libfive_contour:
        libfive_vec2*pts
        uint32_t count

    struct libfive_contours:
        libfive_contour*cs
        uint32_t count

    struct libfive_contour3:
        libfive_vec3*pts
        uint32_t count

    struct libfive_contours3:
        libfive_contour3*cs
        uint32_t count

    struct libfive_mesh:
        libfive_vec3*verts
        libfive_tri*tris
        uint32_t tri_count
        uint32_t vert_count

    struct libfive_mesh_coords:
        libfive_vec3*verts
        uint32_t vert_count
        int32_t*coord_indices
        uint32_t coord_index_count

    struct libfive_pixels:
        cpp_bool*pixels
        uint32_t width
        uint32_t height

    void libfive_contours_delete(libfive_contours*cs)
    void libfive_contours3_delete(libfive_contours3*cs)
    void libfive_mesh_delete(libfive_mesh*m)
    void libfive_mesh_coords_delete(libfive_mesh_coords*m)
    void libfive_pixels_delete(libfive_pixels*ps)
    int libfive_opcode_enum(const char*op)
    int libfive_opcode_args(int op)

    struct libfive_vars:
        _opaque_ptr*vars
        float*values
        uint32_t size
    void libfive_vars_delete(libfive_vars*j)

    struct libfive_tree_
    ctypedef libfive_tree_*libfive_tree

    struct libfive_id_
    ctypedef libfive_id_*libfive_id

    struct libfive_archive_
    ctypedef libfive_archive_*libfive_archive

    struct libfive_evaluator_
    ctypedef libfive_evaluator_*libfive_evaluator

    libfive_tree libfive_tree_x()
    libfive_tree libfive_tree_y()
    libfive_tree libfive_tree_z()
    libfive_tree libfive_tree_var()
    cpp_bool libfive_tree_is_var(libfive_tree t)
    libfive_tree libfive_tree_const(float f)
    float libfive_tree_get_const(libfive_tree t, cpp_bool*success)
    libfive_tree libfive_tree_constant_vars(libfive_tree t)
    libfive_tree libfive_tree_nonary(int op)
    libfive_tree libfive_tree_unary(int op, libfive_tree a)
    libfive_tree libfive_tree_binary(int op, libfive_tree a, libfive_tree b)
    _opaque_ptr libfive_tree_id(libfive_tree t)
    float libfive_tree_eval_f(libfive_tree t, libfive_vec3 p)
    libfive_interval libfive_tree_eval_r(libfive_tree t, libfive_region3 r)
    libfive_vec3 libfive_tree_eval_d(libfive_tree t, libfive_vec3 p)
    cpp_bool libfive_tree_eq(libfive_tree a, libfive_tree b)
    void libfive_tree_delete(libfive_tree ptr)
    libfive_tree libfive_tree_remap(libfive_tree p, libfive_tree x, libfive_tree y, libfive_tree z)
    char*libfive_tree_print(libfive_tree t)
    libfive_contours*libfive_tree_render_slice(
            libfive_tree tree,
            libfive_region2 R,
            float z,
            float res
    )
    libfive_contours3*libfive_tree_render_slice3(
            libfive_tree tree,
            libfive_region2 R,
            float z,
            float res
    )
    libfive_mesh*libfive_tree_render_mesh(libfive_tree tree, libfive_region3 R, float res)
    libfive_mesh_coords*libfive_tree_render_mesh_coords(
            libfive_tree tree,
            libfive_region3 R,
            float res
    )
    cpp_bool libfive_tree_save_mesh(
            libfive_tree tree,
            libfive_region3 R,
            float res, const char*f
    )
    cpp_bool libfive_evaluator_save_mesh(
            libfive_evaluator evaluator,
            libfive_region3 R,
            const char *f
    )
    libfive_pixels*libfive_tree_render_pixels(
            libfive_tree tree,
            libfive_region2 R,
            float z, float res
    )

    libfive_evaluator libfive_tree_evaluator(libfive_tree tree, libfive_vars vars)
    cpp_bool libfive_evaluator_update_vars(libfive_evaluator eval_tree, libfive_vars vars)
    void libfive_evaluator_delete(libfive_evaluator ptr)

    const char*libfive_git_version()
    const char*libfive_git_revision()
    const char*libfive_git_branch()


cdef bytes _uint32_t_format_character():
    if sizeof(unsigned char) == 4:
        return b'B'
    elif sizeof(unsigned short) == 4:
        return b'H'
    elif sizeof(unsigned int) == 4:
        return b'I'
    elif sizeof(unsigned long) == 4:
        return b'L'
    elif sizeof(unsigned long long) == 4:
        return b'Q'
    else:
        raise Exception("no 32-bit wide unsigned integer found on your platform")

cdef bytes uint32_t_format_character = _uint32_t_format_character()

cdef class Interval:
    cdef readonly float lower
    cdef readonly float upper

    def __init__(Interval self, float lower, float upper):
        self.lower = lower
        self.upper = upper

    @staticmethod
    cdef Interval from_libfive_interval(libfive_interval interval):
        cdef Interval new_interval = Interval.__new__(Interval)
        new_interval.lower = interval.lower
        new_interval.upper = interval.upper
        return new_interval

    cdef libfive_interval to_libfive_interval(Interval self):
        cdef libfive_interval new_interval
        new_interval.lower = self.lower
        new_interval.upper = self.upper
        return new_interval

    def __repr__(self):
        return f'{self.__class__.__qualname__}({repr(self.lower)}, {repr(self.upper)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}({str(self.lower)}, {str(self.upper)})'

cdef class Region2D:
    cdef readonly Interval x
    cdef readonly Interval y

    def __init__(Region2D self, Interval x, Interval y):
        self.x = x
        self.y = y

    @staticmethod
    cdef Region2D _from_libfive_region2(libfive_region2 region):
        cdef Region2D new_region = Region2D.__new__(Region2D)
        new_region.x = Interval.from_libfive_interval(region.X)
        new_region.y = Interval.from_libfive_interval(region.Y)
        return new_region

    cdef libfive_region2 to_libfive_region2(Region2D self):
        cdef libfive_region2 new_region
        new_region.X = self.x.to_libfive_interval()
        new_region.Y = self.y.to_libfive_interval()
        return new_region

    def __repr__(self):
        return f'{self.__class__.__qualname__}({repr(self.x)}, {repr(self.y)})'

    def __str__(self):
        prefix = 4 * ' '
        return f'{self.__class__.__qualname__}(\n{prefix + str(self.x)},\n{prefix + str(self.y)},\n)'

cdef class Region3D:
    cdef readonly Interval x
    cdef readonly Interval y
    cdef readonly Interval z

    def __init__(Region3D self, Interval x, Interval y, Interval z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    cdef Region3D _from_libfive_region3(libfive_region3 region):
        cdef Region3D new_region = Region3D.__new__(Region3D)
        new_region.x = Interval.from_libfive_interval(region.X)
        new_region.y = Interval.from_libfive_interval(region.Y)
        new_region.z = Interval.from_libfive_interval(region.Z)
        return new_region

    cdef libfive_region3 to_libfive_region3(Region3D self):
        cdef libfive_region3 new_region
        new_region.X = self.x.to_libfive_interval()
        new_region.Y = self.y.to_libfive_interval()
        new_region.Z = self.z.to_libfive_interval()
        return new_region

    def __repr__(self):
        return f'{self.__class__.__qualname__}({repr(self.x)}, {repr(self.y)}, {repr(self.z)})'

    def __str__(self):
        prefix = 4 * ' '
        return f'{self.__class__.__qualname__}(\n{prefix + str(self.x)},\n{prefix + str(self.y)},{prefix + repr(self.z)},\n)'

cdef class Vector2D:
    cdef readonly float x
    cdef readonly float y

    def __init__(Vector2D self, float x, float y):
        self.x = x
        self.y = y

    @staticmethod
    cdef Vector2D from_libfive_vec2(libfive_vec2 vector):
        cdef Vector2D new_vector = Vector2D.__new__(Vector2D)
        new_vector.x = vector.x
        new_vector.y = vector.y
        return new_vector

    cdef libfive_vec2 to_libfive_vec2(Vector2D self):
        cdef libfive_vec2 new_vector
        new_vector.x = self.x
        new_vector.y = self.y
        return new_vector

    def __repr__(self):
        return f'{self.__class__.__qualname__}({repr(self.x)}, {repr(self.y)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}({str(self.x)}, {str(self.y)})'

cdef class Vector3D:
    cdef readonly float x
    cdef readonly float y
    cdef readonly float z

    def __init__(Vector3D self, float x, float y, float z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    cdef Vector3D from_libfive_vec3(libfive_vec3 vector):
        cdef Vector3D new_vector = Vector3D.__new__(Vector3D)
        new_vector.x = vector.x
        new_vector.y = vector.y
        new_vector.z = vector.z
        return new_vector

    cdef libfive_vec3 to_libfive_vec3(Vector3D self):
        cdef libfive_vec3 new_vector
        new_vector.x = self.x
        new_vector.y = self.y
        new_vector.z = self.z
        return new_vector

    def __repr__(self):
        return f'{self.__class__.__qualname__}({repr(self.x)}, {repr(self.y)}, {repr(self.z)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}({str(self.x)}, {str(self.y)}, {str(self.z)})'

cdef class Vector4D:
    cdef readonly float x
    cdef readonly float y
    cdef readonly float z
    cdef readonly float w

    def __init__(Vector4D self, float x, float y, float z, float w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @staticmethod
    cdef Vector4D from_libfive_vec4(libfive_vec4 vector):
        cdef Vector4D new_vector = Vector4D.__new__(Vector4D)
        new_vector.x = vector.x
        new_vector.y = vector.y
        new_vector.z = vector.z
        new_vector.w = vector.w
        return new_vector

    cdef libfive_vec4 to_libfive_vec4(Vector4D self):
        cdef libfive_vec4 new_vector
        new_vector.x = self.x
        new_vector.y = self.y
        new_vector.z = self.z
        new_vector.w = self.w
        return new_vector

    def __repr__(self):
        return f'{self.__class__.__qualname__}({repr(self.x)}, {repr(self.y)}, {repr(self.z)}, {repr(self.w)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}({str(self.x)}, {str(self.y)}, {str(self.z)}, {str(self.w)})'

cdef class Triangle:
    cdef readonly uint32_t a
    cdef readonly uint32_t b
    cdef readonly uint32_t c

    def __init__(Triangle self, uint32_t a, uint32_t b, uint32_t c):
        self.a = a
        self.b = b
        self.c = c

    @staticmethod
    cdef Triangle from_libfive_tri(libfive_tri triangle):
        cdef Triangle new_triangle = Triangle.__new__(Triangle)
        new_triangle.a = triangle.a
        new_triangle.b = triangle.b
        new_triangle.c = triangle.c
        return new_triangle

    cdef libfive_tri to_libfive_tri(Triangle self):
        cdef libfive_tri new_triangle
        new_triangle.a = self.a
        new_triangle.b = self.b
        new_triangle.c = self.c
        return new_triangle

    def __repr__(self):
        return f'{self.__class__.__qualname__}({repr(self.a)}, {repr(self.b)}, {repr(self.c)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}({str(self.a)}, {str(self.b)}, {str(self.c)})'

cdef class _Contour2D:
    cdef libfive_vec2*_points
    cdef size_t _count
    cdef object _points_owner  # Owned reference makes sure that _points lives as long as its owner.
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]
    cdef bytes _format

    def __init__(_Contour2D self, *args):
        cdef size_t count = len(args)
        self._count = count
        assert not self._points, "__init__ called more than once"
        self._points = <libfive_vec2*> PyMem_Malloc(count * sizeof(libfive_vec2))
        self._points_owner = None
        if not self._points:
            raise MemoryError()

        cdef Vector2D vector
        for i, vector in enumerate(args):
            self._points[i] = vector.to_libfive_vec2()

        self._shape[0] = count
        self._strides[0] = sizeof(libfive_vec2)
        self._format = b'@2f'

    @staticmethod
    cdef from_libfive_contour(libfive_contour contour, object owner):
        cdef _Contour2D new_contour = Contour2D.__new__(Contour2D)
        new_contour._points = contour.pts
        new_contour._count = contour.count
        new_contour._points_owner = owner
        new_contour._shape[0] = contour.count
        new_contour._strides[0] = sizeof(libfive_vec2)
        new_contour._format = b'@2f'
        return new_contour

    cdef libfive_contour to_libfive_contour(_Contour2D self):
        cdef libfive_contour new_contour
        new_contour.pts = self._points
        new_contour.count = self._count
        return new_contour

    def __dealloc__(_Contour2D self):
        if not self._points_owner:
            PyMem_Free(self._points)

    def __getbuffer__(_Contour2D self, Py_buffer *view, int flags):
        assert self._points != NULL, "_points is uninitialized"

        if (flags & buffer.PyBUF_RECORDS_RO) != buffer.PyBUF_RECORDS_RO:
            raise BufferError('writable buffer requested from immutable object')

        view.buf = self._points
        view.obj = self
        view.len = sizeof(libfive_vec2) * self._count
        view.readonly = 1
        view.itemsize = sizeof(libfive_vec2)
        view.format = self._format
        view.ndim = 1
        view.shape = self._shape
        view.strides = self._strides
        view.suboffsets = NULL
        view.internal = NULL

    def __getitem__(_Contour2D self, item):
        cdef size_t count = self._count
        if isinstance(item, int):
            if item < 0:
                item += count
            if 0 <= item < count:
                assert self._points != NULL, "_points is uninitialized"
                return Vector2D.from_libfive_vec2(self._points[item])
            else:
                raise IndexError()
        elif isinstance(item, slice):
            assert self._points != NULL, "_points is uninitialized"
            return [
                Vector2D.from_libfive_vec2(self._points[i])
                for i in range(*item.indices(count))
            ]
        else:
            raise TypeError()

    def __len__(_Contour2D self):
        return self._count


# Workaround because extension types cannot derive from Python classes.
class Contour2D(_Contour2D, abc.Sequence):
    __slots__ = ()

    def __repr__(self):
        return f'{self.__class__.__qualname__}({", ".join(repr(p) for p in self)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}(<{len(self)} points>)'


cdef class _Contours2D:
    cdef libfive_contours*_contours
    cdef void*_raw_data

    def __init__(_Contours2D self, *args):
        cdef size_t j

        assert not self._contours and not self._raw_data, "__init__ called more than once"

        cdef size_t count = len(args)
        cdef size_t count_vectors = sum(len(contour) for contour in args)
        cdef void*_raw_data = PyMem_Malloc(
            sizeof(libfive_contours)
            + count * sizeof(libfive_contour)
            + count_vectors * sizeof(libfive_vec2)
        )

        if not _raw_data:
            raise MemoryError()
        self._raw_data = _raw_data
        self._contours = <libfive_contours*> self._raw_data
        self._contours.cs = <libfive_contour*> ((<libfive_contours*> self._raw_data) + 1)

        self._contours.count = count
        cdef _Contour2D contour
        cdef libfive_vec2*vectors = <libfive_vec2*> (
                <libfive_contour*> ((<libfive_contours*> self._raw_data) + 1) + count)
        for i, contour in enumerate(args):
            self._contours.cs[i].count = contour._count
            self._contours.cs[i].pts = vectors

            for j in range(contour._count):
                self._contours.cs[i].pts[j].x = contour._points[j].x
                self._contours.cs[i].pts[j].y = contour._points[j].y

            vectors += contour._count

    @staticmethod
    cdef from_libfive_contours(libfive_contours*contours):
        cdef _Contours2D new_contours = Contours2D.__new__(Contours2D)
        new_contours._contours = contours
        new_contours._raw_data = NULL
        return new_contours

    cdef libfive_contours*to_libfive_contours(_Contours2D self):
        cdef libfive_contours new_contours
        return self._contours

    def __dealloc__(_Contours2D self):
        if self._raw_data:
            PyMem_Free(self._raw_data)
        elif self._contours != NULL:
            libfive_contours_delete(self._contours)

    def __getitem__(_Contours2D self, item):
        assert self._contours != NULL, "_contours is uninitialized"
        cdef size_t count = self._contours.count
        if isinstance(item, int):
            if item < 0:
                item += count
            if 0 <= item < count:
                assert self._contours.cs != NULL, "_contours.cs is uninitialized"
                return _Contour2D.from_libfive_contour(self._contours.cs[item], <object> self)
            else:
                raise IndexError()
        elif isinstance(item, slice):
            assert self._contours.cs != NULL, "_contours.cs is uninitialized"
            return [
                _Contour2D.from_libfive_contour(self._contours.cs[i], <object> self)
                for i in range(*item.indices(count))
            ]
        else:
            raise TypeError()

    def __len__(_Contours2D self):
        assert self._contours != NULL, "_contours is uninitialized"
        return self._contours.count


# Workaround because extension types cannot derive from Python classes.
class Contours2D(_Contours2D, abc.Sequence):
    __slots__ = ()

    def __repr__(self):
        return f'{self.__class__.__qualname__}({", ".join(repr(c) for c in self)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}(<{len(self)} contours>)'


cdef class _Contour3D:
    cdef libfive_vec3*_points
    cdef size_t _count
    cdef object _points_owner  # Owned reference makes sure that _points lives as long as its owner.
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]
    cdef bytes _format

    def __init__(_Contour3D self, *args):
        cdef size_t count = len(args)
        self._count = count
        assert not self._points, "__init__ called more than once"
        self._points = <libfive_vec3*> PyMem_Malloc(count * sizeof(libfive_vec3))
        self._points_owner = None
        if not self._points:
            raise MemoryError()

        cdef Vector3D vector
        for i, vector in enumerate(args):
            self._points[i] = vector.to_libfive_vec3()

        self._shape[0] = count
        self._strides[0] = sizeof(libfive_vec3)
        self._format = b'@3f'

    @staticmethod
    cdef from_libfive_contour3(libfive_contour3 contour, object owner):
        cdef _Contour3D new_contour = Contour3D.__new__(Contour3D)
        new_contour._points = contour.pts
        new_contour._count = contour.count
        new_contour._points_owner = owner
        new_contour._shape[0] = contour.count
        new_contour._strides[0] = sizeof(libfive_vec3)
        new_contour._format = b'@3f'
        return new_contour

    cdef libfive_contour3 to_libfive_contour3(_Contour3D self):
        cdef libfive_contour3 new_contour
        new_contour.pts = self._points
        new_contour.count = self._count
        return new_contour

    def __dealloc__(_Contour3D self):
        if not self._points_owner:
            PyMem_Free(self._points)

    def __getbuffer__(_Contour3D self, Py_buffer *view, int flags):
        assert self._points != NULL, "_points is uninitialized"

        if (flags & buffer.PyBUF_RECORDS_RO) != buffer.PyBUF_RECORDS_RO:
            raise BufferError('writable buffer requested from immutable object')

        view.buf = self._points
        view.obj = self
        view.len = sizeof(libfive_vec3) * self._count
        view.readonly = 1
        view.itemsize = sizeof(libfive_vec3)
        view.format = self._format
        view.ndim = 1
        view.shape = self._shape
        view.strides = self._strides
        view.suboffsets = NULL
        view.internal = NULL

    def __getitem__(_Contour3D self, item):
        cdef size_t count = self._count
        if isinstance(item, int):
            if item < 0:
                item += count
            if 0 <= item < count:
                assert self._points != NULL, "_points is uninitialized"
                return Vector3D.from_libfive_vec3(self._points[item])
            else:
                raise IndexError()
        elif isinstance(item, slice):
            assert self._points != NULL, "_points is uninitialized"
            return [
                Vector3D.from_libfive_vec3(self._points[i])
                for i in range(*item.indices(count))
            ]
        else:
            raise TypeError()

    def __len__(_Contour3D self):
        return self._count


# Workaround because extension types cannot derive from Python classes.
class Contour3D(_Contour3D, abc.Sequence):
    __slots__ = ()

    def __repr__(self):
        return f'{self.__class__.__qualname__}({", ".join(repr(p) for p in self)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}(<{len(self)} point>)'


cdef class _Contours3D:
    cdef libfive_contours3*_contours
    cdef void*_raw_data

    def __init__(_Contours3D self, *args):
        cdef size_t j

        assert not self._contours and not self._raw_data, "__init__ called more than once"

        cdef size_t count = len(args)
        cdef size_t count_vectors = sum(len(contour) for contour in args)
        cdef void*_raw_data = PyMem_Malloc(
            sizeof(libfive_contours3)
            + count * sizeof(libfive_contour3)
            + count_vectors * sizeof(libfive_vec3)
        )

        if not _raw_data:
            raise MemoryError()
        self._raw_data = _raw_data
        self._contours = <libfive_contours3*> self._raw_data
        self._contours.cs = <libfive_contour3*> ((<libfive_contours3*> self._raw_data) + 1)

        self._contours.count = count
        cdef _Contour3D contour
        cdef libfive_vec3*vectors = <libfive_vec3*> (
                <libfive_contour3*> ((<libfive_contours3*> self._raw_data) + 1) + count)
        for i, contour in enumerate(args):
            self._contours.cs[i].count = contour._count
            self._contours.cs[i].pts = vectors

            for j in range(contour._count):
                self._contours.cs[i].pts[j].x = contour._points[j].x
                self._contours.cs[i].pts[j].y = contour._points[j].y

            vectors += contour._count

    @staticmethod
    cdef from_libfive_contours3(libfive_contours3*contours):
        cdef _Contours3D new_contours = Contours3D.__new__(Contours3D)
        new_contours._contours = contours
        new_contours._raw_data = NULL
        return new_contours

    cdef libfive_contours3*to_libfive_contours3(_Contours3D self):
        cdef libfive_contours3 new_contours
        return self._contours

    def __dealloc__(_Contours3D self):
        if self._raw_data:
            PyMem_Free(self._raw_data)
        elif self._contours != NULL:
            libfive_contours3_delete(self._contours)

    def __getitem__(_Contours3D self, item):
        assert self._contours != NULL, "_contours is uninitialized"
        cdef size_t count = self._contours.count
        if isinstance(item, int):
            if item < 0:
                item += count
            if 0 <= item < count:
                assert self._contours.cs != NULL, "_contours.cs is uninitialized"
                return _Contour3D.from_libfive_contour3(self._contours.cs[item], <object> self)
            else:
                raise IndexError()
        elif isinstance(item, slice):
            assert self._contours.cs != NULL, "_contours.cs is uninitialized"
            return [
                _Contour3D.from_libfive_contour3(self._contours.cs[i], <object> self)
                for i in range(*item.indices(count))
            ]
        else:
            raise TypeError()

    def __len__(_Contours3D self):
        assert self._contours != NULL, "_contours is uninitialized"
        return self._contours.count


# Workaround because extension types cannot derive from Python classes.
class Contours3D(_Contours3D, abc.Sequence):
    __slots__ = ()

    def __repr__(self):
        return f'{self.__class__.__qualname__}({", ".join(repr(c) for c in self)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}(<{len(self)} contours>)'


cdef class _Vertices:
    cdef libfive_vec3*_vertices
    cdef size_t _count
    cdef object _vertices_owner
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]
    cdef bytes _format

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    cdef from_libfive_vec3_array(libfive_vec3*vertices, size_t count, object owner):
        cdef _Vertices new_vertices = Vertices.__new__(Vertices)
        new_vertices._vertices = vertices
        new_vertices._count = count
        new_vertices._vertices_owner = owner
        new_vertices._shape[0] = count
        new_vertices._strides[0] = sizeof(libfive_vec3)
        new_vertices._format = b'@3f'
        return new_vertices

    def __getbuffer__(_Vertices self, Py_buffer *view, int flags):
        assert self._vertices != NULL, "_vertices is uninitialized"

        if (flags & buffer.PyBUF_RECORDS_RO) != buffer.PyBUF_RECORDS_RO:
            raise BufferError('writable buffer requested from immutable object')

        view.buf = self._vertices
        view.obj = self
        view.len = sizeof(libfive_vec3) * self._count
        view.readonly = 1
        view.itemsize = sizeof(libfive_vec3)
        view.format = self._format
        view.ndim = 1
        view.shape = self._shape
        view.strides = self._strides
        view.suboffsets = NULL
        view.internal = NULL

    def __getitem__(_Vertices self, item):
        assert self._vertices != NULL, "_vertices is uninitialized"
        cdef size_t count = self._count
        if isinstance(item, int):
            if item < 0:
                item += count
            if 0 <= item < count:
                return Vector3D.from_libfive_vec3(self._vertices[item])
            else:
                raise IndexError()
        elif isinstance(item, slice):
            return [
                Vector3D.from_libfive_vec3(self._vertices[i])
                for i in range(*item.indices(count))
            ]
        else:
            raise TypeError()

    def __len__(_Vertices self):
        return self._count


class Vertices(_Vertices, abc.Sequence):
    def __repr__(self):
        return f'{self.__class__.__qualname__}({", ".join(repr(v) for v in self)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}(<{len(self)} vertices>)'


cdef class _Triangles:
    cdef libfive_tri*_triangles
    cdef size_t _count
    cdef object _triangles_owner
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]
    cdef bytes _format

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    cdef from_libfive_tri_array(libfive_tri*triangles, size_t count, object owner):
        cdef _Triangles new_triangles = Triangles.__new__(Triangles)
        new_triangles._triangles = triangles
        new_triangles._count = count
        new_triangles._triangles_owner = owner
        new_triangles._shape[0] = count
        new_triangles._strides[0] = sizeof(libfive_tri)
        new_triangles._format = b'@3' + uint32_t_format_character
        return new_triangles

    def __getbuffer__(_Triangles self, Py_buffer *view, int flags):
        assert self._triangles != NULL, "_triangles is uninitialized"

        if (flags & buffer.PyBUF_RECORDS_RO) != buffer.PyBUF_RECORDS_RO:
            raise BufferError('writable buffer requested from immutable object')

        view.buf = self._triangles
        view.obj = self
        view.len = sizeof(libfive_tri) * self._count
        view.readonly = 1
        view.itemsize = sizeof(libfive_tri)
        view.format = self._format
        view.ndim = 1
        view.shape = self._shape
        view.strides = self._strides
        view.suboffsets = NULL
        view.internal = NULL

    def __getitem__(_Triangles self, item):
        assert self._triangles != NULL, "_triangles is uninitialized"
        cdef size_t count = self._count
        if isinstance(item, int):
            if item < 0:
                item += count
            if 0 <= item < count:
                return Triangle.from_libfive_tri(self._triangles[item])
            else:
                raise IndexError()
        elif isinstance(item, slice):
            return [
                Triangle.from_libfive_tri(self._triangles[i])
                for i in range(*item.indices(count))
            ]
        else:
            raise TypeError()

    def __len__(_Triangles self):
        return self._count


class Triangles(_Triangles, abc.Sequence):
    def __repr__(self):
        return f'{self.__class__.__qualname__}({", ".join(repr(t) for t in self)})'

    def __str__(self):
        return f'{self.__class__.__qualname__}(<{len(self)} triangles>)'


cdef class Mesh:
    cdef libfive_mesh*_mesh
    cdef void*_raw_data

    def __init__(Mesh self, vertices, triangles):
        assert not self._mesh and not self._raw_data, "__init__ called more than once"

        if not isinstance(vertices, abc.Iterable) or not isinstance(vertices, abc.Sized) \
                or not isinstance(triangles, abc.Iterable) or not isinstance(triangles, abc.Sized):
            raise TypeError()

        cdef size_t _vertex_count = len(vertices)
        cdef size_t _triangle_count = len(triangles)

        cdef void*_raw_data = PyMem_Malloc(
            sizeof(libfive_mesh)
            + _vertex_count * sizeof(libfive_vec3)
            + _triangle_count * sizeof(libfive_tri)
        )
        if not _raw_data:
            raise MemoryError()
        self._raw_data = _raw_data

        self._mesh = <libfive_mesh*> self._raw_data
        self._mesh.verts = <libfive_vec3*> ((<libfive_mesh*> _raw_data) + 1)
        self._mesh.tris = <libfive_tri*> (
                (<libfive_vec3*> ((<libfive_mesh*> _raw_data) + 1)) + _vertex_count)

        self._mesh.vert_count = _vertex_count
        self._mesh.tri_count = _triangle_count

        cdef Vector3D vertex
        for i, vertex in enumerate(vertices):
            self._mesh.verts[i] = vertex.to_libfive_vec3()

        cdef Triangle triangle
        for i, triangle in enumerate(triangles):
            self._mesh.tris[i] = triangle.to_libfive_tri()

    @staticmethod
    cdef from_libfive_mesh(libfive_mesh*mesh):
        cdef Mesh new_mesh = Mesh.__new__(Mesh)
        new_mesh._mesh = mesh
        new_mesh._raw_data = NULL
        return new_mesh

    @property
    def triangles(Mesh self):
        assert self._mesh != NULL, "_mesh is uninitialized"
        assert self._mesh.tris != NULL, "_mesh.tris is uninitialized"
        return _Triangles.from_libfive_tri_array(self._mesh.tris, self._mesh.tri_count, self)

    @property
    def vertices(Mesh self):
        assert self._mesh != NULL, "_mesh is uninitialized"
        assert self._mesh.tris != NULL, "_mesh.verts is uninitialized"
        return _Vertices.from_libfive_vec3_array(self._mesh.verts, self._mesh.vert_count, self)

    def __dealloc__(Mesh self):
        if self._raw_data:
            PyMem_Free(self._raw_data)
        elif self._mesh != NULL:
            libfive_mesh_delete(self._mesh)

    def __repr__(self):
        assert self._mesh != NULL, "_mesh is uninitialized"
        return f'{self.__class__.__qualname__}([{",".join(repr(v) for v in self.vertices)}], [{",".join(repr(t) for t in self.triangles)}])'

    def __str__(self):
        assert self._mesh != NULL, "_mesh is uninitialized"
        return f'{self.__class__.__qualname__}(<{self._mesh.vert_count} vertices>, <{self._mesh.tri_count} triangles>)'

cdef class Tree:
    cdef libfive_tree _tree

    def __eq__(Tree self, other):
        cdef Tree tree_other
        if isinstance(other, Tree):
            tree_other = other
            return bool(libfive_tree_eq(self._tree, tree_other._tree))
        else:
            return False

    def __hash__(Tree self):
        return <uintptr_t> libfive_tree_id(self._tree)

    def evaluate(Tree self, Vector3D value):
        assert self._tree != NULL, "_tree is uninitialized"
        return libfive_tree_eval_f(self._tree, value.to_libfive_vec3())

    def evaluate_gradient(Tree self, Vector3D value):
        assert self._tree != NULL, "_tree is uninitialized"
        return Vector3D.from_libfive_vec3(libfive_tree_eval_d(self._tree, value.to_libfive_vec3()))

    def __dealloc__(Tree self):
        if self._tree != NULL:
            libfive_tree_delete(self._tree)

    def __neg__(Tree self):
        return Neg(self)

    def __abs__(Tree self):
        return Abs(self)

    def __add__(a, b):
        if isinstance(a, Tree):
            left = a
        elif isinstance(a, numbers.Real):
            left = Const(float(a))
        else:
            return NotImplemented

        if isinstance(b, Tree):
            right = b
        elif isinstance(b, numbers.Real):
            right = Const(float(b))
        else:
            return NotImplemented

        return Add(left, right)

    def __sub__(a, b):
        if isinstance(a, Tree):
            left = a
        elif isinstance(a, numbers.Real):
            left = Const(float(a))
        else:
            return NotImplemented

        if isinstance(b, Tree):
            right = b
        elif isinstance(b, numbers.Real):
            right = Const(float(b))
        else:
            return NotImplemented

        return Sub(left, right)

    def __mul__(a, b):
        if isinstance(a, Tree):
            left = a
        elif isinstance(a, numbers.Real):
            left = Const(float(a))
        else:
            return NotImplemented

        if isinstance(b, Tree):
            right = b
        elif isinstance(b, numbers.Real):
            right = Const(float(b))
        else:
            return NotImplemented

        return Mul(left, right)

    def __truediv__(a, b):
        if isinstance(a, Tree):
            left = a
        elif isinstance(a, numbers.Real):
            left = Const(float(a))
        else:
            return NotImplemented

        if isinstance(b, Tree):
            right = b
        elif isinstance(b, numbers.Real):
            right = Const(float(b))
        else:
            return NotImplemented

        return Div(left, right)

    def __mod__(a, b):
        if isinstance(a, Tree):
            left = a
        elif isinstance(a, numbers.Real):
            left = Const(float(a))
        else:
            return NotImplemented

        if isinstance(b, Tree):
            right = b
        elif isinstance(b, numbers.Real):
            right = Const(float(b))
        else:
            return NotImplemented

        return Mod(left, right)

    def __pow__(a, b, modulo):
        assert modulo is None, 'modulo argument to __pow__ is not supported'

        if isinstance(a, Tree):
            left = a
        elif isinstance(a, numbers.Real):
            left = Const(float(a))
        else:
            return NotImplemented

        if isinstance(b, Tree):
            right = b
        elif isinstance(b, numbers.Real):
            right = Const(float(b))
        else:
            return NotImplemented

        return Pow(left, right)

cdef class NonaryOpTree(Tree):
    def __repr__(NonaryOpTree self):
        return f"{self.__class__.__qualname__}()"

    def __str__(NonaryOpTree self):
        return f"{self.__class__.__qualname__}()"

cdef class X(NonaryOpTree):
    def __init__(X self):
        assert not self._tree, "__init__ called more than once"
        self._tree = libfive_tree_x()

cdef class Y(NonaryOpTree):
    def __init__(Y self):
        assert not self._tree, "__init__ called more than once"
        self._tree = libfive_tree_y()

cdef class Z(NonaryOpTree):
    def __init__(Z self):
        assert not self._tree, "__init__ called more than once"
        self._tree = libfive_tree_z()

cdef class Const(Tree):
    def __init__(Const self, value: float):
        assert not self._tree, "__init__ called more than once"
        self._tree = libfive_tree_const(value)

    @property
    def value(Const self):
        assert self._tree != NULL, "_tree is uninitialized"
        cdef cpp_bool success = False
        cdef float value = libfive_tree_get_const(self._tree, &success)
        assert success, 'failed to get constant value'
        return value

    def __repr__(self):
        return f"{self.__class__.__qualname__}({repr(self.value)})"

    def __str__(self):
        return f"{self.__class__.__qualname__}({str(self.value)})"

cdef class UnaryOpTree(Tree):
    cdef readonly Tree value

    def __repr__(self):
        return f"{self.__class__.__qualname__}({repr(self.value)})"

    def __str__(self):
        prefix = 4 * ' '
        prefixed_value = str(self.value).replace('\n', '\n' + prefix)
        return f"{self.__class__.__qualname__}(\n{prefix + prefixed_value}\n)"

cdef int _square_op = libfive_opcode_enum(b'square')
cdef class Square(UnaryOpTree):
    def __init__(Square self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_square_op, value._tree)

cdef int _sqrt_op = libfive_opcode_enum(b'sqrt')
cdef class Sqrt(UnaryOpTree):
    def __init__(Sqrt self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_sqrt_op, value._tree)

cdef int _neg_op = libfive_opcode_enum(b'neg')
cdef class Neg(UnaryOpTree):
    def __init__(Neg self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_neg_op, value._tree)

cdef int _sin_op = libfive_opcode_enum(b'sin')
cdef class Sin(UnaryOpTree):
    def __init__(Sin self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_sin_op, value._tree)

cdef int _cos_op = libfive_opcode_enum(b'cos')
cdef class Cos(UnaryOpTree):
    def __init__(Cos self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_cos_op, value._tree)

cdef int _tan_op = libfive_opcode_enum(b'tan')
cdef class Tan(UnaryOpTree):
    def __init__(Tan self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_tan_op, value._tree)

cdef int _asin_op = libfive_opcode_enum(b'asin')
cdef class Asin(UnaryOpTree):
    def __init__(Asin self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_asin_op, value._tree)

cdef int _acos_op = libfive_opcode_enum(b'acos')
cdef class Acos(UnaryOpTree):
    def __init__(Acos self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_acos_op, value._tree)

cdef int _atan_op = libfive_opcode_enum(b'atan')
cdef class Atan(UnaryOpTree):
    def __init__(Atan self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_atan_op, value._tree)

cdef int _exp_op = libfive_opcode_enum(b'exp')
cdef class Exp(UnaryOpTree):
    def __init__(Exp self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_exp_op, value._tree)

cdef int _abs_op = libfive_opcode_enum(b'abs')
cdef class Abs(UnaryOpTree):
    def __init__(Abs self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_abs_op, value._tree)

cdef int _log_op = libfive_opcode_enum(b'log')
cdef class Log(UnaryOpTree):
    def __init__(Log self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_log_op, value._tree)

cdef int _recip_op = libfive_opcode_enum(b'recip')
cdef class Recip(UnaryOpTree):
    def __init__(Recip self, Tree value):
        assert not self._tree, "__init__ called more than once"
        self.value = value
        self._tree = libfive_tree_unary(_recip_op, value._tree)

cdef class BinaryOpTree(Tree):
    cdef readonly Tree left
    cdef readonly Tree right

    def __repr__(self):
        return f"{self.__class__.__qualname__}({repr(self.left)}, {repr(self.right)})"

    def __str__(self):
        prefix = 4 * ' '
        prefixed_left = str(self.left).replace('\n', '\n' + prefix)
        prefixed_right = str(self.right).replace('\n', '\n' + prefix)
        prefixed_right = str(self.right).replace('\n', '\n' + prefix)
        return f"{self.__class__.__qualname__}(\n{prefix + prefixed_left},\n{prefix + prefixed_right},\n)"

cdef int _add_op = libfive_opcode_enum(b'add')
cdef class Add(BinaryOpTree):
    def __init__(Add self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_add_op, left._tree, right._tree)

cdef int _mul_op = libfive_opcode_enum(b'mul')
cdef class Mul(BinaryOpTree):
    def __init__(Mul self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_mul_op, left._tree, right._tree)

cdef int _min_op = libfive_opcode_enum(b'min')
cdef class Min(BinaryOpTree):
    def __init__(Min self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_min_op, left._tree, right._tree)

cdef int _max_op = libfive_opcode_enum(b'max')
cdef class Max(BinaryOpTree):
    def __init__(Max self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_max_op, left._tree, right._tree)

cdef int _sub_op = libfive_opcode_enum(b'sub')
cdef class Sub(BinaryOpTree):
    def __init__(Sub self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_sub_op, left._tree, right._tree)

cdef int _div_op = libfive_opcode_enum(b'div')
cdef class Div(BinaryOpTree):
    def __init__(Div self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_div_op, left._tree, right._tree)

cdef int _atan2_op = libfive_opcode_enum(b'atan2')
cdef class Atan2(BinaryOpTree):
    def __init__(Atan2 self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_atan2_op, left._tree, right._tree)

cdef int _pow_op = libfive_opcode_enum(b'pow')
cdef class Pow(BinaryOpTree):
    def __init__(Pow self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_pow_op, left._tree, right._tree)

cdef int _nth_root_op = libfive_opcode_enum(b'nth-root')
cdef class NthRoot(BinaryOpTree):
    def __init__(NthRoot self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_nth_root_op, left._tree, right._tree)

cdef int _mod_op = libfive_opcode_enum(b'mod')
cdef class Mod(BinaryOpTree):
    def __init__(Mod self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_mod_op, left._tree, right._tree)

cdef int _nanfill_op = libfive_opcode_enum(b'nanfill')
cdef class NanFill(BinaryOpTree):
    def __init__(NanFill self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_nanfill_op, left._tree, right._tree)

cdef int _compare_op = libfive_opcode_enum(b'compare')
cdef class Compare(BinaryOpTree):
    def __init__(Compare self, Tree left, Tree right):
        assert not self._tree, "__init__ called more than once"
        self.left = left
        self.right = right
        self._tree = libfive_tree_binary(_compare_op, left._tree, right._tree)

def render_mesh(Tree tree, Region3D region, float resolution):
    cdef libfive_region3 libfive_region = region.to_libfive_region3()
    cdef libfive_mesh* mesh
    with nogil:
        mesh = libfive_tree_render_mesh(
            tree._tree,
            libfive_region,
            resolution,
        )
    return Mesh.from_libfive_mesh(mesh)

def render_slice(Tree tree, Region2D region, float z, float resolution):
    cdef libfive_region2 libfive_region = region.to_libfive_region2()
    cdef libfive_contours* contours
    with nogil:
        contours = libfive_tree_render_slice(
            tree._tree,
            libfive_region,
            z,
            resolution,
        )
    return _Contours2D.from_libfive_contours(contours)

def render_slice_3d(Tree tree, Region2D region, float z, float resolution):
    cdef libfive_region2 libfive_region = region.to_libfive_region2()
    cdef libfive_contours3* contours
    with nogil:
        contours = libfive_tree_render_slice3(
            tree._tree,
            libfive_region,
            z,
            resolution,
        )
    return _Contours3D.from_libfive_contours3(contours)

def version_info():
    cdef bytes git_version = libfive_git_version()
    cdef bytes git_revision = libfive_git_revision()
    cdef bytes git_branch = libfive_git_branch()
    return git_version, git_revision, git_branch

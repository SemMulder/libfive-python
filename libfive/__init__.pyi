#  libfive-python: libfive bindings for Python
#
#  Copyright (C) 2020  Sem Mulder
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this file,
#  You can obtain one at http://mozilla.org/MPL/2.0/.
import numbers
from typing import (
    Sequence,
    Iterable,
    TypeVar,
    Sized,
    Protocol,
    Union,
    overload,
)


class Interval:
    lower: float
    upper: float

    def __init__(self, lower: float, upper: float) -> None: ...


class Region2D:
    x: Interval
    y: Interval

    def __init__(self, x: Interval, y: Interval) -> None: ...


class Region3D:
    x: Interval
    y: Interval
    z: Interval

    def __init__(self, x: Interval, y: Interval, z: Interval) -> None: ...


class Vector2D:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None: ...


class Vector3D:
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float) -> None: ...


class Vector4D:
    x: float
    y: float
    z: float
    w: float

    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...


class Triangle:
    a: int
    b: int
    c: int

    def __init__(self, a: int, b: int, c: int) -> None: ...


class Contour2D(Sequence[Vector2D]):
    def __init__(self, *args: Vector2D) -> None: ...

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, item: int) -> Vector2D: ...

    @overload
    def __getitem__(self, item: slice) -> list[Vector2D]: ...


class Contours2D(Sequence[Contour2D]):
    def __init__(self, *args: Contour2D) -> None: ...

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, item: int) -> Contour2D: ...

    @overload
    def __getitem__(self, item: slice) -> list[Contour2D]: ...


class Contour3D(Sequence[Vector3D]):
    def __init__(self, *args: Vector3D) -> None: ...

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, item: int) -> Vector3D: ...

    @overload
    def __getitem__(self, item: slice) -> list[Vector3D]: ...


class Contours3D(Sequence[Contour3D]):
    def __init__(self, *args: Contour3D) -> None: ...

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, item: int) -> Contour3D: ...

    @overload
    def __getitem__(self, item: slice) -> list[Contour3D]: ...


class Vertices(Sequence[Vector3D]):
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, item: int) -> Vector3D: ...

    @overload
    def __getitem__(self, item: slice) -> list[Vector3D]: ...


class Triangles(Sequence[int]):
    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, item: int) -> int: ...

    @overload
    def __getitem__(self, item: slice) -> list[int]: ...


T = TypeVar("T", covariant=True)


class SizedAndIterable(Iterable[T], Sized, Protocol[T]): ...


class Mesh:
    def __init__(
            self, vertices: SizedAndIterable[Vector3D], triangles: SizedAndIterable[int]
    ) -> None: ...

    @property
    def triangles(self) -> Triangles: ...

    @property
    def vertices(self) -> Vertices: ...


TreeOrReal = Union["Tree", numbers.Real]


class Tree:
    def __eq__(self, other: object) -> bool: ...

    def __hash__(self) -> int: ...

    def evaluate(self, value: Vector3D) -> float: ...

    def evaluate_gradient(self, value: Vector3D) -> Vector3D: ...

    def __neg__(self) -> Neg: ...

    def __abs__(self) -> Abs: ...

    def __add__(self, other: TreeOrReal, /) -> Add: ...

    def __sub__(self, other: TreeOrReal, /) -> Sub: ...

    def __mul__(self, other: TreeOrReal, /) -> Mul: ...

    def __truediv__(self, other: TreeOrReal, /) -> Div: ...

    def __mod__(self, other: TreeOrReal, /) -> Mod: ...

    def __pow__(self, power: TreeOrReal, modulo: None = None, /) -> Pow: ...


class NonaryOpTree(Tree):
    def __init__(self) -> None: ...


class X(NonaryOpTree): ...


class Y(NonaryOpTree): ...


class Z(NonaryOpTree): ...


class Const(Tree):
    def __init__(self, value: float) -> None: ...

    @property
    def value(self) -> float: ...


class UnaryOpTree(Tree):
    def __init__(self, value: Tree) -> None: ...

    @property
    def value(self) -> Tree: ...


class Square(UnaryOpTree): ...


class Sqrt(UnaryOpTree): ...


class Neg(UnaryOpTree): ...


class Sin(UnaryOpTree): ...


class Cos(UnaryOpTree): ...


class Tan(UnaryOpTree): ...


class Asin(UnaryOpTree): ...


class Acos(UnaryOpTree): ...


class Atan(UnaryOpTree): ...


class Exp(UnaryOpTree): ...


class Abs(UnaryOpTree): ...


class Log(UnaryOpTree): ...


class Recip(UnaryOpTree): ...


class BinaryOpTree(Tree):
    def __init__(self, left: Tree, right: Tree) -> None: ...

    @property
    def left(self) -> Tree: ...

    @property
    def right(self) -> Tree: ...


class Add(BinaryOpTree): ...


class Mul(BinaryOpTree): ...


class Min(BinaryOpTree): ...


class Max(BinaryOpTree): ...


class Sub(BinaryOpTree): ...


class Div(BinaryOpTree): ...


class Atan2(BinaryOpTree): ...


class Pow(BinaryOpTree): ...


class NthRoot(BinaryOpTree): ...


class Mod(BinaryOpTree): ...


class NanFill(BinaryOpTree): ...


class Compare(BinaryOpTree): ...


def render_mesh(tree: Tree, region: Region3D, resolution: float) -> Mesh: ...


def render_slice(tree: Tree, region: Region2D, resolution: float) -> Region2D: ...


def render_slice_3d(tree: Tree, region: Region3D, resolution: float) -> Region3D: ...


def version_info() -> tuple[bytes, bytes, bytes]: ...

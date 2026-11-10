"""This module includes the geometry methods that openMotor uses in its calculations"""

import math
from typing import List, Union

import numpy as np
from numpy.typing import NDArray


def circleArea(dia: float) -> float:
    """Returns the area of a circle with diameter dia"""
    return ((dia / 2) ** 2) * math.pi


def circlePerimeter(dia: float) -> float:
    """Returns the perimeter (circumference) of a circle with diameter dia"""
    return dia * math.pi


def circleDiameterFromArea(area: float) -> float:
    """Returns the diameter of a circle with area 'area'"""
    return 2 * ((area / math.pi) ** 0.5)


def tubeArea(dia: float, height: int) -> float:
    """Returns the surface area of a tube (cylinder without endcaps) with diameter 'dia' and height 'height'"""
    return dia * math.pi * height


def cylinderArea(dia: float, height: int) -> float:
    """Returns the surface area of a cylinder with diameter 'dia' and height 'height'"""
    return (2 * circleArea(dia)) + (tubeArea(dia, height))


def cylinderVolume(dia: float, height: int) -> float:
    """Returns the volume of a cylinder with diameter 'dia' and height 'height'"""
    return height * circleArea(dia)


def frustumLateralSurfaceArea(diameterA: int, diameterB: int, length: int) -> float:
    """Returns the surface area of a frustum (truncated cone) with end diameters A and B and length 'length'"""
    radiusA = diameterA / 2
    radiusB = diameterB / 2
    return (
        math.pi * (radiusA + radiusB) * (abs(radiusA - radiusB) ** 2 + length**2) ** 0.5
    )


def frustumVolume(diameterA: int, diameterB: int, length: int) -> float:
    """Returns the volume of a frustum (truncated cone) with end diameters A and B and length 'length'"""
    radiusA = diameterA / 2
    radiusB = diameterB / 2
    return math.pi * (length / 3) * (radiusA**2 + radiusA * radiusB + radiusB**2)


def splitFrustum(
    diameterA: int, diameterB: int, length: int, splitPosition: int
) -> tuple[tuple[int, float, int], tuple[float, int, int]]:
    """Takes in info about a frustum (truncated cone) and a position measured from the "diameterA" and returns
    a tuple of frustums representing the two halves of the original frustum if it were split on the plane at
    distance "position" from the face with diameter "diameterA"
    """
    splitDiameter: float = diameterA + (diameterB - diameterA) * (
        splitPosition / length
    )
    return (diameterA, splitDiameter, splitPosition), (
        splitDiameter,
        diameterB,
        length - splitPosition,
    )


def length(
    contour: NDArray[Union[np.int_, np.float64]],
    mapSize: Union[int, float],
    tolerance: int = 3,
) -> Union[int, float]:
    """Returns the total length of all segments in a contour that aren't within 'tolerance' of the edge of a
    circle with diameter 'mapSize'"""
    offset = np.roll(contour.T, 1, axis=1)
    lengths = np.linalg.norm(contour.T - offset, axis=0)

    centerOffset = np.array([[mapSize / 2, mapSize / 2]])
    radius = np.linalg.norm(contour - centerOffset, axis=1)

    valid = radius < (mapSize / 2) - tolerance

    return np.sum(lengths[valid])


def clean(
    contour: NDArray[Union[np.int_, np.float64]],
    mapSize: Union[int, float],
    tolerance: int,
) -> NDArray[Union[np.int_, np.float64]]:
    """Returns a contour with the same points as the input, omitting any within 'tolerace' of a circle of
    diameter 'mapSize'"""
    offset = np.array([[mapSize / 2, mapSize / 2]])
    lengths = np.linalg.norm(contour - offset, axis=1)
    return contour[lengths < (mapSize / 2) - tolerance]


def cleanSegments(
    contour: NDArray[Union[np.int_, np.float64]],
    mapSize: Union[int, float],
    tolerance: int,
) -> List[NDArray[Union[np.int_, np.float64]]]:
    """Like clean, but returns a list of the separate pieces of the contour that remain. Where points near the edge of
    the circle were removed from the middle of a contour, the contour is split there instead of connecting the two
    remaining ends with a straight line."""
    offset = np.array([[mapSize / 2, mapSize / 2]])
    keep = np.linalg.norm(contour - offset, axis=1) < (mapSize / 2) - tolerance
    if not keep.any():
        return []

    # Indices where runs of kept points start and end
    padded = np.concatenate(([False], keep, [False])).astype(int)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    pieces = [contour[a:b] for a, b in zip(starts, ends)]

    # A closed contour that is cut in the middle of its first and last run is really one piece
    closed = np.allclose(contour[0], contour[-1])
    if closed and len(pieces) > 1 and keep[0] and keep[-1]:
        pieces[0] = np.concatenate((pieces[-1], pieces[0]))
        pieces.pop()
    return pieces


def dist(point1: tuple[int, int], point2: tuple[int, int]) -> int:
    """Returns the distance between two points [x1, y1], [x2, y2]"""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

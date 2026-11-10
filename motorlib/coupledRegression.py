"""
Coupled 3D regression of a slotted grain and the end burner in front of it.

In the independent-grain model an end burner burns its entire face from t=0 while the slotted grain behind it
(for example an X Core) only regresses in 2D. In reality the slot bottoms are the only surface of the end burner that
is lit at ignition, and the flame front then eats into the end burner as a rounded, three dimensional front. This
module treats the end burner and the slotted grain behind it as a single block of propellant. The initial burning
surface is the slot surface (plus the aft face of the slotted grain if it is not inhibited) and the burning surface at
a regression distance 'r' is the set of points at distance 'r' from that initial surface.

Two models are provided:

* Separable 3D: for straight slots with flat bottoms, the 3D distance is exactly separable into the existing 2D
  regression map of the slotted grain and the depth into the end burner. Volumes and areas are computed in closed form
  from a histogram of the 2D map, so the cost is independent of the resolution of the map.
* Voxel 3D: a 3D fast marching calculation on a voxel grid. It currently models the same geometry, so it is mainly a
  cross-check for the separable model and the base for geometries the separable model can not handle.

The results are stored in a CoupledRegressionTable, which two proxy grains use to stand in for the real grains
during a simulation. The real grains are never modified.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import skfmm
from scipy import ndimage
from scipy.signal import savgol_filter

from . import geometry
from .grain import FmmGrain, Grain
from .grains import EndBurningGrain
from .simResult import SimAlert, SimAlertLevel, SimAlertType

MODEL_INDEPENDENT = "Independent grains"
MODEL_SEPARABLE = "Separable 3D"
MODEL_VOXEL = "Voxel 3D"
MODELS = [MODEL_INDEPENDENT, MODEL_SEPARABLE, MODEL_VOXEL]

DEFAULT_VOXEL_SIZE = 2.5e-4
MAX_VOXELS = 5e7


class CoupledRegressionError(Exception):
    """Raised when a coupled regression table can not be generated, for example because it would be too large."""


# ----------------------------------------------------------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------------------------------------------------------


class CoupledRegressionTable:
    """
    Sampled volume and burning surface area of the end burner and the slotted grain as a function of the common
    regression distance. The end burner and the slotted grain each have their own regression axis (rEb, rSlot), because
    they burn out at different distances.
    """

    def __init__(
        self,
        rEb,
        volEb,
        areaEb,
        rSlot,
        volSlot,
        areaSlot,
        rMaxEb: float,
        rMaxSlot: float,
        model: str,
    ) -> None:
        self.rEb = np.asarray(rEb, dtype=float)
        self.V_eb = np.asarray(volEb, dtype=float)
        self.A_eb = np.asarray(areaEb, dtype=float)
        self.rSlot = np.asarray(rSlot, dtype=float)
        self.V_slot = np.asarray(volSlot, dtype=float)
        self.A_slot = np.asarray(areaSlot, dtype=float)
        self.rMaxEb = float(rMaxEb)
        self.rMaxSlot = float(rMaxSlot)
        self.model = model

    def volumeEb(self, r: float) -> float:
        """Volume of end burner propellant left at regression distance 'r'."""
        return float(np.interp(r, self.rEb, self.V_eb))

    def areaEb(self, r: float) -> float:
        """Burning surface area of the end burner at regression distance 'r'."""
        if r >= self.rMaxEb:
            return 0.0
        return float(np.interp(r, self.rEb, self.A_eb))

    def volumeSlot(self, r: float) -> float:
        """Volume of slotted grain propellant left at regression distance 'r'."""
        return float(np.interp(r, self.rSlot, self.V_slot))

    def areaSlot(self, r: float) -> float:
        """Burning surface area of the slotted grain at regression distance 'r'."""
        if r >= self.rMaxSlot:
            return 0.0
        return float(np.interp(r, self.rSlot, self.A_slot))


# ----------------------------------------------------------------------------------------------------------------------
# Pair detection and helpers
# ----------------------------------------------------------------------------------------------------------------------


def findCoupledPair(grains: List[Grain], config=None) -> Optional[Tuple[int, int]]:
    """
    Returns (0, 1) if the first grain is an end burner and the second one is a fast marching grain with the same
    diameter. Otherwise returns None. The slotted grain's top face is always treated as the interface to the end
    burner, whatever its 'inhibitedEnds' value is.
    """
    if len(grains) < 2:
        return None
    eb, slot = grains[0], grains[1]
    if not isinstance(eb, EndBurningGrain) or not isinstance(slot, FmmGrain):
        return None
    dEb = eb.props["diameter"].getValue()
    dSlot = slot.props["diameter"].getValue()
    if dEb <= 0 or not np.isclose(dEb, dSlot, rtol=1e-9, atol=0):
        return None
    if eb.props["length"].getValue() <= 0 or slot.props["length"].getValue() <= 0:
        return None
    return (0, 1)


def isAftExposed(slot: FmmGrain) -> bool:
    """The aft face of the slotted grain burns unless it is inhibited."""
    return slot.props["inhibitedEnds"].getValue() in ("Neither", "Top")


def _validDistances(slot: FmmGrain) -> np.ndarray:
    """Returns the 2D regression distances of the slotted grain in metres for every pixel that is propellant or core."""
    regMap = np.ma.getdata(slot.regressionMap)
    valid = np.logical_not(slot.mask)
    return np.maximum(slot.unNormalize(regMap[valid]), 0.0)


def _asin(x):
    return np.arcsin(np.clip(x, -1.0, 1.0))


# ----------------------------------------------------------------------------------------------------------------------
# Separable model
# ----------------------------------------------------------------------------------------------------------------------


def _separableEndBurner(
    r: np.ndarray, d2: np.ndarray, Le: float, Lx: float, aft: bool, circleArea: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed form volume and burning surface area of the end burner as a function of regression distance.

    Every column of end burner propellant above a pixel with 2D distance d2 has burned down to a depth of
    min(Le, max(sqrt(r^2 - d2^2), aft * (r - Lx))). Columns are binned by d2, assuming a uniform distribution of d2
    inside each bin, and the burned depth is integrated over each bin analytically. Pixels with d2 == 0 (the slot
    footprint) are handled as a point mass.
    """
    weight = circleArea / d2.size  # Area represented by one pixel
    n0 = np.count_nonzero(d2 <= 0)
    positive = d2[d2 > 0]

    if positive.size > 0:
        dMax = float(positive.max())
        numBins = int(np.clip(np.sqrt(positive.size) * 1.5, 50, 1500))
        counts, edges = np.histogram(positive, bins=numBins, range=(0.0, dMax * (1 + 1e-9)))
        lo = edges[:-1][None, :]
        hi = edges[1:][None, :]
        n = counts[None, :].astype(float)
        width = hi - lo
    else:
        lo = hi = n = np.zeros((1, 1))
        width = np.ones((1, 1))
        counts = np.zeros(1)

    volume = np.zeros(r.shape)
    area = np.zeros(r.shape)

    chunk = 256
    for start in range(0, r.size, chunk):
        rc = r[start : start + chunk]
        R = rc[:, None]
        Rs = np.maximum(R, 1e-12)
        gp = np.maximum(R - Lx, 0.0) if aft else np.zeros_like(R)
        allBurned = gp >= Le

        with np.errstate(invalid="ignore", divide="ignore"):
            dL = np.where(R > Le, np.sqrt(np.maximum(R**2 - Le**2, 0.0)), -1.0)
            dg = np.sqrt(np.maximum(R**2 - gp**2, 0.0))
        dL = np.where(allBurned, np.inf, dL)
        dg = np.where(allBurned, np.inf, dg)

        # Antiderivative of sqrt(R^2 - d^2)
        def F(d):
            d = np.clip(d, 0.0, Rs)
            return 0.5 * (d * np.sqrt(np.maximum(Rs**2 - d**2, 0.0)) + Rs**2 * _asin(d / Rs))

        seg1 = np.clip(np.minimum(hi, dL) - lo, 0.0, None) * Le
        a2 = np.maximum(lo, dL)
        b2 = np.minimum(hi, dg)
        active2 = b2 > a2
        seg2 = np.where(active2, F(b2) - F(a2), 0.0)
        len3 = np.clip(hi - np.maximum(lo, dg), 0.0, None)
        seg3 = len3 * np.minimum(gp, Le)
        integral = seg1 + seg2 + seg3

        depth0 = np.minimum(Le, np.maximum(R[:, 0], gp[:, 0]))
        burnedVolume = weight * (n0 * depth0 + np.sum((n / width) * integral, axis=1))
        volume[start : start + chunk] = weight * (n0 + counts.sum()) * Le - burnedVolume

        d2seg = np.where(active2, Rs * (_asin(b2 / Rs) - _asin(a2 / Rs)), 0.0)
        # Where the aft face of the slotted grain has burned through, the end burner also burns from the aft end
        d3seg = np.where(R > Lx, len3, 0.0) if aft else 0.0
        bins = np.where(allBurned, 0.0, d2seg + d3seg)
        a = weight * (n0 * (R[:, 0] < Le) + np.sum((n / width) * bins, axis=1))
        a = np.where(allBurned[:, 0], 0.0, a)
        area[start : start + chunk] = a

    volume = np.maximum(volume, 0.0)
    return volume, area


def buildSeparableTable(
    eb: EndBurningGrain,
    slot: FmmGrain,
    config,
    numSamples: int = 800,
    quick: bool = False,
) -> CoupledRegressionTable:
    """
    Builds the table for the separable 3D model. The slotted grain's simulation setup is run here.

    With 'quick' set the table only covers a tiny regression range. It is cheap to build and only good for values at
    zero regression, like the initial Kn.
    """
    slot.simulationSetup(config)

    D = slot.props["diameter"].getValue()
    Le = eb.props["length"].getValue()
    Lx = slot.props["length"].getValue()
    aft = isAftExposed(slot)
    wallWeb = slot.wallWeb

    d2 = _validDistances(slot)
    circle = geometry.circleArea(D)

    rMaxEb = float(np.sqrt(Le**2 + wallWeb**2))
    rMaxSlot = float(wallWeb)
    if aft:
        rMaxEb = min(rMaxEb, Lx + Le)
        rMaxSlot = min(rMaxSlot, Lx)

    if quick:
        rEb = np.array([0.0, 1e-9])
        rSlot = np.array([0.0, 1e-9])
    else:
        rEb = np.linspace(0.0, rMaxEb, numSamples)
        rSlot = np.linspace(0.0, rMaxSlot, max(100, numSamples // 2))

    volEb, areaEb = _separableEndBurner(rEb, d2, Le, Lx, aft, circle)
    if not quick:
        volEb[-1] = 0.0
        areaEb[-1] = 0.0

    volSlot = np.zeros(rSlot.shape)
    areaSlot = np.zeros(rSlot.shape)
    aftFactor = 1.0 if aft else 0.0
    for i, r in enumerate(rSlot):
        if r >= rMaxSlot and not quick:
            continue
        face = float(slot.getFaceArea(r))
        try:
            perimeter = float(slot.getCorePerimeter(r))
        except ValueError:  # No contour left
            perimeter = 0.0
        length = Lx - aftFactor * r
        volSlot[i] = face * length
        areaSlot[i] = perimeter * length + aftFactor * face

    return CoupledRegressionTable(
        rEb, volEb, areaEb, rSlot, volSlot, areaSlot, rMaxEb, rMaxSlot, MODEL_SEPARABLE
    )


# ----------------------------------------------------------------------------------------------------------------------
# Voxel model
# ----------------------------------------------------------------------------------------------------------------------


class _VoxelField:
    """The 3D distance field generated by the voxel model."""

    def __init__(self, d3, phi, h, nEb, nSlot, fullWeight, crop):
        self.d3 = d3  # Distance, masked array shaped (z, y, x). Possibly only one quadrant.
        self.phi = phi
        self.h = h
        self.nEb = nEb
        self.nSlot = nSlot
        self.weight = fullWeight  # Multiplier for the voxel counts (4 if a quadrant was computed)
        self.crop = crop


def _resampleCore(slot: FmmGrain, nxy: int) -> np.ndarray:
    """Resamples the slotted grain's core map onto an (nxy, nxy) grid of voxel centers. True where there is void."""
    centers = (np.arange(nxy) + 0.5) / nxy * 2 - 1
    idx = np.clip(np.floor((centers + 1) / 2 * (slot.mapDim - 1) + 0.5).astype(int), 0, slot.mapDim - 1)
    return slot.coreMap[np.ix_(idx, idx)] == 0


def _computeVoxelField(eb: EndBurningGrain, slot: FmmGrain, voxelSize: float, maxVoxels: float = MAX_VOXELS):
    """Runs the 3D fast marching method. The slotted grain must have been set up already."""
    D = slot.props["diameter"].getValue()
    Le = eb.props["length"].getValue()
    Lx = slot.props["length"].getValue()
    aft = isAftExposed(slot)

    nxy = max(8, 2 * int(round(D / (2 * voxelSize))))
    h = D / nxy
    nEb = max(1, int(round(Le / h)))
    nSlot = max(1, int(round(Lx / h)))
    nz = nEb + nSlot + (1 if aft else 0)

    core = _resampleCore(slot, nxy)
    centers = (np.arange(nxy) + 0.5) / nxy * 2 - 1
    gx, gy = np.meshgrid(centers, centers)
    casing = gx**2 + gy**2 > 1

    symmetric = np.array_equal(core, core[::-1, :]) and np.array_equal(core, core[:, ::-1])
    half = nxy // 2
    if symmetric:
        core = core[half:, half:]
        casing = casing[half:, half:]
        weight = 4.0
    else:
        weight = 1.0
    ny, nx = core.shape

    numVoxels = float(nz) * ny * nx
    if numVoxels > maxVoxels:
        raise CoupledRegressionError(
            "Voxel 3D model would need {:.2g} voxels (limit {:.2g}). Increase the transition voxel size.".format(
                numVoxels, maxVoxels
            )
        )

    # Height of the bottom of the slot above the end burner interface. Always flat for now, kept as an array so that
    # other slot bottom shapes can be added without changing the solver.
    zBottom = np.zeros(core.shape)

    phi = np.ones((nz, ny, nx))
    sCenters = (np.arange(nSlot) + 0.5) * h
    slotVoid = core[None, :, :] & (sCenters[:, None, None] >= zBottom[None, :, :])
    phi[nEb : nEb + nSlot][slotVoid] = -1.0
    if aft:
        phi[nEb + nSlot] = -1.0

    mask = np.broadcast_to(casing[None, :, :], phi.shape)
    d3 = skfmm.distance(np.ma.MaskedArray(phi, mask), dx=h)
    return _VoxelField(d3, phi, h, nEb, nSlot, weight, symmetric)


def _windowedVolume(sortedD: np.ndarray, prefix: np.ndarray, r: np.ndarray, h: float):
    """
    Returns (count, inWindow) where count is the effective number of voxels left at regression r and inWindow is the
    number of voxels in the shell around r. Each voxel is spread out over a distance h centered on its own distance.
    """
    tol = 1e-9 * h
    hi = np.searchsorted(sortedD, r + h / 2 + tol, side="right")
    lo = np.searchsorted(sortedD, r - h / 2 + tol, side="right")
    hi = np.maximum(hi, lo)
    full = sortedD.size - hi
    partial = ((prefix[hi] - prefix[lo]) + (h / 2 - r) * (hi - lo)) / h
    return full + partial, (hi - lo)


def buildVoxelTable(
    eb: EndBurningGrain,
    slot: FmmGrain,
    config,
    voxelSize: Optional[float] = None,
    maxVoxels: float = MAX_VOXELS,
) -> CoupledRegressionTable:
    """
    Builds the table with a 3D fast marching calculation on a voxel grid. The slotted grain's simulation setup is run
    here. Raises CoupledRegressionError if the grid would have too many voxels.
    """
    slot.simulationSetup(config)
    if voxelSize is None:
        voxelSize = config.getProperty("transitionVoxelSize")

    Le = eb.props["length"].getValue()
    Lx = slot.props["length"].getValue()
    field = _computeVoxelField(eb, slot, voxelSize, maxVoxels)
    h = field.h

    d3 = np.ma.getdata(field.d3)
    propellant = np.logical_and(field.phi > 0, np.logical_not(np.ma.getmaskarray(field.d3)))

    def regionTable(zStart, zEnd, lengthScale):
        d = np.sort(d3[zStart:zEnd][propellant[zStart:zEnd]])
        prefix = np.concatenate(([0.0], np.cumsum(d)))
        rMax = float(d[-1] + h / 2) if d.size else 0.0
        count = max(int(rMax / (h / 2)), 16)
        r = np.linspace(0.0, rMax, count + 1)
        n, nIn = _windowedVolume(d, prefix, r, h)
        scale = field.weight * lengthScale
        vol = n * h**3 * scale
        area = nIn * h**2 * scale
        if area.size >= 11:
            area = np.maximum(savgol_filter(area, 11, 3), 0.0)
        vol[-1] = 0.0
        area[-1] = 0.0
        return r, vol, area, rMax

    rEb, volEb, areaEb, rMaxEb = regionTable(0, field.nEb, Le / (field.nEb * h))
    rSlot, volSlot, areaSlot, rMaxSlot = regionTable(
        field.nEb, field.nEb + field.nSlot, Lx / (field.nSlot * h)
    )

    return CoupledRegressionTable(
        rEb, volEb, areaEb, rSlot, volSlot, areaSlot, rMaxEb, rMaxSlot, MODEL_VOXEL
    )


# ----------------------------------------------------------------------------------------------------------------------
# Proxy grains
# ----------------------------------------------------------------------------------------------------------------------


class _CoupledProxy(Grain):
    """Base for the proxy grains. They share the properties of the wrapped grain and get all volumes and areas
    from the table."""

    def __init__(self, wrapped: Grain, table: CoupledRegressionTable) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.table = table
        self.props = wrapped.props
        self.geomName = wrapped.geomName

    def getEndPositions(self, regDist):
        return self.wrapped.getEndPositions(regDist)

    def getDetailsString(self, lengthUnit="m"):
        return self.wrapped.getDetailsString(lengthUnit)

    def getGeometryErrors(self):
        return self.wrapped.getGeometryErrors()

    def simulationSetup(self, config):
        pass  # The table has been built already


class CoupledEndBurnerProxy(_CoupledProxy):
    """Stands in for the end burner. Its burning surface is the part of the 3D front that is inside of it."""

    def getSurfaceAreaAtRegression(self, regDist):
        return self.table.areaEb(regDist)

    def getVolumeAtRegression(self, regDist):
        if regDist >= self.table.rMaxEb:
            return 0.0
        return self.table.volumeEb(regDist)

    def getWebLeft(self, regDist):
        return self.table.rMaxEb - regDist

    def getMassFlux(self, massIn, dTime, regDist, dRegDist, position, density):
        return 0

    def getPeakMassFlux(self, massIn, dTime, regDist, dRegDist, density):
        return 0

    def getPortArea(self, regDist):
        return None


class CoupledSlotProxy(_CoupledProxy):
    """Stands in for the slotted grain behind the end burner."""

    def getSurfaceAreaAtRegression(self, regDist):
        return self.table.areaSlot(regDist)

    def getVolumeAtRegression(self, regDist):
        if regDist >= self.table.rMaxSlot:
            return 0.0
        return self.table.volumeSlot(regDist)

    def getWebLeft(self, regDist):
        return self.table.rMaxSlot - regDist

    def getPortArea(self, regDist):
        if regDist >= self.table.rMaxSlot:
            return geometry.circleArea(self.props["diameter"].getValue())
        return self.wrapped.getPortArea(regDist)

    def getMassFlux(self, massIn, dTime, regDist, dRegDist, position, density):
        massFlow = massIn + (self.getVolumeSlice(regDist, dRegDist) * density / dTime)
        return massFlow / self.getPortArea(regDist + dRegDist)

    def getPeakMassFlux(self, massIn, dTime, regDist, dRegDist, density):
        return self.getMassFlux(massIn, dTime, regDist, dRegDist, 0, density)


# ----------------------------------------------------------------------------------------------------------------------
# Entry point for the motor
# ----------------------------------------------------------------------------------------------------------------------


def buildSimulationGrains(grains: List[Grain], config, quick: bool = False) -> Tuple[List[Grain], List[SimAlert]]:
    """
    Returns the list of grains a simulation should use, and any alerts generated while preparing it. With the
    independent grains model (or if there is no end burner followed by a slotted grain), the list is just the grains
    passed in. The grains are not modified, but all of them have their simulation setup run by this function.

    With 'quick' set, the table is only built to be accurate at zero regression and the voxel model is replaced with
    the separable one, which is equivalent at that point.
    """
    mode = config.getProperty("slottedEndBurnerModel")
    alerts: List[SimAlert] = []

    pair = None if mode == MODEL_INDEPENDENT else findCoupledPair(grains, config)

    if mode != MODEL_INDEPENDENT and pair is None:
        alerts.append(
            SimAlert(
                SimAlertLevel.WARNING,
                SimAlertType.CONSTRAINT,
                "Slotted end burner model '{}' needs an end burner followed by a slotted grain of the same diameter. "
                "Using independent grains.".format(mode),
                "Motor",
            )
        )

    paired = set(pair) if pair is not None else set()
    for gid, grain in enumerate(grains):
        if gid not in paired:
            grain.simulationSetup(config)

    if pair is None:
        return list(grains), alerts

    eb, slot = grains[pair[0]], grains[pair[1]]
    try:
        if mode == MODEL_VOXEL and not quick:
            table = buildVoxelTable(eb, slot, config)
        else:
            table = buildSeparableTable(eb, slot, config, quick=quick)
    except CoupledRegressionError as exc:
        alerts.append(SimAlert(SimAlertLevel.ERROR, SimAlertType.CONSTRAINT, str(exc), "Motor"))
        return list(grains), alerts

    simGrains = list(grains)
    simGrains[pair[0]] = CoupledEndBurnerProxy(eb, table)
    simGrains[pair[1]] = CoupledSlotProxy(slot, table)

    if not quick:
        alerts.append(
            SimAlert(
                SimAlertLevel.MESSAGE,
                SimAlertType.VALUE,
                "Grains 1 and 2 were simulated with the coupled '{}' slotted end burner model.".format(mode),
                "Motor",
            )
        )
    return simGrains, alerts


# ----------------------------------------------------------------------------------------------------------------------
# Axial section preview
# ----------------------------------------------------------------------------------------------------------------------


def _lineCoordinates(slot: FmmGrain, angle: float, numPoints: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns pixel coordinates (row, col) in the slotted grain's 2D maps along a line through the center at 'angle'
    (degrees), and the normalized position along the line (-1 to 1)."""
    t = np.linspace(-1.0, 1.0, numPoints)
    theta = np.radians(angle)
    xn, yn = t * np.cos(theta), t * np.sin(theta)
    col = (xn + 1) / 2 * (slot.mapDim - 1)
    row = (yn + 1) / 2 * (slot.mapDim - 1)
    return row, col, t


def getAxialSectionData(
    eb: EndBurningGrain,
    slot: FmmGrain,
    config,
    mapDim: int,
    angle: float = 0.0,
    numContours: int = 15,
    model: Optional[str] = None,
):
    """
    Returns a tuple (image, d3Image, contours, contourLevels) describing a cut through the motor axis at 'angle'
    (degrees, 0 is along the x axis of the grain face), from the head end (left) through the end burner and the slotted
    grain. 'image' is a masked array that is 0 in the slots and 1 in propellant, 'd3Image' the 3D regression distance
    in metres and 'contours' a list of lists of (row, col) point arrays at 'contourLevels', which are evenly spaced
    regression distances. The model defaults to the one from the config, and 'mapDim' is the horizontal resolution.
    """
    from skimage import measure

    if model is None:
        model = config.getProperty("slottedEndBurnerModel")

    # Work on a config with the requested resolution, so the slotted grain's maps are small
    cfg = type(config)()
    cfg.setProperties(config.getProperties())
    cfg.setProperty("mapDim", max(250, int(mapDim)))
    slot.simulationSetup(cfg)

    D = slot.props["diameter"].getValue()
    Le = eb.props["length"].getValue()
    Lx = slot.props["length"].getValue()
    aft = isAftExposed(slot)

    numRho = int(mapDim)
    numZ = int(np.clip(round(mapDim * (Le + Lx) / D), 32, 2000))
    z = np.linspace(-Le, Lx, numZ)  # Negative is in the end burner. Index 0 is the head end.
    rho = np.linspace(-0.5 * D, 0.5 * D, numRho)

    row, col, t = _lineCoordinates(slot, angle, numRho)
    inCasing = np.abs(t) <= 1.0

    if model == MODEL_VOXEL:
        h = max(config.getProperty("transitionVoxelSize"), D / 120)
        field = _computeVoxelField(eb, slot, h)
        d3 = np.ma.getdata(field.d3)
        solid = np.where(field.phi > 0, 1.0, 0.0)
        if field.crop:  # Rebuild the full domain by mirroring the quadrant
            d3, solid = _mirrorQuadrant(d3), _mirrorQuadrant(solid)
        nxy = d3.shape[1]
        centers = (np.arange(nxy) + 0.5) / nxy * 2 - 1
        casing = (centers[:, None] ** 2 + centers[None, :] ** 2) > 1
        d3 = _fillInvalid(d3, np.broadcast_to(casing[None, :, :], d3.shape))
        solid = _fillInvalid(solid, np.broadcast_to(casing[None, :, :], solid.shape))
        zVox = (np.arange(d3.shape[0]) + 0.5) * field.h - field.nEb * field.h
        zCoord = np.interp(z, zVox, np.arange(d3.shape[0]))
        theta = np.radians(angle)
        px = (t * np.cos(theta) + 1) / 2 * nxy - 0.5
        py = (t * np.sin(theta) + 1) / 2 * nxy - 0.5
        zz, pp = np.meshgrid(zCoord, np.arange(numRho), indexing="ij")
        coords = np.array([zz, py[pp], px[pp]])
        d3Line = ndimage.map_coordinates(np.maximum(d3, 0.0), coords, order=1, mode="nearest")
        solidLine = ndimage.map_coordinates(solid, coords, order=0, mode="nearest")
        d3Image = d3Line.T  # (rho, z)
        void = solidLine.T < 0.5
    else:
        d2Line = ndimage.map_coordinates(
            _fillInvalid(np.ma.getdata(slot.regressionMap), np.ma.getmaskarray(slot.regressionMap)),
            [row, col],
            order=1,
            mode="nearest",
        )
        d2Line = np.maximum(slot.unNormalize(d2Line), 0.0)
        coreLine = ndimage.map_coordinates(slot.coreMap.astype(float), [row, col], order=0, mode="nearest") == 0

        Z, D2 = np.meshgrid(z, d2Line)  # shape (rho, z)
        inEb = Z < 0
        d3Image = np.where(inEb, np.sqrt(D2**2 + Z**2), D2)
        if aft:
            d3Image = np.minimum(d3Image, Lx - Z)
        void = np.logical_and(np.logical_not(inEb), coreLine[:, None])

    outside = np.broadcast_to(np.logical_not(inCasing)[:, None], d3Image.shape)
    image = np.ma.MaskedArray(np.where(void, 0, 1), outside)
    d3Masked = np.ma.MaskedArray(np.where(void, 0.0, d3Image), np.logical_or(outside, void))

    rMax = float(np.max(np.where(void, 0.0, d3Image)[np.logical_not(outside)]))
    levels = np.linspace(0, rMax, numContours)
    contours: List[List[np.ndarray]] = []
    filled = np.ma.getdata(d3Masked)
    valid = np.logical_not(np.ma.getmaskarray(d3Masked))
    for level in levels:
        try:
            found = measure.find_contours(filled, level, mask=valid)
        except (ValueError, RuntimeError):
            found = []
        contours.append(found)
    return (image, d3Masked, contours, levels)


def _fillInvalid(arr: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    """Replaces the entries of 'arr' where 'invalid' is True with the value of the nearest valid entry. The maps of the
    slotted grain are masked outside of the casing, and interpolating towards those entries would pull the contours at
    the casing wall to zero."""
    if not invalid.any():
        return arr
    idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def _mirrorQuadrant(arr: np.ndarray) -> np.ndarray:
    full = np.concatenate((arr[:, ::-1, :], arr), axis=1)
    return np.concatenate((full[:, :, ::-1], full), axis=2)


def getAreaCurves(
    eb: EndBurningGrain, slot: FmmGrain, config, numPoints: int = 100
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns the total burning surface area versus regression distance for the coupled model ('coupled') and for the
    independent grains model ('legacy'), each as (r, area). The slotted grain must have been set up with the config's
    map resolution, which this function takes care of.
    """
    model = config.getProperty("slottedEndBurnerModel")
    cfg = type(config)()
    cfg.setProperties(config.getProperties())

    if model == MODEL_VOXEL:
        table = buildVoxelTable(eb, slot, cfg, voxelSize=max(cfg.getProperty("transitionVoxelSize"), slot.props["diameter"].getValue() / 120))
    else:
        table = buildSeparableTable(eb, slot, cfg, numSamples=300)

    rMax = max(table.rMaxEb, table.rMaxSlot)
    r = np.linspace(0, rMax, numPoints)
    coupled = np.array([table.areaEb(x) + table.areaSlot(x) for x in r])

    legacy = []
    thres = 1e-5
    for x in r:
        a = geometry.circleArea(eb.props["diameter"].getValue()) if x < eb.props["length"].getValue() else 0.0
        if slot.isWebLeft(x, thres):
            a += slot.getSurfaceAreaAtRegression(x)
        legacy.append(a)
    return {"coupled": (r, coupled), "legacy": (r, np.array(legacy))}

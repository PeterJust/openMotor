import unittest

import numpy as np

import motorlib.coupledRegression as cr
import motorlib.grains
import motorlib.motor
import motorlib.propellant
from motorlib.simResult import SimAlertLevel


def makeConfig(model, mapDim=500, voxelSize=5e-4):
    config = motorlib.motor.MotorConfig()
    config.setProperties(
        {
            "timestep": 0.03,
            "burnoutThrustThres": 0.1,
            "burnoutWebThres": 2.54e-5,
            "ambPressure": 101325,
            "maxPressure": 1e7,
            "maxMassFlux": 1e4,
            "maxMachNumber": 0.7,
            "minPortThroat": 1,
            "sepPressureRatio": 0.4,
            "flowSeparationWarnPercent": 0.05,
        }
    )
    config.setProperty("mapDim", mapDim)
    config.setProperty("slottedEndBurnerModel", model)
    config.setProperty("transitionVoxelSize", voxelSize)
    return config


def makeGrains(inhibitedEnds="Both", slotWidth=0.003, slotLength=0.008, diameter=0.0254, ebLength=0.04, xLength=0.01):
    eb = motorlib.grains.EndBurningGrain()
    eb.setProperties({"diameter": diameter, "length": ebLength})
    slot = motorlib.grains.XCore()
    slot.setProperties(
        {
            "diameter": diameter,
            "length": xLength,
            "inhibitedEnds": inhibitedEnds,
            "slotWidth": slotWidth,
            "slotLength": slotLength,
        }
    )
    return eb, slot


def makeMotor(model, **kwargs):
    motor = motorlib.motor.Motor()
    motor.config = makeConfig(model, mapDim=500)
    eb, slot = makeGrains(**kwargs)
    motor.grains = [eb, slot]
    motor.nozzle.setProperties({"throat": 0.004, "exit": 0.008, "efficiency": 1.0})
    motor.propellant = motorlib.propellant.Propellant()
    motor.propellant.setProperties(
        {
            "name": "KNSU",
            "density": 1890,
            "tabs": [
                {
                    "minPressure": 0,
                    "maxPressure": 1e7,
                    "a": 0.000101,
                    "n": 0.319,
                    "t": 1720,
                    "m": 41.98,
                    "k": 1.133,
                }
            ],
        }
    )
    return motor


class FindCoupledPairTests(unittest.TestCase):
    def test_validPair(self):
        eb, slot = makeGrains()
        self.assertEqual(cr.findCoupledPair([eb, slot]), (0, 1))

    def test_invalidPairs(self):
        eb, slot = makeGrains()
        bates = motorlib.grains.BatesGrain()
        bates.setProperties({"diameter": 0.0254, "length": 0.01, "coreDiameter": 0.01})
        self.assertIsNone(cr.findCoupledPair([]))
        self.assertIsNone(cr.findCoupledPair([eb]))
        self.assertIsNone(cr.findCoupledPair([slot, eb]))
        self.assertIsNone(cr.findCoupledPair([eb, bates]))  # Not a fast marching grain
        eb2, slot2 = makeGrains()
        slot2.setProperty("diameter", 0.03)
        self.assertIsNone(cr.findCoupledPair([eb2, slot2]))  # Different diameters


class SeparableTableTests(unittest.TestCase):
    def setUp(self):
        self.config = makeConfig(cr.MODEL_SEPARABLE)

    def build(self, **kwargs):
        eb, slot = makeGrains(**kwargs)
        table = cr.buildSeparableTable(eb, slot, self.config)
        return eb, slot, table

    def test_initialVolumesMatchIndependentGrains(self):
        for inhibited in ("Both", "Top", "Bottom", "Neither"):
            eb, slot, table = self.build(inhibitedEnds=inhibited)
            self.assertAlmostEqual(table.volumeEb(0) / eb.getVolumeAtRegression(0), 1, delta=2e-3)
            self.assertAlmostEqual(table.volumeSlot(0) / slot.getVolumeAtRegression(0), 1, delta=2e-3)

    def test_initialAreaIsSlotSurface(self):
        eb, slot, table = self.build(inhibitedEnds="Both")
        # The end burner only burns the slot footprint at first
        footprint = slot.getFaceArea(0)
        footprint = eb.getSurfaceAreaAtRegression(0) - footprint  # Area of the cross
        self.assertAlmostEqual(table.areaEb(0) / footprint, 1, delta=0.03)
        walls = slot.getCorePerimeter(0) * slot.props["length"].getValue()
        self.assertAlmostEqual(table.areaSlot(0) / walls, 1, delta=0.01)

    def test_aftFaceAddsToInitialArea(self):
        _, slotInhibited, tableInhibited = self.build(inhibitedEnds="Both")
        _, slotOpen, tableOpen = self.build(inhibitedEnds="Top")
        extra = tableOpen.areaSlot(0) - tableInhibited.areaSlot(0)
        self.assertAlmostEqual(extra / slotOpen.getFaceArea(0), 1, delta=0.01)

    def test_areaIntegratesToVolume(self):
        for inhibited in ("Both", "Top"):
            _, _, table = self.build(inhibitedEnds=inhibited)
            self.assertAlmostEqual(np.trapz(table.A_eb, table.rEb) / table.V_eb[0], 1, delta=5e-3)
            # The slotted grain's area comes from the perimeter of its 2D map and its volume from pixel counts, like in
            # the independent model, so they agree a bit less well.
            self.assertAlmostEqual(np.trapz(table.A_slot, table.rSlot) / table.V_slot[0], 1, delta=0.04)
            self.assertEqual(table.V_eb[-1], 0)

    def test_endBurnerAreaApproachesFullFace(self):
        eb, slot, table = self.build(inhibitedEnds="Both")
        full = eb.getSurfaceAreaAtRegression(0)
        r = 0.5 * (slot.wallWeb * 3 + table.rMaxEb * 0.8)
        self.assertGreater(r, slot.wallWeb * 1.5)
        self.assertAlmostEqual(table.areaEb(r) / full, 1, delta=0.02)

    def test_burnoutPoint(self):
        eb, slot, table = self.build(inhibitedEnds="Both")
        Le = eb.props["length"].getValue()
        self.assertAlmostEqual(table.rMaxEb, np.sqrt(Le**2 + slot.wallWeb**2))
        self.assertAlmostEqual(table.rMaxSlot, slot.wallWeb)
        # Burning from the aft face limits both
        _, slotOpen, tableOpen = self.build(inhibitedEnds="Neither")
        Lx = slotOpen.props["length"].getValue()
        self.assertAlmostEqual(tableOpen.rMaxSlot, min(Lx, slotOpen.wallWeb))
        self.assertAlmostEqual(tableOpen.rMaxEb, min(np.sqrt(Le**2 + slotOpen.wallWeb**2), Le + Lx))

    def test_quickTable(self):
        eb, slot = makeGrains()
        quick = cr.buildSeparableTable(eb, slot, self.config, quick=True)
        full = cr.buildSeparableTable(eb, slot, self.config)
        self.assertAlmostEqual(quick.areaEb(0), full.areaEb(0))
        self.assertAlmostEqual(quick.areaSlot(0), full.areaSlot(0))


class VoxelTableTests(unittest.TestCase):
    def test_matchesSeparable(self):
        for inhibited in ("Both", "Top"):
            eb, slot = makeGrains(inhibitedEnds=inhibited)
            config = makeConfig(cr.MODEL_VOXEL, voxelSize=4e-4)
            sep = cr.buildSeparableTable(eb, slot, config)
            vox = cr.buildVoxelTable(eb, slot, config)

            self.assertAlmostEqual(vox.volumeEb(0) / sep.volumeEb(0), 1, delta=0.02)
            self.assertAlmostEqual(vox.volumeSlot(0) / sep.volumeSlot(0), 1, delta=0.02)
            self.assertAlmostEqual(vox.rMaxEb / sep.rMaxEb, 1, delta=0.03)
            self.assertAlmostEqual(vox.rMaxSlot / sep.rMaxSlot, 1, delta=0.03)

            # Compare the end burner's area and volume at points along the burn
            for fraction in (0.1, 0.25, 0.5, 0.75):
                r = fraction * sep.rMaxEb
                self.assertAlmostEqual(vox.volumeEb(r) / sep.volumeEb(r), 1, delta=0.05)
                self.assertAlmostEqual(vox.areaEb(r) / sep.areaEb(r), 1, delta=0.08)
            # The initial burning surface of the whole block
            self.assertAlmostEqual(
                (vox.areaEb(0) + vox.areaSlot(0)) / (sep.areaEb(0) + sep.areaSlot(0)), 1, delta=0.08
            )

    def test_tooManyVoxels(self):
        eb, slot = makeGrains()
        config = makeConfig(cr.MODEL_VOXEL)
        slot.simulationSetup(config)
        with self.assertRaises(cr.CoupledRegressionError):
            cr._computeVoxelField(eb, slot, 1e-4, maxVoxels=1000)


class ProxyAndMotorTests(unittest.TestCase):
    def test_independentModelIsUnchanged(self):
        motor = makeMotor(cr.MODEL_INDEPENDENT)
        grains, alerts = motor.buildSimulationGrains()
        self.assertEqual(alerts, [])
        self.assertIs(grains[0], motor.grains[0])
        self.assertIs(grains[1], motor.grains[1])

    def test_coupledModelDoesNotModifyMotor(self):
        motor = makeMotor(cr.MODEL_SEPARABLE)
        originals = list(motor.grains)
        grains, alerts = motor.buildSimulationGrains()
        self.assertIsInstance(grains[0], cr.CoupledEndBurnerProxy)
        self.assertIsInstance(grains[1], cr.CoupledSlotProxy)
        self.assertEqual(motor.grains, originals)
        self.assertEqual(grains[0].props, originals[0].props)
        self.assertEqual(grains[1].geomName, originals[1].geomName)

    def test_noValidPairFallsBack(self):
        motor = makeMotor(cr.MODEL_SEPARABLE)
        motor.grains.reverse()
        grains, alerts = motor.buildSimulationGrains()
        self.assertEqual([a.level for a in alerts], [SimAlertLevel.WARNING])
        self.assertIs(grains[0], motor.grains[0])

    def test_voxelLimitIsAnError(self):
        motor = makeMotor(cr.MODEL_VOXEL)
        motor.config.setProperty("transitionVoxelSize", 5e-5)
        grains, alerts = motor.buildSimulationGrains()
        self.assertEqual([a.level for a in alerts], [SimAlertLevel.ERROR])
        res = motor.runSimulation()
        self.assertFalse(res.success)
        self.assertEqual(len(res.getAlertsByLevel(SimAlertLevel.ERROR)), 1)

    def test_simulation(self):
        legacy = makeMotor(cr.MODEL_INDEPENDENT).runSimulation()
        separable = makeMotor(cr.MODEL_SEPARABLE).runSimulation()
        self.assertTrue(legacy.success)
        self.assertTrue(separable.success)

        # The same propellant is burned in either model
        self.assertAlmostEqual(separable.getPropellantMass() / legacy.getPropellantMass(), 1, delta=1e-3)
        # The end burner face is not lit at once, so Kn starts lower than with independent grains
        self.assertLess(separable.channels["kn"].getData()[0], 0.8 * legacy.channels["kn"].getData()[0])
        self.assertIn(
            "coupled 'Separable 3D'", [a.description for a in separable.alerts][0]
        )
        # The grains of the result are the real ones
        self.assertIsInstance(separable.motor.grains[0], motorlib.grains.EndBurningGrain)

        # All the mass of the propellant is accounted for in the mass flow
        mass = separable.channels["mass"].getData()
        self.assertAlmostEqual(sum(mass[0]) / separable.getPropellantMass(), 1, delta=1e-6)
        self.assertEqual(len(mass[0]), 2)

    def test_quickResults(self):
        legacy = makeMotor(cr.MODEL_INDEPENDENT).getQuickResults()
        coupled = makeMotor(cr.MODEL_SEPARABLE).getQuickResults()
        self.assertAlmostEqual(coupled["propellantMass"] / legacy["propellantMass"], 1, delta=2e-3)
        self.assertAlmostEqual(coupled["volumeLoading"] / legacy["volumeLoading"], 1, delta=1e-3)
        self.assertLess(coupled["initialKn"], legacy["initialKn"])


class AxialSectionTests(unittest.TestCase):
    def test_sections(self):
        for model in (cr.MODEL_SEPARABLE, cr.MODEL_VOXEL):
            eb, slot = makeGrains()
            config = makeConfig(model, mapDim=250, voxelSize=5e-4)
            for angle in (0, 45):
                image, d3, contours, levels = cr.getAxialSectionData(eb, slot, config, 250, angle)
                self.assertEqual(image.shape, d3.shape)
                self.assertEqual(len(contours), len(levels))
                self.assertGreater(len(contours[1]), 0)
                # The far end of the end burner is the farthest from the slots
                self.assertGreater(np.ma.max(d3), 0.9 * eb.props["length"].getValue())

    def test_areaCurves(self):
        eb, slot = makeGrains()
        curves = cr.getAreaCurves(eb, slot, makeConfig(cr.MODEL_SEPARABLE, mapDim=250))
        rCoupled, aCoupled = curves["coupled"]
        rLegacy, aLegacy = curves["legacy"]
        self.assertEqual(rCoupled.shape, aCoupled.shape)
        self.assertLess(aCoupled[0], aLegacy[0])


if __name__ == "__main__":
    unittest.main()

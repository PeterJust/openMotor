from PyQt6.QtWidgets import QLabel

import motorlib.grain
import motorlib.nozzle
import motorlib.motor
import motorlib.coupledRegression

from .collectionEditor import CollectionEditor
from .grainPreviewWidget import GrainPreviewWidget
from .nozzlePreviewWidget import NozzlePreviewWidget

class MotorEditor(CollectionEditor):
    def __init__(self, parent):
        super().__init__(parent, True)

        self.expRatioLabel = QLabel("Expansion ratio: -")
        self.expRatioLabel.hide()
        self.stats.addWidget(self.expRatioLabel)

        self.grainPreview = GrainPreviewWidget()
        self.grainPreview.hide()
        self.stats.addWidget(self.grainPreview)

        self.nozzlePreview = NozzlePreviewWidget()
        self.nozzlePreview.hide()
        self.stats.addWidget(self.nozzlePreview)

        self.objType = None

        # The motor that is being edited and the index of the grain in it, used for previews that depend on the
        # grains around the edited one
        self.motor = None
        self.grainIndex = None

    def setMotorContext(self, motor, grainIndex=None):
        """Tells the editor which motor the next loaded object belongs to, and which grain of it is being edited.
        Call this before loadObject."""
        self.motor = motor
        self.grainIndex = grainIndex

    def getCoupledContext(self, grain):
        """If the grain is the slotted grain behind an end burner and a coupled slotted end burner model is selected,
        returns the end burner and the motor config. Otherwise returns None."""
        if self.motor is None or self.grainIndex != 1 or not hasattr(grain, 'regressionMap'):
            return None
        if self.motor.config.getProperty('slottedEndBurnerModel') == motorlib.coupledRegression.MODEL_INDEPENDENT:
            return None
        grains = list(self.motor.grains[:2])
        if len(grains) < 2:
            return None
        grains[1] = grain
        if motorlib.coupledRegression.findCoupledPair(grains) is None:
            return None
        return (grains[0], self.motor.config)

    def propertyUpdate(self):
        if issubclass(self.objType, motorlib.nozzle.Nozzle):
            exitDia = self.propertyEditors['exit'].getValue()
            throatDia = self.propertyEditors['throat'].getValue()
            if throatDia == 0:
                self.expRatioLabel.setText('Expansion ratio: -')
            else:
                self.expRatioLabel.setText('Expansion ratio: {:.3f}'.format((exitDia / throatDia) ** 2))
            nozzle = self.objType()
            nozzle.setProperties(self.getProperties())
            self.nozzlePreview.loadNozzle(nozzle)

        if issubclass(self.objType, motorlib.grain.PerforatedGrain):
            testGrain = self.objType()
            testGrain.setProperties(self.getProperties())
            self.grainPreview.loadGrain(testGrain, self.getCoupledContext(testGrain))

    def loadObject(self, obj):
        self.objType = type(obj)
        self.loadProperties(obj)

        if issubclass(self.objType, motorlib.grain.PerforatedGrain):
            self.grainPreview.show()
            self.nozzlePreview.hide()
            self.expRatioLabel.hide()
            self.propertyUpdate()

        if issubclass(self.objType, motorlib.nozzle.Nozzle):
            self.expRatioLabel.show()
            self.nozzlePreview.show()
            self.grainPreview.hide()

        if issubclass(self.objType, motorlib.motor.MotorConfig):
            self.expRatioLabel.hide()
            self.nozzlePreview.hide()
            self.grainPreview.hide()

    def cleanup(self):
        self.expRatioLabel.hide()
        self.grainPreview.hide()
        self.nozzlePreview.hide()
        self.grainPreview.cleanup()
        super().cleanup()

from threading import Thread

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal

import motorlib
import motorlib.coupledRegression

from ..views.GrainPreview_ui import Ui_GrainPreview
from .grainPreviewGraph import GrainPreviewGraph

class GrainPreviewWidget(QWidget):

    previewReady = pyqtSignal(tuple)

    axialAngles = (0, 45)

    def __init__(self):
        super().__init__()
        self.ui = Ui_GrainPreview()
        self.ui.setupUi(self)

        self.ui.tabFace.setupImagePlot()
        self.ui.tabRegression.setupImagePlot()
        self.ui.tabAreaGraph.setupGraphPlot()

        # The axial section tab is only shown when the grain is simulated together with an end burner. It is created
        # here instead of in the generated UI file.
        # There is one for a section through a slot, and one for a section between two slots.
        self.axialTabs = []
        self.axialTabIndexes = []
        for angle in self.axialAngles:
            tab = GrainPreviewGraph()
            tab.setupImagePlot()
            index = self.ui.tabWidget.addTab(tab, "Axial {}\u00b0".format(angle))
            self.axialTabs.append(tab)
            self.axialTabIndexes.append(index)
        self._setAxialTabsVisible(False)

        # Used to navigate back to the tab the user was on after they clear alerts
        self.lastNonAlertTab = 1

        self.ui.tabWidget.currentChanged.connect(self.onTabChanged)

        self.previewReady.connect(self.updateView)

    def _setAxialTabsVisible(self, visible):
        for index in self.axialTabIndexes:
            self.ui.tabWidget.setTabVisible(index, visible)

    def _cleanupAxialTabs(self):
        for tab in self.axialTabs:
            tab.cleanup()

    def loadGrain(self, grain, coupledContext=None):
        """Shows previews of the grain. If 'coupledContext' is not None it must be a tuple of the end burner in front
        of the grain and the motor config, and the previews of the coupled slotted end burner model are added."""
        self._setAxialTabsVisible(coupledContext is not None)
        if coupledContext is None:
            self._cleanupAxialTabs()

        geomAlerts = grain.getGeometryErrors()

        self.ui.tabAlerts.clear()
        for err in geomAlerts:
            self.ui.tabAlerts.addItem(err.description)

        for alert in geomAlerts:
            if alert.level == motorlib.simResult.SimAlertLevel.ERROR:
                # Go to alerts tab and clear up graph/images
                self.ui.tabWidget.setCurrentIndex(0)
                self.ui.tabFace.cleanup()
                self.ui.tabRegression.cleanup()
                self.ui.tabAreaGraph.cleanup()
                self._cleanupAxialTabs()
                return

        # If they were on the alert tab, go to their last image/graph tab. Otherwise, let them stay
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.tabWidget.setCurrentIndex(self.lastNonAlertTab)

        # Generate the contents to show on the image/graph tabs
        dataThread = Thread(target=self._genData, args=[grain, coupledContext])
        dataThread.start()

    def _genData(self, grain, coupledContext):
        out = grain.getRegressionData(250, coreBlack=False)
        axial = None
        if coupledContext is not None:
            try:
                axial = self._genCoupledData(grain, *coupledContext)
            except Exception as exc:  # The preview should never take the editor down
                print(exc)
        self.previewReady.emit((out, axial))

    @staticmethod
    def _genCoupledData(grain, endBurner, config):
        # Use the low resolution maps of the preview
        previewConfig = type(config)()
        previewConfig.setProperties(config.getProperties())
        previewConfig.setProperty("mapDim", 250)
        sections = [
            motorlib.coupledRegression.getAxialSectionData(
                endBurner, grain, previewConfig, 250, angle
            )
            for angle in GrainPreviewWidget.axialAngles
        ]
        curves = motorlib.coupledRegression.getAreaCurves(endBurner, grain, previewConfig)
        return (sections, curves)

    def updateView(self, data):
        (coreIm, regImage, contours, contourLengths), axial = data

        self.ui.tabFace.cleanup()
        self.ui.tabFace.showImage(coreIm)

        if regImage is not None:
            self.ui.tabRegression.cleanup()
            self.ui.tabRegression.showImage(regImage)
            self.ui.tabRegression.showContours(contours)

            points = [[], []]

            for k in contourLengths.keys():
                points[0].append(k)
                points[1].append(contourLengths[k])

            self.ui.tabAreaGraph.cleanup()
            if axial is None:
                self.ui.tabAreaGraph.showGraph(points)

        self._cleanupAxialTabs()
        if axial is not None:
            sections, curves = axial
            # Sections with contours of equal regression distance
            for tab, (image, _, sectionContours, _) in zip(self.axialTabs, sections):
                tab.showImage(image)
                tab.showContours(sectionContours)

            # Total burning surface of the coupled model against the independent grains
            self.ui.tabAreaGraph.cleanup()
            self.ui.tabAreaGraph.showGraph(curves["legacy"], color="gray", linestyle="--")
            self.ui.tabAreaGraph.showGraph(curves["coupled"], color="b")

    def onTabChanged(self, tabIndex):
        if tabIndex != 0:
            self.lastNonAlertTab = tabIndex

    def cleanup(self):
        self.lastNonAlertTab = 1
        self.ui.tabAlerts.clear()
        self.ui.tabRegression.cleanup()
        self.ui.tabFace.cleanup()
        self.ui.tabAreaGraph.cleanup()
        self.ui.tabAreaGraph.resetGraphBounds()
        self._cleanupAxialTabs()
        self._setAxialTabsVisible(False)

"""This submodule houses the nozzle object and functions related to isentropic flow"""
import math

from scipy.optimize import fsolve

from .properties import FloatProperty, PropertyCollection
from . import geometry
from .simResult import SimAlert, SimAlertLevel, SimAlertType
from .constants import gasConstant, maximumRefDiameter, maximumRefLength

def eRatioFromPRatio(k, pRatio):
    """Returns the expansion ratio of a nozzle given the pressure ratio it causes."""
    return (((k+1)/2)**(1/(k-1))) * (pRatio ** (1/k)) * ((((k+1)/(k-1))*(1-(pRatio**((k-1)/k))))**0.5)

class Nozzle(PropertyCollection):
    """An object that contains the details about a motor's nozzle."""
    def __init__(self):
        super().__init__()
        self.props['throat'] = FloatProperty('Throat Diameter', 'm', 0, maximumRefDiameter)
        self.props['exit'] = FloatProperty('Exit Diameter', 'm', 0, maximumRefDiameter)
        self.props['efficiency'] = FloatProperty('Efficiency', '', 0, 2)
        self.props['divAngle'] = FloatProperty('Divergence Half Angle', 'deg', 0, 90)
        self.props['convAngle'] = FloatProperty('Convergence Half Angle', 'deg', 0, 90)
        self.props['throatLength'] = FloatProperty('Throat Length', 'm', 0, maximumRefLength / 10)
        self.props['slagCoeff'] = FloatProperty('Slag Buildup Coefficient', '(m*Pa)/s', 0, 1e6)
        self.props['erosionCoeff'] = FloatProperty('Throat Erosion Coefficient', 'm/(s*Pa)', 0, 1e6)

    def getDetailsString(self, lengthUnit='m'):
        """Returns a human-readable string containing some details about the nozzle."""
        return 'Throat: {}'.format(self.props['throat'].dispFormat(lengthUnit))

    def calcExpansion(self):
        """Returns the nozzle's expansion ratio."""
        return (self.props['exit'].getValue() / self.props['throat'].getValue()) ** 2

    def getThroatArea(self, dThroat=0):
        """Returns the area of the nozzle's throat. The optional parameter is added on to the nozzle throat diameter
        allow erosion or slag buildup during a burn."""
        return geometry.circleArea(self.props['throat'].getValue() + dThroat)

    def getExitArea(self):
        """Return the area of the nozzle's exit."""
        return geometry.circleArea(self.props['exit'].getValue())

    def getExitPressure(self, k, inputPressure):
        """Solves for the nozzle's exit pressure, given an input pressure and the gas's specific heat ratio."""
        return fsolve(lambda x: (1/self.calcExpansion()) - eRatioFromPRatio(k, x / inputPressure), 0)[0]

    def getDivergenceLosses(self):
        """Returns nozzle efficiency losses due to divergence angle"""
        divAngleRad = math.radians(self.props["divAngle"].getValue())
        return (1 + math.cos(divAngleRad)) / 2

    def getThroatLosses(self, dThroat=0):
        """Returns the losses caused by the throat aspect ratio as described in this document:
        http://rasaero.com/dloads/Departures%20from%20Ideal%20Performance.pdf"""
        throatAspect = self.props['throatLength'].getValue() / (self.props['throat'].getValue() + dThroat)
        if throatAspect > 0.45:
            return 0.95
        return 0.99 - (0.0333 * throatAspect)

    def getSkinLosses(self):
        """Returns the losses due to drag on the nozzle surface as described here:
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a099791.pdf. This is a constant for now, as people likely don't have
        a way to measure this themselves."""
        return 0.99

    def getIdealThrustCoeff(self, chamberPres, ambPres, gamma, dThroat, exitPres=None):
        """Calculates C_f, the ideal thrust coefficient for the nozzle, given the propellant's specific heat ratio, the
        ambient and chamber pressures. If nozzle exit presure isn't provided, it will be calculated. dThroat is the 
        change in throat diameter due to erosion or slag accumulation."""
        if chamberPres == 0:
            return 0

        if exitPres is None:
            exitPres = self.getExitPressure(gamma, chamberPres)
        exitArea = self.getExitArea()
        throatArea = self.getThroatArea(dThroat)

        term1 = (2 * (gamma ** 2)) / (gamma - 1)
        term2 = (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))
        term3 = 1 - ((exitPres / chamberPres) ** ((gamma - 1) / gamma))

        momentumThrust = (term1 * term2 * term3) ** 0.5
        pressureThrust = ((exitPres - ambPres) * exitArea) / (throatArea * chamberPres)

        return momentumThrust + pressureThrust

    def getAdjustedThrustCoeff(self, chamberPres, ambPres, gamma, dThroat, exitPres=None):
        """Calculates adjusted thrust coefficient for the nozzle, given the propellant's specific heat ratio, the
        ambient and chamber pressures. If nozzle exit presure isn't provided, it will be calculated. dThroat is the 
        change in throat diameter due to erosion or slag accumulation. This method uses a combination of the techniques
        described in these resources to adjust the thrust coefficient:
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a099791.pdf
        http://rasaero.com/dloads/Departures%20from%20Ideal%20Performance.pdf"""
        thrustCoeffIdeal = self.getIdealThrustCoeff(chamberPres, ambPres, gamma, dThroat, exitPres)
        divLoss = self.getDivergenceLosses()
        throatLoss = self.getThroatLosses(dThroat)
        skinLoss = self.getSkinLosses()
        efficiency = self.getProperty('efficiency')
        return divLoss * throatLoss * efficiency * (skinLoss * thrustCoeffIdeal + (1 - skinLoss))

    def getIdealExitVelocity(self, chamberPres, gamma, T, molarMass, exitPres=None):
        """Calculates the ideal (isentropic) exit velocity of the nozzle, given the chamber pressure, the propellant's
        specific heat ratio, combustion temperature and molar mass. If nozzle exit pressure isn't provided, it will be
        calculated. This is the momentum term of the ideal thrust coefficient (C_f * c*), so it doesn't depend on the
        ambient pressure: under- or overexpansion only changes the pressure thrust, not the exit velocity."""
        if chamberPres == 0:
            return 0

        if exitPres is None:
            exitPres = self.getExitPressure(gamma, chamberPres)

        term1 = (2 * gamma) / (gamma - 1)
        term2 = (gasConstant / molarMass) * T
        term3 = 1 - ((exitPres / chamberPres) ** ((gamma - 1) / gamma))

        return (term1 * term2 * term3) ** 0.5

    def getAdjustedExitVelocity(self, chamberPres, gamma, T, molarMass, dThroat, exitPres=None):
        """Calculates the adjusted exit velocity of the nozzle: the momentum-equivalent (mass-averaged), axial exit
        speed of the flow including losses. Only losses that reduce the speed of the flow are applied: throat, skin
        friction and the catch-all efficiency (heat loss to the nozzle, incomplete combustion, ...). The divergence loss
        is left out on purpose. It reduces the axial momentum (thrust), but the gas still leaves at full speed, and the
        Mach number depends on the speed only. Unlike the thrust coefficient, the skin loss is a plain factor here, as
        the (1 - skinLoss) offset is meaningless for a velocity."""
        velocityIdeal = self.getIdealExitVelocity(chamberPres, gamma, T, molarMass, exitPres)
        throatLoss = self.getThroatLosses(dThroat)
        skinLoss = self.getSkinLosses()
        efficiency = self.getProperty('efficiency')
        return throatLoss * skinLoss * efficiency * velocityIdeal

    def calcExitMachNumber(self, chamberPres, exitPres, gamma):
        """Calculates the ideal Mach number at the nozzle exit given the chamber pressure, exit pressure, 
        and isentropic exponent. -- suboptimal solution, better calculate in harmony with the rest of the program!"""
        if exitPres > 0:
            return math.sqrt((2 / (gamma - 1) * ((chamberPres / exitPres) ** ((gamma - 1) / gamma) - 1)))
        else:
            return 0
    
    def calcNonIdealExitMachNumber(self, velocity, T, gasConstant, molarMass, gamma):
        """Calculates the non-ideal Mach number at the nozzle exit given the exit velocity, temperature, gas constant, 
        and isentropic exponent. The (gamma -1) /2 belongs to the u**2 and converts the stagnation temperature to the 
        local speed of sound. T has to be the stagnation temperature, which stays constant in adiabatic flow with
        friction. The result is an effective Mach number belonging to the adjusted exit velocity."""
        if velocity > 0:
            term = gamma * gasConstant * T / molarMass / velocity**2 - ((gamma - 1) / 2)
            if term <= 0:  # Velocity above the maximum possible one for this temperature (efficiency too high)
                return 0
            return 1 / term ** 0.5
        else:
            return 0

    def getGeometryErrors(self):
        """Returns a list containing any errors with the nozzle's properties."""
        errors = []
        if self.props['throat'].getValue() == 0:
            aText = 'Throat diameter must not be 0'
            errors.append(SimAlert(SimAlertLevel.ERROR, SimAlertType.GEOMETRY, aText, 'Nozzle'))
        if self.props['exit'].getValue() < self.props['throat'].getValue():
            aText = 'Exit diameter must not be smaller than throat diameter'
            errors.append(SimAlert(SimAlertLevel.ERROR, SimAlertType.GEOMETRY, aText, 'Nozzle'))
        if self.props['efficiency'].getValue() == 0:
            aText = 'Efficiency must not be 0'
            errors.append(SimAlert(SimAlertLevel.ERROR, SimAlertType.CONSTRAINT, aText, 'Nozzle'))
        return errors

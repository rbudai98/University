from numpy.polynomial.polynomial import Polynomial
# example for polynomial p(x) = -2 + 5x**2 + x**3
pcoefs = [-2., 0., 5., 1.]
poly = Polynomial(pcoefs)

from pyqsp.angle_sequence import QuantumSignalProcessingPhases
phi = QuantumSignalProcessingPhases(poly, signal_operator="Wx", method="laurent")

pyqsp.response.PlotQSPResponse(phi, target=poly, signal_operator="Wx")
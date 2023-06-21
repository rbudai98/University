# Exercise 9.2 c)
pcoefs = [0., -1/3., 0., 5/4.]
poly = Polynomial(pcoefs)
phi = QuantumSignalProcessingPhases(poly, signal_operator="Wx", method="laurent")
#pyqsp.response.PlotQSPResponse(phi, target=poly, signal_operator="Wx")
plt.plot(x_range, [(5 * a**3 / 4  - a / 3) for a in x_range], 'b')
plt.plot(x_range, [np.real(project1(qsp(phi, a))) for a in x_range], 'r+')

plt.show()
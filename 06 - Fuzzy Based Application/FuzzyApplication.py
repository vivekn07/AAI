import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# New Antecedent/Consequent objects hold universe variables and membership functions
quality = ctrl.Antecedent(np.arange(0, 10, 0.1), 'quality')
service = ctrl.Antecedent(np.arange(0, 10, 0.1), 'service')
tip = ctrl.Consequent(np.arange(0, 25, 0.1), 'tip')

# Define membership functions
quality['poor'] = fuzz.zmf(quality.universe, 0, 5)
quality['average'] = fuzz.gaussmf(quality.universe, 5, 1)
quality['good'] = fuzz.smf(quality.universe, 5, 10)

service['poor'] = fuzz.zmf(service.universe, 0, 5)
service['average'] = fuzz.gaussmf(service.universe, 5, 1)
service['good'] = fuzz.smf(service.universe, 5, 10)

tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# View membership functions
quality['average'].view()
service['poor'].view()
tip['medium'].view()

# Define rules
rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] & quality['good'], tip['high'])

# Control System Creation and Simulation
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Inputs
tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8

# Compute the result
tipping.compute()

# Output
print(tipping.output['tip'])
tip.view(sim=tipping)

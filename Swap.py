import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from solver import BSDESolver
from XvaSolver import XvaSolver
import xvaEquation as eqn
import RecursiveEquation as receqn
import munch
from scipy.stats import norm

 

if __name__ == "__main__":
    dim = 1 #dimension of brownian motion
    P = 2048 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval=252
    schedule_float = np.array([2,50,98])
    schedule_fix = np.array([1,50,99])
    r = 0.02
    a = 0.1
    b = 0.02
    sigma=0.05
    r_init=0.025
    risk_free = 0.01
    config = {
                "eqn_config": {
                    "_comment": "a swap contract",
                    "eqn_name": "Swap",
                    "rate_model": "CIR",
                    "schedule_float": schedule_float,
                    "schedule_fix": schedule_fix,
                    "a": a,
                    "b": b,
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval,
                    "r_fixed":r,
                    "risk_free":risk_free,
                    "sigma":sigma,
                    "r_init":r_init

                },
                "net_config": {
                    "y_init_range": [-5, 5],
                    "num_hiddens": [dim+20, dim+20],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 4000,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
                }
    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    
    #apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()  

    
    #apply trained model to evaluate value of the forward contract via Monte Carlo
    simulations = bsde_solver.model.simulate_path(bsde.sample(P))    
     
    #estimated epected positive and negative exposure
    time_stamp = np.linspace(0,1,num_time_interval+1)
    epe = np.mean(np.exp(-r*time_stamp)*np.maximum(simulations,0),axis=0)    
    ene = np.mean(np.exp(-r*time_stamp)*np.minimum(simulations,0),axis=0)

    plt.figure()
     # plt.plot(time_stamp,[epe_exact[0]]+list(epe_exact),'b--',label='DEPE = exact solution')
    plt.plot(time_stamp,epe[0],'b',label='DEPE = deep solver approximation')

     # plt.plot(time_stamp,[ene_exact[0]]+list(ene_exact),'r--',label='DNPE = exact solution')
    plt.plot(time_stamp,ene[0],'r',label='DNPE = deep solver approximation')

    plt.xlabel('t')
    plt.legend()
    plt.show()
    
# from QuantLib import *
# import matplotlib.pyplot as plt
# import numpy as np

# # Set up QuantLib objects
# calculation_date = Date(1, 1, 2023)
# Settings.instance().evaluationDate = calculation_date

# calendar = UnitedStates()
# settlement_date = calendar.advance(calculation_date, 2, Days)  # Assuming a 2-day settlement delay

# fixed_rate = 0.02

# # Simulate floating rate using CIR process
# a = 0.1
# b = 0.02
# sigma = 0.05
# r0 = 0.025
# T = 1
# dt = 1/252  # Daily time step

# # Simulate CIR process
# time, cir_rates = cir_simulation(a, b, sigma, r0, T, dt)

# # Assuming the simulation covers the swap tenor, you can use the simulated rates directly
# floating_rates = [cir_rates[int(t / dt)] for t in np.arange(0, T, floating_leg_tenor)]

# # Create fixed leg
# fixed_leg_tenor = Period(1, Years)
# fixed_leg_daycount = Actual360()
# fixed_schedule = Schedule(settlement_date, settlement_date + fixed_leg_tenor, fixed_leg_tenor, calendar, Unadjusted, Unadjusted, DateGeneration.Backward, False)
# fixed_leg = FixedRateLeg(fixed_schedule, fixed_leg_daycount, [100.0], [fixed_rate])

# # Create floating leg using simulated rates
# floating_leg_tenor = Period(6, Months)
# floating_leg_daycount = Actual360()
# floating_schedule = Schedule(settlement_date, settlement_date + fixed_leg_tenor, floating_leg_tenor, calendar, Unadjusted, Unadjusted, DateGeneration.Backward, False)
# floating_leg = IborLeg([100.0], floating_schedule, Index(), floating_leg_daycount, 0, [], [], [], floating_rates)

# # Create interest rate swap
# interest_rate_swap = VanillaSwap(VanillaSwap.Payer, fixed_leg, floating_leg)

# # Set up the yield curve
# risk_free_rate = 0.01
# discount_curve = YieldTermStructureHandle(FlatForward(settlement_date, QuoteHandle(SimpleQuote(risk_free_rate)), Actual360()))

# # Set up the pricing engine
# swap_engine = DiscountingSwapEngine(discount_curve)
# interest_rate_swap.setPricingEngine(swap_engine)

# # Get the simulated price
# simulated_price = interest_rate_swap.NPV()

# print("Simulated Price of Interest Rate Swap:", simulated_price)

   # # bsde_solver.model.save('testmodel.tf',save_format='tf')
   
   # # XVA computation step. 
   #  r_f = 0.04
   #  configFVA = {
   #              "eqn_config": {
   #                  "_comment": "XVA on a forward",
   #                  "eqn_name": "FVA",
   #                  "total_time": total_time,
   #                  "num_time_interval": num_time_interval,
   #                  "r":r,
   #                  "r_fl": r_f,
   #                  "r_fb": r_f,
   #                  "r_cl": 0.00,
   #                  "r_cl": 0.00,
   #                  "clean_value": bsde,
   #                  "clean_value_model": bsde_solver.model
   #              },
   #              "net_config": {
   #                  "y_init_range": [-5, 5],
   #                  "num_hiddens": [dim+20, dim+20],
   #                  "lr_values": [5e-2, 5e-3],
   #                  "lr_boundaries": [2000],
   #                  "num_iterations": 4000,
   #                  "batch_size": batch_size,
   #                  "valid_size": 256,
   #                  "logging_frequency": 100,
   #                  "dtype": "float64",
   #                  "verbose": True
   #              }
   #              }
   #  configFVA = munch.munchify(configFVA) 
   #  fvabsde = getattr(receqn, configFVA.eqn_config.eqn_name)(configFVA.eqn_config) 
   #  tf.keras.backend.set_floatx(configFVA.net_config.dtype)
    
   #  #apply algorithm 3
   #  xva_solver = XvaSolver(config, fvabsde)
   #  xva_training_history = xva_solver.train()  
    
   #  fva_simulations = xva_solver.model.simulate_path(fvabsde.sample(P))    
    
   #  print("Exact Values from analytic formulas")
   #  exactVhat = r_init - strike*np.exp(-r * total_time)
   #  exactV = np.exp(-(r_f - r) * total_time)*r_init - strike*np.exp(-r_f * total_time)
   #  exactFVA = exactVhat - exactV
   #  print("exactV = " + str(exactV))
   #  print("exactVhat = " + str(exactVhat))
   #  print("exactFVA = " + str(exactFVA))
    
    
   #  print("FVA from Algorithm 3")
   #  fvaFromSolver = fva_simulations[0,0,0]
   #  print("fvaFromSolver = " +str(fvaFromSolver) )
   #  fvaError = fva_simulations[0,0,0] - exactFVA
   #  print("error = "+ str(fvaError))
    
    


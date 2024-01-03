import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from solver import BSDESolver
import xvaEquation as eqn
import munch
import pandas as pd
import QuantLib as ql

 

if __name__ == "__main__":
    dim = 1 #dimension of brownian motion
    P = 2048 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval = 200
    strike = 100
    r = 0.01
    barrier = 99.9
    sigma = 0.25
    x_init = 100
    config = {
                "eqn_config": {
                    "_comment": "a basket call option",
                    "eqn_name": "BarrierOption",
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval,
                    "strike":strike,
                    "r":r,
                    "sigma":sigma,
                    "x_init":x_init,
                    "strike":strike,
                    "barrier":barrier

                },
                "net_config": {
                    "y_init_range": [9, 11],
                    "num_hiddens": [dim+20, dim+20],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 4000,
                    "batch_size": batch_size,
                    "valid_size": 1024,
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

    #Simulate the BSDE after training - MtM scenarios
    simulations = bsde_solver.model.simulate_path(bsde.sample(P))
    
    #estimated expected positive and negative exposure
    time_stamp = np.linspace(0,1,num_time_interval+1)
    epe = np.mean(np.exp(-r*time_stamp)*np.maximum(simulations,0),axis=0)    
    ene = np.mean(np.exp(-r*time_stamp)*np.minimum(simulations,0),axis=0)
        
        
        
    #exact solution using quantlib
    from QuantLib import *
    # Barrier Option: Up-and-Out Call 
    # Strike 100, Barrier 150, Rebate 50, Exercise date 4 years 
    
    #Set up the global evaluation date to today
    today = Date(23,April,2023)
    Settings.instance().evaluationDate = today
    
    # Specify option
    option = BarrierOption(Barrier.DownOut, 97, 0, PlainVanillaPayoff(Option.Call, 100.0), EuropeanExercise(Date(23, April, 2024)))
    
    # We will now pass the market data: spot price : 100, risk-free rate: 1% and sigma: 30% 
    # Underlying Price
    u = SimpleQuote(100)
    # Risk-free Rate
    r = SimpleQuote(0.01)
    # Sigma 
    sigma = SimpleQuote(0.25)
    
    # Build flat curves and volatility
    riskFreeCurve = FlatForward(0, TARGET(), QuoteHandle(r), Actual360())
    volatility = BlackConstantVol(0, TARGET(), QuoteHandle(sigma), Actual360())
    
    # Model and Pricing Engine
    # Build the pricing engine by encapsulating the market data in a Black-Scholes process
    
    # Stochastic Process
    process = BlackScholesProcess(QuoteHandle(u), YieldTermStructureHandle(riskFreeCurve), BlackVolTermStructureHandle(volatility))
    
    # Build the engine (based on an analytic formula) and set it to the option for evaluation
    option.setPricingEngine(AnalyticBarrierEngine(process))
    
    # Market Data Changes
    # Change the market data to get new option pricing. 
    
    # Set initial value and define h
    u0 = u.value(); h=0.01
    P0 = option.NPV()

    
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    time_stamp = np.linspace(0,1,201)
    
    epe_exact = np.array([P0 for s in time_stamp[0:]])
    ene_exact = np.array([0.0 for s in time_stamp[0:]])

    
    
    
        
    fig = plt.figure()
    plt.plot(time_stamp,list(epe_exact),'b--',label='DEPE = exact solution',)
    epe_1 = epe.squeeze()
    plt.plot(time_stamp,epe_1,'b',label='DEPE = deep solver approximation')

    plt.plot(time_stamp,list(ene_exact),'r--',label='DNPE = exact solution',)
    ene_1 = ene.squeeze()
    plt.plot(time_stamp,ene_1,'r',label='DNPE = deep solver approximation')

    plt.xlabel('t')
    plt.legend()

    plt.show()
    fig.savefig(config.eqn_config.eqn_name + '.pdf',format = 'pdf')
    
    df = pd.DataFrame(simulations[:,0,:])
    filepath = 'exposure' + config.eqn_config.eqn_name + '.xlsx'
    df.to_excel(filepath, index=False)

"""
Test implementation involving 'illegal' assignment based on structural unsuitability
Integrate in to main module
"""

from flask import Flask
import ghhops_server as hs

import rhino3dm
import numpy as np
import scipy.optimize
import time

from mip import Model, xsum, BINARY, INTEGER, minimize, maximize

app = Flask(__name__)
hops = hs.Hops(app)

print_variables = False
    
    
@hops.component(
    "/solve",
    name="Solve",
    description="Solve the ILP Problem",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsString("Method", "M", "Method"),
        hs.HopsNumber("Stock", "S", "Stock Lengths", hs.HopsParamAccess.LIST),
        hs.HopsNumber("PartLengths","P","Part Lengths", hs.HopsParamAccess.LIST),
        hs.HopsNumber("PartCounts","C","Part Counts", hs.HopsParamAccess.LIST),
        hs.HopsNumber("PartRejections","R","Part Rejections", hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsNumber("Selection","S","Solved Result", hs.HopsParamAccess.LIST)
    ],
)
def solve(method,stock_lengths, part_lengths, part_requests, part_rejections):
    """
    Entry point function from hops
    """
    time_start = time.time()
    model = Model()
    model.max_mip_gap_abs = 1.5
    model.max_mip_gap = .1
    #model.cuts = 3
    if method == "default":
        solve_function = solve_default
    elif method == "waste":
        solve_function = solve_waste
    elif method == "max":
        solve_function = solve_max
        
    model = solve_function(model,stock_lengths, part_lengths, part_requests, part_rejections)
    
    # optimizing the model
    model.optimize(max_nodes=10000, max_seconds=30)

    # printing the solution
    print('')
    print('Objective value: {model.objective_value:.3}'.format(**locals()))
    print('Solution: ', end='')
    
    if print_variables:
        for v in model.vars:
            if v.x > 1e-5:
                print('{v.name} = {v.x}'.format(**locals()))
                print('          ', end='')    
                
    print(model.objective_value)
    print(model.objective_bound)
            
            
    output = [float(v.x) for v in model.vars]
    
    time_end = time.time()
    time_elapsed = round(time_end - time_start,3)
    
    
    log_string = method
    log_string += f" stock_{len(stock_lengths)}"
    log_string += f" part_types_{len(part_lengths)}"
    log_string += f" part_total_{sum(part_requests)}"
    log_string += f" time_{time_elapsed}"
    log_string += "\n"
    
    with open ("log.txt","a") as f:
        f.write(log_string)
    
    
    return output
    
def solve_default(model, stock_lengths, part_lengths, part_requests, part_rejections):
    """
    Strategy most similar to stock cutting example. 
    Built using loops and explicit utilization boolean variables
    Trends towards utilizing largest stock first
    """
    part_lengths = np.array(part_lengths)
    part_count = len(part_lengths)

    part_requests = np.array(part_requests)

    stock_lengths = np.array(stock_lengths)
    stock_count = len(stock_lengths)
    
    max_parts = np.max(part_requests)
    
    # Amount of each part used in that piece
    part_usage = {(i, j): model.add_var(obj=0, var_type=INTEGER, name="part_usage[%d,%d]" % (i, j), lb=0, ub=max_parts)
         for i in range(part_count) for j in range(stock_count)}
    # Whether the piece is used
    stock_usage = {j: model.add_var(obj=1, var_type=BINARY, name="stock_usage[%d]" % j)
         for j in range(stock_count)}

    # Constraints
    # Ensure enough parts are produced
    for i in range(part_count):
        model.add_constr(xsum(part_usage[i, j] for j in range(stock_count)) == part_requests[i])
    # Ensure the used amount of the bar is <= the usable amount of the bar (0 if unused)
    for j in range(stock_count):
        model.add_constr(xsum(part_lengths[i] * part_usage[i, j] for i in range(part_count)) <= stock_lengths[j] * stock_usage[j])
        

    # additional constraints to reduce symmetry
    # Put unused bars at end of list (reduces search space)
    # Not appropriate for this type
    #for j in range(1, stock_count):
    #    model.add_constr(stock_usage[j - 1] >= stock_usage[j])
        
    model.objective = minimize(xsum(stock_usage[i] for i in range(stock_count)))

    return(model)
    
# Part rejections    : 1 Can't Use, 1000 Can Use
# min(1, part_usage) : 0 Not using, 1 Is Using
# 1- above           : 1 Not using, 0 Is using
def solve_waste(model, stock_lengths, part_lengths, part_requests, part_rejections):
    """
    Optimizes for minimizing waste from used piecess
    Does not attempt leftover usability
    """
    part_lengths = np.array(part_lengths)
    part_count = len(part_lengths)

    part_requests = np.array(part_requests)
    
    stock_lengths = np.array(stock_lengths)
    stock_count = len(stock_lengths)
    
    part_rejections = np.array(part_rejections).reshape( (stock_count,part_count) )
    part_rejections = part_rejections.astype(int)
    
    print(part_rejections)
    
    smallest_part = np.min(part_lengths)
    longest_stock = np.max(stock_lengths)
    max_part_guess = int(longest_stock / smallest_part)
    
    max_parts = np.max(part_requests)

    # Variable : Amount of each part used in that piece
    part_usage = {(i, j): model.add_var(var_type=INTEGER, 
                                        name="part_usage[%d,%d]" % (i, j), 
                                        lb=0, 
                                        ub=max_part_guess)
                          for i in range(part_count) for j in range(stock_count)}
    # Variable : Whether the stock piece is used
    stock_usage = {j: model.add_var(var_type=BINARY, 
                                    name="stock_usage[%d]" % j)
                      for j in range(stock_count)}

    # Constraint : Ensure enough parts are produced
    for i in range(part_count):
        model.add_constr(xsum(part_usage[i, j] for j in range(stock_count)) 
                         == 
                         part_requests[i])
    # Constraint : Ensure the used amount of the bar is <= the usable amount of the bar (0 if unused)
    # Note, the multiplication by stock_usage here prevents a var/var multiplication in the objective
    for j in range(stock_count):
        model.add_constr(xsum(part_lengths[i] * part_usage[i, j] for i in range(part_count)) 
                         <=
                         stock_lengths[j] * stock_usage[j])
                         
      
    # Account for structural rejections
    for j in range(stock_count):
        for i in range(part_count):
            model.add_constr( part_usage[i, j] <= part_rejections[j,i])
    
        
    model.objective = minimize(xsum((stock_lengths[j] * stock_usage[j]) -
                                    xsum((part_lengths[i] * part_usage[i, j]) for i in range(part_count))
                                    for j in range(stock_count)))

    return(model)
    
    
def solve_max(model, stock_lengths, part_lengths, part_requests, part_rejections):
    """
    Ignores the utilized variable, tries to optimize the square of leftovers 
    Uses some SOS nonsense
    """
    part_lengths = np.array(part_lengths)
    part_count = len(part_lengths)

    part_requests = np.array(part_requests)

    stock_lengths = np.array(stock_lengths)
    stock_count = len(stock_lengths)
    
    max_parts = np.max(part_requests)
    largest_stock = np.max(stock_lengths)
 
    # Amount of each part used in that piece
    part_usage = {(i, j): model.add_var(var_type=INTEGER, name="part_usage[%d,%d]" % (i, j), lb=0, ub=max_parts)
         for i in range(part_count) for j in range(stock_count)}

    # Constraint : Ensure enough parts are produced
    for i in range(part_count):
        model.add_constr(xsum(part_usage[i, j] for j in range(stock_count)) 
                         == 
                         part_requests[i])
    # Constraint : Ensure the used amount of the bar is <= the usable amount of the bar
    for j in range(stock_count):
        model.add_lazy_constr(xsum(part_lengths[i] * part_usage[i, j] for i in range(part_count)) 
                              <= 
                              stock_lengths[j])
    
    # Create the nonlinear function for the objective
    score = [model.add_var(f"score_{i}") for i in range(stock_count)]
    for j in range(stock_count):
        d_count = 6
        v = [stock_lengths[j] * (v / (d_count - 1)) for v in range(d_count)] # X values for pow function
        vn = [pow(stock_lengths[j] - v[n],2) for n in range(d_count)]
        w = [model.add_var(f"w_{j}_{v}") for v in range(d_count)]
        model.add_constr(xsum(w) == 1)
        
        model.add_constr(xsum((part_lengths[i] * part_usage[i, j])
                                                 for i in range(part_count))
                         ==
                         xsum(v[k] * w[k] for k in range(d_count)))
        model.add_constr(score[j] == xsum(vn[k] * w[k] for k in range(d_count)))
        model.add_sos([(w[k],v[k]) for k in range(d_count)],2)
    
    
        
    model.objective = maximize(xsum(score[i] for i in range(stock_count)))

    return(model)
    
@hops.component(
    "/fit_curve",
    name="Fit Curve",
    description="Fit a curve to the data",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsNumber("XValues","X","X Values", hs.HopsParamAccess.LIST),
        hs.HopsNumber("YValues","Y","Y Values", hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsNumber("Values","V","Solved Values", hs.HopsParamAccess.LIST)
    ],
)
def solve(x,y):
    popt, _ = scipy.optimize.curve_fit(objective, x, y)
    a, b, c = popt
    return list(popt)


def objective(x,a,b,c):
    return (a * x) + (b * (x**2)) + c


if __name__ == "__main__":
    app.run()
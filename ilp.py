import json
import math
import mip
import multiprocessing
import numpy as np
import psutil
import time

from mip import Model, xsum, BINARY, INTEGER, minimize, maximize, OptimizationStatus

debug_print_variables = False


def solve_ilp(method, stock_lengths, part_lengths, part_requests, model_args=None) -> tuple[mip.OptimizationStatus, list[int]]:
    """
    Solves a material matching problem using integer linear programming
    :param str method: Goal definition used. Currently, may be 'default' (minimize stock pieces), 'waste' (minimize
    stock cutoff), or 'max' (maximize contiguous remaining length)
    :param [float] stock_lengths: The length of each member of the currently considered stock
    :param [float] part_lengths: The length of each unique part type used in the design
    :param [int] part_requests: The number of each unique part type required by the design
    :param dict model_args: Pass tuning parameters directly to the model. Currently relevant for testing model
    efficiency tests

    """
    if model_args is None:
        model_args = {}

    print(f"Method : {method}")
    print(f"{len(stock_lengths)} stock pieces between {np.min(stock_lengths)} and {np.max(stock_lengths)}")
    print(f"{len(part_lengths)} part types between {np.min(part_lengths)} and {np.max(part_lengths)}")
    print(f"{sum(part_requests)} total part requests")
    time_start = time.time()
    model = Model()
    model.max_mip_gap_abs = 1.5
    model.max_mip_gap = .1

    if "save_scenario" in model_args and model_args["save_scenario"]:
        output_dict = {"method": method,
                       "stock_lengths": stock_lengths,
                       "part_lengths": part_lengths,
                       "part_requests": part_requests}
        with open("scenarios/scenario.json", 'w') as f:
            json.dump(output_dict, f)

    if "clique" in model_args:
        model.clique = int(model_args["clique"])

    # 0 (No cuts) allows for finding feasible solutions within the 30-second window
    if "cuts" in model_args:
        model.cuts = int(model_args["cuts"])
    else:
        model.cuts = 0

    if "emphasis" in model_args:
        model.emphasis = int(model_args["emphasis"])
    else:
        model.emphasis = 1

    if "lp_method" in model_args:
        model.lp_method = int(model_args["lp_method"])

    # Improves speed
    if "preprocess" in model_args:
        model.preprocess = int(model_args["preprocess"])
    else:
        model.preprocess = 1

    if method == "default":
        solve_function = _solve_default
    elif method == "waste":
        solve_function = _solve_waste
    elif method == "max":
        solve_function = _solve_max
    else:
        print(f"Bad method argument '{method}'")
        return OptimizationStatus.NO_SOLUTION_FOUND, [0]

    model = solve_function(model, stock_lengths, part_lengths, part_requests)
    model.threads = -1

    max_nodes = 10000
    if "max_nodes" in model_args:
        max_nodes = int(model_args["max_nodes"])

    max_seconds = 30
    if "max_seconds" in model_args:
        max_seconds = int(model_args["max_seconds"])

    # optimizing the model
    status: OptimizationStatus = model.optimize(max_nodes=max_nodes, max_seconds=max_seconds)

    print('')
    print(f"Optimization Status : {status}")

    if status == OptimizationStatus.INFEASIBLE or status == OptimizationStatus.NO_SOLUTION_FOUND:
        return status, [0]

    # printing the solution

    print('Objective value: {model.objective_value:.3}'.format(**locals()))
    print('Solution: ', end='')

    if debug_print_variables:
        for v in model.vars:
            if v.x > 1e-5:
                print('{v.name} = {v.x}'.format(**locals()))
                print('          ', end='')

    print(model.objective_value)
    print(model.objective_bound)

    output = [float(v.x) for v in model.vars]

    time_end = time.time()
    time_elapsed = round(time_end - time_start, 3)

    """
    CSV Record:
    Context
    Max threads : multiprocessing.cpu_count()
    Memory psutil.virtual_memory()[0] / 1024^3
    CPU: psutil.cpu_freq()[0]

    Parameters: 288 variations
    0 clique -1 (automatic) 0 (none) 1 (enabled) 2(aggressive)
    1 cuts : -1 (automatic) 0( disabled) 1(moderate) 2 (aggressive) 3(most)
    2 lp_method: 0: auto, 1 (dual) 2(primal) 3(barrier)
    3 emphasis: 0 (balance), 1( feasibility), or 2(optimality)
    4 preprocess: -1 (automatic) 0 (off) 1( on)


    Output:
    Time (seconds)
    state( integer flag)
    Objective Value (float)
    """

    # log_string = "threads, mem, cpu, clique, cuts, lp_method, emphasis, preprocess, goal_method, state, objective, time"
    log_string = str(model_args["id"]) + ","
    log_string += str(multiprocessing.cpu_count()) + ","
    log_string += str(round(psutil.virtual_memory()[0] / math.pow(1024, 3), 2)) + ","
    log_string += str(psutil.cpu_freq()[0]) + ","

    log_string += (str(model_args["clique"]) if "clique" in model_args else "") + ","
    log_string += (str(model_args["cuts"]) if "cuts" in model_args else "") + ","
    log_string += (str(model_args["emphasis"]) if "emphasis" in model_args else "") + ","
    log_string += (str(model_args["lp_method"]) if "lp_method" in model_args else "") + ","
    log_string += (str(model_args["preprocess"]) if "preprocess" in model_args else "") + ","

    log_string += method + ","

    log_string += str(status) + ","
    log_string += str(round(model.objective_value, 3)) + ","
    log_string += str(time_elapsed) + "\n"

    log_filepath = model_args["log_filepath"] if "log_filepath" in model_args else "log.csv"
    print(log_filepath)
    with open(log_filepath, "a") as f:
        f.write(log_string)

    return status, output


def _solve_default(model, stock_lengths, part_lengths, part_requests):
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
    # Ensure the used amount of the stock piece is <= the usable amount of the stock piece (0 if unused)
    for j in range(stock_count):
        model.add_constr(
            xsum(part_lengths[i] * part_usage[i, j] for i in range(part_count)) <= stock_lengths[j] * stock_usage[j])

    # additional constraints to reduce symmetry
    # Put unused bars at end of list (reduces search space)
    # Not appropriate for problems with different stock lengths
    # for j in range(1, stock_count):
    #    model.add_constr(stock_usage[j - 1] >= stock_usage[j])

    model.objective = minimize(xsum(stock_usage[i] for i in range(stock_count)))

    return model


def _solve_waste(model, stock_lengths, part_lengths, part_requests):
    """
    Optimizes for minimizing waste from used pieces
    Does not attempt leftover usability
    """
    part_lengths = np.array(part_lengths)
    part_count = len(part_lengths)

    part_requests = np.array(part_requests)

    stock_lengths = np.array(stock_lengths)
    stock_count = len(stock_lengths)

    # Variable : Amount of each part used in that piece
    # Regarding the upper bound (ub) here:
    # - Simplest : maximum of all part requests
    # - Stock aware : max count is the floor of the longest stock divided by the smallest part
    # - Item aware : The floor of the current stock length divided by the current part length
    part_usage = {(i, j): model.add_var(var_type=INTEGER,
                                        name="part_usage[%d,%d]" % (i, j),
                                        lb=0,
                                        ub=int(stock_lengths[j] / part_lengths[i]))
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

    print(f"Model created with {len(model.vars)} variables and {len(model.constrs)} constraints")

    model.objective = minimize(xsum((stock_lengths[j] * stock_usage[j]) -
                                    xsum((part_lengths[i] * part_usage[i, j]) for i in range(part_count))
                                    for j in range(stock_count)))

    return model


def _solve_max(model, stock_lengths, part_lengths, part_requests):
    """
    Ignores the utilized variable, tries to optimize the square of leftovers Because we're trying to maximize a
    convex function, uses special-ordered-sets to approximate the function with linear segments
    See :
    https://python-mip.readthedocs.io/en/latest/sos.html
    https://python-mip.readthedocs.io/en/latest/examples.html#exsos
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
        v = [stock_lengths[j] * (v / (d_count - 1)) for v in range(d_count)]  # X values for pow function
        vn = [pow(stock_lengths[j] - v[n], 2) for n in range(d_count)]
        w = [model.add_var(f"w_{j}_{v}") for v in range(d_count)]
        model.add_constr(xsum(w) == 1)

        model.add_constr(xsum((part_lengths[i] * part_usage[i, j])
                              for i in range(part_count))
                         ==
                         xsum(v[k] * w[k] for k in range(d_count)))
        model.add_constr(score[j] == xsum(vn[k] * w[k] for k in range(d_count)))
        model.add_sos([(w[k], v[k]) for k in range(d_count)], 2)

    model.objective = maximize(xsum(score[i] for i in range(stock_count)))

    return model
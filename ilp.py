"""
Implementation of 1D Cutting Stock Problem using Integer Linear Programming
Agnostic to caller
"""

import json
import math
import mip
import multiprocessing
import numpy as np
import psutil
import time

from mip import Model, xsum, BINARY, INTEGER, minimize, maximize, OptimizationStatus

debug_print_variables = False


def solve_ilp(method, stock_lengths, part_lengths, part_requests, model_args=None) -> tuple[
    mip.OptimizationStatus, list[int], str]:
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

    log_filepath = model_args["log_filepath"] if "log_filepath" in model_args else "log.csv"
    log_string = ""

    print(f"Method : {method}")
    print(f"{len(stock_lengths)} stock pieces between {np.min(stock_lengths)} and {np.max(stock_lengths)}")
    print(f"{len(part_lengths)} part types between {np.min(part_lengths)} and {np.max(part_lengths)}")
    print(f"{sum(part_requests)} total part requests")
    print(f"Log filepath at : {log_filepath}")
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
    elif method == "order":
        solve_function = _solve_order
    elif method == "homogenous":
        solve_function = _solve_homogenous
    else:
        print(f"Bad method argument '{method}'")
        log_string = f"{(str(model_args['id']) if 'id' in model_args else 'no_id')},{mip.OptimizationStatus.ERROR},{-1},{0}"

        log_line(log_string, log_filepath)
        return OptimizationStatus.NO_SOLUTION_FOUND, [0], log_string

    model, extra = solve_function(model, stock_lengths, part_lengths, part_requests)
    model.threads = -1

    max_nodes = 10001
    if "max_nodes" in model_args:
        max_nodes = int(model_args["max_nodes"])
    print(f"Max Nodes : {max_nodes}")

    max_seconds = 30
    if "max_seconds" in model_args:
        max_seconds = int(model_args["max_seconds"])

    max_nodes_same_incumbent = 1073741824
    if "max_nodes_same_incumbent" in model_args:
        max_nodes_same_incumbent = int(model_args["max_nodes_same_incumbent"])

    max_seconds_same_incumbent = float('inf')
    if "max_seconds_same_incumbent" in model_args:
        max_seconds_same_incumbent = int(model_args["max_seconds_same_incumbent"])

    # optimizing the model
    status: OptimizationStatus = model.optimize(max_nodes=max_nodes,
                                                max_seconds=max_seconds,
                                                max_nodes_same_incumbent=max_nodes_same_incumbent,
                                                max_seconds_same_incumbent=max_seconds_same_incumbent)

    time_end = time.time()
    time_elapsed = round(time_end - time_start, 3)

    print('')
    print(f"Optimization Status : {status}")

    if status == OptimizationStatus.INFEASIBLE or status == OptimizationStatus.NO_SOLUTION_FOUND or status == OptimizationStatus.ERROR:
        log_string = f"{(str(model_args['id']) if 'id' in model_args else 'no_id')},{status},0,{time_elapsed}"
        log_line(log_string, log_filepath)
        return status, [0], log_string

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
    log_string = (str(model_args["id"]) if "id" in model_args else "no_id") + ","
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

    value_waste = -1
    value_score = -1

    waste_total = 0
    score_total = 0

    # Reconstructing objectives
    if method in ["default", "waste", "max", "homogenous"]:
        part_count = len(part_lengths)
        stock_count = len(stock_lengths)
        x = np.array([float(n) for n in model.vars[0:len(stock_lengths) * part_count]])
        x = x.reshape([part_count, stock_count])
        x = x.transpose()
        x = x.flatten()

        if method == 'homogenous' or method == 'max':
            x2 = np.array([float(n) for n in model.vars[0:len(stock_lengths) * part_count]]).reshape(
                [part_count, stock_count])
            y = np.sum(x2, axis=0) > 0
        else:
            y = np.array([float(n) for n in model.vars[len(x):len(x) + stock_count]])

        np_stock = np.array([float(n) for n in stock_lengths])
        np_part_lengths = np.array([float(n) for n in part_lengths])

        for i in range(stock_count):
            usage = np.sum(x[i * part_count: (i + 1) * part_count] * np_part_lengths) * y[i]
            available = stock_lengths[i] * y[i]
            waste_total += available - usage

            score_total += (stock_lengths[i] - usage) ** 2
    elif method == 'order':
        waste_values = extra

        output = np.array(output).reshape((len(part_lengths), len(stock_lengths)))
        output = output.transpose()

        waste_total = np.sum(waste_values * output)

        waste_array = np.sum(waste_values * output, axis=1)
        leftover_array = np.maximum(waste_array, np.array(stock_lengths) * (1 - waste_array))
        leftover_array = leftover_array * leftover_array

        score_total = leftover_array.sum()

    # Simplified log
    log_string = f"{(str(model_args['id']) if 'id' in model_args else 'no_id')},{status},{round(model.objective_value, 3)},{time_elapsed},{waste_total},{score_total}"

    log_line(log_string, log_filepath)

    return status, output, log_string


def log_line(s, log_filepath):
    s += "\n"
    with open(log_filepath, "a") as f:
        f.write(s)


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

    # Amount of each part used in that piece
    part_usage = {(i, j): model.add_var(obj=0,
                                        var_type=INTEGER,
                                        name="part_usage[%d,%d]" % (i, j),
                                        lb=0,
                                        ub=int(stock_lengths[j] / part_lengths[i]))
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

    return model, None


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

    return model, None


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

    # Amount of each part used in that piece
    part_usage = {(i, j): model.add_var(var_type=INTEGER,
                                        name="part_usage[%d,%d]" % (i, j),
                                        lb=0,
                                        ub=int(stock_lengths[j] / part_lengths[i]))
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

    return model, None


def _solve_homogenous(model, stock_lengths, part_lengths, part_requests):
    """
    Optimizes for using the fewest number of unique part types in each stock type
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

    # Variable : Whether each part type appears in a particular stock piece
    part_type_usage = {(i, j): model.add_var(var_type=BINARY,
                                             name="part_type_usage[%d,%d]" % (i, j))
                       for i in range(part_count) for j in range(stock_count)}

    M = 1000
    # Constraint : Align 'part_type_usage' to refer to 'part_usage'
    for i in range(part_count):
        for j in range(stock_count):
            model.add_constr(part_usage[i, j] >= 1 - (M * (1 - part_type_usage[i, j])))
            model.add_constr(part_usage[i, j] <= M * part_type_usage[i, j])

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
                         stock_lengths[j])

    print(f"Model created with {len(model.vars)} variables and {len(model.constrs)} constraints")

    model.objective = minimize(xsum(xsum(part_type_usage[i, j] for i in range(part_count))
                                    for j in range(stock_count)))

    return model, None


def _solve_order(model, stock_lengths, part_lengths, _):
    """
    Optimizes for minimizing waste from used pieces
    Does not attempt leftover usability
    """
    part_lengths = np.array(part_lengths)
    part_count = len(part_lengths)

    stock_lengths = np.array(stock_lengths)
    stock_count = len(stock_lengths)

    # Helper constant : If stock j starts at part i, it will be able to cover a total of n parts
    stock_extents = []
    stock_wastes = []
    for j in range(stock_count):
        stock_extents.append([])
        stock_wastes.append([])
        for i in range(part_count):
            extent = 0
            countdown = stock_lengths[j]
            while (i + extent) < len(part_lengths) and countdown >= part_lengths[i + extent]:
                countdown -= part_lengths[i + extent]
                extent += 1
            stock_wastes[-1].append(countdown)
            stock_extents[-1].append(extent)
    stock_extents = np.array(stock_extents)
    stock_wastes = np.array(stock_wastes)
    #print(stock_extents)
    #print(stock_wastes)

    # Variable : Boolean whether a particular stock starts at a particular part
    stock_start = {(i, j): model.add_var(var_type=BINARY, name=f"stock_start[{i},{j}]")
                   for i in range(part_count) for j in range(stock_count)}

    # Constraint : Each stock piece can start at most one part
    for i in range(part_count):
        model.add_constr(xsum(stock_start[i, j] for j in range(stock_count)) <= 1)

    # Constraint : Each part can start at most once
    for j in range(stock_count):
        model.add_constr(xsum(stock_start[i, j] for i in range(part_count)) <= 1)

    # Constraint : The first part must be a start
    model.add_constr(xsum(stock_start[0, j] for j in range(stock_count)) == 1)

    # Constraint : Every stock start must come at the end of another
    for i in range(1, part_count):
        model.add_constr(i * xsum(stock_start[i, j] for j in range(stock_count))
                         <=
                         xsum(xsum(stock_start[i2, j] * stock_extents[j, i2] for i2 in range(i)) for j in
                              range(stock_count)))
        model.add_constr(i + ((1 - xsum(stock_start[i, j] for j in range(stock_count))) * part_count)
                         >=
                         xsum(xsum(stock_start[i2, j] * stock_extents[j, i2] for i2 in range(i)) for j in
                              range(stock_count)))

    # Constraint : Utilized stock must cover all parts
    model.add_constr(xsum(xsum(stock_start[i, j] * stock_extents[j, i] for i in range(part_count)) for j in
                          range(stock_count))
                     ==
                     part_count)

    print(f"Model created with {len(model.vars)} variables and {len(model.constrs)} constraints")

    # model.objective = minimize(xsum(xsum(stock_start[i, j] for i in range(part_count)) for j in range(stock_count)))
    model.objective = minimize(
        xsum(xsum(stock_start[i, j] * stock_wastes[j, i] for i in range(part_count)) for j in range(stock_count)))

    print()

    return model, stock_wastes


def _demo_homogenous():
    stock_lengths = [10, 12, 16, 14, 14, 30, 12]
    part_lengths = [2, 4, 6, 8]
    part_requests = [4, 3, 5, 3]

    model = Model()
    model.preprocess = 1
    model = _solve_homogenous(model, stock_lengths, part_lengths, part_requests)
    status: OptimizationStatus = model.optimize()

    print('')
    print(f"Optimization Status : {status}")

    output = np.array([float(v.x) for v in model.vars])

    offset = len(part_lengths) * len(stock_lengths)
    shape = (len(part_lengths), len(stock_lengths))
    output_x = output[0:offset].reshape(shape)
    print("Output X:")
    print(output_x)

    output_s = output[offset:offset * 2].reshape(shape)
    print("Output S:")
    print(output_s)

    y = np.sum(output_x, axis=0) > 0
    print(y)

    print("Objective: ")
    print(model.objective_value)


def _demo_order():
    stock_lengths = [10, 12, 16, 14, 14, 30, 12]
    part_lengths = [5, 8, 4, 2, 1, 2, 4, 6, 5, 5, 5, 5, 5, 1, 3, 3, 4]

    model = Model()
    model.preprocess = 1
    model, extra = _solve_order(model, stock_lengths, part_lengths, None)
    status: OptimizationStatus = model.optimize()

    print('')
    print(f"Optimization Status : {status}")

    output = [float(v.x) for v in model.vars]
    output = np.array(output).reshape((len(part_lengths), len(stock_lengths)))
    output = output.transpose()
    print("Output:")
    print(output)
    print("Objective: ")
    print(model.objective_value)

    print('Waste matrix: ')
    print(extra)

    waste_array = np.sum(output * extra, axis = 1)
    leftover_array = np.maximum(waste_array, np.array(stock_lengths) * (1-waste_array) )
    leftover_array = leftover_array * leftover_array

    print(waste_array)
    print(leftover_array)
    print(leftover_array.sum())


if __name__ == "__main__":
    _demo_order()
    # _demo_homogenous()

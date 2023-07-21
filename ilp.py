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
from typing import Optional


def solve_ilp(method, stock_lengths, part_lengths, part_requests, model_args=None) -> tuple[
    mip.OptimizationStatus, list[float], str]:
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

    if method == "order_split":
        return solve_ilp_order_split(method, stock_lengths, part_lengths, model_args)

    log_filepath = model_args.get("log_filepath", "log.csv")
    log_string = ""

    print(f"Method : {method}")
    print(f"{len(stock_lengths)} stock pieces between {np.min(stock_lengths)} and {np.max(stock_lengths)}")
    print(f"{len(part_lengths)} part types between {np.min(part_lengths)} and {np.max(part_lengths)}")
    print(f"{sum(part_requests)} total part requests")
    print(f"Log filepath at : {log_filepath}")
    time_start = time.time()

    model = _get_basic_model(model_args)

    # Export the scenario if necessary
    if "save_scenario" in model_args and model_args["save_scenario"]:
        output_dict = {"method": method,
                       "stock_lengths": stock_lengths,
                       "part_lengths": part_lengths,
                       "part_requests": part_requests}
        with open("scenarios/scenario.json", 'w') as f:
            json.dump(output_dict, f)

    # Setup the model specifics
    if method == "default":
        solve_function = _model_default
    elif method == "waste":
        solve_function = _model_waste
    elif method == "max":
        solve_function = _model_max
    elif method == "order":
        solve_function = _model_order
        part_lengths = np.repeat(part_lengths, part_requests, axis=0)
        print(f"Part count new : {len(part_lengths)}")
    elif method == "homogenous":
        solve_function = _model_homogenous
    else:
        print(f"Bad method argument '{method}'")
        log_string = f"{(str(model_args['id']) if 'id' in model_args else 'no_id')},{mip.OptimizationStatus.ERROR},{-1},{0}"
        log_line(log_string, log_filepath)
        return OptimizationStatus.NO_SOLUTION_FOUND, [0], log_string

    model, stock_wastes, stock_extents = solve_function(model, stock_lengths, part_lengths, part_requests)

    max_nodes = int(model_args.get("max_nodes", 10000))
    max_seconds = int(model_args.get("max_seconds", 30))
    max_nodes_same_incumbent = int(model_args.get("max_nodes_same_incumbent", 1073741824))
    max_seconds_same_incumbent = int(model_args.get("max_seconds_same_incumbent", 10000000))

    # Attempt to optimize the model
    status: OptimizationStatus = model.optimize(max_nodes=max_nodes,
                                                max_seconds=max_seconds,
                                                max_nodes_same_incumbent=max_nodes_same_incumbent,
                                                max_seconds_same_incumbent=max_seconds_same_incumbent)

    time_end = time.time()
    time_elapsed = round(time_end - time_start, 3)

    print('')
    print(f"Optimization Status : {status}")

    # Return if not at least feasible
    if status in [OptimizationStatus.INFEASIBLE, OptimizationStatus.NO_SOLUTION_FOUND, OptimizationStatus.ERROR, OptimizationStatus.UNBOUNDED]:
        log_string = f"{(str(model_args['id']) if 'id' in model_args else 'no_id')},{status},0,{time_elapsed}"
        log_line(log_string, log_filepath)
        return status, [0], log_string

    # Print solution
    print('Objective value: {model.objective_value:.3}'.format(**locals()))
    print('Objective bound: {model.objective_bound:.3}'.format(**locals()))
    print('Solution: ', end='')

    # Extract variables
    output = [float(v.x) for v in model.vars]

    waste_total = 0
    score_total = 0

    # Reconstruct objectives from variables
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
        usage = np.array(output).reshape((len(part_lengths), len(stock_lengths)))
        usage = usage.transpose()
        usage_array = usage.sum(axis=1)

        waste_total = np.sum(stock_wastes * usage)

        waste_array = np.sum(stock_wastes * usage, axis=1)
        leftover_array = np.maximum(waste_array, np.array(stock_lengths) * (1 - usage_array))
        leftover_array = leftover_array * leftover_array

        score_total = leftover_array.sum()

    # Calculate 'best' possible score
    best_total = 0
    part_lengths_np = np.array([float(n) for n in part_lengths])
    part_requests_np = np.array([float(n) for n in part_requests])
    stock_np = np.array([float(n) for n in stock_lengths])
    stock_np.sort()
    design_sum = (part_lengths_np * part_requests_np).sum()
    for stock in stock_np:
        if design_sum <= 0:
            best_total += stock ** 2
        elif design_sum >= stock:
            design_sum -= stock
        elif design_sum < stock:
            stock -= design_sum
            best_total += stock ** 2
            design_sum = 0

    # Scale the output score
    score_total = score_total / best_total

    # Assemble standard log line
    # id, status, objective value, waste value, score value, total time, last improved time
    log_id = str(model_args['id']) if 'id' in model_args else 'no_id'
    log_objective_value = round(model.objective_value, 3)
    log_improvement_time = _progress_log_last_time(model)

    if method == 'max':
        log_objective_value = score_total

    log_string = f"{log_id},{status},{log_objective_value},{waste_total},{score_total},{time_elapsed},{log_improvement_time}"
    log_line(log_string, log_filepath)

    return status, output, log_string


def solve_ilp_order_split(method, stock_lengths, part_lengths, model_args=None) -> tuple[
    mip.OptimizationStatus, list[float], str]:
    """
    Note : in scenarios close to the feasible edge, the models integer tolerance was causing issues
    (e.g. was returning values of 0.999999 which was messing with np.where. Remember to use np.rint on all relevant output)
    """
    stock_lengths = np.array(stock_lengths)

    log_filepath = model_args.get("log_filepath", "log.csv")
    log_id = str(model_args.get('id', 'no_id'))

    print(f"Method : {method}")
    print(f"{len(stock_lengths)} stock pieces between {np.min(stock_lengths)} and {np.max(stock_lengths)}")
    print(f"{len(part_lengths)} part types between {np.min(part_lengths)} and {np.max(part_lengths)}")
    print(f"Log filepath at : {log_filepath}")

    max_seconds = int(model_args.get("max_seconds", 30))

    final_assignment = np.repeat(-1, len(stock_lengths))

    split_point = len(part_lengths) // 2
    part_lengths_a = part_lengths[:split_point]
    part_lengths_b = part_lengths[split_point:]

    print(f"Parts A Shape : {len(part_lengths_a)}")
    print(f"Parts B Shape : {len(part_lengths_b)}")

    time_start = time.time()

    # Run model A
    model_a: mip.Model = _get_basic_model(model_args)
    model_a, wastes_a, extents_a = _model_order(model_a, stock_lengths, part_lengths_a, None)
    model_a.max_seconds = max_seconds
    status_a: OptimizationStatus = model_a.optimize()

    # Return if not at least feasible
    if status_a in [OptimizationStatus.INFEASIBLE, OptimizationStatus.NO_SOLUTION_FOUND, OptimizationStatus.ERROR]:
        elapsed_time = time.time() - time_start
        log_string = f"{log_id},{status_a},0,0,0,{elapsed_time},{elapsed_time}"
        log_line(log_string, log_filepath)
        return status_a, [0], log_string

    # Get output A
    output_a = [float(v.x) for v in model_a.vars]
    output_a = np.array(output_a).reshape((len(part_lengths_a), len(stock_lengths)))
    output_a = output_a.transpose()
    output_a = np.rint(output_a)

    # Fill the final assignment array for the stock pieces used this round
    print(output_a)
    print(set(output_a.flatten()))
    where_a = np.where(output_a.astype(int) == 1)
    print(where_a)
    final_assignment[where_a[0]] = where_a[1]

    # Extract unused stock lengths, and their original position in the array
    used_stock = output_a.sum(axis=1).astype('int')
    print(used_stock)
    stock_lengths_b = stock_lengths[used_stock == 0]
    original_ids = np.array(range(len(stock_lengths)))[used_stock == 0]

    print(f"Used stock count A : {used_stock.sum()}")
    print(f"Stock  count B : {len(stock_lengths_b)}")

    print(final_assignment)

    coverage_check = 0
    for i, a in enumerate(final_assignment):
        if a != -1:
            print((i,a))
            coverage_check += extents_a[i,a]
    print(f"Coverage check Just A : {coverage_check}")

    # Run Model B
    model_b: mip.Model = _get_basic_model(model_args)
    model_b, wastes_b, extents_b = _model_order(model_b, stock_lengths_b, part_lengths_b, None)
    model_b.max_seconds = max_seconds
    status_b: OptimizationStatus = model_b.optimize()

    # Return if not at least feasible
    if status_b in [OptimizationStatus.INFEASIBLE, OptimizationStatus.NO_SOLUTION_FOUND, OptimizationStatus.ERROR]:
        elapsed_time = time.time() - time_start
        log_string = f"{log_id},{status_b},0,0,0,{elapsed_time},{elapsed_time}"
        log_line(log_string, log_filepath)
        return status_b, [0], log_string

    time_end = time.time()
    time_elapsed = round(time_end - time_start, 3)

    # Get output B
    output_b = [float(v.x) for v in model_b.vars]
    output_b = np.array(output_b).reshape((len(part_lengths_b), len(stock_lengths_b)))
    output_b = output_b.transpose()
    output_b = np.rint(output_b)

    # Fill the final assignment array for the stock pieces used in round B
    where_b = np.where(output_b == 1)
    final_assignment[original_ids[where_b[0]]] = where_b[1] + len(part_lengths_a)

    # Create base for the combined waste matrix
    waste_complete = wastes_a.transpose()
    extents_complete = extents_a.transpose()

    # Extend complete waste matrix
    for i in range(len(part_lengths_b)):
        waste_complete = np.append(waste_complete, np.zeros((1, len(stock_lengths))), axis=0)
        extents_complete = np.append(extents_complete, np.zeros((1, len(stock_lengths))), axis=0)

    # Fill in complete waste matrix
    for i in range(len(stock_lengths_b)):
        waste_complete[len(part_lengths_a):, original_ids[i]] = wastes_b[i]
        extents_complete[len(part_lengths_a):, original_ids[i]] = extents_b[i]

    waste_total = 0
    score_total = 0
    for i, v in enumerate(final_assignment):
        if v == -1:
            score_total += math.pow(stock_lengths[i], 2)
        else:
            waste_total += waste_complete[v, i]
            score_total += math.pow(waste_complete[v, i], 2)

    # Assemble standard log line
    # id, status, objective value, waste value, score value, total time, last improved time
    log_objective_value = round(model_a.objective_value + model_b.objective_value, 3)
    log_improvement_time = _progress_log_last_time(model_a) + _progress_log_last_time(model_b)

    log_string = f"{log_id},{model_a.status}/{model_b.status},{log_objective_value},{waste_total},{score_total},{time_elapsed},{log_improvement_time}"
    log_line(log_string, log_filepath)

    output = final_assignment.tolist()

    print(f"Final Assignment :  {output}")

    print(waste_complete)
    print(extents_complete)

    coverage_check = 0
    coverage_a = 0
    coverage_b = 0
    for i, a in enumerate(output):
        if a != -1:
            coverage_check += extents_complete[a, i]
            if a < len(part_lengths_a):
                coverage_a += extents_complete[a, i]
            else:
                coverage_b += extents_complete[a, i]
    print(f"Coverage check : {coverage_check}")

    print(f"Objective A : {model_a.objective_value}")
    print(f"Objective B : {model_b.objective_value}")

    print(f"Coverage a : {coverage_a}")
    print(f"Coverage B : {coverage_b}")

    print(waste_complete.shape)
    print(extents_complete.shape)

    return model_a.status, output, log_string


def _get_basic_model(model_args) -> mip.Model:
    if 'use_gurobi' in model_args and model_args['use_gurobi']:
        model = Model(solver_name=mip.GUROBI)
    else:
        model = Model()

    model.max_mip_gap = 0.01  # 1% gap
    model.threads = -1  # Use all available cores
    model.store_search_progress_log = True  # Necessary to check last improved time

    if "clique" in model_args:
        model.clique = int(model_args["clique"])

    if "cuts" in model_args:
        model.cuts = int(model_args["cuts"])

    if "emphasis" in model_args:
        model.emphasis = int(model_args["emphasis"])
    else:
        model.emphasis = 1  # Look for feasible solutions first

    if "lp_method" in model_args:
        model.lp_method = int(model_args["lp_method"])

    if "preprocess" in model_args:
        model.preprocess = int(model_args["preprocess"])
    else:
        model.preprocess = 1

    return model


def _get_model_log_string(method, model, model_args, time_elapsed):
    """
    Assembles a log string for use when testing model parameters
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

    log_string += str(model.status) + ","
    log_string += str(round(model.objective_value, 3)) + ","
    log_string += str(time_elapsed) + "\n"

    return log_string


def _progress_log_last_time(model: mip.Model) -> Optional[float]:
    if len(model.search_progress_log.log) == 0:
        return None

    log = model.search_progress_log.log
    last_bound = log[-1][1][1]

    for i, entry in enumerate(reversed(log)):
        if i == 0:
            continue
        if entry[1][1] != last_bound:
            return entry[0]

    return log[0][0]

    print(log[-1])
    print(log[-2])
    print(log[-3])


def _model_default(model, stock_lengths, part_lengths, part_requests):
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

    return model, None, None


def _model_waste(model, stock_lengths, part_lengths, part_requests):
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

    model.cuts = 3

    return model, None, None


def _model_max(model, stock_lengths, part_lengths, part_requests):
    """
    Ignores the utilized variable, tries to optimize the square of leftovers. Because we're trying to maximize a
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
        model.add_constr(xsum(part_lengths[i] * part_usage[i, j] for i in range(part_count))
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

    model.clique_merge()
    model.cuts = 3
    
    print(f"Model created with {len(model.vars)} variables and {len(model.constrs)} constraints")

    model.objective = maximize(xsum(score[i] for i in range(stock_count)))

    return model, None, None


def _model_homogenous(model, stock_lengths, part_lengths, part_requests):
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

    return model, None, None


def _model_order(model: mip.Model, stock_lengths: [float], part_lengths: [float], _):
    """
    Optimizes for minimizing waste from used pieces
    Does not attempt leftover usability
    Lazy constraints were tested but not conclusively useful and may be the wrong venue for them
    Setting the cuts parameter aggressive was tested but not conclusive
    """
    part_lengths = np.array(part_lengths)
    part_count = len(part_lengths)

    stock_lengths = np.array(stock_lengths)
    stock_count = len(stock_lengths)

    # Helper constant : If stock j starts at part i, it will be able to cover a total of n parts
    stock_extents, stock_wastes = _calculate_order_constants(stock_lengths, part_lengths)

    # print(stock_extents)
    # print(stock_wastes)

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
        model.add_constr(i
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

    model.clique_merge()

    model.pump_passes = 20

    print(f"Model created with {len(model.vars)} variables and {len(model.constrs)} constraints")

    model.objective = minimize(
        xsum(xsum(stock_start[i, j] * stock_wastes[j, i] for i in range(part_count)) for j in range(stock_count)))

    return model, stock_wastes, stock_extents


def solve_order_two():
    pass


def log_line(s, log_filepath):
    s += "\n"
    with open(log_filepath, "a") as f:
        f.write(s)


def _calculate_order_constants(stock_lengths, part_lengths):
    stock_extents = []
    stock_wastes = []
    for j in range(len(stock_lengths)):
        stock_extents.append([])
        stock_wastes.append([])
        for i in range(len(part_lengths)):
            extent = 0
            countdown = stock_lengths[j]
            while (i + extent) < len(part_lengths) and countdown >= part_lengths[i + extent]:
                countdown -= part_lengths[i + extent]
                extent += 1
            stock_wastes[-1].append(countdown)
            stock_extents[-1].append(extent)
    stock_extents = np.array(stock_extents)
    stock_wastes = np.array(stock_wastes)

    return stock_extents, stock_wastes


def _demo_homogenous():
    stock_lengths = [10, 12, 16, 14, 14, 30, 12]
    part_lengths = [2, 4, 6, 8]
    part_requests = [4, 3, 5, 3]

    model = Model()
    model.preprocess = 1
    model, extra = _model_homogenous(model, stock_lengths, part_lengths, part_requests)
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
    model_solver = mip.GUROBI

    time = 600
    stock_lengths = [10, 12, 16, 14, 14, 30, 12]
    part_lengths = [5, 8, 4, 2, 1, 2, 4, 6, 5, 5, 5, 5, 5, 1, 3, 3, 4, 3.5,2.5,1.5]

    # ~20 to be equal with dome
    rep = 1

    stock_lengths = np.repeat(stock_lengths, rep)
    part_lengths = np.repeat(part_lengths, rep)

    # Run model full
    """
    model = Model(solver_name=model_solver)
    model.preprocess = 1
    model.emphasis = 1
    model.store_search_progress_log = True
    model, waste_values = _model_order(model, stock_lengths, part_lengths, None)
    model.max_seconds = time
    status: OptimizationStatus = model.optimize()

    output = [float(v.x) for v in model.vars]
    usage = np.array(output).reshape((len(part_lengths), len(stock_lengths)))
    usage = usage.transpose()
    usage_array = usage.sum(axis=1)

    waste_total = np.sum(waste_values * usage)

    waste_array = np.sum(waste_values * usage, axis=1)
    leftover_array = np.maximum(waste_array, np.array(stock_lengths) * (1 - usage_array))
    leftover_array = leftover_array * leftover_array

    score_total = leftover_array.sum()

    print(status)

    return 0
    """

    print(f"Demo with {len(stock_lengths)} stock pieces and {len(part_lengths)} part requests")

    final_assignment = np.repeat(-1, len(stock_lengths))

    split_point = len(part_lengths) // 2
    part_lengths_a = part_lengths[:split_point]
    part_lengths_b = part_lengths[split_point:]

    # Run model A
    model_a = Model(solver_name=model_solver)
    model_a.preprocess = 1
    model_a.emphasis = 1
    model_a, extra_a, extents_a = _model_order(model_a, stock_lengths, part_lengths_a, None)
    model_a.max_seconds = time
    status_a: OptimizationStatus = model_a.optimize()

    # Get output A
    output = [float(v.x) for v in model_a.vars]
    output = np.array(output).reshape((len(part_lengths_a), len(stock_lengths)))
    output = output.transpose()

    # Fill the final assignment array for the stock pieces used this round
    where_a = np.where(output == 1)
    final_assignment[where_a[0]] = where_a[1]

    # Extract unused stock lengths, and their original position in the array
    used_stock = output.sum(axis=1)
    stock_lengths_b = stock_lengths[used_stock == 0]
    original_ids = np.array(range(len(stock_lengths)))[used_stock == 0]

    # Run Model B
    model_b = Model(solver_name=model_solver)
    model_b.preprocess = 1
    model_b.emphasis = 1
    model_b, extra_b, extents_b = _model_order(model_b, stock_lengths_b, part_lengths_b, None)
    model_b.max_seconds = time
    status_b: OptimizationStatus = model_b.optimize()

    # Get output B
    output = [float(v.x) for v in model_b.vars]
    output = np.array(output).reshape((len(part_lengths_b), len(stock_lengths_b)))
    output = output.transpose()

    # Fill the final assignment array for the stock pieces used in round B
    where_b = np.where(output == 1)
    final_assignment[original_ids[where_b[0]]] = where_b[1] + len(part_lengths_a)

    # Create base for the combined waste matrix
    waste_complete = extra_a.transpose()
    extents_complete = extents_a.transpose()

    # Extend complete waste matrix
    for i in range(len(part_lengths_b)):
        waste_complete = np.append(waste_complete, np.zeros((1, len(stock_lengths))), axis=0)
        extents_complete = np.append(extents_complete, np.zeros((1, len(stock_lengths))), axis=0)

    # Fill in complete waste matrix
    for i in range(len(stock_lengths_b)):
        waste_complete[len(part_lengths_a):, original_ids[i]] = extra_b[i]
        extents_complete[len(part_lengths_a):, original_ids[i]] = extents_b[i]

    if True:
        print('Complete Waste matrix: ')
        np.set_printoptions(linewidth=np.inf)
        np.set_printoptions(threshold=np.inf)
        print(waste_complete)
        print(extents_complete)

    waste_total = 0
    score_total = 0
    for i, v in enumerate(final_assignment):
        if v == -1:
            score_total += math.pow(stock_lengths[i], 2)
        else:
            waste_total += waste_complete[v, i]
            score_total += math.pow(waste_complete[v, i], 2)

    print('')
    print(f"Optimization Status A : {status_a}")
    print(f"Optimization Status B : {status_b}")
    print(f"Objective A : {model_a.objective_value}")
    print(f"Objective B : {model_b.objective_value}")
    print(f"Final Assignment : {final_assignment}")
    print(f"Waste : {waste_total}")
    print(f"Score : {score_total}")

    coverage_check = 0
    for i, a in enumerate(final_assignment):
        if a != -1:
            coverage_check += extents_complete[a, i]
    print(f"Coverage check : {coverage_check}")


if __name__ == "__main__":
    _demo_order()
    # _demo_homogenous()

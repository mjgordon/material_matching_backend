import mip
import numpy as np
import time

from mip import Model, xsum, BINARY, INTEGER, minimize, maximize, OptimizationStatus

print_variables = False


def solve_ilp(method, stock_lengths, part_lengths, part_requests) -> tuple[mip.OptimizationStatus, list[int]]:
    print(method)
    print(stock_lengths)
    print(part_lengths)
    print(part_requests)
    time_start = time.time()
    model = Model()
    model.max_mip_gap_abs = 1.5
    model.max_mip_gap = .1
    # model.cuts = 3
    if method == "default":
        solve_function = solve_default
    elif method == "waste":
        solve_function = solve_waste
    elif method == "max":
        solve_function = solve_max

    model = solve_function(model, stock_lengths, part_lengths, part_requests)

    # optimizing the model
    status = model.optimize(max_nodes=10000, max_seconds=30)

    print('')
    print(f"Optimization Status : {status}")

    if status == OptimizationStatus.INFEASIBLE:
        return status, [0]

    # printing the solution

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
    time_elapsed = round(time_end - time_start, 3)

    log_string = method + ","
    log_string += str(len(stock_lengths)) + ","
    log_string += str(len(part_lengths)) + ","
    log_string += str(sum(part_requests)) + ","
    log_string += str(time_elapsed) + "\n"

    with open("log.txt", "a") as f:
        f.write(log_string)

    return status, output


def solve_default(model, stock_lengths, part_lengths, part_requests):
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
        model.add_constr(
            xsum(part_lengths[i] * part_usage[i, j] for i in range(part_count)) <= stock_lengths[j] * stock_usage[j])

    # additional constraints to reduce symmetry
    # Put unused bars at end of list (reduces search space)
    # Not appropriate for this type
    # for j in range(1, stock_count):
    #    model.add_constr(stock_usage[j - 1] >= stock_usage[j])

    model.objective = minimize(xsum(stock_usage[i] for i in range(stock_count)))

    return model


def solve_waste(model, stock_lengths, part_lengths, part_requests):
    """
    Optimizes for minimizing waste from used piecess
    Does not attempt leftover usability
    """
    part_lengths = np.array(part_lengths)
    part_count = len(part_lengths)

    part_requests = np.array(part_requests)

    stock_lengths = np.array(stock_lengths)
    stock_count = len(stock_lengths)

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

    model.objective = minimize(xsum((stock_lengths[j] * stock_usage[j]) -
                                    xsum((part_lengths[i] * part_usage[i, j]) for i in range(part_count))
                                    for j in range(stock_count)))

    return model


def solve_max(model, stock_lengths, part_lengths, part_requests):
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

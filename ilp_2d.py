"""
Implementation of 2D Cutting Stock Problem using Integer Linear Programming

Additions:
Objective that considers end of strip waste
Objective taht considers waste above each part
Solving on sheets instead of roll
Allow for part flipping
"""

import mip



from mip import Model, BINARY, minimize, xsum


def solve_ilp_2d_roll(part_widths, part_heights, stock_width, model_args=None) -> tuple[
    mip.OptimizationStatus, list[int]]:
    if model_args is None:
        model_args = {}

    part_count = len(part_widths)
    I = set(range(part_count))
    S = [[j for j in I if part_heights[j] <= part_heights[i]] for i in I]
    G = [[j for j in I if part_heights[j] >= part_heights[i]] for i in I]

    model = Model()

    grouping = [{j: model.add_var(var_type=BINARY) for j in S[i]} for i in I]
    # flip = [model.add_var(var_type=BINARY) for i in I]

    model.objective = minimize(xsum(part_heights[i] * grouping[i][i] for i in I))

    # Constraint : each part has only one parent, which may be itself
    for i in I:
        model.add_constr(xsum(grouping[j][i] for j in G[i])
                         ==
                         1)

    # Constraint : If a part is a header, it opens the remaining width in the row for the sum of its children
    for i in I:
        model.add_constr(xsum(part_widths[j] * grouping[i][j] for j in S[i] if j != i)
                         <=
                         (stock_width - part_widths[i]) * grouping[i][i])

    model.optimize()

    output = [float(v.x) for v in model.vars]

    return model.status, output


def solve_ilp_2d_roll_area(part_widths, part_heights, stock_width, model_args=None) -> tuple[
    mip.OptimizationStatus, list[int]]:
    if model_args is None:
        model_args = {}

    part_count = len(part_widths)
    I = set(range(part_count))
    S = [[j for j in I if part_heights[j] <= part_heights[i]] for i in I]
    G = [[j for j in I if part_heights[j] >= part_heights[i]] for i in I]

    model = Model()

    grouping = [{j: model.add_var(var_type=BINARY) for j in S[i]} for i in I]
    # flip = [model.add_var(var_type=BINARY) for i in I]

    model.objective = minimize(xsum(part_heights[i] * grouping[i][i]
                                    for i in I) * stock_width
                               +
                               xsum(((stock_width - xsum(part_widths[j] * grouping[i][j] for j in S[i]))
                                    * part_heights[i])
                                    * grouping[i][i]
                                    for i in I))

    # Constraint : each part has only one parent, which may be itself
    for i in I:
        model.add_constr(xsum(grouping[j][i] for j in G[i])
                         ==
                         1)

    # Constraint : If a part is a header, it opens the remaining width in the row for the sum of its children
    for i in I:
        model.add_constr(xsum(part_widths[j] * grouping[i][j] for j in S[i] if j != i)
                         <=
                         (stock_width - part_widths[i]) * grouping[i][i])

    model.optimize()

    output = [float(v.x) for v in model.vars]

    print(model.objective_value)
    print(model.objective_bound)

    return model.status, output


def solve_ilp_2d_sheets_(part_widths, part_heights, stock_widths, stock_heights, model_args=None) -> tuple[
    mip.OptimizationStatus, list[int]]:
    if model_args is None:
        model_args = {}

    part_count = len(part_widths)
    I = set(range(part_count))
    S = [[j for j in I if part_heights[j] <= part_heights[i]] for i in I]
    G = [[j for j in I if part_heights[j] >= part_heights[i]] for i in I]

    model = Model()

    grouping = [{j: model.add_var(var_type=BINARY) for j in S[i]} for i in I]
    # flip = [model.add_var(var_type=BINARY) for i in I]

    model.objective = minimize(xsum(part_heights[i] * grouping[i][i] for i in I))

    # Constraint : each part has only one parent, which may be itself
    for i in I:
        model.add_constr(xsum(grouping[j][i] for j in G[i])
                         ==
                         1)

    # Constraint : If a part is a header, it opens the remaining width in the row for the sum of its children
    for i in I:
        model.add_constr(xsum(part_widths[j] * grouping[i][j] for j in S[i] if j != i)
                         <=
                         (stock_width - part_widths[i]) * grouping[i][i])

    model.optimize()

    output = [float(v.x) for v in model.vars]

    return model.status, output

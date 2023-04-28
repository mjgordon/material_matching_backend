"""
Used for local solving. Creates a hops-visible flask server, and calls the ilp functions.
"""

import ghhops_server as hs
import rhino3dm
import numpy as np
import scipy.optimize

from flask import Flask

import ilp
import ilp_2d

app = Flask(__name__)
hops = hs.Hops(app)


@hops.component(
    "/hops_ilp",
    name="Solve",
    description="Solve the ILP Problem",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsString("Method", "M", "Method"),
        hs.HopsNumber("Stock", "S", "Stock Lengths", hs.HopsParamAccess.LIST),
        hs.HopsNumber("PartLengths", "P", "Part Lengths", hs.HopsParamAccess.LIST),
        hs.HopsNumber("PartCounts", "C", "Part Counts", hs.HopsParamAccess.LIST),
        hs.HopsString("Name", "N", "Project or test name")
    ],
    outputs=[
        hs.HopsNumber("Selection", "S", "Solved Result", hs.HopsParamAccess.LIST),
        hs.HopsString("Log","L","Log String")
    ]
)
def hops_ilp(method, stock_lengths, part_lengths, part_requests,name):
    status, output, log = ilp.solve_ilp(method, stock_lengths, part_lengths, part_requests)
    return output, log


@hops.component(
    "/hops_ilp_2d",
    name="solve_2d",
    description="Solve a 2d rectangle packing problem",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsNumber("PartWidths", "W", "Part Widths", hs.HopsParamAccess.LIST),
        hs.HopsNumber("PartHeights", "H", "Part Heights", hs.HopsParamAccess.LIST),
        hs.HopsNumber("StockWidth", "S", "Stock Width"),
    ],
    outputs=[
        hs.HopsNumber("X", "X", "Groupings", hs.HopsParamAccess.LIST),
    ]
)
def hops_ilp_2d(part_widths, part_heights, stock_width):
    #status, X = ilp_2d.solve_ilp_2d_roll_area(part_widths, part_heights, stock_width)
    status, X = ilp_2d.solve_ilp_2d_roll(part_widths, part_heights, stock_width)
    print(status)
    return X


@hops.component(
    "/fit_curve",
    name="Fit Curve",
    description="Fit a curve to the data",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsNumber("XValues", "X", "X Values", hs.HopsParamAccess.LIST),
        hs.HopsNumber("YValues", "Y", "Y Values", hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsNumber("Values", "V", "Solved Values", hs.HopsParamAccess.LIST)
    ],
)
def fit_curve(x, y):
    popt, _ = scipy.optimize.curve_fit(objective, x, y)
    #a, b, c = popt
    return list(popt)


def objective(x, a, b, c):
    return (a * x) + (b * (x**2)) + c


if __name__ == "__main__":
    app.run()

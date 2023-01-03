import ghhops_server as hs
import rhino3dm
import scipy.optimize

from flask import Flask

import ilp

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
        hs.HopsNumber("PartCounts", "C", "Part Counts", hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsNumber("Selection", "S", "Solved Result", hs.HopsParamAccess.LIST)
    ]
)
def hops_ilp(method, stock_lengths, part_lengths, part_requests):
    output = ilp.solve_ilp(method, stock_lengths, part_lengths, part_requests)
    return output


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

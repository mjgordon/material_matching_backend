"""
SocketIO client that connects to the dispatcher as a user. Also creates a hops-visible flask server, and passes solve
requests from hops to the dispatcher
"""

import argparse
import json
import ghhops_server as hs
import rhino3dm
import scipy.optimize
import socketio
import time

from flask import Flask

import ilp

app = Flask(__name__)
hops = hs.Hops(app)
sio = socketio.Client()

solving_flag: bool = False
response_usage = None

log_path: str = ""

save_scenario = False

log_string = ""


def main():
    parser = argparse.ArgumentParser(description='Give dispatcher IP address')
    parser.add_argument("-i", "--ip")
    args = vars(parser.parse_args())
    print(args["ip"])

    sio.connect("http://" + args["ip"] + ":52323")

    app.run()


@sio.event
def connect():
    print('connection established')
    sio.emit("client_id", {'type': 'user', 'name': 'rhino'})


@sio.on('solve_response')
def solve_response(data):
    global response_usage, solving_flag, log_path,log_string
    response_usage = data["usage"]
    log_string = data["log_string"] + "\n"
    with open(log_path, "a") as f:
        f.write(log_string)
    solving_flag = False


@sio.on('solve_infeasible')
def solve_infeasible(data):
    global solving_flag,log_string
    print("Infeasible")
    log_string = data["log_string"] + "\n"
    with open(log_path, "a") as f:
        f.write(log_string)
    solving_flag = False


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
def hops_ilp(method, stock_lengths, part_lengths, part_requests, name):
    global solving_flag, log_path, log_strings
    sio.emit("solve_request", {'method': method,
                               'stock_lengths': stock_lengths,
                               'part_lengths': part_lengths,
                               'part_requests': part_requests,
                               'model_args': {'log_filepath': f"logs/{name}.csv",
                                              'max_nodes': 100000,
                                              'max_seconds': 120}})
    log_path = f"logs/{name}.csv"
    solving_flag = True

    if save_scenario:
        output_dict = {"name": name,
                       "method": method,
                       "stock_lengths": stock_lengths,
                       "part_lengths": part_lengths,
                       "part_requests": part_requests}
        with open("scenarios/scenario.json", 'w') as f:
            json.dump(output_dict, f)

    while solving_flag:
        time.sleep(0.1)

    return response_usage,log_string


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
    main()

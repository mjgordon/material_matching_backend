"""
Socketio client that connects to the dispatcher as a solver.
Receives solve requests and calls the ilp functions
"""

import argparse
import datetime
import mip
import socketio
import sys

import ilp


sio = socketio.Client()

solver_name = ""


def main():
    global solver_name
    parser = argparse.ArgumentParser(description='Give dispatcher IP address')
    parser.add_argument("-i","--ip")
    parser.add_argument("-n","--name")
    args = vars(parser.parse_args())

    print(f"Looking for dispatcher at {args['ip']}")

    if "name" in args:
        solver_name = args['name']

    if "ip" in args:
        sio.connect(f"http://{args['ip']}:52323")
    else:
        sio.connect('http://localhost:52323')

    sio.wait()


@sio.event
def connect():
    print('Connected to Dispatcher')
    sio.emit("client_id", {'type': 'solver', 'name': f"{solver_name}"})


@sio.event
def disconnect():
    print('Disconnected from Dispatcher')


@sio.on("solve_request")
def solve_request(data):
    print("\n========================================================================")
    print(f"Received solve request at {str(datetime.datetime.now())}")
    method = data["method"]
    stock_lengths = [float(n) for n in data["stock_lengths"]]
    part_lengths = [float(n) for n in data["part_lengths"]]
    part_requests = [int(n) for n in data["part_requests"]]
    model_args = data["model_args"] if "model_args" in data else {}

    status, solve_output, log_string = ilp.solve_ilp(method, stock_lengths, part_lengths, part_requests, model_args=model_args)

    if status in [mip.OptimizationStatus.INFEASIBLE, mip.OptimizationStatus.NO_SOLUTION_FOUND, mip.OptimizationStatus.ERROR]:
        response = {'requester_sid': data['requester_sid'], 'log_string': log_string}
        sio.emit("solve_infeasible", response)
    else:
        if method not in ['order', 'order_split']:
            solve_output = solve_output[0:-len(stock_lengths)]
        response = {'requester_sid': data['requester_sid'], 'usage': solve_output, 'log_string': log_string}
        sio.emit("solve_response", response)


def signal_handler(signal, frame):
    sio.disconnect()
    sys.exit(0)

#Turned this off as it was causing the entry script to hang when scrolling on windows
#signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
    main()

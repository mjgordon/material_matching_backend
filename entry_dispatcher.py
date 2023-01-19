import argparse
import datetime
import mip
import signal
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
    print(args["ip"])
    if "ip" in args:
        sio.connect(f"http://{args['ip']}:52323")
    else:
        sio.connect('http://localhost:52323')

    if "name" in args:
        solver_name = args['name']

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

    print(f"\nReceived solve request at {str(datetime.datetime.now())}")
    method = data["method"]
    stock_lengths = [float(n) for n in data["stock_lengths"]]
    part_lengths = [float(n) for n in data["part_lengths"]]
    part_requests = [int(n) for n in data["part_requests"]]

    status, solve_output = ilp.solve_ilp(method, stock_lengths, part_lengths, part_requests)

    if status == mip.OptimizationStatus.INFEASIBLE:
        response = {'requester_sid': data['requester_sid']}
        sio.emit("solve_infeasible", response)
    else:
        solve_output = solve_output[0:-len(stock_lengths)]
        response = {'requester_sid': data['requester_sid'], 'usage': solve_output}
        sio.emit("solve_response", response)


def signal_handler(signal, frame):
    sio.disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
    main()

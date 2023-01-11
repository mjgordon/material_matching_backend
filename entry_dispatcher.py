import argparse
import datetime
import mip
import socketio

import ilp


sio = socketio.Client()


def main():
    parser = argparse.ArgumentParser(description='Give dispatcher IP address')
    parser.add_argument("-i","--ip")
    args = vars(parser.parse_args())
    print(args["ip"])
    if "ip" in args:
        sio.connect(f"http://{args['ip']}:52323")
    else:
        sio.connect('http://localhost:52323')

    sio.wait()


@sio.event
def connect():
    print('connection established')
    sio.emit("client_id", {'type': 'solver'})


@sio.event
def disconnect():
    print('disconnected from server')


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
        print(solve_output)


if __name__ == '__main__':
    main()

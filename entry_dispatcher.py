import datetime
import socketio

import ilp

sio = socketio.Client()


def main():
    sio.connect('http://localhost:5000')
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
    print(f"Received solve request at {str(datetime.datetime.now())}")
    method = data["method"]
    stock_lengths = [float(n) for n in data["stock_lengths"]]
    part_lengths = [float(n) for n in data["part_lengths"]]
    part_requests = [int(n) for n in data["part_requests"]]

    solve_output = ilp.solve_ilp(method, stock_lengths, part_lengths, part_requests)
    solve_output = solve_output[0:-len(stock_lengths)]

    response = {'requester_sid': data['requester_sid'],
                'usage': solve_output}

    sio.emit("solve_response", response)

    print(solve_output)


if __name__ == '__main__':
    main()

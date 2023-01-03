import socketio

sio = socketio.Client()


def main():
    sio.connect('http://localhost:5000')
    default_message()
    # sio.wait()


@sio.event
def connect():
    print('connection established')
    sio.emit("client_id", {'type': 'user'})


@sio.on('solve_response')
def solve_response(data):
    print('Solved with : ', data)


@sio.event
def disconnect():
    print('disconnected from server')


def default_message():
    sio.emit("solve_request", {'method': 'waste',
                               'stock_lengths': [10, 5, 4],
                               'part_lengths': [3, 2],
                               'part_requests': [3, 3]})


if __name__ == '__main__':
    main()

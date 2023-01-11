import eventlet
import json
import socketio

import socket
import fcntl
import struct



# TODO: using this with wildcard is bad, see if it can just be set to webserver ip
sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

solver_sids = []
user_sids = []


def main():
    ip = get_ip_address('eth0')
    #eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 52323)), app)
    eventlet.wsgi.server(eventlet.listen((ip, 52323)), app)


@sio.event
def connect(sid, environ):
    #emit("hello world from server")
    print('connect ', sid)


@sio.event
def my_message(sid, data):
    print('message ', data)


@sio.on('client_id')
def client_id(sid, data):
    if isinstance(data, str):
        data = json.loads(data)

    print(f"Client at {sid} is a {data['type']}")

    if data['type'] == 'solver':
        solver_sids.append(sid)
    elif data['type'] == 'user':
        user_sids.append(sid)
    else:
        print(f"Bad client type : {data['type']}")


@sio.on('solve_request')
def solve_request(sid, data):
    if isinstance(data, str):
        data = json.loads(data)
    print('Received solve request : ', data)
    data["requester_sid"] = sid
    sio.emit('solve_request', data, sid=solver_sids[0])


@sio.on('solve_response')
def solve_response(sid, data):
    print('Received solve response : ', data)
    sio.emit('solve_response', data, sid=data['requester_sid'])


@sio.on('solve_infeasible')
def solve_infeasible(sid, data):
    print('Received infeasible solve : ', data)
    sio.emit('solve_infeasible', data, sid=data['requester_sid'])


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])


if __name__ == '__main__':
    main()

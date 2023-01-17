import curses
import curses.textpad
import datetime
import eventlet
import json
import os
import signal
import socketio
import sys


# TODO: using this with wildcard is bad, see if it can just be set to webserver ip
sio = socketio.Server(cors_allowed_origins='*')

app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})
solver_sids = []
user_sids = []

stdscr = None
win_solvers = None
win_users = None
win_messages = None
message_list = []


class PrintRedirect:
    def write(self, string: str):
        if string != '\n':
            message_list.insert(0, str(datetime.datetime.now()) + " : " + string)
        if len(message_list) > 100:
            message_list.pop(-1)

    def flush(self):
        pass


print_redirect = PrintRedirect()


def main():
    global stdscr, win_solvers, win_users, win_messages
    try:
        ip = os.popen('ip addr show eth0').read().split("inet ")[1].split("/")[0]
    except IndexError:
        ip = "127.0.0.1"

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    sys.stdout = print_redirect
    sys.stderr = print_redirect

    redraw_curses()

    eventlet.wsgi.server(eventlet.listen((ip, 52323)), app)


def handler(signum, frame):
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    exit(0)

signal.signal(signal.SIGINT, handler)


def redraw_curses():
    global stdscr, win_solvers, win_users, win_messages

    stdscr.clear()

    rows, cols = stdscr.getmaxyx()

    win_solvers = curses.newwin(rows // 2, cols // 2, 0, 0)
    win_users = curses.newwin(rows // 2, cols // 2, 0, cols // 2)
    win_messages = curses.newwin(rows // 2, cols, rows // 2, 0)

    win_solvers.addstr(1, 1, f"Solvers ({len(solver_sids)})")
    for i, sid in enumerate(solver_sids):
        win_solvers.addstr(2 + i, 1, sid)
    curses.textpad.rectangle(win_solvers, 0, 0, rows // 2 - 1, cols // 2 - 2)
    win_solvers.refresh()

    win_users.addstr(1, 1, f"Users ({len(user_sids)})")
    for i, sid in enumerate(user_sids):
        win_users.addstr(2 + i, 1, sid)
    curses.textpad.rectangle(win_users, 0, 0, rows // 2 - 1, cols // 2 - 2)
    win_users.refresh()

    win_messages.addstr(1, 1, "Messages")
    for i, message in enumerate(message_list):
        if i > rows // 2 - 5:
            break
        win_messages.addstr(rows // 2 - 3 - i, 1,message)
    curses.textpad.rectangle(win_messages, 0, 0, rows // 2 - 2, cols - 1)
    win_messages.refresh()


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

    redraw_curses()


@sio.on('solve_request')
def solve_request(sid, data):
    if isinstance(data, str):
        data = json.loads(data)
    print('Received solve request : Length ', len(data))
    data["requester_sid"] = sid
    sio.emit('solve_request', data, sid=solver_sids[0])

    redraw_curses()


@sio.on('solve_response')
def solve_response(sid, data):
    print('Received solve response : ', data)
    sio.emit('solve_response', data, sid=data['requester_sid'])

    redraw_curses()


@sio.on('solve_infeasible')
def solve_infeasible(sid, data):
    print('Received infeasible solve : ', data)
    sio.emit('solve_infeasible', data, sid=data['requester_sid'])

    redraw_curses()


@sio.event
def disconnect(sid):
    print('disconnect ', sid)

    redraw_curses()





if __name__ == '__main__':
    main()

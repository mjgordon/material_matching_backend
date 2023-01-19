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

solver_usage = {}
user_usage = {}

solver_nametable = {}

stdscr = None
win_solvers = None
win_users = None
win_messages = None
message_list = []

ip = None


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
    global stdscr, win_solvers, win_users, win_messages, ip
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


def exit_handler(signum, frame):
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()
    exit(0)


signal.signal(signal.SIGINT, exit_handler)


def redraw_curses():
    global stdscr, win_solvers, win_users, win_messages, ip

    stdscr.clear()

    rows, cols = stdscr.getmaxyx()

    win_solvers = curses.newwin(rows // 2, cols // 2, 0, 0)
    win_users = curses.newwin(rows // 2, cols // 2, 0, cols // 2)
    win_messages = curses.newwin(rows // 2, cols, rows // 2, 0)

    win_solvers.addstr(1, 1, f"Solvers ({len(solver_sids)})")
    for i, sid in enumerate(solver_sids):
        solver_status = "*" if sid in solver_usage and solver_usage[sid] else " "
        win_solvers.addstr(2 + i, 1, f"[{solver_status}] {sid} ({solver_nametable[sid]})")
    curses.textpad.rectangle(win_solvers, 0, 0, rows // 2 - 1, cols // 2 - 2)
    win_solvers.refresh()

    win_users.addstr(1, 1, f"Users ({len(user_sids)})")
    for i, sid in enumerate(user_sids):
        user_status = "*" if sid in user_usage and user_usage[sid] else " "
        win_users.addstr(2 + i, 1, f"[{user_status}] {sid}")
    curses.textpad.rectangle(win_users, 0, 0, rows // 2 - 1, cols // 2 - 2)
    win_users.refresh()

    win_messages.addstr(1, 1, f"Messages | This IP {ip}")
    for i, message in enumerate(message_list):
        if i > rows // 2 - 5:
            break
        win_messages.addstr(rows // 2 - 3 - i, 1, message)
    curses.textpad.rectangle(win_messages, 0, 0, rows // 2 - 2, cols - 1)
    win_messages.refresh()


@sio.event
def connect(sid, environ):
    print(f'Connection from {sid}')

    redraw_curses()


@sio.event
def disconnect(sid):
    print(f'Disconnection from {sid}')
    if sid in solver_sids:
        solver_sids.remove(sid)
    elif sid in user_sids:
        user_sids.remove(sid)

    redraw_curses()


@sio.on('client_id')
def client_id(sid, data):
    if isinstance(data, str):
        data = json.loads(data)

    print(f"Client at {sid} is a {data['type']}")

    if data['type'] == 'solver':
        solver_sids.append(sid)
        solver_nametable[sid] = data["name"] if "name" in data else ""
    elif data['type'] == 'user':
        user_sids.append(sid)
    else:
        print(f"Bad client type : {data['type']}")

    redraw_curses()


@sio.on('solve_request')
def solve_request(sid, data):
    if isinstance(data, str):
        data = json.loads(data)
    print(f"[->] solve request for method '{data['method']}' from user : {sid}")
    data["requester_sid"] = sid

    solver_id = 0
    for i, solver_sid in enumerate(solver_sids):
        if solver_sid not in solver_usage or solver_usage[solver_sid] is False:
            solver_usage[solver_sid] = True
            solver_id = i
            break

    sio.emit('solve_request', data, room=solver_sids[solver_id])

    solver_usage[solver_sids[solver_id]] = True
    user_usage[sid] = True

    redraw_curses()


@sio.on('solve_response')
def solve_response(sid, data):
    print(f'[<-] feasible solver response for user : {data["requester_sid"]}')
    sio.emit('solve_response', data, room=data['requester_sid'])

    solver_usage[sid] = False
    user_usage[data['requester_sid']] = False

    redraw_curses()


@sio.on('solve_infeasible')
def solve_infeasible(sid, data):
    print(f'[x-] infeasible solve response for user : {data["requester_sid"]}')
    sio.emit('solve_infeasible', data, room=data['requester_sid'])

    solver_usage[sid] = False
    user_usage[data['requester_sid']] = False

    redraw_curses()


if __name__ == '__main__':
    main()

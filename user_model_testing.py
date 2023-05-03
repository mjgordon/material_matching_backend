"""
SocketIO client that connects to the dispatcher as a user. Starting from a locally saved json file describing the
matching problem scenario, iterates through relevant model variables and logs the results
"""

import argparse
import json
import socketio
import time

sio = socketio.Client()

solving_flag: bool = False

log_path = ""


def main():
    global solving_flag, log_path

    parser = argparse.ArgumentParser(description='Give dispatcher IP address')
    parser.add_argument("-i", "--ip")
    args = vars(parser.parse_args())
    print(f"Looking for dispatcher at {args['ip']}")
    if "ip" in args:
        sio.connect(f"http://{args['ip']}:52323")
    else:
        sio.connect('http://localhost:52323')

    # Load problem file
    if "file" in args:
        filepath = args["file"]
    else:
        filepath = "scenarios/scenario_test_inventory.json"
    with open(filepath) as file:
        scenario_json = json.load(file)

    log_path = f"logs/{scenario_json['name']}.csv"

    items = []
    a_cliques = [-1, 0, 1, 2]
    b_cuts = [-1, 0, 1, 2, 3]
    c_emphasis = [0, 1, 2]
    d_lp_method = [0, 1, 2, 3]
    e_preprocess = [-1, 0, 1]

    """
    for aa in a_cliques:
        for ab in b_cuts:
            for ac in c_emphasis:
                for ad in d_lp_method:
                    for ae in e_preprocess:
                        for i in range(10):
                            item = {"id": len(items),
                                    "clique": aa,
                                    "cuts": ab,
                                    "emphasis": ac,
                                    "lp_method": ad,
                                    "preprocess": ae,
                                    "max_nodes": 10000,  # 10,000 - 100,000
                                    "max_seconds": 30}   # 30 - 300
                            items.append(item)
    
    

    print(f"Test run has {len(items)} items")

    # Offset from start if iteration has been stopped early and restarted
    start = 0
    for i in range(start, len(items)):
        item = items[i]
        scenario_json["model_args"] = item
        print(item)
        sio.emit("solve_request", scenario_json)

        solving_flag = True
        while solving_flag:
            time.sleep(0.1)
    """

    print(len(scenario_json["stock_lengths"]))

    stock_lengths = scenario_json["stock_lengths"]
    count = 0
    for i in range(0, len(stock_lengths)):
        for j in range(1):
            scenario_json["model_args"] = {"id": i, # set back to count if j > 1
                                           "max_nodes": 1073741824,
                                           "max_seconds": 300,
                                           "max_nodes_same_incumbent": 10000}
            scenario_json["stock_lengths"] = stock_lengths[0:len(stock_lengths) - i]
            sio.emit("solve_request", scenario_json)
            print(f"{count} : {i}")
            solving_flag = True
            while solving_flag:
                time.sleep(0.1)
            count += 1




@sio.event
def connect():
    print('connection established')
    sio.emit("client_id", {'type': 'user', 'name': 'Model Testing'})


@sio.on('solve_response')
def solve_response(data):
    global solving_flag
    solving_flag = False
    log_string = data["log_string"] + "\n"
    with open(log_path, "a") as f:
        f.write(log_string)
    print("Received solve response")


@sio.on('solve_infeasible')
def solve_infeasible(data):
    global solving_flag
    print("Infeasible")
    log_string = data["log_string"] + "\n"
    with open(log_path, "a") as f:
        f.write(log_string)
    solving_flag = False


@sio.event
def disconnect():
    print('disconnected from server')


if __name__ == '__main__':
    main()

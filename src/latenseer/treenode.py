from collections import defaultdict, OrderedDict
from typing import Dict

from .probdist import CDF, PMF


class AutoDict(dict):
    def __init__(self, factory):
        self.factory = factory

    def __missing__(self, key):
        self[key] = self.factory(key)
        return self[key]


class TreeNode(object):
    def __init__(self, endpoint) -> None:
        self.endpoint: str = endpoint
        self.service: str = None
        self.name: str = None
        self.count: int = 0
        self.tracecount: int = 0
        self.setnode = False
        self.parent: TreeNode = None
        self.depth: int = -1
        self.children: AutoDict = AutoDict(lambda key: TreeNode(key))
        
        self.latency = PMF()  # span duration distribution
        self.latency_cdf = None
        self.siblings = dict()  # sibling succession latency distribution
        self.siblings_cdf = dict()
        self.relative_start = PMF()  # Relative start time to its parent
        self.relative_start_cdf = None
        self.relative_finish = PMF()  # Relative finish time to its parent
        self.relative_finish_cdf = None
        self.DAG = OrderedDict() # Causal dependency graph for sibling nodes
        self.predict_latency = None
        
        self.joint = PMF()
        # FIXME: ignore merge for now
        # self.trigger = defaultdict(float)
        self.trigger = defaultdict(lambda: defaultdict(float))
        
        # slack analysis related, just for propagation process    
        self.slack = PMF()

    def merge(self, other):
        assert self.endpoint == other.endpoint

        self.count += other.count
        self.tracecount += other.tracecount
        for k, v in other.children.items():
            self.children[k].merge(v)

    # Syntactic suger for getting a child node by endpoint
    def __getitem__(self, key):
        return self.children[key]      

    def print(self, indent=0):
        if self.setnode:
            print("*" * indent + f"S { {self.endpoint} } count={self.count}")
        # for key in self.parallel_groups:
        #   print(key, end=',')
        #   print(self.parallel_groups[key]['first'].endpoint, self.parallel_groups[key]['last'].endpoint)
        if self.setnode is False:
            print(
                " " * indent + f"{self.service} depth={self.depth}"
            )
        for c in self.children.values():
            c.print(indent + 1)
            
    def print_test(self, indent=0):
        if self.setnode:
            # print("*" * indent + f"S { {self.endpoint} } count={self.count}")
            # if len(self.DAG) >= 2:
            #     print("*" * indent + f"S { {self.endpoint} } count={self.count}")
            for i in self.DAG:
                nodes = self.DAG[i]['nodes']
                for serial_nodes in nodes:
                    print(" " * indent + f"p{i}: {len(serial_nodes)}", end=" ")
                print()
        # for key in self.parallel_groups:
        #   print(key, end=',')
        #   print(self.parallel_groups[key]['first'].endpoint, self.parallel_groups[key]['last'].endpoint)
        # if self.setnode is False:
        #     print(
        #         " " * indent + f"{self.service} {self.trigger}"
        #     )
        for c in self.children.values():
            c.print_test(indent + 1)
            
    def print_siblings(self, indent=0):
        if self.setnode:
            print(" " * indent + f"S { {self.endpoint} } count={self.count}")
            print(" " * indent + f"******DAG******")
            for i in self.DAG:
                nodes = self.DAG[i]['nodes']
                print(" " * indent + f"first node: {self.DAG[i]['first'].endpoint}")
                print(" " * indent + f"last node: {self.DAG[i]['last'].endpoint}")
                for serial_nodes in nodes:
                    print(" " * indent + f"{i}:", end=" ")
                    for node in serial_nodes:
                        print(" " * indent + f"{node.endpoint}", end=",")
                    print()
        for c in self.children.values():
            c.print_siblings(indent + 1)

    def print_server_latency(self, indent=0):
        print(" " * indent + f"{self.endpoint} {self.latency}")
        for c in self.children.values():
            c.print_server_latency(indent + 1)

    def print_nol_latency(self, indent=0):
        if self.setnode:
            print(" " * indent + f"S {self.count}")
            print(" " * indent + f"{self.siblings}")
        else:
            print(" " * indent + f"{self.endpoint}")

        for c in self.children.values():
            c.print_nol_latency(indent + 1)
            
    def clean_node(self):
        self.predict_latency = None
        self.trigger = defaultdict(lambda: defaultdict(float))
        for c in self.children.values():
            c.clean_node()

    def dot_nodes(self, str, thresh=1):
        if self.count < thresh:
            return
        str.append(f'n{id(self)} [label="{self.endpoint}"]')
        for c in self.children.values():
            c.dot_nodes(str, thresh)

    def dot_edges(self, str, parentid, thresh=1):
        if self.count < thresh:
            return
        str.append(
            f'n{parentid} -> n{id(self)} [label="{self.count}/{self.tracecount}"]'
        )
        for c in self.children.values():
            c.dot_edges(str, id(self), thresh)

    def dot_graph_collect(self, nodes, edges):
        nodes[self.endpoint] = id(self)
        for c in self.children.values():
            edges[(self.endpoint, c.endpoint)] += c.count
            c.dot_graph_collect(nodes, edges)

    def to_json(self):
        return {
            "endpoint": self.endpoint,
            "count": self.count,
            "tracecount": self.tracecount,
            "setnode": self.setnode,
            "siblings": self.siblings,
            "latency": self.latency,
            "children": [v.to_json() for k, v in self.children.items()],
        }
        
    def merge_json(self, node):
        if node["setnode"]:
            self.setnode = True
            for pair in node["siblings"]:
                if pair in self.siblings:
                    
                    self.siblings[pair] += node["siblings"][pair]
                else:
                    self.siblings[pair] = node["siblings"][pair]
            self.count += node["count"]
            self.tracecount += node["tracecount"]
            # parentnode = self.parent
            # setnode = parentnode[node['endpoint']]
            # setnode.parent = parentnode
            # setnode.setnode = True
            # setnode.sibings = node['siblings']
        else:
            assert self.endpoint == node["endpoint"]
            self.latency += node["latency"]
            self.count += node["count"]
            self.tracecount += node["tracecount"]

        for c in node["children"]:
            self.children[c["endpoint"]].merge_json(c)

    def num_nodes(self):
        return 1 + sum([c.num_nodes() for c in self.children.values()])
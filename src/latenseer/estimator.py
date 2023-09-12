from collections import namedtuple, defaultdict
from typing import List

from .treenode import TreeNode
from .tree import Tree
from .probdist import PMF, CDF
from .utils import removeprefix, removesuffix

class Estimator(object):
    
    def __init__(self, tree: Tree):
        self.tree = tree
        self.slack_profile = {}
        self.agg_service_slack = defaultdict(PMF)
        self.agg_endpoint_slack = defaultdict(lambda: defaultdict(PMF))
        
    def __str__(self):
        return str(self.tree)

    def __repr__(self):
        return str(self)
    
    def estimate(self, es_type: str):
        if es_type == "latency":
            self.estimate_latency()
        elif es_type == "slack":
            self.clean_slack()
            root = self.tree.root
            # Initialize the slack for root
            root.slack.update_pmf(0)
            
            self.estimate_slack(root)
            
        else:
            raise ValueError(f"Unknown estimation type: {es_type}")
    
    def estimate_slack(self, node: TreeNode, thre=50):
        
        setnodes = list(node.children.values())
        for snode in setnodes:
            # print('setnode: ', snode.endpoint)
            snode.slack = node.slack.normalize()
            for latency_tuple in snode.joint.keys():
                self._propagate_slack(latency_tuple, snode)
        
            for child_node in snode.children.values():
                self.estimate_slack(child_node)

    def _propagate_slack(self, 
                         latency_tuple, 
                         setnode: TreeNode,
                         ):
        """
        Calculate slack for services in a given setnode and update slack_profile.
        To simply the calculation, we only consider the individual slack
        for each service under a given setnode. 
        The aggregated slack will be calculated later.
        """
        
        def __get_serial_latency(latencies, 
                                 serial_nodes: List[TreeNode]):
            """
            Notes: we neglect the network latency between serial nodes; however,
            if we are only considering client-side latency, we don't need to 
            worry about the network latency.
            """
            total_latency = 0
            for snode in serial_nodes:
                node_latency = getattr(latencies, snode.endpoint)
                total_latency += node_latency
            return total_latency
        
        def __cal_slack_helper(latency_tuple,
                               mix_nodes: List[List[TreeNode]]):
            sub_parallel_latencies = []
            for serial_nodes in mix_nodes:
                sub_parallel_latencies.append(__get_serial_latency(latency_tuple, serial_nodes))
            
            max_latency = max(sub_parallel_latencies)

            for i in range(len(mix_nodes)):
                serial_nodes = mix_nodes[i]
                delta = max_latency - sub_parallel_latencies[i]
                for node in serial_nodes:
                    for key, prob in setnode.slack.items():
                        node.slack.update_pmf(key + delta, prob)
                    clean_interface = removeprefix(node.name, "_")
                    slack_key = (node.service, clean_interface, node.depth)
                    
                    if slack_key not in self.slack_profile:
                        self.slack_profile[slack_key] = node.slack.normalize()
                    else:
                        self.slack_profile[slack_key] += node.slack.normalize()
                        
        
        for pid in setnode.DAG:
            # the nodes in a parallel group
            mix_sp_nodes = setnode.DAG[pid]['nodes']
            __cal_slack_helper(latency_tuple, mix_sp_nodes)
            
            
    def aggregate_slack(self):
        """
        Aggregate slack information stored in slack_profile:
            - Remove client-side slack
            - Calculate minimum distribution for a same group
            - Return 1) groupby service 2) groupby endpoint
        """
        for key, pmf in self.slack_profile.items():
            service = key[0]
            endpoint = key[1]
            # if (removesuffix(service, 'service') == removesuffix(endpoint, 'server'))\
            #     or endpoint.endswith('client'):
            if endpoint.endswith('client'):
                continue
            
            self.agg_service_slack[service] += pmf
            self.agg_endpoint_slack[service][endpoint] += pmf
        

    def get_slack_profile(self):
        for key in self.slack_profile:
            self.slack_profile[key].normalize()
        
        return self.slack_profile
    
    def clean_slack(self):
        self.slack_profile = {}
        self.agg_service_slack = defaultdict(PMF)
        self.agg_endpoint_slack = defaultdict(lambda: defaultdict(PMF))

    
    def estimate_latency(self):
        # Need to reorganize the code (move the propagation func here)
        raise NotImplementedError()
        
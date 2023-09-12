from collections import defaultdict, namedtuple
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
import random

from .treenode import TreeNode
from .probdist import ADD_OP, MAX_OP, MERGE_OP, PMF, CDF
from .utils import removeprefix


class Tree(object):
    def __init__(self, trace_path=None, threshold=50, trace_df=None, twr_traces=None, twr_traceids=[]) -> None:
        self.trace_path: str = trace_path
        self.trace_df: pd.DataFrame = trace_df
        self.root: TreeNode = None
        self.threshold: int = threshold
        self.count_thre: int = 0
        self.leafnodes = set()
        self.num_nodes = -1
        
        ## test twrtrace tree
        self.twr_traces = twr_traces
        self.twr_traceids = twr_traceids

    def _set_latency_measures_helper(self, node):
        """Calculate the latency cdf of sibling pairs"""
        if node.setnode:
            # Calcualte the cdf of succession times
            for pair in node.siblings:
                if node.siblings[pair] is not None:
                    node.siblings[pair].normalize()
                    node.siblings_cdf[pair] = CDF(node.siblings[pair])
                else:
                    node.siblings_cdf[pair] = None
        else:
            # Calculate the latency cdf of each node
        #     node.latency_cdf = CDF(node.latency)
            # node.latency.normalize()
            # node.relative_start.normalize()
            # node.relative_finish.normalize()
            node.relative_start_cdf = CDF(node.relative_start)
            node.relative_finish_cdf = CDF(node.relative_finish)
        for c in node.children.values():
            self._set_latency_measures_helper(c)
    
    def set_latency_measures(self):
        """Calculate the latency cdf of each node and sibling pairs"""
        for c in self.root.children.values():
            self._set_latency_measures_helper(c)

    def _search_by_service_helper(self, node, services, targets, issetnode):
        # test_endpoints = ['memcached', 'redis', 'mongo', 'server']
        test_endpoints = ['memcached', 'redis', 'mongo']
        if node.service in services \
            and not any(ele in node.endpoint for ele in test_endpoints) \
            and node.setnode is issetnode:
            targets.append(node)
        for c in node.children.values():
            self._search_by_service_helper(c, services, targets, issetnode)
    
    def _search_by_endpoint_helper(self, node, target_name, targets, issetnode):
        if node.endpoint == target_name and node.setnode is issetnode:
            targets.append(node)
        for c in node.children.values():
            self._search_by_endpoint_helper(c, target_name, targets, issetnode)
            
    def search_by_name(self, name_type='service', target_name=None, issetnode=False):
        """
        Search the node by name
        """
        targets = []
        if target_name == 'root':
            return [self.root]
        if name_type == 'service':
            for node in self.root.children.values():
                self._search_by_service_helper(node, target_name, targets, issetnode)
        else:
            for node in self.root.children.values():
                self._search_by_endpoint_helper(node, target_name, targets, issetnode)
        
        if len(targets) == 0:
            raise Exception(f'No node named {target_name} found')
        return targets
    
    def find_injection_nodes(self, services):
        """
        Traverse the tree and find nodes that should be delayed.
        """
        # Find the nodes with the given service names as target nodes
        target_nodes = []
        for node in self.root.children.values():
            self._search_by_service_helper(node, services, target_nodes, False)
        
        # for node in target_nodes:
        #     print(node.service, ', ', node.endpoint)
        
        # Attach the direct child nodes to the target nodes
        child_nodes = []
        for tnode in target_nodes:
            setnodes = tnode.children.values()
            for snode in setnodes:
                child_nodes.extend(list(snode.children.values()))
        
        target_nodes.extend(child_nodes)
        return set(target_nodes)

    
    def from_trace(self, num_traces, trace_type='alibaba', sample=False, min_spans=-1, max_spans=-1, max_width=-1):
        def update_sibling_pairs(setnode, trace, siblings, sibling_node_names):
            """ 
            Record the succession time of sibling pairs
            """
            n = len(siblings)
            for i in range(n-1):
                s1_starttime = trace.iloc[siblings[i]]['timestamp']
                s1_finishtime = trace.iloc[siblings[i]]['timestamp'] \
                                + abs(trace.iloc[siblings[i]]['rt'])
                for j in range(i+1, n):
                    s2_starttime = trace.iloc[siblings[j]]['timestamp']
                    s2_finishtime = trace.iloc[siblings[j]]['timestamp'] \
                                + abs(trace.iloc[siblings[j]]['rt'])
                    # to avoid duplicate pairs (s1, s2) and (s2, s1)
                    pair = (sibling_node_names[i], sibling_node_names[j]) \
                            if sibling_node_names[i] >= sibling_node_names[j] \
                            else (sibling_node_names[j], sibling_node_names[i])
                    # we don't need to compare s1_starttime and s2_starttime, because we have already sorted the siblings by their start time
                    succession_time = s2_starttime - s1_finishtime
                    if pair in setnode.siblings:
                        setnode.siblings[pair].update_pmf(succession_time)
                    else:
                        setnode.siblings[pair] = PMF()
                        setnode.siblings[pair].update_pmf(succession_time)
            if n == 1:
                pair = (sibling_node_names[0], None)
                setnode.siblings[pair] = None
        
        def traverse_siblings(trace, children):
            """
            Traverse the siblings of a setnode and return the list of sibling names
            """
            # TODO: Pre-merging the known parallel siblings
            
            # Sort the sibling spans by relative start time
            initial_sibling_starttimes = [trace.iloc[i]['timestamp'] for i in children]
            sort_index = np.argsort(initial_sibling_starttimes)
            siblings = [children[i] for i in sort_index]
            
            # Get the sibling span names
            sibling_span_names = [trace.iloc[i]["dm"] + "_" + trace.iloc[i]["interface"] for i in siblings]
            
            # Get the sibling node names by 
            name_records = defaultdict(int)
            sibling_node_names = []
            ordered_sibling_spans = []
            i = 0
            j = 1
            while i < len(siblings):
                # Inplicitly assume the consecutive sibling spans with same service name and endpoint (interface) are parallel,
                # and we only keep the first one
                while j < len(siblings) and sibling_span_names[j] == sibling_span_names[i]:
                    j += 1
                name_records[sibling_span_names[i]] += 1
                sibling_node_names.append(sibling_span_names[i] + "_" + str(name_records[sibling_span_names[i]]))
                ordered_sibling_spans.append(siblings[i])
                i = j
                j += 1
            
            return sibling_node_names, ordered_sibling_spans

        def get_durations(span, trace, i, ordered_sibling_spans, sibling_node_names, durations):
            """
            Get the sub-durations of a parent span
            """
            n = len(ordered_sibling_spans)
            childspan = trace.iloc[ordered_sibling_spans[i]]
            prev_span = trace.iloc[ordered_sibling_spans[i-1]] if i >= 1 else None
            p_st = span['timestamp']
            c_st = childspan['timestamp']
            p_dt = abs(span['rt'])
            c_dt = abs(childspan['rt'])
            
            if n == 1:
                relative_start_duration = c_st - p_st
                relative_finish_duration = p_st + p_dt - c_st - c_dt
                # sub_durations = [relative_start_duration,
                #                  c_dt,
                #                  relative_finish_duration]
                durations[sibling_node_names[0]] = c_dt
                durations['rs'] = relative_start_duration
                durations['rf'] = relative_finish_duration
            else:
                if i == 0:
                    relative_start_duration = c_st - p_st
                    # sub_durations = [relative_start_duration,
                    #                  c_dt]
                    durations[sibling_node_names[i]] = c_dt
                    durations['rs'] = relative_start_duration
                elif i == n - 1:
                    relative_finish_duration = p_st + p_dt - c_st - c_dt
                    successtion_duration = c_st - prev_span['timestamp'] - abs(prev_span['rt'])
                    # sub_durations = [successtion_duration,
                    #                  c_dt,
                    #                  relative_finish_duration]
                    durations[sibling_node_names[i]] = c_dt
                    durations['rf'] = relative_finish_duration
                    durations[sibling_node_names[i-1] + '_' + sibling_node_names[i]] = successtion_duration
                else:
                    successtion_duration = c_st - prev_span['timestamp'] - abs(prev_span['rt'])
                    # sub_durations = [successtion_duration,
                    #                  c_dt]
                    durations[sibling_node_names[i]] = c_dt
                    durations[sibling_node_names[i-1] + '_' + sibling_node_names[i]] = successtion_duration
                    
        def propagate_counts(trace, curidx, traceview, treenode, depth):
            
            span = trace.iloc[curidx]
            treenode.count += 1
            
            if trace_type == 'alibaba':
                span_rpcid = span['rpcid']
                children = traceview[span_rpcid]
            elif trace_type == 'dsb':
                children = traceview[span['spanid']]
            else:
                raise Exception("Unknown trace type")
            
            pstarttime = span['timestamp']
            pduration = abs(span['rt'])
            
            # Create a setnode and check if this setnode exists
            if len(children):
                sibling_node_names, ordered_sibling_spans \
                    = traverse_siblings(trace, children)
                setnode_name = "|".join(sorted(sibling_node_names))
                setnode = treenode[setnode_name]
                setnode.count += 1
                setnode.setnode = True
                setnode.parent = treenode
                setnode.latency.update_pmf(pduration)
                update_sibling_pairs(setnode, trace, ordered_sibling_spans, sibling_node_names)
            
                # Create regular nodes
                durations = {}
                for i in range(len(ordered_sibling_spans)):
                    childspan = trace.iloc[ordered_sibling_spans[i]]
                    subnode = setnode[sibling_node_names[i]]
                    subnode.parent = setnode
                    subnode.depth = depth
                    subnode.service = childspan['dm']
                    subnode.name = childspan['interface']
                    subnode.latency.update_pmf(abs(childspan['rt']))
                    subnode.count += 1
                    relative_start_duration = (childspan['timestamp']-pstarttime)
                    relative_finish_duration = (
                                                pstarttime
                                                + pduration
                                                - childspan['timestamp']
                                                - childspan['rt']
                                                )
                    subnode.relative_start.update_pmf(relative_start_duration)
                    subnode.relative_finish.update_pmf(relative_finish_duration)
                    
                    get_durations(span, trace, i, ordered_sibling_spans, sibling_node_names, durations)
                    if i == len(ordered_sibling_spans) - 1:
                        attrnames = [sibling_node_names[i] for i in range(len(sibling_node_names))]
                        attrnames.extend(['rs', 'rf'])
                        attrnames.extend([sibling_node_names[i-1] + '_' + sibling_node_names[i] 
                                          for i in range(1, len(sibling_node_names))])                        
                        Joint = namedtuple("Joint", attrnames)
                        cur_joint = Joint(**durations)
                        setnode.joint.update_pmf(cur_joint)
                        # setnode.test_joint.update_pmf(tuple(durations))
                        
                    # Recursively propagate the counts
                    propagate_counts(trace, ordered_sibling_spans[i], traceview, subnode, depth+1)

        def rpcid_make_trace_tree(trace):
            root_rpcid = trace['rpcid'].min()
            treeview = defaultdict(list)
            rootidx = 0
            for index, span in trace.iterrows():
                if span['rpcid'] != root_rpcid:
                    parent_rpcid = span['rpcid'][:-2]
                    treeview[parent_rpcid].append(index)
                else:
                    rootidx = index
            return treeview, rootidx
        
        def parentid_make_trace_tree(trace):
            treeview = defaultdict(list)
            rootidx = 0
            for index, span in trace.iterrows():
                if not pd.isna(span['parentid']):
                    treeview[span['parentid']].append(index)
                else:
                    rootidx = index
            return treeview, rootidx


        # Initialize the aggregation tree
        rootnode = TreeNode("root")
        
        if trace_type == 'alibaba':
            # Add traces to the tree
            if self.trace_path is not None:
                df = pd.read_csv(self.trace_path, dtype={'rpcid':str}).fillna('None')
            elif self.trace_df is not None:
                df = self.trace_df
            else:
                raise ValueError("Either trace_path or trace_df must be provided")
            
            traceids = df['traceid'].unique()
        
            # tmp_remove_ids = df[df['dm'] == 'd2708']['traceid'].unique()
            # traceids = set(traceids) - set(tmp_remove_ids)
            # df = df[df['traceid'].isin(traceids)]
            
            # tmp_remove_ids2 = df[(df['dm'] == 'd7067') & (df['interface'] == 'i1185')]['traceid'].unique()
            # traceids = df['traceid'].unique()
            # traceids = list(set(traceids) - set(tmp_remove_ids2))

            if sample:
                sample_traceids = random.sample(list(traceids), num_traces)
            else:
                sample_traceids = traceids
                
            for traceid in tqdm(sample_traceids):
                trace_df = df[df['traceid'] == traceid]
                trace_df = trace_df.replace('(?)', 'None')
                trace_df = trace_df[~((trace_df['rpctype'] == 'rpc') & (trace_df['rt'] < 0))].reset_index(drop=True)
                # if len(trace_df) != 0 and trace_df['rpcid'].min() == '0':
                if len(trace_df) != 0:
                    if len(trace_df) > 1000:
                        print(traceid)
                        continue
                    traceview, rootidx = rpcid_make_trace_tree(trace_df)
                    rootnode.latency.update_pmf(abs(trace_df.iloc[rootidx]['rt']))
                    propagate_counts(trace_df, rootidx, traceview, rootnode, 0)
        
        elif trace_type == 'dsb':
            if self.trace_path is not None:
                df = pd.read_csv(self.trace_path)
            elif self.trace_df is not None: 
                df = self.trace_df
            else:
                raise ValueError("Either trace_path or trace_df must be provided")

            traceids = df['traceid'].unique()
            
            if sample:
                sample_traceids = random.sample(list(traceids), num_traces)
            else:
                sample_traceids = list(traceids)
            
            for traceid in tqdm(sample_traceids):
                trace_df = df[df['traceid'] == traceid].reset_index(drop=True)
                traceview, rootidx = parentid_make_trace_tree(trace_df)
                rootnode.latency.update_pmf(abs(trace_df.iloc[rootidx]['rt']))
                propagate_counts(trace_df, rootidx, traceview, rootnode, 0)
                
        elif trace_type == 'twitter':
            traceids = self.from_twitter_traces(rootnode, sample, num_traces, min_spans, max_spans, max_width)
        else:
            raise ValueError("trace_type must be one of 'alibaba', 'dsb', or 'twitter'")
        
        self.root = rootnode
        
        # set cdf for latency measures
        self.set_latency_measures()

    
    def from_twitter_traces(self, rootnode, sample=False, num_traces=1000, min_spans=-1, max_spans=-1, max_width=-1):
        
        def twr_traverse_siblings(trace, children):
            # PARALLEL = ['service_1373_endpoint_181', 'service_1581_endpoint_19']
            PARALLEL = ['service_1373', 'service_1581', 'service_944', 
                        'service_1284', 'service_1413', 'service_205', 
                        'service_941', 'service_181', 'service_964',
                        'service_843', 'service_1538', 'service_925',
                        'service_1589', 'service_1280', 'service_1009',
                        'service_748',
                        ]
            
            initial_sibling_starttimes = []
            for i in children:
                span = trace[i]
                times = {}
                for d in span['annotations']:
                    times[d['value']] = d['timestamp']
                initial_sibling_starttimes.append(times['cs'])
            
            sort_index = np.argsort(initial_sibling_starttimes)
            siblings = [children[i] for i in sort_index]
            
            # Get the sibling span names
            sibling_span_names = []
            for i in siblings:
                servicename = trace[i]['remoteEndpoint']['serviceName'].replace('-', '_') if len(trace[i]['remoteEndpoint']['serviceName']) else 'tmp'
                name = trace[i]["name"].replace('-', '_') if len(trace[i]["name"]) else 'tmp'
                tmp_name = removeprefix(servicename + "_" + name, '_')
                sibling_span_names.append(tmp_name)
                
            # sibling_span_names = [removeprefix(trace[i]['remoteEndpoint']['serviceName'].replace('-', '_') + "_" + trace[i]["name"].replace('-', '_'), '_')
            #                       for i in siblings]
            
            name_records = defaultdict(int)
            sibling_node_names = []
            ordered_sibling_spans = []
            
            i = 0
            j = 1
            while i < len(siblings):
                # Inplicitly assume the consecutive sibling spans with same service name and endpoint (interface) are parallel,
                # and we only keep the first one
                # while j < len(siblings) and sibling_span_names[j] == sibling_span_names[i]:
                while j < len(siblings) and sibling_span_names[j] == sibling_span_names[i] and any(name in sibling_span_names[i] for name in PARALLEL):
                    j += 1
                name_records[sibling_span_names[i]] += 1
                sibling_node_names.append(sibling_span_names[i] + "_" + str(name_records[sibling_span_names[i]]))
                ordered_sibling_spans.append(siblings[i])
                i = j
                j += 1
            
            # if (len(sibling_span_names) - len(sibling_node_names)) > 20:
            #     print(len(sibling_span_names), len(sibling_node_names))
            #     print(sibling_node_names)
            
            # print(len(sibling_span_names), len(sibling_node_names), len(ordered_sibling_spans))
            
            return sibling_node_names, ordered_sibling_spans
        
        def twr_update_sibling_pairs(setnode, trace, siblings, sibling_node_names):
            n = len(siblings)
            for i in range(n-1):
                s1 = trace[siblings[i]]
                s1_times = {}
                for d in s1['annotations']:
                    s1_times[d['value']] = d['timestamp']
                
                # for j in range(i+1, n):
                j = i + 1
                s2 = trace[siblings[j]]
                s2_times = {}
                for d in s2['annotations']:
                    s2_times[d['value']] = d['timestamp']
                
                pair = (sibling_node_names[i], sibling_node_names[j]) \
                        if sibling_node_names[i] >= sibling_node_names[j] \
                        else (sibling_node_names[j], sibling_node_names[i])
                succession_time = s2_times['cs'] - s1_times['cr']
                if pair in setnode.siblings:
                    setnode.siblings[pair].update_pmf(succession_time)
                else:
                    setnode.siblings[pair] = PMF()
                    setnode.siblings[pair].update_pmf(succession_time)
            if n == 1:
                pair = (sibling_node_names[0], None)
                setnode.siblings[pair] = None
        
        def twr_get_durations(span, trace, i, ordered_sibling_spans, sibling_node_names, durations):
            
            n = len(ordered_sibling_spans)
            childspan = trace[ordered_sibling_spans[i]]
            prev_span = trace[ordered_sibling_spans[i-1]] if i >= 1 else None
            
            child_times = {}
            for d in childspan['annotations']:
                child_times[d['value']] = d['timestamp']
            
            if i >= 1:
                prev_times = {}
                for d in prev_span['annotations']:
                    prev_times[d['value']] = d['timestamp']
                
            p_times = {}
            for d in span['annotations']:
                p_times[d['value']] = d['timestamp']
            
            if span['is_root']:
                p_times['cs'] = p_times['ss']
                p_times['cr'] = p_times['sr']
            
            if n == 1:
                relative_start_duration = child_times['cs'] - p_times['cs']
                relative_finish_duration = p_times['cr'] - child_times['cs']

                durations[sibling_node_names[0]] = child_times['cr'] - child_times['cs']
                durations['rs'] = relative_start_duration
                durations['rf'] = relative_finish_duration
            else:
                if i == 0:
                    relative_start_duration = child_times['cs'] - p_times['cs']
                    # sub_durations = [relative_start_duration,
                    #                  c_dt]
                    durations[sibling_node_names[i]] = child_times['cr'] - child_times['cs']
                    durations['rs'] = relative_start_duration
                elif i == n - 1:
                    relative_finish_duration = p_times['cr'] - child_times['cs']
                    successtion_duration = child_times['cs'] - prev_times['cr']
                    durations[sibling_node_names[i]] = child_times['cr'] - child_times['cs']
                    durations['rf'] = relative_finish_duration
                    durations[sibling_node_names[i-1] + '_' + sibling_node_names[i]] = successtion_duration
                else:
                    successtion_duration = child_times['cs'] - prev_times['cr']
                    durations[sibling_node_names[i]] = child_times['cr'] - child_times['cs']
                    durations[sibling_node_names[i-1] + '_' + sibling_node_names[i]] = successtion_duration
        
        def twr_propagate_counts(trace, curidx, traceview, treenode, depth):
            span = trace[curidx]
            treenode.count += 1
            
            if span['is_root']:
                pstarttime = span['timestamp']
                pduration = span['duration']
                pfinish = pstarttime + pduration
            else:
                times = {}
                for d in span['annotations']:
                    times[d['value']] = d['timestamp']
                pstarttime = times['cs']
                pduration = times['cr'] - times['cs']
                pfinish = times['cr']
            
            children = traceview[span['id']]
            
            # Create a setnode and check if this setnode exists
            if len(children):
                sibling_node_names, ordered_sibling_spans \
                    = twr_traverse_siblings(trace, children)
                setnode_name = "|".join(sorted(sibling_node_names))
                setnode = treenode[setnode_name]
                setnode.count += 1
                setnode.setnode = True
                setnode.parent = treenode
                setnode.latency.update_pmf(pduration)
                twr_update_sibling_pairs(setnode, trace, ordered_sibling_spans, sibling_node_names)
            
                # Create regular nodes
                durations = {}
                for i in range(len(ordered_sibling_spans)):
                    childspan = trace[ordered_sibling_spans[i]]
                    ctimes = {}
                    for d in childspan['annotations']:
                        ctimes[d['value']] = d['timestamp']
                    subnode = setnode[sibling_node_names[i]]
                    subnode.parent = setnode
                    subnode.depth = depth
                    subnode.service = childspan['remoteEndpoint']['serviceName'] if len(childspan['remoteEndpoint']['serviceName']) else 'tmp'
                    subnode.name = childspan['name'] if len(childspan['name']) else 'tmp'
                    subnode.latency.update_pmf(ctimes['cr'] - ctimes['cs'])
                    subnode.count += 1
                    relative_start_duration = (ctimes['cs']-pstarttime)
                    relative_finish_duration = (pfinish - ctimes['cr'])
                    subnode.relative_start.update_pmf(relative_start_duration)
                    subnode.relative_finish.update_pmf(relative_finish_duration)
                    
                    twr_get_durations(span, trace, i, ordered_sibling_spans, sibling_node_names, durations)
                    
                    if i == len(ordered_sibling_spans) - 1:
                        # print(curidx)
                        attrnames = [removeprefix(sibling_node_names[i], '_') for i in range(len(sibling_node_names))]
                        attrnames.extend(['rs', 'rf'])
                        attrnames.extend([removeprefix(sibling_node_names[i-1] + '_' + sibling_node_names[i], '_')
                                          for i in range(1, len(sibling_node_names))])     
                                
                        Joint = namedtuple("Joint", attrnames)
                        cur_joint = Joint(**durations)
                        setnode.joint.update_pmf(cur_joint)
                        # setnode.test_joint.update_pmf(tuple(durations))
                        
                    # Recursively propagate the counts
                    twr_propagate_counts(trace, ordered_sibling_spans[i], traceview, subnode, depth+1)
        
        def twr_make_trace_tree(trace: List[Dict[str, Any]]):
            treeview = defaultdict(list)
            rootidx = 0
            for index, span in enumerate(trace):
                if span['is_root']:
                    rootidx = index
                    continue
                if span['kind'] == 'SERVER':
                    continue
                treeview[span['parentId']].append(index)
            
            width = [len(w) for w in treeview.values()]
            return treeview, rootidx, width
        
        
        if len(self.twr_traces) == 0:
            raise ValueError("Twitter traces not loaded")
        
        traceids = list(self.twr_traces.keys())
        
        if min_spans > -1 and max_spans > -1:
            new_traceids = []
            for traceid in traceids:
                if len(self.twr_traces[traceid]) // 2 <= max_spans \
                    and len(self.twr_traces[traceid]) // 2 >= min_spans:
                    new_traceids.append(traceid)
            traceids = new_traceids
        
        if sample:
            sample_traceids = random.sample(list(traceids), num_traces)
        else:
            sample_traceids = traceids
            
        if len(self.twr_traceids):
            sample_traceids = [traceid for traceid in sample_traceids 
                                        if traceid in self.twr_traceids]
        
        i = 0
        for traceid in tqdm(sample_traceids):
            trace = self.twr_traces[traceid]
            # print(traceid)
            # if len(trace) // 2 > 2000:
            #     continue
            traceview, rootidx, widths = twr_make_trace_tree(trace)
            # print(traceid, len(trace)//2, np.percentile(widths, 95), np.percentile(widths, 99))
            # if width // 200 > max_width:
            #     print(i, traceid, len(trace)//2, width//2)
            #     continue
            # Only root node use the serverside duration
            rootnode.latency.update_pmf(trace[rootidx]['duration'])
            twr_propagate_counts(trace, rootidx, traceview, rootnode, 0)
            i += 1
        print(i, "traces processed")
        return traceids


    def extract_dependency(self):
        """
        Predict the causal dependencies between sibling nodes for the tree
        """
        def set_node_id(nodenames):
            """
            Set the node id for each node in the tree
            """
            name2id = {}
            id2name = {}
            i = 0
            for pair in nodenames:
                if pair[0] not in name2id:
                    name2id[pair[0]] = i
                    id2name[i] = pair[0]
                    i += 1
                if pair[1] not in name2id:
                    name2id[pair[1]] = i
                    id2name[i] = pair[1]
                    i += 1
            return name2id, id2name
        
        def get_first_last_node(nodes):
            """Get the first and last node in a parallel cluster"""
            rs_latencies = [node.relative_start_cdf.Percentile(self.threshold) 
                            for node in nodes]
            rf_latencies = [node.relative_finish_cdf.Percentile(self.threshold)
                            for node in nodes]

            min_rs = min(rs_latencies)
            min_rs_index = rs_latencies.index(min_rs)

            min_fs = min(rf_latencies)
            min_fs_index = rf_latencies.index(min_fs)

            return nodes[min_rs_index], nodes[min_fs_index]
        
        def predict_sibling_dependencies(node):
            """
            Predict the causal dependencies between child nodes 
            for the given node
            """
            # Only setnode stores the sibling pairs
            if node.setnode:
                # if len(node.siblings) > 100:
                #     print(len(node.siblings))
                #     print(node.endpoint)
                name2id, id2name = set_node_id(node.siblings.keys())
                # Initialize the dependency graph by adding edges between 
                # parallel nodes
                G = nx.Graph()
                for pair in node.siblings_cdf:
                    # the sibling node here is the only child of the setnode,
                    # so we just keep the node in the dependency graph
                    if node.siblings_cdf[pair] is None and pair[1] is None:
                        G.add_node(name2id[pair[0]])
                    else:
                        # Add the edge between the two PARALLEL sibling nodes
                        if node.siblings_cdf[pair].Percentile(self.threshold) >= 0:
                            G.add_node(name2id[pair[0]])
                            G.add_node(name2id[pair[1]])
                        else:
                            G.add_edge(name2id[pair[0]], name2id[pair[1]])
                            
                # Group the parallel nodes into clusters
                connected_clusters = list(nx.algorithms.connected_components(G)) ## for twitter
                # connected_clusters = list(nx.algorithms.find_cliques(G)) 
                # print(connected_clusters)
                parallel_clusters = {}
                parallel_clusters_rs = {}
                p = 0
                for cluster in connected_clusters:
                    # subG is used for identifying the serial behaviors within
                    # the parallel cluster
                    subG = nx.Graph()
                    if len(cluster) == 1:
                        subG.add_node(list(cluster)[0])
                        # continue
                    else:
                        all_cluster_pairs = set([(i, j) 
                                            for i in cluster 
                                            for j in cluster if i != j])
                        all_pairs = set([(name2id[pair[0]], name2id[pair[1]])
                                        for pair in node.siblings])
                        cluster_pairs = all_cluster_pairs.intersection(all_pairs)
                        for pair in cluster_pairs:
                            # Add the edge between the two SERIAL sibling nodes
                            if node.siblings_cdf[(id2name[pair[0]], id2name[pair[1]])] \
                                .Percentile(self.threshold) >= 0:
                                subG.add_edge(pair[0], pair[1])
                            else:
                                subG.add_node(pair[0])
                                subG.add_node(pair[1])
                    
                    # Group the serial nodes into clusters
                    sub_parallel_lists = list(nx.algorithms.find_cliques(subG))
                    
                    # Every sublist in this list is a serial cluster
                    sub_parallel_nodes = [[node[id2name[id]] 
                                          for id in sub_list]
                                          for sub_list in sub_parallel_lists]
                    sub_parallel_cluster = []
                    sub_parallel_first_nodes = []
                    for serial_nodes in sub_parallel_nodes:
                        # Sort the serial nodes by their relative start time
                        # FIXME: handle the case where the tdigest object for 
                        # relative start time is empty
                        rs = [node.relative_start_cdf \
                                            .Percentile(self.threshold)
                                            for node in serial_nodes]
                        nodes = [x for _, x in sorted(zip(rs,serial_nodes),
                                                      key = lambda x: x[0])]
                        sub_parallel_cluster.append(nodes)
                        
                        # Get the first node in the serial cluster
                        sub_parallel_first_nodes.append(nodes[0])
                    
                    # Get the first and last node in the parallel cluster
                    parallel_cluster_nodes = [node[id2name[id]] for id in cluster]
                    fnode, lnode = get_first_last_node(parallel_cluster_nodes)
                    
                    
                    # parallel_clusters[p] = sub_parallel_cluster
                    parallel_clusters[p] = {'first': fnode, 
                                            'last': lnode, 
                                            'nodes': sub_parallel_cluster}
                    parallel_clusters_rs[p] = min([
                        node.relative_start_cdf.Percentile(self.threshold) 
                        for node in sub_parallel_first_nodes
                        ])
                    p += 1
                
                # Sort the parallel clusters by their relative start time and 
                # build the DAG for the child nodes
                sorted_pids = sorted(parallel_clusters_rs, 
                                     key=parallel_clusters_rs.get)
                for i in range(len(sorted_pids)):
                    node.DAG[i] = parallel_clusters[sorted_pids[i]]
            
            for c in node.children.values():
                predict_sibling_dependencies(c)
        
        # Start from the child nodes of the root
        for c in self.root.children.values():
            predict_sibling_dependencies(c)

    def get_leaf_nodes(self, node):
        """Return all leaf nodes"""
        if node.setnode is False:
            if len(node.children) == 0:
                self.leafnodes.add(node)
        for c in node.children.values():
            self.get_leaf_nodes(c)
            
    def node_latency_prediction(self, node:TreeNode):
        """
        Calculate the latency from the DAG of a given node
        Given a causal dependency DAG of parent node A:
               -> B ->           -> E ->
             /         \       /         \ 
        start           sync ->           sync -> G -> end
             \         /       \         /
                C -> D           -> F ->

        Denotations:
        l(x) ~ Latency of node x
        st(x, y) ~ Successtion time between node x and y
        rs(x) ~ Relative start time of node x to parent node
        rf(x) ~ Relative finish time of node x to parent node
        
        In this case, assume we have 
        DAG = {0: {'first': B, 'last': D, 'nodes': [[B], [C, D]]},
               1: {'first': E, 'last': F, 'nodes': [[E], [F]]},
               2: {'first': G, 'last': G, 'nodes': [[G]]}}
               }
        Then, the latency of node A is:
        l(A) = 
            max(rs(B)+l(B), rs(C)+l(C)+l(D)+st(C,D))
            + max(st(D,E)+l(E), st(D,F)+l(F))
            + st(F,G)+l(G)+rf(G)
        """
        
        def predict_parallel_cluster_latency(nodes, setnode, ref, lastnode):
            """
            Predict the latency of a parallel cluster
            """
            serial_latencies = []
            print("number of serial activities within one parallel cluster: ", len(nodes))
            for serial_nodes in nodes:
                # get each node latency
                latencies = [node.latency 
                             if node.predict_latency is None 
                             else node.predict_latency
                             for node in serial_nodes]
                # get successtion time between each pair of serial nodes
                for i in range(len(serial_nodes) - 1):
                    s1_name = serial_nodes[i].endpoint
                    s2_name = serial_nodes[i+1].endpoint
                    pair = (s1_name, s2_name) if s1_name >= s2_name \
                                                else (s2_name, s1_name)
                    if pair in setnode.siblings:
                        latencies.append(setnode.siblings[pair])
                # Add latencies between parallel clusters
                # First parallel cluster only needs to add relative_start
                if ref == 'only':
                    latencies.append(serial_nodes[0].relative_start)
                    latencies.append(serial_nodes[-1].relative_finish)
                elif ref == 'first':
                    latencies.append(serial_nodes[0].relative_start)
                # Other parallel clusters need to add successtion time
                # between the last node of the previous parallel cluster and
                # the first node of the current parallel cluster
                else:
                    s1_name = serial_nodes[0].endpoint
                    s2_name = lastnode.endpoint
                    pair = (s1_name, s2_name) if s1_name >= s2_name \
                                                else (s2_name, s1_name)
                    if pair in setnode.siblings:
                        latencies.append(setnode.siblings[pair])

                    if ref == 'last':
                        latencies.append(serial_nodes[-1].relative_finish)
                
                predicted_serial_latency = ADD_OP(latencies)
                serial_latencies.append(predicted_serial_latency)
            
            # Return the predicted latency of the parallel cluster
            predicted_parallel_cluster_latency = MAX_OP(serial_latencies)
            return predicted_parallel_cluster_latency

        setnodes: List[TreeNode] = node.children.values()
        
        setnode_latencies = []
        setnode_weights = []
        for s in setnodes:
            if s.count < self.count_thre:
                continue
            
            parallel_group_latencies = []
            for pid in s.DAG:
                print("parallel cluster:", pid)
                nodes = s.DAG[pid]['nodes']
                
                if len(s.DAG) == 1:
                    # The only parallel cluster
                    pred_pcl = \
                                predict_parallel_cluster_latency(
                                                nodes, 
                                                s, 
                                                ref='only', 
                                                lastnode=None)
                elif pid == 0:
                    # The first parallel cluster
                    pred_pcl = \
                                predict_parallel_cluster_latency(
                                                nodes, 
                                                s, 
                                                ref='first', 
                                                lastnode=None
                                                )
                elif pid == len(s.DAG) - 1:
                    # The last parallel cluster
                    pred_pcl = \
                                predict_parallel_cluster_latency(
                                                nodes, 
                                                s, 
                                                ref='last', 
                                                lastnode=s.DAG[pid-1]['last']
                                                )
                else:
                    # The middle parallel cluster
                    pred_pcl = \
                                predict_parallel_cluster_latency(
                                                nodes, 
                                                s, 
                                                ref='middle', 
                                                lastnode=s.DAG[pid-1]['last']
                                                )
                parallel_group_latencies.append(pred_pcl)
            
            # Add the latency of the parallel clusters
            predicted_setnode_latency = ADD_OP(parallel_group_latencies)
            setnode_latencies.append(predicted_setnode_latency)
            setnode_weights.append(s.count)
            
        if len(setnode_latencies) == 1:
            return setnode_latencies[0]
        else:
            normalized_setnode_weights = [w / sum(setnode_weights) \
                                            for w in setnode_weights]
            return MERGE_OP(setnode_latencies, normalized_setnode_weights)

    def propagate_latency_from_bottom(self):
        """
        Propagate the latency information from the leaf nodes to the root
        """
        # self.root.clean_node()
        self.get_leaf_nodes(self.root)
        # Update the predict_latency of the leaf nodes as propagation trigger
        for node in self.leafnodes:
            node.predict_latency = node.latency
        
        affected_nodes = defaultdict(set)
        for node in self.leafnodes:
            affected_nodes[node.depth].add(node)
        cur_depth = max(affected_nodes.keys())

        while cur_depth >= 0:
            print('cur_depth: ', cur_depth)
            # cur_nodes have the same depth (starting the bottom of the tree)
            cur_nodes = affected_nodes[cur_depth]
            pnodes = set()
            for node in cur_nodes:
                # node.parent.parent is the parent of the node
                # node.parent is the setnode
                pnodes.add(node.parent.parent)
            for pnode in pnodes:
                print('pnode: ', pnode.endpoint)
                pnode.predict_latency = self.node_latency_prediction(pnode)
                affected_nodes[pnode.depth].add(pnode)
            cur_depth -= 1
    
    
    def joint_setnode_latency_prediction(self, setnode:TreeNode):
        """
        TODO: rewrite the comments
        """ 
        
        def get_parallel_delta(parallel_deltas,
                               parallel_latencies):
            origin_max = max(parallel_latencies)
            if len(parallel_deltas) != len(parallel_latencies):
                raise ValueError('The length of parallel_deltas and \
                                 parallel_latencies should be the same')
            n = len(parallel_deltas)
            
            # To reduce the computation complexity, we consider the expected
            # delta for each parallel cluster
            expected_parallel_deltas = [np.average(list(deltas.keys()), 
                                                  weights=list(deltas.values()))
                                       for deltas in parallel_deltas]
            # print(expected_parllel_deltas)                        
            new_max = max([expected_parallel_deltas[i] + parallel_latencies[i] 
                           for i in range(n)])
            
            # delta = new_max - origin_max if new_max > origin_max else 0
            delta = new_max - origin_max
            return delta

        def get_serial_delta(latencies, 
                             serial_nodes: List[TreeNode]):
            # FIXME: this function is not correct
            delta_dict = defaultdict(float)
            total_latency = 0
            for node in serial_nodes:
                name = node.endpoint
                latency = getattr(latencies, name)
                total_latency += latency
                for l in node.latency.keys():
                    if len(node.trigger[l]) == 0:
                        node.trigger[l] = {0: 1}
                node_deltas = node.trigger[latency]

                for delta in node_deltas:
                    delta_dict[delta] += node_deltas[delta]
            return delta_dict, total_latency
        
        def cal_delta(latencies, dependencies):
            """Calculate the delta for each latency combination of siblings
            Return original latency values and the corresponding deltas
            """
            origin_latency = sum(latencies)
            total_delta = 0
            
            for pid in dependencies:
                parallel_deltas = []
                parallel_latencies = []
                for serial_nodes in dependencies[pid]['nodes']:
                    delta_dict, serial_latency = get_serial_delta(latencies, 
                                                             serial_nodes)
                    parallel_deltas.append(delta_dict)
                    parallel_latencies.append(serial_latency)
                    
                p_delta = get_parallel_delta(parallel_deltas, 
                                             parallel_latencies)
                total_delta += p_delta
            
            return origin_latency, total_delta
        
        trigger = defaultdict(lambda: defaultdict(int))
        
        for latency_tuple in setnode.joint.keys():
            # print(latency_tuple)
            origin_latency, delta = cal_delta(latency_tuple, setnode.DAG)
            # print('origin_latency: ', origin_latency, 'delta: ', delta)
            trigger[origin_latency] = {delta: 1}
        setnode.trigger = trigger
    
    def joint_node_latency_prediction(self, node:TreeNode):
        setnodes: List[TreeNode] = list(node.children.values())
        # print('joint_node_latency_prediction:', node.endpoint)
        # Calculate the latency delay for each branch
        for s in setnodes:
            # print('setnode_latency_prediction:', s.endpoint, s.count)
            self.joint_setnode_latency_prediction(s)
        
        ## update the trigger of the node
        trigger = defaultdict(lambda: defaultdict(float))
        total_count = sum([s.count for s in setnodes])
        for s in setnodes:
            for latency in s.trigger.keys():
                for delta in s.trigger[latency].keys():
                    # trigger[latency][delta] += s.count / total_count
                    trigger[latency][delta] += 1
        
        # normalize the trigger
        for latency in trigger:
            total_weight = sum(trigger[latency].values())
            for delta in trigger[latency]:
                trigger[latency][delta] /= total_weight
        
        # Add the node's self-delayed latency
        if len(node.trigger) == 0:
            node.trigger = trigger
        else:
            if len(node.trigger) != len(trigger):
                raise ValueError('The length of node.trigger and trigger \
                                 should be the same')
            # Assume the original injected latency is constant
            origin_node_delay = list(list(node.trigger.values())[0].keys())[0]
            new_node_trigger = defaultdict(lambda: defaultdict(float))
            for latency in trigger:
                for delta in trigger[latency]:
                    new_node_trigger[latency][delta + origin_node_delay] = \
                        trigger[latency][delta]
            node.trigger = new_node_trigger
        
    
    def marginal_setnode_latency_prediction(self, setnode:TreeNode):
        
        def predict_serial_latency(serial_nodes: List[TreeNode],
                                   ref: str,
                                   lastnode: TreeNode):
            latencies = [node.latency
                         if node.predict_latency is None
                         else node.predict_latency
                         for node in serial_nodes]

            # Get succession time between each pair of serial nodes
            for i in range(len(serial_nodes) - 1):
                s1_name = serial_nodes[i].endpoint
                s2_name = serial_nodes[i + 1].endpoint
                pair = (s1_name, s2_name) if s1_name >= s2_name \
                                            else (s2_name, s1_name)
                if pair in setnode.siblings:
                    latencies.append(setnode.siblings[pair])
            
            # Add latencies between parallel clusters
            # First parallel cluster: only asdd the relative_start
            if ref == 'only':
                latencies.append(serial_nodes[0].relative_start)
                latencies.append(serial_nodes[-1].relative_finish)
            elif ref == 'first':
                latencies.append(serial_nodes[0].relative_start)
            # Other parallel clusters need to add successtion time
            # between the last node of the previous parallel cluster and
            # the first node of the current parallel cluster
            else:
                s1_name = serial_nodes[0].endpoint
                s2_name = lastnode.endpoint
                pair = (s1_name, s2_name) if s1_name >= s2_name \
                                            else (s2_name, s1_name)
                if pair in setnode.siblings:
                    latencies.append(setnode.siblings[pair])

                if ref == 'last':
                    latencies.append(serial_nodes[-1].relative_finish)
            
            # Return the predicted latency of the serial nodes
            return ADD_OP(latencies)
        
        def predict_parallel_cluster_latency(nodes, ref, lastnode):
            """
            Predict the latency of the parallel cluster
            """
            serial_latencies = []
            for serial_nodes in nodes:
                predicted_serial_latency = predict_serial_latency(serial_nodes, 
                                                                  ref, 
                                                                  lastnode)
                serial_latencies.append(predicted_serial_latency)
            
            return MAX_OP(serial_latencies)
        
        if setnode.count < self.count_thre:
            return
        
        parallel_group_latencies = []
        for pid in setnode.DAG:
            # print('pid: ', pid)
            nodes = setnode.DAG[pid]['nodes']
            if len(setnode.DAG) == 1:
                # The only parallel group
                pred_pcl = predict_parallel_cluster_latency(nodes,
                                                            ref='only',
                                                            lastnode=None)
            elif pid == 0:
                # The first parallel group
                pred_pcl = predict_parallel_cluster_latency(nodes,
                                                            ref='first',
                                                            lastnode=None)
            elif pid == len(setnode.DAG) - 1:
                # The last parallel group
                pred_pcl = predict_parallel_cluster_latency(nodes,
                                                            ref='last',
                                                            lastnode=setnode.DAG[pid-1]['last'])
            else:
                # The middle parallel groups
                pred_pcl = predict_parallel_cluster_latency(nodes,
                                                            ref='middle',
                                                            lastnode=setnode.DAG[pid-1]['last'])
            
            parallel_group_latencies.append(pred_pcl)
        
        return ADD_OP(parallel_group_latencies)
        
    def marginal_node_latency_prediction(self, node:TreeNode):
        """
        Because of we cannot do deconvolution of probability distributions,
        we cannot propagate the deltas. Instead, we propagate the full
        probability distribution of the latency.
        """
        setnodes: List[TreeNode] = list(node.children.values())
        for s in setnodes:
            print(s.endpoint)
            s.predict_latency = self.marginal_setnode_latency_prediction(s)

        if len(setnodes) == 1:
            node.predict_latency = setnodes[0].predict_latency
        else:
            # setnode_weights = [s.count for s in setnodes]
            # total_weight = sum(setnode_weights)
            # normalized_setnode_weights = [w / total_weight \
            #                                 for w in setnode_weights]
            setnode_latencies = [s.predict_latency for s in setnodes]
            
            node.predict_latency = MERGE_OP(setnode_latencies)


    def latency_propagation(self, delayed_nodes, assumption):
        """
        Propagate the latency variables from the affected nodes to the root
        with the asssumption of dependent or independent random variables
        """
        affected_nodes = defaultdict(set)
        for node in delayed_nodes:
            affected_nodes[node.depth].add(node)
    
        cur_depth = max(affected_nodes.keys())
        
        if assumption == 'dependent':
            node_latency_prediction = self.joint_node_latency_prediction
        elif assumption == 'independent':
            node_latency_prediction = self.marginal_node_latency_prediction
        else:
            raise ValueError('The assumption should be either dependent or independent')
        
        while cur_depth >= 0:
            print('cur_depth: ', cur_depth)
            # cur_nodes have the same depth (starting the bottom of the tree)
            cur_nodes = affected_nodes[cur_depth]
            pnodes = set()
            for node in cur_nodes:
                pnodes.add(node.parent.parent)
            for pnode in pnodes:
                node_latency_prediction(pnode)
                affected_nodes[pnode.depth].add(pnode)
            cur_depth -= 1
            
    def apply_trigger(self, services:List[str], 
                      delta:int, 
                      assumption='dependent',
                      inject_latency= None):
        """
        Apply the trigger to the node
        """
        # affeced_nodes = []
        # for nodename in nodenames:
        #     nodes = self.search_by_name(name_type='endpoint', 
        #                                 target_name=nodename, 
        #                                 issetnode=False)
        #     affeced_nodes.extend(nodes)
        
        affeced_nodes = self.find_injection_nodes(services)
        
        if len(affeced_nodes) == 0:
            raise ValueError('No affected nodes')
        
        for node in affeced_nodes:
            latencies = node.latency.keys()
            for latency in latencies:
                # node.trigger[latency] = delta
                if assumption == 'dependent':
                    node.trigger[latency] = {delta: 1}
                elif assumption == 'independent':
                    if inject_latency is None:
                        raise ValueError('inject_latency should not be None')
                    # node.predict_latency = inject_latency
                    node.predict_latency = PMF()
                    for key, prob in node.latency.items():
                      node.predict_latency[key + inject_latency] = prob
                else:
                    raise ValueError('The assumption should be either \
                                     dependent or independent')
        
        self.latency_propagation(affeced_nodes, assumption)
    
    def get_prediction(self, node: TreeNode):
        """
        For merge operator, update the return_pmf with value * prob.
        For given value, we may have multiple deltas with different probs.
        """
        return_pmf = defaultdict(float)
        for key, value in node.latency.items():
            for delta, prob in node.trigger[key].items():
                return_pmf[key + delta] += value * prob
        return PMF(return_pmf)
    
    
    def print_stats(self):
        
        self.num_nodes = 1
        
        def _count_nodes(node):
            for c in node.children.values():
                # if c.setnode is False:
                self.num_nodes += 1
                _count_nodes(c)

        _count_nodes(self.root)
        print('total number of nodes: ', self.num_nodes)
        
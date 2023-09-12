import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import requests
import json
import multiprocessing as mp
import yaml
import csv


class Parser(object):
    def __init__(self) -> None:
        pass
        
class CSVParser(object):
    
    def __init__(self, trace_path=None, trace_df=None) -> None:
        super.__init__()
        self.trace_path: str = trace_path
        self.trace_df: pd.DataFrame = trace_df
        
    def _get_rename_map(self, origin_list, prefix=""):
        i = 0
        rename_map = {}
        origin_set = set(origin_list)
        for obj in origin_set:
            name = prefix + str(i)
            if obj not in rename_map:
                rename_map[obj] = name
            i += 1
          
        return rename_map
    
    def get_traces(self):
        if self.trace_path is not None:
            df = pd.read_csv(self.trace_path, dtype={'rpcid':str}).fillna('None')
        elif self.trace_df is not None:
            df = self.trace_df
        else:
            raise ValueError('trace_path or trace_df must be provided')
        return df

    def rename(self, filename: str):
        df = pd.read_csv(filename)
        df['traceid'] = df['traceid'].map(self._get_rename_map(df['traceid'].to_list(), 't'))
        df['um'] = df['um'].map(self._get_rename_map(df['um'].to_list(), 'u'))
        df['dm'] = df['dm'].map(self._get_rename_map(df['dm'].to_list(), 'd'))
        df['interface'] = df['interface'].map(self._get_rename_map(df['interface'].to_list(), 'i'))
        new_filename = filename + '_cleaned.csv'
        df.to_csv(new_filename, index=False)
        
    def clean_trace(self, trace_df):
        trace_df = trace_df.replace('(?)', 'None')
        trace_df = trace_df[~((trace_df['rpctype'] == 'rpc') 
                            & (trace_df['rt'] < 0))].reset_index(drop=True)
        return trace_df

        
    def _remove_http_chaos(self, sub_df):
        # df = self.get_traces()
        traceids = sub_df['traceid'].unique()
        remove_traceids = []
        # for traceid in tqdm(traceids):
        for traceid in traceids:
            trace_df = sub_df[sub_df['traceid'] == traceid]
            trace_df = trace_df.replace('(?)', 'None')
            trace_df = trace_df[~((trace_df['rpctype'] == 'rpc') 
                                & (trace_df['rt'] < 0))].reset_index(drop=True)
            if len(trace_df[trace_df['rpctype'] == 'http']) == 0:
                # remove traces without http calls
                # print(f'{traceid} has no http calls')
                # df = df[df['traceid'] != traceid]
                remove_traceids.append(traceid)
            else:
                # print(f'{traceid} has http calls')
                http_df = trace_df[(trace_df['rpctype'] == 'http')]
                lengths = http_df['rpcid'].str.len()
                argmax = np.where(lengths == lengths.max())[0][0]
                keep_rpcid = http_df.iloc[argmax]['rpcid']
                if len(keep_rpcid) > 5:
                    remove_traceids.append(traceid)
                else:
                    rm_index = sub_df[(sub_df['traceid'] == traceid) 
                                & (sub_df['rpctype'] == 'http') 
                                & (sub_df['rpcid'] != keep_rpcid)].index
                    sub_df.drop(rm_index, inplace=True)
        cleaned_sub_df = sub_df[~sub_df['traceid'].isin(remove_traceids)]
        # sub_df.to_csv('no_http_chaos.csv', index=False)
        return cleaned_sub_df
    
    def _remove_missing_spans(self, sub_df):
        
        traceids = sub_df['traceid'].unique()
        sound_traceids = []
        for traceid in traceids:
            trace_df = sub_df[sub_df['traceid'] == traceid]
            
            root_rpcid = trace_df[trace_df['rpctype'] == 'http']['rpcid'].iloc[0]
            
            lengths = trace_df['rpcid'].str.len()
            argmax = np.where(lengths == lengths.max())[0][0]
            cur_rpcid = trace_df.iloc[argmax]['rpcid']
            
            
            while cur_rpcid != root_rpcid:
                prev_rpcid = cur_rpcid[0:len(cur_rpcid)-2]
                if prev_rpcid not in trace_df['rpcid'].to_list():
                    break
                else:
                    cur_rpcid = prev_rpcid
            if cur_rpcid == root_rpcid:
                # keep traces with root 0, 0.1, and 0.1.1
                sound_traceids.append(traceid)
        return sub_df[sub_df['traceid'].isin(sound_traceids)]
        
    
    def multiprocess(self, func_type='chaos_http', num_workers=32):
        
        start_time = time.time()
        df = self.get_traces()
        traceids = df['traceid'].unique()
        sub_traceids = np.array_split(np.array(traceids), num_workers)
        sub_dfs = [df[df['traceid'].isin(traceids)] for traceids in sub_traceids]
        
        if func_type == 'chaos_http':
            func = self._remove_http_chaos
            filename = 'no_http_chaos.csv'
        elif func_type == 'missing_span':
            func = self._remove_missing_spans
            filename = 'no_missing_span.csv'
        else:
            raise ValueError('func_type must be chaos_http or missing_span')
        
        pool = mp.Pool(processes=num_workers)
        results = pool.map(func, sub_dfs)
        finish_time = time.time()
        print(f'finish multiprocess in {finish_time - start_time} seconds')

        return_df = pd.concat(results, axis=0)
        return_df.to_csv(filename, index=False)
        return return_df
    

class JSONParser(Parser):
    def __init__(self, config_path=None) -> None:
        super().__init__()
        self.config_path = config_path
        
    def pull_traces(self):
        """
        Pull traces from jaeger trace endpoint
        Raises:
            err: HTTP error
        Returns:
            list: a list of all traces passing through a service
        """
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        dsb_pull_cfg = config['DSB']['pull']
        starttime = dsb_pull_cfg['starttime']
        finishtime = dsb_pull_cfg['finishtime']
        service = dsb_pull_cfg['service']
        num_traces = dsb_pull_cfg['num_traces']
        host_ip = dsb_pull_cfg['host_ip']
        
        if starttime > finishtime:
            raise ValueError('starttime must be less than finishtime')
        
        url = "http://" \
            + host_ip + ":16686/api/traces?limit=" \
            + str(num_traces) \
            + "&service=" + service \
            + "&start=" + str(starttime) \
            + "&end=" + str(finishtime)
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise err

        response = json.loads(response.text)
        traces = response['data']
        return traces
    
    def convert_json_to_csv(self, traces, filepath='traces.csv'):
        """
        Convert json traces to csv format
        Columns: traceid, spanid, parentid, dm, interface, timestamp, rt, nodeid
        
        Args:
            traces (list): a list of traces
        """
        # TODO: add clock skews
        header = ['traceid', 
                  'spanid', 
                  'parentid',
                  'dm', 
                  'interface', 
                  'timestamp', 
                  'rt',
                  'nodeid',
                  ]

        with open(self.config_path + 'service_node_mapping.yml') as f:
            mapping = yaml.safe_load(f)
        
        with open(filepath, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for trace in tqdm(traces):
                traceId = trace['traceID']
                spans = trace['spans']
                for span in spans:
                    spanId = span['spanID']
                    operationName = span['operationName'].replace('-', '_').replace('/', '_')
                    if spanId != traceId:
                        parentId = span['references'][0]['spanID']
                    else:
                        parentId = None
                    # duration =round(span['duration'] / 1000, 2)
                    duration = span['duration']
                    processId = span['processID']
                    service = trace['processes'][processId]['serviceName']
                    service = service.replace('-', '_')
                    startTime = span['startTime']
                    nodeId = mapping[service]
                    span_data = [traceId, 
                                 spanId, 
                                 parentId, 
                                 service, 
                                 operationName, 
                                 startTime, 
                                 duration,
                                 nodeId
                                 ]
                    writer.writerow(span_data)
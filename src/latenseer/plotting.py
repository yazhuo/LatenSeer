import matplotlib.pyplot as plt
from random import choices
import pandas as pd

from .probdist import CDF, PMF

COLORS = {
    'single-measured': 'black',
    'multi-measured': 'black',
    'multi-pred': 'red',
}

LINESTYLES = {
    'single-measured': ':',
    'multi-measured': '-',
    'multi-pred': '--',
}

LABELS = {
    'single-measured': 'single-site (measured)',
    'multi-measured': 'multi-site (measured)',
    'multi-pred': 'multi-site (predicted)',
}


def plot_cdfs(stats, figname='result.png', colors=COLORS, linestyles=LINESTYLES, labels=LABELS):
    """
    stats: a dict of e2e latency distribution
    """
    plt.figure(figsize=(8, 5))
    percentiles = [i/100 for i in range(101)]

    local = CDF(stats['single-measured'])
    remote = CDF(stats['multi-measured'])
    pred = CDF(stats['multi-pred'])
    
    local.plot(xscale=1000, lw=5, ls=linestyles['single-measured'], color=colors['single-measured'], label=labels['single-measured'])
    remote.plot(xscale=1000, lw=5, ls=linestyles['multi-measured'], color=colors['multi-measured'], label=labels['multi-measured'])
    pred.plot(xscale=1000, lw=5, ls=linestyles['multi-pred'], color=colors['multi-pred'], label=labels['multi-pred'])
    
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel('E2E Latency (ms)', fontsize=22)
    plt.ylabel('CDF', fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(fontsize=16)
    plt.savefig(figname, bbox_inches='tight')



def plot_slacks(agg_service_slack, figname="slack_result.png"):
    slack_dict = {}
    for service, pmf in agg_service_slack.items():
        if CDF(pmf).Percentile(50) > 0:
            values = [x/1000 for x in list(pmf.keys())]
            weights = list(pmf.values())
            slack_dict[service] = choices(values, weights, k=1000)

    data_df = pd.DataFrame(slack_dict)
    plt.figure(figsize=(12, 8))

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    lss = ['-', '--', ':', '-.']
    
    i = 0
    for key in slack_dict:
        latency = slack_dict[key]
        CDF(PMF(latency)).plot(xscale = 1, ls=lss[i], lw=8, marker='', color=colors[i], label=key)
        i += 1

    plt.xlabel('Slack Latency (ms)', fontsize=36)
    plt.ylabel('CDF', fontsize=36)
    plt.tick_params(axis='both', which='major', labelsize=36)
    plt.legend(loc='best', fontsize=32)

    plt.grid()
    
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')
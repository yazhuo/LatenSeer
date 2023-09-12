#!/bin/bash

## Change the following list to your own server list on cloudlab
server_list=(
 yazhuoz@clnode072.clemson.cloudlab.us
 yazhuoz@hp054.utah.cloudlab.us
 yazhuoz@hp042.utah.cloudlab.us
 yazhuoz@hp065.utah.cloudlab.us
 yazhuoz@hp070.utah.cloudlab.us
 yazhuoz@hp053.utah.cloudlab.us
 yazhuoz@hp058.utah.cloudlab.us
 yazhuoz@hp046.utah.cloudlab.us
 yazhuoz@hp072.utah.cloudlab.us
 yazhuoz@c220g1-031128.wisc.cloudlab.us
 yazhuoz@c220g1-031117.wisc.cloudlab.us
 yazhuoz@c220g1-031106.wisc.cloudlab.us
 yazhuoz@c220g1-031112.wisc.cloudlab.us
 yazhuoz@c220g1-031122.wisc.cloudlab.us
 yazhuoz@c220g1-031101.wisc.cloudlab.us
 yazhuoz@c220g1-031110.wisc.cloudlab.us
 yazhuoz@c220g1-031104.wisc.cloudlab.us
)


for i in ${server_list[@]}; do
	(ssh -p 22 $i "cd  /proj/latencymodel-PG0/yazhuoz/scripts && ./setup_cloudlab_env.sh")&
done
wait

for i in ${server_list[@]}; do
	(ssh -p 22 $i "git clone https://github.com/yazhuo/DeathStarBench.git")
	# echo $i
done
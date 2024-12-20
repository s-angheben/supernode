#!/bin/bash

echo "Compute datasets statistics";
python count_concepts.py >> dataset_stats.txt;

#echo "Running HIV test";
#python run_HIV.py;
#python run_HIV_supernode_homogeneous.py --concepts="cyclebasis";
#python run_HIV_supernode_homogeneous.py --concepts="maxcliques";
#python run_HIV_supernode_hetero_multi.py --concepts="cyclebasis";
#python run_HIV_supernode_hetero_multi.py --concepts="maxcliques";
#python run_HIV_supernode_hetero_multi.py --concepts="cycb_maxcliq_star2_minl_maxl";

echo "Running PROTEINS test";
python run_PROTEINS.py;
python run_PROTEINS_supernode_homogeneous.py --concepts="cyclebasis";
python run_PROTEINS_supernode_homogeneous.py --concepts="maxcliques";
python run_PROTEINS_supernode_hetero_multi.py --concepts="cyclebasis";
python run_PROTEINS_supernode_hetero_multi.py --concepts="maxcliques";
python run_PROTEINS_supernode_hetero_multi.py --concepts="cycb_maxcliq_star2_minl_maxl";

echo "Running IMDBb test";
python run_IMDBb.py;
python run_IMDBb_supernode_homogeneous.py --concepts="cyclebasis";
python run_IMDBb_supernode_homogeneous.py --concepts="maxcliques";
python run_IMDBb_supernode_hetero_multi.py --concepts="cyclebasis";
python run_IMDBb_supernode_hetero_multi.py --concepts="maxcliques";
python run_IMDBb_supernode_hetero_multi.py --concepts="cycb_maxcliq_star2_minl_maxl";

#!/bin/bash

echo "Running HIV test";
python run_HIV.py;
python run_HIV_supernode_homogeneous.py --concepts="cyclebasis";
python run_HIV_supernode_homogeneous.py --concepts="maxcliques";
python run_PROTEINS_supernode_hetero_multi.py --concepts="cyclebasis";
python run_PROTEINS_supernode_hetero_multi.py --concepts="maxcliques";
python run_PROTEINS_supernode_hetero_multi.py --concepts="cycb_maxcliq_star2_minl_maxl";

echo "Running PROTEINS test";
python run_PROTEINS.py;
python run_PROTEINS_supernode_homogeneous.py --concepts="cyclebasis";
python run_PROTEINS_supernode_homogeneous.py --concepts="maxcliques";
python run_PROTEINS_supernode_heterogeneous.py --concepts="cyclebasis";
python run_PROTEINS_supernode_heterogeneous.py --concepts="maxcliques";
python run_PROTEINS_supernode_heterogeneous.py --concepts="cycb_maxcliq_star2_minl_maxl";

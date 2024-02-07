from data.builder import create_dataset_tree_cycle

if __name__ == "__main__":
    create_dataset_tree_cycle("/home/sam/Documents/network/project/dataset", "simple", 100,
                              node_num=15, cycle_level=2)


#create_dataset_tree_cycle("/home/sam/Documents/network/project/dataset", "d1",
#                           10000, node_num=40, cycle_level=10)

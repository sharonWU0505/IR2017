from priority_queue import Heap
import numpy as np


class Cluster:
    def __init__(self, doc, doc_idx, ini_sim_list):
        self.docs = [doc]
        self.doc_indices = [doc_idx]
        self.size = 1
        self.centroid = np.array(doc)
        self.pri_queue = Heap()
        self.pri_queue.build_heap(ini_sim_list)

    def merge(self, new_cluster):
        self.docs.extend(new_cluster.docs)
        self.doc_indices.extend(new_cluster.doc_indices)

        n1 = self.size
        n2 = new_cluster.size
        self.centroid = (n1 * self.centroid + n2 * new_cluster.centroid) / (n1 + n2)

        self.size += new_cluster.size

    def get_most_sim(self):
        sim_idx = self.pri_queue.heapList[1][0]
        sim_value = self.pri_queue.heapList[1][1]
        return [sim_idx, sim_value]

class Heap:
    def __init__(self):
        self.heapList = [[-1, -1]]
        self.currentSize = 0

    def _move_up(self, i):
        while i // 2 > 0:
            parent_index = i // 2
            if self.heapList[i][1] < self.heapList[parent_index][1]:
                temp = self.heapList[parent_index]
                self.heapList[parent_index] = self.heapList[i]
                self.heapList[i] = temp
                i = parent_index
            else:
                break

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize += 1
        self._move_up(self.currentSize)

    def _move_down(self, i):
        while (i * 2) <= self.currentSize:
            mc = self._min_child(i)
            if self.heapList[i][1] > self.heapList[mc][1]:
                temp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = temp
                i = mc
            else:
                break

    def _min_child(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2][1] < self.heapList[i * 2 + 1][1]:
                return i * 2
            else:
                return i * 2 + 1

    def delete(self, k):
        i = [x for x, y in enumerate(self.heapList) if y[0] == k][0]
        self.heapList[i] = self.heapList[self.currentSize]
        self.heapList = self.heapList[:-1]
        self.currentSize -= 1
        self._move_down(i)

    def build_heap(self, sim_list):
        i = len(sim_list) // 2
        self.currentSize = len(sim_list)
        self.heapList = [[-1, -1]] + sim_list
        while i > 0:
            self._move_down(i)
            i = i - 1

    def clear(self):
        self.heapList = [(-1, -1)]
        self.currentSize = 0

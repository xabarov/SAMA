import time


class LabelsIds:
    def __init__(self, labels_size):
        self.labels_size = labels_size
        self.max_id = labels_size - 1
        self.labels_ids = set([x for x in range(labels_size)])  # Начальное состояние - по порядку без пропусков

    def get_unique_label_id(self):
        all_set = set([x for x in range(self.max_id)])

        labels_inside_range = list(all_set - self.labels_ids)

        if len(labels_inside_range) > 0:
            labels_inside_range.sort()
            unique = labels_inside_range[0]
            self.labels_ids.add(unique)
            return unique

        else:
            self.max_id += 1
            self.labels_ids.add(self.max_id)
            return self.max_id

    def remove_label_id(self, id):
        labels_list = list(self.labels_ids)
        labels_list.sort()
        if id == labels_list[-1]:
            self.max_id -= 1
        self.labels_ids.discard(id)

    def print_labels(self):

        print("Labels:\n")
        print(self.labels_ids)
        print(f"Max id: {self.max_id}")


if __name__ == '__main__':
    import numpy as np

    size = 300000

    start = time.process_time()

    labels = LabelsIds(size)

    spent_time = time.process_time() - start

    print(f"Время создания: {spent_time:^5.3f} c из {size} элементов")

    delete_size = 15
    random_ids = np.random.randint(0, size, delete_size)
    start = time.process_time()

    for label_id in random_ids:
        labels.remove_label_id(label_id)

    spent_time = time.process_time() - start

    print(f"Время удаления элементов: {spent_time:^5.3f} c из {delete_size} элементов")

    start = time.process_time()
    unique_ids = []
    unique_size = 50
    for i in range(unique_size):
        unique_ids.append(labels.get_unique_label_id())
    spent_time = time.process_time() - start

    print(unique_ids)

    print(f"Время поиска уникальных {unique_size} элементов: {spent_time:^5.3f}")

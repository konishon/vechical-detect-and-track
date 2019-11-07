from collections import OrderedDict

from tracking.mathhelper import iou_from_shapely


class IOUTracker:
    def __init__(self, max_disappeared=50, iou_ratio=0.90):
        self.max_disappeared = max_disappeared
        self.iou_ratio = iou_ratio

        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.next_object_id = 0

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, boxes):
        for box in boxes:
            (start_x, start_y, end_x, end_y) = box
            center_x = (start_x + end_x) / 2
            center_y = (end_x + end_y) / 2
            self.register((int(center_x), int(center_y)))

        for index, box in enumerate(boxes):
            (start_x, start_y, end_x, end_y) = box

            c = (start_x + end_x) / 2
            d = (start_y + end_y) / 2

            for box_to_check in boxes:
                (start_x, start_y, end_x, end_y) = box_to_check
                a = (start_x + end_x) / 2
                b = (start_y + end_y) / 2

                overlap_ratio = iou_from_shapely((c, d), (a, b))
                if overlap_ratio > self.iou_ratio:
                    key_to_update = list(self.objects.keys())[index]
                    self.objects[key_to_update] = (int(a), int(b))
                
        return self.objects
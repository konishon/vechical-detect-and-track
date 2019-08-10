class TrackableObject:
    def __init__(self, objectID, centroid, lastTenCentroids,label):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = centroid
        self.label = label
        self.lastTenCentroids = lastTenCentroids
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False

    def __repr__(self):
        return "ID: {0} centroid: {1}".format(self.objectID,self.centroids)

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from geopy.distance import distance
from shapely.geometry import MultiPoint

class DBSCANUtil():
    
    def __init__(self, training_df, testing_df, epsilon):
        self.training_df = training_df
        self.testing_df = testing_df
        self.epsilon = epsilon

        self.locations = training_df[['latitude', 'longitude']].values
        self.station_labels = 0
        self.centermost_station_points = 0

    def fit(self):
        kms_per_radian = 6371.0088
        epsilon = self.epsilon / kms_per_radian
        min_samples = 1

        # Fit DBSCAN model with location data points
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
        dbscan = dbscan.fit(np.radians(self.locations))

        self.station_labels = dbscan.labels_

    @staticmethod
    def compute_centermost_point(cluster):
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)

        return tuple(centermost_point)

    def get_centermost_point(self):
        num_clusters = len(set(self.station_labels))
        station_clusters = pd.Series([self.locations[self.station_labels == i] for i in range(num_clusters)])
        self.centermost_station_points = station_clusters.map(DBSCANUtil.compute_centermost_point)

    def classify(self):
        location_validation_label = []

        # Iterate through all test points
        for _, test_station in self.testing_df.iterrows():
            test_station = (test_station['latitude'], test_station['longitude'])
            neareast_distance = float('inf')
            label = -1
            
            # Iterate through all center points, choose closest to classify test point
            for cluster, center_point in enumerate(self.centermost_station_points):
                curr_distance = distance(test_station, center_point)
                
                # Update for nearest point
                if curr_distance < neareast_distance:
                    neareast_distance = curr_distance
                    label = cluster
            
            # Add label
            location_validation_label.append(label)

        self.testing_df['label'] = location_validation_label

    def get_average_metric(self):
        num_clusters = len(set(self.station_labels))
        
        # Get furthest distance between test location and boundary point within their cluster
        station_clusters = pd.Series([self.locations[self.station_labels == i] for i in range(num_clusters)])
        distance_metric = []

        # Iterate through all test points to get distance metric
        for _, test_station in self.testing_df.iterrows():
            test_location = (test_station['latitude'], test_station['longitude'])
            label = int(test_station['label'])
            max_distance = 0
            
            # Iterate through all stations, compare for largest distance metric
            for station in station_clusters[label]:
                station = (station[0], station[1])
                curr_distance = distance(station, test_location)
            
                # Swap
                if curr_distance > max_distance:
                    max_distance = curr_distance
            
            # Append metric for one label iteration
            distance_metric.append(max_distance.km)

        self.testing_df['distance_metric'] = distance_metric

        return self.testing_df['distance_metric'].mean()

    def output_performance(self):
        DBSCANUtil.fit(self)
        DBSCANUtil.get_centermost_point(self)
        DBSCANUtil.classify(self)

        metric = DBSCANUtil.get_average_metric(self)
        return metric
    
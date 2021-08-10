from geopy.distance import geodesic
import math

def get_adjency_matrix_weight(p1,p2,epsilon,lamda):
    distance = geodesic(p1,p2).km
    weight = math.exp(-((distance ** 2)/(lamda ** 2)))
    if weight >= epsilon:
        return weight
    else:
        return 0
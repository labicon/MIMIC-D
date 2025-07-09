from discrete import *
import sys

print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)
fast_frechet = FastDiscreteFrechetMatrix(euclidean)

arr1 = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]], dtype=np.float64)
arr2 = np.array([[2,2],[3,3],[4,4],[5,5],[6,6]], dtype=np.float64)
frechet_dist = fast_frechet.distance(arr1,arr2)
print(frechet_dist)
from sklearn.utils import resample

a = [0, 2, 3, 4, 5, 6 ,7, 8]
b = [ 7, 6, 5, 4, 3, 2, 7,9]

a,b = resample(a,b, replace=False)

print(a, '\n', b)


a,b = resample(a,b, replace=True)

print(a, '\n', b)
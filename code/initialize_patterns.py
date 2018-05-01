sparsity = 0.2
image_size = 10
numPatterns = 20
patterns = np.zeros(shape = (image_size**2, numPatterns))
x = int(round(image_size**2*sparsity))
r = random.sample(range(0, image_size**2-1), x*numPatterns)
r = np.asarray(r)
ry = r.reshape(x, numPatterns)
for i in range(numPatterns):
    patterns[ry[:,i],i] = 1

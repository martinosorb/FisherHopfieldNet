sparsity = 0.5
pattern1 = binary_images[60,:]
pattern2 = binary_images[30,:]
pattern3 = binary_images[40,:]
pattern4 = binary_images[50,:]
pattern5 = binary_images[70,:]
pattern6 = binary_images[80,:]
pattern7 = binary_images[10,:]
pattern8 = binary_images[20,:]
pattern9 = binary_images[5,:]
pattern10= binary_images[15,:]

# image_size = 10
# numPatterns = 20
# patterns = np.random.rand(image_size, image_size, numPatterns)
# patterns[patterns < 1-sparsity] = 0
# patterns[patterns > 1-sparsity] = 1
# pattern1 = patterns[:,:,0]
# pattern2 = patterns[:,:,1]
# pattern3 = patterns[:,:,2]
# pattern4 = patterns[:,:,3]
# pattern5 = patterns[:,:,4]
# pattern6 = patterns[:,:,5]
# pattern7 = patterns[:,:,6]
# pattern8 = patterns[:,:,7]
# pattern9 = patterns[:,:,8]
# pattern10 = patterns[:,:,9]
# pattern11 = patterns[:,:,10]
# pattern12 = patterns[:,:,11]
# pattern13 = patterns[:,:,12]
# pattern14 = patterns[:,:,13]
# pattern15 = patterns[:,:,14]
# pattern16 = patterns[:,:,15]
#
# import random
# sparsity = 0.1
# pattern1 = np.zeros(image_size**2, numPatterns)
# r = random.sample(range(0, image_size**2-1), int(round(image_size**2*sparsity)*numPatterns)
# r = r.reshape(int(round(image_size**2*sparsity), numPatterns)
# pattern1[r] = 1
# for i in range(numPatterns):


plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(pattern1.reshape([image_size, image_size]))
plt.subplot(1,2,2)
plt.imshow(pattern2.reshape([image_size, image_size]))
plt.xticks([])
plt.yticks([])
plt.show()

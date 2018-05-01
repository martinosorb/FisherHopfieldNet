def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

current_index = 0
images = np.zeros(shape = (num_classes * samples_per_class, 32, 32))

index_to_show = 72

for i in range(num_classes):
    for j in range(samples_per_class):
        current_picture = sample_pictures[i, j,:,:,:]
        current_picture.squeeze()
        images[current_index, :,:] = rgb2gray(current_picture)
        current_index += 1

image_to_show = sample_pictures[math.floor(index_to_show/num_classes), index_to_show%samples_per_class,:,:,:]
plt.figure(figsize=(10,15))
plt.subplot(1, 4, 1)
plt.imshow(image_to_show.astype('uint8'))
plt.subplot(1, 4, 2)
plt.imshow(images[index_to_show,:,:].astype('uint8'))

#resize images just for faster processing speed
image_size = 10
resized_images = np.zeros(shape = (num_classes*samples_per_class, image_size, image_size))
for i in range(num_classes*samples_per_class):
    resized_images[i,:,:] = imresize(images[i,:,:], (image_size, image_size))

images = resized_images

#linearize
lin_images = np.reshape(images, (num_classes * samples_per_class, image_size*image_size))
#recovered_image = np.reshape(lin_images, (num_classes * samples_per_class, 32, 32))

plt.subplot(1, 4, 3)
plt.imshow(images[index_to_show,:,:].astype('uint8'))
#plt.figure()
#plt.imshow(recovered_image[32,:,:].astype('uint8'))

#get maximum value and scale it the image to values between 0 and 1. Then, threshold the image at 0.5 to get a
#binary output
max_values = np.amax(lin_images, axis = 1)
scaled_images = lin_images / max_values[:, np.newaxis]
small_values = scaled_images < 0.5
large_values = scaled_images >= 0.5
binary_images = scaled_images
binary_images[small_values] = 0
binary_images[large_values] = 1
#binary_images = binary_images - np.mean(binary_images)
#print(np.mean(binary_images))
binary_image_recoverd = np.reshape(binary_images, (num_classes * samples_per_class, image_size, image_size))
plt.subplot(1, 4, 4)
plt.imshow(binary_image_recoverd[index_to_show,:,:].astype('uint8'))
#plt.show()

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 10
sample_pictures = np.zeros(shape = (10,10,32,32,3))
current_class = 0
current_sample = 0

for y_hat, cls in enumerate(classes):
    current_sample = 0
    idxs = np.flatnonzero(y == y_hat)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y_hat + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
        sample_pictures[current_class, current_sample,:,:,:] = X[idx]
        current_sample += 1
    current_class += 1

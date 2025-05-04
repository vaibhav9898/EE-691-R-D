class Config:
    # Dataset paths
    train_data_dir = 'Downloads/MNIST/train'
    test_data_dir = 'Downloads/MNIST/test'

    # Data parameters
    mnist_mean = 0.1307
    mnist_std = 0.3081
    batch_size = 256  # For data loaders
    gen_batch_size = 1024  # For generator training

    # Model parameters
    nc = 1  # Number of channels
    ncf = 64  # Classifier feature dimension
    nz = 100  # Latent vector size
    ngf = 64  # Generator feature dimension
    n_classes = 6  # Number of classes
    labels_per_class = 10

    # Training parameters
    num_epochs = 2000
    lambda_l1 = 1e-5
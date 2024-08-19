import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import sys
from pycocotools.coco import COCO
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import numpy as np
import os
import requests
import time
import math
def setup_training_parameters():
    """
    Sets up the training parameters for the model.

    Returns:
    - dict: A dictionary containing all the training parameters.
    """
    params = {
        "batch_size": 64,
        "vocab_threshold": 4,
        "vocab_from_file": True,
        "embed_size": 256,
        "hidden_size": 512,
        "num_epochs": 1,
        "save_every": 1,
        "print_every": 1,
        "log_file": 'training_log.txt',
    }
    return params

def setup_transformations():
    """
    Sets up the image transformations for the training data.

    Returns:
    - transform_train: torchvision.transforms.Compose, composed transformations for training images.
    """
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform_train

def build_data_loader(transform_train, batch_size, vocab_threshold, vocab_from_file):
    """
    Builds the data loader for training.

    Parameters:
    - transform_train: torchvision.transforms.Compose, transformations to apply to the images.
    - batch_size: int, number of images per batch.
    - vocab_threshold: int, minimum word count threshold for vocabulary.
    - vocab_from_file: bool, whether to load the vocabulary from a file.

    Returns:
    - data_loader: DataLoader, data loader for the training data.
    """
    data_loader = get_loader(transform=transform_train,
                             mode='train',
                             batch_size=batch_size,
                             vocab_threshold=vocab_threshold,
                             vocab_from_file=vocab_from_file)
    return data_loader

def initialize_models(embed_size, hidden_size, vocab_size):
    """
    Initializes the Encoder and Decoder models.

    Parameters:
    - embed_size: int, dimensionality of image and word embeddings.
    - hidden_size: int, number of features in hidden state of the RNN decoder.
    - vocab_size: int, the size of the vocabulary.

    Returns:
    - encoder: EncoderCNN, the initialized encoder model.
    - decoder: DecoderRNN, the initialized decoder model.
    """
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    return encoder, decoder

def setup_device_and_criterion():
    """
    Sets up the device and the loss function for training.

    Returns:
    - device: torch.device, the device to be used for training (CPU or GPU).
    - criterion: torch.nn.CrossEntropyLoss, the loss function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    return device, criterion

def setup_optimizer(encoder, decoder, lr=0.001):
    """
    Sets up the optimizer for training.

    Parameters:
    - encoder: EncoderCNN, the encoder model.
    - decoder: DecoderRNN, the decoder model.
    - lr: float, learning rate.

    Returns:
    - optimizer: torch.optim.Adam, the initialized optimizer.
    """
    params = list(decoder.parameters()) + list(encoder.embed.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08)
    return optimizer

def train_model(encoder, decoder, criterion, optimizer, data_loader, device, vocab_size, total_step, params):
    """
    Trains the Encoder-Decoder model.
    Parameters:
    - encoder: EncoderCNN, the initialized encoder model.
    - decoder: DecoderRNN, the initialized decoder model.
    - criterion: torch.nn.CrossEntropyLoss, the loss function.
    - optimizer: torch.optim.Adam, the optimizer.
    - data_loader: DataLoader, the data loader for the training data.
    - device: torch.device, the device to be used for training (CPU or GPU).
    - vocab_size: int, the size of the vocabulary.
    - total_step: int, total number of steps per epoch.
    - params: dict, the training parameters.
    """
    # Open the training log file.
    f = open(params['log_file'], 'w')

    old_time = time.time()
    # response = requests.get("http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token",
    #                         headers={"Metadata-Flavor":"Google"})

    for epoch in range(1, params['num_epochs'] + 1):
        for i_step in range(1, total_step + 1):
            if time.time() - old_time > 60:
                old_time = time.time()

            # Randomly sample a caption length, and sample indices with that length.
            indices = data_loader.dataset.get_train_indices()
            # Create and assign a batch sampler to retrieve a batch with the sampled indices.
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            data_loader.batch_sampler.sampler = new_sampler

            # Obtain the batch.
            images, captions = next(iter(data_loader))

            # Move batch of images and captions to GPU if CUDA is available.
            images = images.to(device)
            captions = captions.to(device)

            # Zero the gradients.
            decoder.zero_grad()
            encoder.zero_grad()

            # Pass the inputs through the CNN-RNN model.
            features = encoder(images)
            outputs = decoder(features, captions)

            # Calculate the batch loss.
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            # Get training statistics.
            stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, params['num_epochs'], i_step, total_step, loss.item(), np.exp(loss.item()))

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()

            # Print training statistics to file.
            f.write(stats + '\n')
            f.flush()

            # Print training statistics (on different line).
            if i_step % params['print_every'] == 0:
                print('\r' + stats)

        # Save the weights.
        if epoch % params['save_every'] == 0:
            torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))
            torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))

    # Close the training log file.
    f.close()


def main():
    # Setup training parameters
    params = setup_training_parameters()

    # Setup transformations
    transform_train = setup_transformations()

    # Build data loader
    data_loader = build_data_loader(transform_train,
                                    params['batch_size'],
                                    params['vocab_threshold'],
                                    params['vocab_from_file'])

    # Get vocab size
    vocab_size = len(data_loader.dataset.vocab)

    # Initialize models
    encoder, decoder = initialize_models(params['embed_size'], params['hidden_size'], vocab_size)

    # Setup device and loss function
    device, criterion = setup_device_and_criterion()
    encoder.to(device)
    decoder.to(device)

    # Setup optimizer
    optimizer = setup_optimizer(encoder, decoder)

    # Set total number of training steps per epoch
    total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

    # Train the model
    train_model(encoder, decoder, criterion, optimizer, data_loader, device, vocab_size, total_step, params)

if __name__ == "__main__":
    main()



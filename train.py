import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset import DoodleDataset
from src.trainer import fit

# Setting seed for reproducability
torch.manual_seed(42)

# Constants
MODELS_PATH = './models'
NUM_CLASSES = 50
DATA_DIR = './data/unzip'
# Taking first <NUM_CLASSES> files in the directory
CLASS_FILES = os.listdir(DATA_DIR)[:NUM_CLASSES]
IM_PER_CLASS = 5000
BATCH_SIZE = 128
NUM_EPOCHS_F = 20
NUM_EPOCHS = 5
IMAGE_SIZE = 224


def main():
    full_dataset = DoodleDataset(
        DATA_DIR,
        class_list=CLASS_FILES,
        im_per_class=IM_PER_CLASS,
        transform=transforms.Compose([
            transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]))

    # Splitting the data
    train_size = int(0.7 * len(full_dataset))
    test_size = int(0.2 * len(full_dataset))
    val_size = len(full_dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size, val_size]
    )

    dataloader = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False),
    }

    # Here it is decided where to make all the computations: gpu or cpu?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loading the Resnet18 with pretrained weights and freezing the models layers.
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Adding FC layer to train classifier. Required_grad is true by default here.
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training the model for the first time
    model = fit(dataloader, model, criterion,
                optimizer, num_epochs=NUM_EPOCHS_F)

    # Here I am unfreezing all model's layers to make fine-tuning.
    for param in model.parameters():
        param.requires_grad = True

    # Empltying cuda's cache ot release memory.
    torch.cuda.empty_cache()

    # Here I am defining scheduler that will decrease learning rate by factor of 0.1 every 7 epochs.
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

    model = fit(dataloader, model, criterion, optimizer,
                scheduler=exp_lr_scheduler, num_epochs=NUM_EPOCHS)


if __name__ == 'main':
    main()

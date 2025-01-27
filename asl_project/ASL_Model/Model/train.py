import os
import torch
import dataSetup, engine, modelSetup, utils

from pathlib import Path
from torchvision import transforms

MUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LR = 0.001

data_path = Path('archive/asl_alphabet_train/asl_alphabet_train')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transform = transforms.Compose([
    transforms.Resize((255,255)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])


if __name__ == '__main__':

    train_dataloader, test_dataloader, class_names = dataSetup.create_dataloaders(
        data_dir=data_path,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    model = modelSetup.ASLModel(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)


    engine.train(model= model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=MUM_EPOCHS,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    utils.save_model(model=model,
                     target_dir="models",
                     model_name='ASLalphabet')
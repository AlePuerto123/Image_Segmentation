import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import SegmentationDataset
from model import get_model

def train():

    dataset = SegmentationDataset(config.TRAIN_IMAGES,config.TRAIN_MASKS)

    # Create a DataLoader to iterate through the dataset in batches
    loader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)

    model = get_model()

    optimizer = torch.optim.Adam(model.parameters(),lr=config.LR)

    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(config.NUM_EPOCHS):

        loop = tqdm(loader)

        for imgs, masks in loop:

            imgs = imgs.to(config.DEVICE)
            masks = masks.to(config.DEVICE)


            preds = model(imgs)

            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), config.MODEL_PATH)
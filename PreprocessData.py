import json
import random

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from ScranBot import ScranModel
from ScranDataSet import ScranDataset
from ScranGetter import ScranGetter

# Image transformation
image_transform = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor()
])

# Text tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Flatten structured fields
def flatten_text(entry):
    return f"{entry['host']} | {entry['year']} | {entry['country']} | {entry['title']} | {entry['desc']} | {entry['price']}"

# Image loader
def load_image_tensor(path):
    image = Image.open(path).convert("RGB")
    return image_transform(image)

# Compute label
def get_label(score_l, score_r):
    return 0 if float(score_l.strip('%')) > float(score_r.strip('%')) else 1

# Full preprocess
def preprocess_pair(pair, scores):
    left, right = pair

    # Image
    img_l = load_image_tensor(left["image"])
    img_r = load_image_tensor(right["image"])

    # Text
    text_l = flatten_text(left)
    text_r = flatten_text(right)

    tok_l = tokenizer(text_l, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    tok_r = tokenizer(text_r, return_tensors="pt", padding="max_length", truncation=True, max_length=32)

    label = torch.tensor(get_label(scores[0], scores[1]), dtype=torch.float)

    return {
        "img_left": img_l,
        "img_right": img_r,
        "tok_left": tok_l,
        "tok_right": tok_r,
        "label": label
    }

def collate_fn(batch):
    # Stack image tensors
    img_left = torch.stack([item['img_left'] for item in batch])
    img_right = torch.stack([item['img_right'] for item in batch])

    # Stack tokenized text (input_ids, attention_mask, etc.)
    tok_left = {
        key: torch.stack([item['tok_left'][key] for item in batch])
        for key in batch[0]['tok_left']
    }
    tok_right = {
        key: torch.stack([item['tok_right'][key] for item in batch])
        for key in batch[0]['tok_right']
    }

    labels = torch.stack([item['label'] for item in batch])

    return {
        'img_left': img_left,
        'img_right': img_right,
        'tok_left': tok_left,
        'tok_right': tok_right,
        'label': labels
    }

def load_pair(l_index, r_index):
    with open('data.json', 'r') as df, open('key.json', 'r') as kf:
        data = json.load(df)
        keys = json.load(kf)

        return [data[l_index], data[r_index]], [keys[l_index], keys[r_index]]

def plot_transformed_images(transform, seed=42):
    random.seed(seed)
    pair, score = load_pair(random.randint(0, 184), random.randint(0, 184))
    for item in pair:
        image_path = item["image"]
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

    plt.show()

def get_pairs(n=100, seed=42):
    random.seed(seed)
    data_pairs, score_pairs = [], []
    for i in range(n):
        data_pair, score_pair = load_pair(random.randint(0, 184), random.randint(0, 184))
        data_pairs.append(data_pair)
        score_pairs.append(score_pair)

    return data_pairs, score_pairs

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    for batch in dataloader:
        # 1. Forward pass
        logits = model(batch['img_left'], batch['img_right'],
                       batch['tok_left'], batch['tok_right'])
        print(logits)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(logits, batch['label'])
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
    #     y_pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    #     train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    #
    #     # Adjust metrics to get average loss and accuracy per batch
    # train_loss = train_loss / len(dataloader)
    # train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch in dataloader:
            # 1. Forward pass
            test_logits = model(batch['img_left'], batch['img_right'],
                           batch['tok_left'], batch['tok_right'])

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_logits, batch['label'])
            test_loss += loss.item()

            # Calculate and accumulate accuracy
    #         test_pred_labels = test_logits.argmax(dim=1)
    #         test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
    #
    # # Adjust metrics to get average loss and accuracy per batch
    # test_loss = test_loss / len(dataloader)
    # test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


from tqdm.auto import tqdm


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        # 4. Print out what's happening
        # print(
        #     f"Epoch: {epoch + 1} | "
        #     f"train_loss: {train_loss:.4f} | "
        #     f"train_acc: {train_acc:.4f} | "
        #     f"test_loss: {test_loss:.4f} | "
        #     f"test_acc: {test_acc:.4f}"
        # )
        #
        # # 5. Update results dictionary
        # # Ensure all data is moved to CPU and converted to float for storage
        # results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        # results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        # results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        # results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

def learn(num_epochs=1):
    model = ScranModel()
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

    data_pairs, score_pairs = get_pairs()

    dataset = ScranDataset(data_pairs, score_pairs, preprocess_pair)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Setup loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()

    # Train model_0
    model_results = train(model=model,
                            train_dataloader=dataloader,
                            test_dataloader=dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=num_epochs)

    torch.save(model.state_dict(), "model_weights.pth")

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

def play():
    model = ScranModel()
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
    model.eval()
    print("got ScranBot")
    getter = ScranGetter()
    with torch.inference_mode():
        for i in range(10):
            data_pair = getter.record()

            dataset = ScranDataset([data_pair], [["0%", "0%"]], preprocess_pair)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

            for batch in dataloader:
                logit = model(batch['img_left'], batch['img_right'],
                           batch['tok_left'], batch['tok_right'])

                print(logit)

                if logit[0] < 0:
                    getter.play_binary(choice=False)
                else:
                    getter.play_binary(choice=True)

    getter.quit()



learn(2)
# play()


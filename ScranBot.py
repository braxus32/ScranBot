import torch
import torch.nn as nn
from transformers import AutoModel

class ScranModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 80 * 80, 512),
            nn.ReLU()
        )

        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768 + 512 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, img_left, img_right,
                tok_left, tok_right):

        img_feat_left = self.image_encoder(img_left)
        img_feat_right = self.image_encoder(img_right)

        input_ids_left = tok_left["input_ids"].squeeze(1)
        attention_mask_left = tok_left["attention_mask"].squeeze(1)
        input_ids_right = tok_right["input_ids"].squeeze(1)
        attention_mask_right = tok_right["attention_mask"].squeeze(1)

        text_feat_left = self.text_encoder(
            input_ids=input_ids_left,
            attention_mask=attention_mask_left
        ).pooler_output

        text_feat_right = self.text_encoder(
            input_ids=input_ids_right,
            attention_mask=attention_mask_right
        ).pooler_output

        # Combine all
        features = torch.cat([img_feat_left, text_feat_left,
                              img_feat_right, text_feat_right], dim=1)

        return self.classifier(features).squeeze(1)

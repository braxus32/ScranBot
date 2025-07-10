from torch.utils.data import Dataset

class ScranDataset(Dataset):
    def __init__(self, data_pairs, score_pairs, preprocess_fn):
        self.data_pairs = data_pairs
        self.score_pairs = score_pairs
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.preprocess(self.data_pairs[idx], self.score_pairs[idx])
        return item

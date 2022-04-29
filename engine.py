import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

class Engine(object):
    """
    The search engine that maintains a database and retrieve the real images that
    are the most similar to the input doodle.

    The only optimization is the cuda acceleration.
    Higher speedup can be achieved via batch preprocessing of the database.
    """

    def __init__(self, dataset, doodle_model, real_model):
        self.doodle_model = doodle_model
        self.real_model = real_model
        self.doodle_model.eval()
        self.real_model.eval()

        self.database = self.get_database_from_dataset(dataset)  # format: {vec: (img, label)}

        print(f'Engine ready. Database size: {len(self.database)}')

    def query(self, doodle_img, topk=1):
        doodle_img = doodle_img.reshape(1, 64, 64)

        def get_doodle_transforms(X, size):
            T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((X/255).mean(axis=(0, 1, 2)),
                                    (X/255).std(axis=(0, 1, 2)))])

            return T

        doodle_preprocess = get_doodle_transforms(doodle_img, 64)
        doodle_img = doodle_preprocess(doodle_img).unsqueeze(0)

        with torch.no_grad():
            _, query_vector = self.doodle_model(doodle_img, return_feats=True)
        sims, retrieved_samples = [], []
        for vec_db, (img_db, label_db) in self.database.items():
            sims.append(F.cosine_similarity(query_vector, vec_db, dim=1).item())
            retrieved_samples.append((img_db, label_db))
        topk_id = np.argpartition(sims, len(sims) - topk)[-topk:]
        return [retrieved_samples[x] for x in topk_id]

    def get_database_from_dataset(self, dataset):
        # take a dataset object as input
        def get_real_transforms(X, size):
            T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((X/255).mean(), (X/255).std())])
            
            return T

        def get_doodle_transforms(X, size):
            T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((X/255).mean(axis=(0, 1, 2)),
                                    (X/255).std(axis=(0, 1, 2)))])

            return T

        real_data, real_label = dataset.X, dataset.Y  # np arrays
        real_preprocess = get_real_transforms(real_data, 64)

        pairs = {}
        for i, (data, label) in enumerate(zip(real_data, real_label)):
            data_processed = real_preprocess(data)
            with torch.no_grad():
                # here we use a batch size of 1.
                # A larger value can lead to speedup, but it requires some engineering
                _, vec = self.real_model(data_processed.unsqueeze(0), return_feats=True)
            img_label_pair = (data, label)
            pairs[vec] = img_label_pair

            if i % 1000 == 0:
                print(f'building database... [{i} / {len(real_data)}]')

        return pairs


if __name__ == "__main__":
    from dataset import ImageDataset
    from training_config import doodles, reals, doodle_size, real_size, NUM_CLASSES

    train_set = ImageDataset(doodles, reals, doodle_size, real_size, train=True)
    val_set = ImageDataset(doodles, reals, doodle_size, real_size, train=False)

    # model hyper params.
    dropout = 0.2
    add_layer = True

    # models
    from models import ExampleCNN

    doodle_model = ExampleCNN(1, NUM_CLASSES, dropout, add_layer)  # please use checkpoint in real runs
    real_model = ExampleCNN(3, NUM_CLASSES, dropout, add_layer)

    # engine
    doodle, doodle_label, _, _ = train_set[0]  # a random doodle sample
    engine = Engine(val_set, doodle_model, real_model)  # pass in the val_set and the trained models

    # search
    out_samples = engine.query(doodle)
    print(len(out_samples))

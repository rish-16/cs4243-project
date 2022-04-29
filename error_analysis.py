from dataset_collection import *
from model_training import *

def plot_errors(x, y, yhat, n=5):
    idx = torch.where(y!=yhat)[0]
    plot_row(sample_array(x[idx], n))
    
def analyse_errors(model, dataset, classes, n=10):
    dl = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x, y = next(iter(dl))
    model = model.eval()
    with torch.no_grad():
        preds, _ = model(x, return_feat=True)
        _, yhat = torch.max(preds, 1)
    for i, c in enumerate(classes):
        print(c)
        idx = torch.where(yhat == i)[0]
        plot_errors(dataset.X[idx], y[idx], yhat[idx], n=n)
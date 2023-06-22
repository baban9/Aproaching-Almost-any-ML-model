import torch 
import torch.nn as nn
from tqdm import tqdm

def train(data_loader, model, optimizer, device): 
    model.train() 
    
    for data in tqdm(data_loader, desc="training sample"):
        reviews = data['review']
        targets = data['target']

        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float32) 

        optimizer.zero_grad() 

        predictions = model(reviews) 

        loss = nn.BCEWithLogitsLoss()(
            predictions, targets.view(-1,1)
        )

        loss.backward() 
        optimizer.step() 

def evaluate(data_loader, model, device):
    final_predictions = [] 
    final_targets = [] 

    model.eval() 
    with torch.no_grad():
        for data in tqdm(data_loader, desc="eval samples"):
            reviews = data['review']
            targets = data['target'] 
            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float) 

            predictions = model(reviews)
            predictions = predictions.cpu().numpy().tolist()
            targets = data["target"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)

    return final_predictions, final_targets
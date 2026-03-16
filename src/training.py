import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def accuracy(out, yb): return (out.argmax(dim=1)==yb).float().mean()
def report(loss, preds, yb): print(f'loss: {loss:.5f}, accuracy: {accuracy(preds, yb):.3f}')

class Dataset:
    def __init__(self, x, y): self.x,self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i],self.y[i]

def evaluate(model, valid_dl, loss_func, ae_loss=False):
    # pred_list = []
    # label_list = []
    total_loss = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for xb,yb in valid_dl:
            pred = model(xb)
            # pred_list.append(pred)
            # if ae_loss: label_list.append(xb)
            # else: label_list.append(yb)
            n = len(xb)
            count += n
            if ae_loss: loss = loss_func(pred,xb).item()*n
            else: loss = loss_func(pred,yb).item()*n
            total_loss += loss
        # preds = torch.cat(pred_list, dim=0)
        # labels = torch.cat(label_list, dim=0)
        # loss = loss_func(preds, labels)
        print(f'loss: {total_loss/count:.5f}')
        # report(loss, preds, labels)

def fit(model, train_dl, valid_dl, opt, loss_func, epochs, ae_loss = False):
    print(f"initial performance ===============")
    evaluate(model, valid_dl, loss_func, ae_loss)
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
            if ae_loss: loss = loss_func(model(xb), xb)
            else: loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(f"end of epoch {epoch} performance ===============")
        evaluate(model, valid_dl, loss_func, ae_loss)


def get_dls(train_ds, valid_ds, bs, collate_fn, min_bs = 8):
    last_train_batch_len = len(train_ds) % bs
    train_dl = DataLoader(
        train_ds,
        bs,
        shuffle=True,
        drop_last=(last_train_batch_len > 0) and (last_train_batch_len < min_bs),
        collate_fn=collate_fn
        )
    valid_dl = DataLoader(
        valid_ds,
        bs,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
        )
    return train_dl, valid_dl

def get_model_opt(input_dim, hidden_dim, n_cls, lr=0.5):
    model = nn.Sequential(
        nn.Linear(input_dim,hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim,n_cls)
    )
    opt = optim.SGD(model.parameters(), lr=lr)
    return model, opt
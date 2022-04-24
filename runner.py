def trainer(
    max_epoch, 
    model, 
    train_loader, 
    test_loader, 
    loss_fn,
    optimizer,
    scheduler,
    writer
):
    for ep in range(1, max_epoch+1):
        train(ep, model, train_loader, loss_fn, optimizer, writer)
        if scheduler is not None:
            scheduler.step()

        if True:
            acc = test(ep, model, test_loader, writer)

def tester():
    pass

def train(ep, model, train_loader, loss_fn, optimizer, writer):
    model.train()
    mean_loss = 0.0
    for batch in train_loader:
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        out = model(x)

        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()

@torch.no_grad()
def test(ep, model, test_loader, writer):
    model.eval()

    for batch in test_loader:
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        out = model(x)


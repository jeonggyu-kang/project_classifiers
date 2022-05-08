import torch

from evaluation import calc_accuracy


def trainer(
    max_epoch, 
    model, 
    train_loader, 
    test_loader, 
    loss_fn,
    optimizer,
    scheduler,
    meta, 
    writer = None,
):

    save_every = meta['save_every']
    print_every = meta['print_every']
    test_every = meta['test_every']


    for ep in range(1, max_epoch+1):
        train(ep, max_epoch, model, train_loader, loss_fn, optimizer, writer, print_every)
        if scheduler is not None:
            scheduler.step()

        if True:
        # if ep % test_every == 0:
            acc = test(ep, max_epoch, model, test_loader, writer)
            #TODO : implement
            #writer.update(acc)
        
        if ep% save_every == 0:
            # wariter.save()
            pass
    

def tester():
    pass

def train(ep, max_epoch, model, train_loader, loss_fn, optimizer, writer, _print_every):
    model.train()
    mean_loss = 0.0

    print_every = len(train_loader) // _print_every

    preds = []
    gts = []

    step = 0
    step_cnt = 1

    for i, batch in enumerate(train_loader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        out = model(x)

        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        step += 1

        preds.append(out.data.cpu())
        gts.append(y.data.cpu())

        if (i+1) % print_every == 0:
            mean_loss /= step
            print('Epoch [{}/{}] Step[{}/{}] Loss: {:.4f}'.format(
                ep, max_epoch, step_cnt, _print_every, mean_loss))
            
            mean_loss = 0.0
            step = 0
            step_cnt += 1
        
    preds = torch.cat(preds)
    gts = torch.cat(gts)

    acc = calc_accuracy(preds, gts)

    print('Epoch[{}/{}] Train Acc: {:.4f}'.format(ep,max_epoch, acc))
    print('='*40)


@torch.no_grad()                                         # stop calculating gradient
def test(ep, max_epoch, model, test_loader, writer):
    model.eval()

    preds = []
    gts = []

    for batch in test_loader:
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        out = model(x)
        preds.append(out.data.cpu())
        gts.append(y.data.cpu())

    preds = torch.cat(preds)
    gts = torch.cat(gts)

    acc = calc_accuracy(preds, gts)
    print ('Epoch[{}/{}] TestAcc: {:.4f}'.format(ep, max_epoch, acc))
    print ('+'*40)
    return acc

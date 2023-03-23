import torch


def save_weight(model, epoch, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_weight(path):
    check_point = torch.load(path)
    return check_point

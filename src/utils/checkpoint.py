import os
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save training checkpoint with full state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
        if optimizer and 'optimizer_state' in ckpt and ckpt['optimizer_state']:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if scheduler and 'scheduler_state' in ckpt and ckpt['scheduler_state']:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        return ckpt.get('metrics', {}), ckpt.get('epoch', 0)
    else:
        model.load_state_dict(ckpt)
        return {}, 0


def load_model_weights(model, weights_path, device='cpu'):
    """Load model weights for inference."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model

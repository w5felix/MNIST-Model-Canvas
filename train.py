import os
import argparse
import importlib
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from digitsvis.models import SimpleCNN
from digitsvis.data import get_dataloaders

# Resolve project base directory (this file's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = loss_sum / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def export_to_onnx(checkpoint_path: str, onnx_out: Optional[str] = None) -> str:
    """Export a trained SimpleCNN checkpoint to ONNX.

    Returns the ONNX output path.
    """
    ckpt_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    default_onnx = os.path.join(BASE_DIR, 'static', 'mnist_cnn.onnx')
    onnx_path = onnx_out or default_onnx
    onnx_path = os.path.abspath(os.path.expanduser(onnx_path))
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    device = torch.device('cpu')
    model = SimpleCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 1, 28, 28, device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=['input'],
        output_names=['logits'],
        opset_version=13,
        dynamic_axes=None,
    )
    return onnx_path


def train(
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
    out_path: str = "mnist_cnn.pt",
    dataset: str = "mnist",
    data_root: str = "data",
    num_workers: int = 2,
    dataset_plugin: Optional[str] = None,
    onnx_out: Optional[str] = None,
    skip_onnx: bool = False,
) -> None:
    # Optionally import a dataset plugin module which registers a custom dataset
    if dataset_plugin:
        print(f"Loading dataset plugin module: {dataset_plugin}")
        importlib.import_module(dataset_plugin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader = get_dataloaders(dataset=dataset, root=data_root, batch_size=batch_size, num_workers=num_workers)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] loss: {running_loss/200:.4f}")
                running_loss = 0.0

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch} validation - loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, out_path)
    print(f"Saved trained model to {out_path}")

    # Optional: export to ONNX for in-browser inference (served from /static)
    if not skip_onnx:
        try:
            # Resolve default ONNX output
            default_onnx = os.path.join(BASE_DIR, 'static', 'mnist_cnn.onnx')
            onnx_path = onnx_out or default_onnx
            onnx_path = os.path.abspath(os.path.expanduser(onnx_path))
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

            # Always export from CPU for maximal compatibility
            dummy = torch.randn(1, 1, 28, 28)
            model_cpu = model.to('cpu').eval()
            torch.onnx.export(
                model_cpu,
                dummy,
                onnx_path,
                input_names=['input'],
                output_names=['logits'],
                opset_version=13,
                dynamic_axes=None,
            )
            print(f"Exported ONNX model to {onnx_path}")
        except Exception as e:
            print(f"ONNX export skipped/failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SimpleCNN on a selectable dataset or export ONNX.")
    parser.add_argument('--epochs', type=int, default=int(os.getenv("EPOCHS", "3")), help='Number of epochs')
    parser.add_argument('--lr', type=float, default=float(os.getenv("LR", "0.001")), help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=int(os.getenv("BATCH", "64")), help='Batch size')
    parser.add_argument('--model-out', type=str, default=os.getenv("MODEL_OUT", "mnist_cnn.pt"), help='Model checkpoint path (for saving or exporting)')
    parser.add_argument('--dataset', type=str, default=os.getenv("DATASET", "mnist"), help='Dataset name (registered)')
    parser.add_argument('--data-root', type=str, default=os.getenv("DATA_ROOT", "data"), help='Dataset root directory')
    parser.add_argument('--num-workers', type=int, default=int(os.getenv("NUM_WORKERS", "2")), help='DataLoader workers')
    parser.add_argument('--dataset-plugin', type=str, default=os.getenv("DATASET_PLUGIN", ""), help='Module path to import to register a custom dataset')
    parser.add_argument('--onnx-out', type=str, default=os.getenv("ONNX_OUT", ""), help='Optional ONNX output path (defaults to static/mnist_cnn.onnx)')
    parser.add_argument('--skip-onnx', action='store_true', help='Skip ONNX export after training')
    parser.add_argument('--export-only', action='store_true', help='Only export ONNX from an existing checkpoint (no training)')
    args = parser.parse_args()

    plugin = args.dataset_plugin or None

    if args.export_only:
        out = export_to_onnx(args.model_out, args.onnx_out or None)
        print(f"Exported ONNX to: {out}")
    else:
        train(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            out_path=args.model_out,
            dataset=args.dataset,
            data_root=args.data_root,
            num_workers=args.num_workers,
            dataset_plugin=plugin,
            onnx_out=(args.onnx_out or None),
            skip_onnx=bool(args.skip_onnx),
        )

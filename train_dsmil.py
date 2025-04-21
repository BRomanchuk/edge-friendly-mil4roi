from pipeline import train_dsmil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pooling model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")

    args = parser.parse_args()

    train_dsmil(device=args.device, epochs=args.epochs)
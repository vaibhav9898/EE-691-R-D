import torch
from data_loader import get_data_loaders
from models import Classifier, Generator
from utils import weights_initialization_gen_normal, generate_sorted_input_pdf
from train import train
from config import Config

def main():
    # Load configuration
    cfg = Config()

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, test_loader = get_data_loaders(cfg)

    # Initialize models
    csf = Classifier(cfg.nc, cfg.ncf).to(device)
    csf.load_state_dict(torch.load('best_classifier.pth'))
    csf.eval()

    gen = Generator(cfg.nz, cfg.ngf, cfg.nc, cfg.n_classes).to(device)
    gen.apply(weights_initialization_gen_normal)
    gen.train()

    # Generate fixed input for visualization
    fixed_noise, fixed_input_pdf = generate_sorted_input_pdf(
        cfg.labels_per_class * cfg.n_classes, cfg.n_classes, cfg.nz
    )
    fixed_noise = fixed_noise.to(device)
    fixed_input_pdf = fixed_input_pdf.to(device)

    # Train the generator
    train(csf, gen, train_loader, fixed_noise, fixed_input_pdf, cfg, device)

if __name__ == "__main__":
    main()
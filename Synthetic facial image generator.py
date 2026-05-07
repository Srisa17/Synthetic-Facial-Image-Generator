import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import time
import os

# --- 1. CONFIGURATION & PATHS ---
CHECKPOINT_DIR = "Checkpoints"
OUTPUT_DIR = "Outputs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  
BATCH_SIZE = 128      
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_EPOCHS = 20      

# --- 2. YOUR ORIGINAL MODELS (Restored for Checkpoint Compatibility) ---
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(CHANNELS_IMG, FEATURES_DISC, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(FEATURES_DISC, FEATURES_DISC*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_DISC*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(FEATURES_DISC*2, FEATURES_DISC*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_DISC*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(FEATURES_DISC*4, FEATURES_DISC*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_DISC*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(FEATURES_DISC*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x): return self.disc(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, FEATURES_GEN*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURES_GEN*16),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURES_GEN*16, FEATURES_GEN*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURES_GEN*8, FEATURES_GEN*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURES_GEN*4, FEATURES_GEN*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(FEATURES_GEN*2, CHANNELS_IMG, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, x): return self.gen(x)

# --- 3. MAIN EXECUTION ---
if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = datasets.ImageFolder(root="data", transform=data_transforms) 
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    gen.apply(weights_init)
    disc.apply(weights_init)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

    # --- 4. AUTO-RESUME (Silent Load) ---
    START_EPOCH = 0
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth.tar")]
    if checkpoints:
        latest_cp = sorted(checkpoints)[-1]
        cp_path = os.path.join(CHECKPOINT_DIR, latest_cp)
        print(f"[INFO] Resuming from: {cp_path}")
        # weights_only=False ensures optimizer and epoch load correctly
        checkpoint = torch.load(cp_path, map_location=device, weights_only=False)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1
    else:
        print("[INFO] No checkpoint found. Starting fresh.")

    # --- 5. TRAINING LOOP ---
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        start_time = time.time()
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            b_size = real.size(0)

            # 1. Train Discriminator
            disc.zero_grad()
            # Label Smoothing (0.9) for cleaner image detail
            label = torch.full((b_size,), 0.9, device=device)
            output = disc(real).view(-1)
            loss_d_real = criterion(output, label)
            loss_d_real.backward()

            noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
            fake = gen(noise)
            label.fill_(0.1) # Smoothed label for fake
            output = disc(fake.detach()).view(-1)
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward()
            
            loss_d = loss_d_real + loss_d_fake
            opt_disc.step()

            # 2. Train Generator
            gen.zero_grad()
            label.fill_(1.0) 
            output = disc(fake).view(-1)
            loss_g = criterion(output, label)
            loss_g.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

        # --- SAVE PER EPOCH ---
        with torch.no_grad():
            fake_display = gen(fixed_noise).detach().cpu()
            img_filename = os.path.join(OUTPUT_DIR, f"output_epoch_{epoch:03d}.png")
            vutils.save_image(fake_display, img_filename, normalize=True)
            
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"gan_checkpoint_epoch_{epoch:03d}.pth.tar")
        torch.save({
            'gen_state_dict': gen.state_dict(), 'disc_state_dict': disc.state_dict(), 
            'opt_gen_state_dict': opt_gen.state_dict(), 'opt_disc_state_dict': opt_disc.state_dict(), 
            'epoch': epoch
        }, checkpoint_path)
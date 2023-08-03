#Import libraries

from sklearn import model_selection
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import HAM10000_Dataset
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import torch
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1024, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=7, help="number of classes for dataset")
parser.add_argument("--img_height", type=int, default=450, help="size of height image dimension")
parser.add_argument("--img_widht", type=int, default=600, help="size of widht image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--save_folder", type=str, default="images_cgan2", help="interval between image sampling")
opt = parser.parse_args()

# Start wandb configs
args_dict = vars(opt)
wandb.init(
    project='MML_project',
    config=args_dict
)

os.makedirs(opt.save_folder, exist_ok=True)
img_shape = (opt.channels, opt.img_height, opt.img_widht)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

num_parametros_gen = sum(p.numel() for p in Generator.parameters(Generator))
print("Número total de parámetros del generador:", num_parametros_gen)
num_parametros_dis = sum(p.numel() for p in Discriminator.parameters(Discriminator))
print("Número total de parámetros del discriminador:", num_parametros_dis)

breakpoint()

# Categories of the diferent lesions
labels_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Get data
csv_path = "data/HAM10000_metadata.csv"
df_data=pd.read_csv(csv_path).set_index('image_id')
df_data.dx=df_data.dx.astype('category',copy=True)
df_data['label']=df_data.dx.cat.codes # Create a new column with the encoded categories
df_data['lesion_type']= df_data.dx.map(labels_dict) # Create a new column with the lesion type
df_data['path'] = "data/resized/"+df_data.index + '.jpg' # Create a new column with the path to the image

# Save relation between label and lesion_type
label_list = df_data['label'].value_counts().keys().tolist()
lesion_list = df_data['lesion_type'].value_counts().keys().tolist()
label_to_lesion = dict(zip(label_list, lesion_list))

# Split data into train, validation and test
test_size = 0.2
val_size = 0.15
df_train, df_test = model_selection.train_test_split(df_data, test_size=test_size)
df_train_idx = df_train.reset_index().copy()
df_train, df_val = model_selection.train_test_split(df_train_idx, test_size=val_size)

# Configure data loader
training_data = HAM10000_Dataset(df_train)
validation_data = HAM10000_Dataset(df_val)
test_data = HAM10000_Dataset(df_test)

print("Training data: ", len(training_data))
print("Validation data: ", len(validation_data))
print("Test data: ", len(test_data))

train_dataloader = DataLoader(training_data, batch_size=opt.batch_size, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=opt.batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Verify cuda
cuda = True if torch.cuda.is_available() else False

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, batches_done):
    """Saves a grid of generated images ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, f"{opt.save_folder}/{batches_done}.png", nrow=n_row, normalize=True)

def val():
    generator.eval()
    discriminator.eval()
    total_val_loss = 0.0
    total_val_samples = 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(val_dataloader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, valid)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            total_val_loss += d_loss.item() * batch_size
            total_val_samples += batch_size

    generator.train()
    discriminator.train()
    return total_val_loss / total_val_samples


def test():
    generator.eval()
    discriminator.eval()
    total_test_loss = 0.0
    total_test_samples = 0

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_dataloader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, valid)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            total_test_loss += d_loss.item() * batch_size
            total_test_samples += batch_size

    return total_test_loss / total_test_samples

#  Training
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(train_dataloader):
    
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        #  Train Generator
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        #  Train Discriminator
        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(train_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=opt.n_classes, batches_done=batches_done)

        # Validation at the end of each epoch
        val_loss = val()
        print(f"Validation Loss after Epoch {epoch}: {val_loss}")

        log_dict = {}
        log_dict['d_loss'] = d_loss.item()
        log_dict['d_real_loss'] = d_real_loss.item()
        log_dict['d_fake_loss'] = d_fake_loss.item()
        log_dict['g_loss'] = g_loss.item()
        log_dict['val_loss'] = val_loss
        log_dict['Epoch'] = epoch
        wandb.log(log_dict)

# Testing after the training is complete
test_loss = test()
print(f"Test Loss: {test_loss}")
wandb.log({'test_loss': test_loss})

wandb.finish()


# Import libraries
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import seaborn as sns
import pandas as pd
import os

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

# Visualize some examples

lesion_groups = df_data.sort_values(["lesion_type"]).groupby("lesion_type")
num_columns = 3
num_rows = len(lesion_groups)
fig, axs = plt.subplots(num_rows, num_columns, figsize=(12, 3*num_rows))  # Adjust figsize based on the number of rows

for row, (lesion_name,images_id) in enumerate(lesion_groups):
    for col, (_, c_row) in enumerate(images_id.sample(num_columns).iterrows()):
        if col == 0:
            axs[row,col].set_ylabel(lesion_name, fontsize=20)
        ax = axs[row, col]
        ax.imshow(imread(os.path.join("data", "HAM10000_images", c_row.name + '.jpg')))
        ax.set_xticks([])  
        ax.set_yticks([])

plt.tight_layout()
fig.savefig('lesion_examples.png', dpi=300)

# Plot some histograms to caracterize the data
plt.style.use('seaborn')

plt.figure()
lesion = df_data['lesion_type'].value_counts()
plt.bar(lesion.keys().tolist(), lesion.tolist(), color=sns.color_palette("husl", 9))
plt.title('Lesion type')
plt.xticks(rotation=20)
plt.savefig('lesion_histogram.png', dpi=300)

plt.figure()
localization = df_data['localization'].value_counts()
plt.bar([x[:8] for x in localization.keys().tolist()],localization.tolist(), color=sns.color_palette("husl", 9))
plt.title('Lesion localization')
plt.xticks(rotation=30)
plt.savefig('localization_histogram.png', dpi=300)

plt.figure()
age = df_data['age'].value_counts()
plt.bar([int(x) for x in age.keys().tolist()], age.tolist(), color=sns.color_palette("husl", 9))
plt.title('Patient Age')
plt.xticks(rotation=30)
plt.savefig('age_histogram.png', dpi=300)

plt.figure()
sex = df_data['sex'].value_counts()
plt.bar(sex.keys().tolist(), sex.tolist(), color=sns.color_palette("husl", 9))
plt.title("Patient Sex")
plt.xticks(rotation=30)
plt.savefig('sex_histogram.png', dpi=300)

def resize_images_in_folder(folder_path, desired_size):

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        try:
            # Open the image using PIL
            img = Image.open(filepath)

            # Resize the image
            resized_img = img.resize(desired_size, Image.ANTIALIAS)

            # Save the resized image 
            new_folder = os.path.join("data","resized")
            os.makedirs(new_folder, exist_ok=True)
            new_filepath = os.path.join(new_folder, filename)
            resized_img.save(new_filepath)

            print(f"Resized {filename} to {desired_size}")

        except Exception as e:
            print(f"Error resizing {filename}: {e}")

# Call the function to resize images in the folder
# resize_images_in_folder("data/HAM10000_images", (128,128))

class HAM10000_Dataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame):
        self.df = df
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # read dataframe row
        row = self.df.iloc[idx]
        # get image label and path
        lesion_class = row["label"]
        image_file = row['path']
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Couldn't find image {image_file}")
        # read image 
        image = imread(image_file)
        image = self.toTensor(image)

        return image, lesion_class


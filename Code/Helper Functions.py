def create_dir_structure(root_dir, source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_name in os.listdir(source_dir):
        if file_name.endswith(('.jpg','.jpeg','.png')):
            person_id = file_name.split('_')[0]
            person_dir = os.path.join(dest_dir, person_id)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(person_dir, file_name))

class Organize_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [] # creates a list of image_paths(person_dir+img_name) for the images present within the root directory
        self.labels = [] # creates a list of labels(person ID) corresponding to the image_path in image_paths list. contains duplicates as the same person can have multiple images.

        for label in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, label)
            if os.path.isdir(person_dir):
                for img_name in os.listdir(person_dir):
                    if img_name.endswith(('.jpg','.jpeg','.png')):
                        self.image_paths.append(os.path.join(person_dir, img_name))
                        self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label




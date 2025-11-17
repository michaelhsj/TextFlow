from torchvision.transforms import v2


#define data augmentation

class RandomSubsetTransform:
    """Randomly apply k out of n transforms to each image"""
    
    def __init__(self, num_transforms=2, intensity=0.5):
        """
        Args:
            num_transforms: Number of transforms to apply per image
        """
        self.num_transforms = num_transforms
        
        # Define pool of available transforms
        self.transform_pool = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomGrayscale(p=1.0),
        ]
    
    def __call__(self, img):
        # Randomly select k transforms
        selected = random.sample(self.transform_pool, 
                                min(self.num_transforms, len(self.transform_pool)))
        
        # Apply selected transforms sequentially
        for transform in selected:
            img = transform(img)
        
        # Always convert to tensor at the end
        img = transforms.ToTensor()(img)
        return img

# - slight rotation, noise, distortion, blur?
# - choice of 2 of 4 out of given image, apply each augmentation with intensity defined by t

if __name__ == "__main__":
    detection_model = ...
    recognition_mdoel = ...

    dataset = TextOCRDoctrDetDataset(
    images_dir=DATASET_DIR / "train_val_images",
    json_path=DATASET_DIR / "TextOCR_0.1_train.json",
    num_samples=TRAIN_NUM_SAMPLES,
    )

    loader = DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=doctr_detection_collate,
)

    epochs = 100
    max_t = 100

    for i in range(epochs):
        t = (i / epochs - 1) * max_t
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device)
            #transform images
            #...abs

            output = model(images, targets)
            loss = output["loss"]
            preds = output["preds"]

            







#import trained recognition and detection models

#slot data augmentation into detection dataset loader

#define small test dataset

#t from 1 to 100 - defines scale of augmentation

#for each t (or linrange of t), pass our model over randomly augmented inputs

#evaluate detection and recognition loss across t
#optionally (but I think importantly, define accuracy metrics and calculate?)

#graph, save graph. 
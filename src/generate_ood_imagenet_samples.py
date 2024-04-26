import os

import dotenv
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2

SEED = 33096
NUM_SAMPLES = 500


def main():
    dotenv.load_dotenv()
    DATA_ROOT = os.environ["DATA_ROOT"]

    dataset = torchvision.datasets.ImageNet(
        root=os.path.join(DATA_ROOT, "imagenet"),
        split="train",
        transform=torchvision.transforms.v2.Compose(
            [
                torchvision.transforms.v2.ToImage(),
                torchvision.transforms.v2.ToDtype(torch.uint8, scale=True),
                torchvision.transforms.v2.Resize((32, 32)),
            ]
        ),
    )

    rng = np.random.default_rng(SEED)
    target_classes = np.concatenate(
        (
            np.arange(300, 330),
            np.arange(410, 420),
            np.arange(445, 465),
            np.arange(473, 481),
            np.arange(485, 509),
            np.arange(518, 534),
            np.arange(537, 546),
            np.arange(548, 554),
            np.arange(588, 594),
            np.arange(596, 626),
            np.arange(629, 653),
            np.arange(666, 669),
            np.arange(672, 674),
            np.arange(676, 689),
            np.arange(691, 692),
            np.arange(695, 716),
            np.arange(718, 723),
            np.arange(725, 729),
            np.arange(731, 999),
        ),
        axis=0,
    )
    assert len(target_classes) == NUM_SAMPLES  # one sample per class

    all_labels = np.array(dataset.targets)

    ood_images = torch.empty((NUM_SAMPLES, 3, 32, 32), dtype=torch.uint8)
    for sample_idx, sample_class in enumerate(target_classes):
        candidate_indices = np.argwhere(all_labels == sample_class)[:, 0]
        imagenet_idx = rng.choice(candidate_indices)
        image, _ = dataset[imagenet_idx]
        assert image.dtype == torch.uint8 and image.shape == (3, 32, 32)
        ood_images[sample_idx] = image

    output_path = os.path.join(os.path.abspath(os.path.basename(__file__)), os.pardir, "ood_imagenet_samples.pt")
    torch.save(ood_images, output_path)
    print("Saved images to", output_path)


if __name__ == "__main__":
    main()

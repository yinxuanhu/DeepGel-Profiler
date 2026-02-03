import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from sklearn.decomposition import PCA

# --- Configuration ---
# Path to your image directory
IMAGE_DIR = './data/images'
# Output directory for results
OUTPUT_DIR = './results'
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """
    Load pre-trained ResNet-34 model and remove the classification layer.
    Returns:
        model: PyTorch model acting as a feature extractor.
    """
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    # Remove the final fully connected layer to output 512-dim features
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    """
    Define standard image preprocessing steps for ResNet.
    """
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_features(image_dir, model, transform):
    """
    Iterate through images, extract features, and parse filenames.
    """
    features = []
    metadata = []

    valid_exts = ('.jpg', '.png', '.tif', '.tiff')
    file_list = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)])

    print(f"Starting feature extraction for {len(file_list)} images...")

    with torch.no_grad():
        for filename in file_list:
            try:
                # 1. Load and Preprocess
                img_path = os.path.join(image_dir, filename)
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(DEVICE)

                # 2. Extract Feature Vector (512 dimensions)
                feat = model(img_t).cpu().numpy().flatten()
                features.append(feat)

                # 3. Parse Metadata (Format: 12WPI_0.3MF_4h_01.tif)
                parts = filename.split('_')
                if len(parts) >= 3:
                    metadata.append({
                        'Filename': filename,
                        'WPI': parts[0],
                        'MF': parts[1],
                        'Time': parts[2]
                    })
                else:
                    metadata.append({
                        'Filename': filename,
                        'WPI': 'Unknown', 'MF': 'Unknown', 'Time': 'Unknown'
                    })

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return np.array(features), pd.DataFrame(metadata)


def perform_pca_and_plot(features, df_meta, output_dir):
    """
    Perform PCA and generate two distinct visualization plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Filter for 12% WPI Matrix only
    mask = df_meta['WPI'].astype(str).str.contains('12')
    X_subset = features[mask]
    df_subset = df_meta[mask].copy()

    if len(df_subset) == 0:
        print("No 12% WPI samples found.")
        return

    # 2. Run PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_subset)
    evr = pca.explained_variance_ratio_

    df_subset['PC1'] = pcs[:, 0]
    df_subset['PC2'] = pcs[:, 1]

    print(f"PCA Variance Explained: PC1={evr[0]:.1%}, PC2={evr[1]:.1%}")

    # Save Data for Origin
    df_subset.to_csv(os.path.join(output_dir, 'pca_data_origin.csv'), index=False)

    # --- Plot A: Kinetics (Heating Time Effect) ---
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid", context="talk")

    time_order = ['4h', '8h', '12h']
    # Add Control if exists
    if df_subset['Time'].str.contains('0MF').any() or df_subset['MF'].str.contains('0MF').any():
        # Adjust logic based on your specific naming for control
        pass

    sns.scatterplot(
        x='PC1', y='PC2', hue='Time',
        data=df_subset, palette='autumn_r', hue_order=time_order,
        s=300, alpha=0.9, edgecolor='black'
    )
    plt.title(f'Figure A: Kinetic Trajectory (Time Effect)\nVariance: {evr[0] + evr[1]:.1%}')
    plt.xlabel(f'PC1 ({evr[0]:.1%})')
    plt.ylabel(f'PC2 ({evr[1]:.1%})')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Induction Time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'FigA_Kinetics.png'), dpi=300)
    plt.close()

    # --- Plot B: Dosage (Concentration Effect - 12h Only) ---
    df_mature = df_subset[df_subset['Time'] == '12h'].copy()
    df_mature = df_mature.sort_values(by='MF')

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2', hue='MF',
        data=df_mature, palette='viridis',
        s=300, alpha=0.9, edgecolor='black'
    )
    plt.title(f'Figure B: MF Concentration Effect (12h Mature Gels)')
    plt.xlabel(f'PC1 ({evr[0]:.1%})')
    plt.ylabel(f'PC2 ({evr[1]:.1%})')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="MF Conc")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'FigB_Dosage.png'), dpi=300)
    plt.close()

    print(f"Analysis complete. Results saved to {output_dir}")


def main():
    if not os.path.exists(IMAGE_DIR):
        print(f"Please create {IMAGE_DIR} and put your .tif images there.")
        return

    # Pipeline execution
    model = load_model()
    transform = get_transform()
    features, df_meta = extract_features(IMAGE_DIR, model, transform)

    if len(features) > 0:
        perform_pca_and_plot(features, df_meta, OUTPUT_DIR)
    else:
        print("No features extracted.")


if __name__ == '__main__':
    main()
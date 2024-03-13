import matplotlib.pyplot as plt
import cv2
from feature_extractor import extract_hog_features, extract_lbp_features

if __name__ == "__main__":
    # Example usage for HOG feature extraction:
    image_path = 'files/brown_spot.jpg'
    image = cv2.imread(image_path)
    save_path_hog = 'extracted_images/hog_image.jpg'
    features_hog, hog_image = extract_hog_features(image, save_path=save_path_hog)

    # Visualize the HOG image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap=plt.cm.gray)
    plt.title('HOG Features')
    plt.axis('off')

    plt.show()

    # Example usage for LBP feature extraction:
    save_path_lbp = 'extracted_images/lbp_image.jpg'
    lbp_image = extract_lbp_features(image, save_path=save_path_lbp)

    # Visualize the LBP image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lbp_image, cmap='gray')
    plt.title('LBP Features')
    plt.axis('off')

    plt.show()

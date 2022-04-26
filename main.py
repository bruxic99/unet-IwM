import cv2
import numpy as np
import matplotlib.pyplot as plt


def gamma(img_gray, gamma):
    return (np.power(img_gray / 255, gamma).clip(0, 1) * 255).astype(np.uint8)


def preprocess_image(image):
    red, green, blue = cv2.split(image)

    green_blured = cv2.GaussianBlur(green, (19, 19), 10)
    green_sharpened = cv2.addWeighted(green, 1.5, green_blured, -0.5, 0)

    contrast_green = cv2.equalizeHist(green_sharpened)
    return contrast_green


def process_image(image):
    kernels = [(5, 5), (11, 11), (23, 23)]
    morph = image
    for kernel in kernels:
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel), iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel), iterations=1)

    substract_image = cv2.subtract(morph, image)

    contrast_substract_image = cv2.equalizeHist(substract_image)

    _, thresh_image = cv2.threshold(contrast_substract_image, 15, 255, cv2.THRESH_BINARY)

    return contrast_substract_image, thresh_image


def post_process_image(image, tresh):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(tresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    im = cv2.bitwise_and(image, image, mask=mask)

    _, thresh_im = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)

    eroded = cv2.erode(thresh_im, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    blood_vessels = cv2.bitwise_not(eroded)

    return blood_vessels


def statistic(original, result, vessels):
    mask = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mask = gamma(mask, 1.5)
    _, mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)

    noice = result - cv2.bitwise_not(mask)
    noice = cv2.dilate(noice, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    noice = cv2.bitwise_not(noice)

    result = cv2.bitwise_and(result, noice)

    false_positive = result - vessels
    false_negative = vessels - result
    true_positive = cv2.bitwise_and(result, vessels)
    true_negative = cv2.bitwise_not(result + vessels) - mask

    FP = np.sum(false_positive == 255)
    FN = np.sum(false_negative == 255)
    TP = np.sum(true_positive == 255)
    TN = np.sum(true_negative == 255)

    acurracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return result, [acurracy, sensitivity, specificity]


filename_eye = str(input("Podaj nazwę pliku zawierającego zdjęcie oka: "))
filename_vessel = str(input("Podaj nazwę pliku zawierającego zdjęcie naczynek: "))

filename_eye = "./eye/" + filename_eye + ".jpg"
filename_vessel = "./vessels/" + filename_vessel + ".tif"

print(filename_eye)
img = cv2.imread("zdjecia_oczy/"+filename_eye)
vessels = cv2.imread("zdjecia_naczynka/"+filename_vessel)

if img is not None and vessels is not None:
    vessels = cv2.cvtColor(vessels, cv2.COLOR_BGR2GRAY)
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original = cv2.resize(original, (700, 605))
    vessels = cv2.resize(vessels, (700, 605))

    preimage = preprocess_image(original)
    image, tresh = process_image(preimage)
    result = post_process_image(image, tresh)
    end_image, stats = statistic(original, result, vessels)

    plt.title("Wykryte naczynka oka")
    plt.imshow(end_image, cmap="gray")
    plt.show()
    plt.title("Rzeczywiste naczynka oka")
    plt.imshow(vessels, cmap="gray")
    plt.show()

    print("Trafność: {} \nCzułość: {}, \nSwoistość: {}.".format(stats[0], stats[1], stats[2]))
elif img is None and vessels is not None:
    print("Plik zawierający zdjęcie oka nie istnieje.")
elif img is not None and vessels is None:
    print("Plik zawierający zdjęcie naczynek nie istnieje.")
else:
    print("Podane pliki nie istnieją.")

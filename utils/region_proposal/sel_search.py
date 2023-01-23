import cv2
from selective_search import selective_search

def main():
    image_path = "../../objectness_classification/test_data/e8e54409-495b-49e5-b325-556c812d6ff4/pnr_275.jpg"
    image = cv2.imread(image_path)

    boxes = selective_search(image, mode='single')
    for box in boxes:
        cv2.rectangle(image, (box[0],box[1]), (box[2], box[3]), (0,0,255), thickness=1)

    cv2.imwrite("./selective_search.png", image)


if __name__ == '__main__':
    main()

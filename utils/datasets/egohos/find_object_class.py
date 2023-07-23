import os
import glob
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm
import gensim
import numpy as np


RAW_VIDEO_INFO_FILE = "./video_scene.txt"
SUMMARY_FILE = "./scene_summary.txt"
RESHAPED_SUMMARY_FILE = "./scene_summary_reshaped.txt"


def get_target_video_info():
    home = os.path.expanduser("~")
    dataset_dir = os.path.join(home, "Documents/datasets")

    egohos_dir = os.path.join(dataset_dir, "egohos")
    files = glob.glob(f"{egohos_dir}/*/image/ego4d_*.jpg")

    fileinfo = [
        os.path.splitext(os.path.basename(p))[0].split("_")[1] 
        for p in files
    ]

    return fileinfo


def get_scenario_from_web():
    video_ids = get_target_video_info()
    video_ids = set(video_ids)

    url = "https://visualize.ego4d-data.org/"
    access_id = "AKIATEEVKTGZKZPI4QGL"

    options = Options()
    options.headless = True
    # options.add_experimental_option("detach", True)
    chrome = webdriver.Chrome(options=options)
    chrome.implicitly_wait(30)

    chrome.get(url)

    aws_id_input = chrome.find_element(By.CLASS_NAME, "bp3-input")
    aws_id_input.send_keys(access_id)
    login_button = chrome.find_element(By.CLASS_NAME, "bp3-button-text")
    login_button.click()

    for i, video in enumerate(tqdm(sorted(video_ids))):
        # last line num on txt file
        if i < 859:
            continue

        search_video_on_website(video)

    def search_video_on_website(video):
        search_box = chrome.find_element(By.CLASS_NAME, "searchbox-input")
        search_box.clear()
        search_box.send_keys(video)

        card = chrome.find_element(By.ID, f"item-{video}")
        card.click()

        senario = chrome.find_element(By.XPATH, '//*[@id="root"]/main/div/div/div[2]/div[1]/div/div/div[5]')
        senario.click()

        name = chrome.find_element(By.XPATH, '//*[@id="root"]/main/div/div/div[2]/div[1]/div/div/div[6]/div/span[2]').text

        with open(RAW_VIDEO_INFO_FILE, mode='a') as f:
            f.write(f"{video} {name}\n")

        header = chrome.find_element(By.CLASS_NAME, "header-logo")
        header.click()


def summarize_scene_info():
    video_list = get_target_video_info()

    with open(RAW_VIDEO_INFO_FILE, mode='r') as f:
        result = f.readlines()
    result_dict = {l.split(" ")[0]:l.strip()[40:] for l in result}

    video_list = [result_dict[video] for video in video_list]
    for senario, num in dict(collections.Counter(video_list)).items():
        with open(SUMMARY_FILE, mode="a") as f:
            f.write(f"{senario};{num}\n")


def find_word2vec():
    with open("./scene_summary_reshaped.txt", mode="r") as f:
        lines = f.readlines()

    lines = [" ".join(l.split(";")[0].split(" / ")[0].split("/")) for l in lines]
    reference = lines[5]

    path = "/Users/shionyamadate/Downloads/GoogleNews-vectors-negative300.bin.gz"
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    def average_sentence(sentence, model, num_features):
        words = sentence.split(" ")
        feature_vec = np.zeros((num_features,), dtype="float32")
        for word in words:
            feature_vec = np.add(feature_vec, model[word])
        if len(word) > 0:
            feature_vec = np.divide(feature_vec, len(words))
        return feature_vec

    def similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    base = average_sentence(reference, model, 300)
    for l in lines:
        v = average_sentence(l, model, 300)
        sim = similarity(base, v)
        with open("./wordvec.txt", mode="a") as f:
            f.write(f"{l},{sim}\n")


if __name__ == '__main__':
    find_word2vec()

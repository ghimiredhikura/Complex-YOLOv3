
# This script serves as downloading any file from google drive. 
# Deepak Ghimire, ghmdeepak@gmail.com 
# 2019/08/14

import requests

def download_file_from_google_drive(id, file_name):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, file_name):
        CHUNK_SIZE = 32768

        with open(file_name, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, file_name)    

# downlad pretrained complex-yolo-v3 weight files 
print("Downloading file yolov3_ckpt_epoch-298.pth ...")
download_file_from_google_drive("1e7PCqeV3tS68KtBIUbX34Uhy81XnsYZV", "yolov3_ckpt_epoch-298.pth")

print("Downloading file tiny-yolov3_ckpt_epoch-220.pth ...")
download_file_from_google_drive("19Qvpq2kQyjQ5uhQgi-wcWmSqFy4fcvny", "tiny-yolov3_ckpt_epoch-220.pth")

print("Completed!")
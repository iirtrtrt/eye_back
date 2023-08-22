from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import cv2
import dlib
import numpy as np
import torch
from flask_cors import CORS

from eye_open_close_cnn import EyeOpenCloseCNN

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

left_model = EyeOpenCloseCNN()
left_model.load_state_dict(torch.load("trained_model/left_500_new.pt"))
left_model.to(device)

right_model = EyeOpenCloseCNN()
right_model.load_state_dict(torch.load(f"trained_model/right_384_new.pt"))
right_model.to(device)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

eye_labels = ["Closed", "Open"]

result = {}

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def distance_between_points(a, b):
    x_diff = a.x - b.x
    y_diff = a.y - b.y
    return np.sqrt(x_diff * x_diff + y_diff * y_diff)


def detect_eyes(image, detector, predictor):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)

    if len(faces) > 0:
        landmarks = predictor(gray_image, faces[0])

        left_eyebrow_to_pupil = distance_between_points(
            landmarks.part(19), landmarks.part(37)
        )
        right_eyebrow_to_pupil = distance_between_points(
            landmarks.part(24), landmarks.part(44)
        )
        left_eye_width = distance_between_points(landmarks.part(36), landmarks.part(39))
        right_eye_width = distance_between_points(
            landmarks.part(42), landmarks.part(45)
        )

        left_max_side = max(2 * left_eyebrow_to_pupil, left_eye_width)
        right_max_side = max(2 * right_eyebrow_to_pupil, right_eye_width)

        left_center = landmarks.part(38)
        right_center = landmarks.part(45)

        left_eye = image[
            int(left_center.y - left_max_side // 2) : int(
                left_center.y + left_max_side // 2
            ),
            int(left_center.x - left_max_side // 2) : int(
                left_center.x + left_max_side // 2
            ),
        ]
        right_eye = image[
            int(right_center.y - right_max_side // 2) : int(
                right_center.y + right_max_side // 2
            ),
            int(right_center.x - right_max_side // 2) : int(
                right_center.x + right_max_side // 2
            ),
        ]

        return left_eye, right_eye
    else:
        return None, None


@app.route("/get_prediction_result", methods=["POST"])
def get_prediction_result():
    image_data = request.files["image"].read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    left_eye, right_eye = detect_eyes(image, detector, predictor)

    is_left = True

    if left_eye is not None and right_eye is not None:
        for eye_img in [left_eye, right_eye]:
            eye_img_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

            final_image = cv2.resize(eye_img_gray, (48, 48))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0

            dataa = torch.from_numpy(final_image)
            dataa = dataa.type(torch.FloatTensor)
            dataa = dataa.to(device)

            if is_left:
                outputs = left_model(dataa)
                predictions = torch.argmax(outputs, dim=1).item()
                is_open = predictions == 1

                result["left_eye"] = is_open
            else:
                outputs = right_model(dataa)
                predictions = torch.argmax(outputs, dim=1).item()
                is_open = predictions == 1

                result["right_eye"] = is_open

            is_left = False

        return jsonify({"success": True, "data": result})
    else:
        result["no_face"] = True

        return jsonify({"success": True, "data": result})


@app.route("/test", methods=["GET"])
def test_api():
    return jsonify({"message": "Server is working!"})


if __name__ == "__main__":
    app.run(debug=True)

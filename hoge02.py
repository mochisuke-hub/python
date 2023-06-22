import cv2
import dlib

# 顔検出器の初期化
detector = dlib.get_frontal_face_detector()

# 顔器官検出器の初期化
predictor = dlib.shape_predictor('C:\\Users\\user\\PycharmProjects\\pythonProject\\hoge20230622\\dlib-models-master\\shape_predictor_68_face_landmarks.dat')

# 画像の読み込み
image = cv2.imread('C:\\Users\\user\\PycharmProjects\\pythonProject\\hoge20230622\\face\\face01.jpg')

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔の検出
faces = detector(gray)

for face in faces:
    # 顔器官の検出
    landmarks = predictor(gray, face)

    # 鼻の位置
    nose = (landmarks.part(30).x, landmarks.part(30).y)
    cv2.circle(image, nose, 3, (0, 255, 0), -1)  # 鼻を円で描画

    # 左眉毛の位置
    left_eyebrow = (landmarks.part(21).x, landmarks.part(21).y)
    cv2.circle(image, left_eyebrow, 3, (0, 0, 255), -1)  # 左眉毛を円で描画

    # 右眉毛の位置
    right_eyebrow = (landmarks.part(22).x, landmarks.part(22).y)
    cv2.circle(image, right_eyebrow, 3, (0, 0, 255), -1)  # 右眉毛を円で描画

    # 上唇の位置
    upper_lip = (landmarks.part(51).x, landmarks.part(51).y)
    cv2.circle(image, upper_lip, 3, (255, 0, 0), -1)  # 上唇を円で描画

    # 下唇の位置
    lower_lip = (landmarks.part(57).x, landmarks.part(57).y)
    cv2.circle(image, lower_lip, 3, (255, 0, 0), -1)  # 下唇を円で描画

# 結果の表示
cv2.imshow('Facial Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

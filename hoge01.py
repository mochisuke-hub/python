import cv2

# 事前に学習された顔検出器を読み込む
#GitHubでのソース
face_cascade_path = 'C:\\Users\\user\\Desktop\\python\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# 画像ファイルやカメラから入力を取得する
image = cv2.imread('C:\\Users\\user\\Desktop\\python\\face\\fuji03.jpg')  # 画像ファイルの場合
# capture = cv2.VideoCapture(0)  # カメラからの入力の場合

# 入力をグレースケールに変換する
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔を検出する
#「scaleFactor」 ：スケールの縮小量を定義。(>1.0)の決まり。
# 　　           ：1.1の場合、顔検出アルゴリズムは元の画像から1.1倍、1.21倍、1.331倍、といったようにスケールを変化させながら顔領域を検出します。スケールファクターが小さいほど、より細かいスケールでの検出が行われ、詳細な特徴を持つ顔領域を見逃すリスクが低くなりますが、処理時間が増える傾向があります。
#「minNeighbors」：値が小さいほど、より多くの検出窓が顔領域として検出。その判定の厳しさの値。
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(30, 30))

# 検出された顔の周囲に矩形を描画する
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 結果を表示する
cv2.imshow('Face Detection', image)
cv2.waitKey(0)

# ウィンドウを閉じる
cv2.destroyAllWindows()
import cv2
import numpy as np
import streamlit as st


def order_points(pts):
  rect = np.zeros((4, 2), dtype="float32")
  s = pts.sum(axis=1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
  diff = np.diff(pts, axis=1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  return rect


def four_point_transform(image, pts):
  rect = order_points(pts)
  (tl, tr, br, bl) = rect
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
  dst = np.array(
      [
          [0, 0],
          [maxWidth - 1, 0],
          [maxWidth - 1, maxHeight - 1],
          [0, maxHeight - 1],
      ],
      dtype="float32",
  )
  M = cv2.getPerspectiveTransform(rect, dst)
  return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def enhance_old_photo(image):
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
  cl = clahe.apply(l)
  limg = cv2.merge((cl, a, b))
  enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
  hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  s = np.clip(cv2.multiply(s, 1.15), 0, 255).astype(np.uint8)
  return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


st.title("📸 Онлайн-сканер старых фотографий")
st.write(
    "Сделайте снимок старого фото на камеру телефона или загрузите файл для"
    " автоматического выравнивания и цветокоррекции."
)

input_method = st.radio("Выберите способ:", ["Сделать фото камерой", "Загрузить файл"])

file_to_process = None
if input_method == "Сделать фото камерой":
  file_to_process = st.camera_input("Нажмите для снимка")
else:
  file_to_process = st.file_uploader(
      "Выберите фото", type=["jpg", "jpeg", "png"]
  )

if file_to_process is not None:
  file_bytes = np.asarray(bytearray(file_to_process.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)

  st.image(
      cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
      caption="Исходное изображение",
      use_container_width=True,
  )

  # Обработка
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  edged = cv2.Canny(blurred, 75, 200)

  contours, _ = cv2.findContours(
      edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
  )
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

  screen_cnt = None
  for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4 and cv2.contourArea(c) > 5000:
      screen_cnt = approx
      break

  if screen_cnt is not None:
    processed = four_point_transform(image, screen_cnt.reshape(4, 2))
  else:
    processed = image

  final_image = enhance_old_photo(processed)

  st.image(
      cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB),
      caption="Результат (обрезка + цветокоррекция)",
      use_container_width=True,
  )

  success, encoded_image = cv2.imencode(".jpg", final_image)
  if success:
    st.download_button(
        label="📥 Скачать готовый JPEG",
        data=encoded_image.tobytes(),
        file_name="restored_photo.jpg",
        mime="image/jpeg",
    )
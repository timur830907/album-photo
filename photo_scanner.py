import os
import cv2
import numpy as np


def order_points(pts):
  """Сортировка углов прямоугольника: [верх-лево, верх-право, низ-право, низ-лево]"""
  rect = np.zeros((4, 2), dtype="float32")
  s = pts.sum(axis=1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]

  diff = np.diff(pts, axis=1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  return rect


def four_point_transform(image, pts):
  """Выполняет трансформацию перспективы (выпрямляет фото анфас)"""
  rect = order_points(pts)
  (tl, tr, br, bl) = rect

  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))

  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))

  dst = np.array(
      [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
      dtype="float32",
  )

  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  return warped


def main():
  # Создаем папку для сохранения сканов, если ее нет
  output_dir = "scanned_photos"
  os.makedirs(output_dir, exist_ok=True)

  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("Ошибка: Не удалось получить доступ к камере.")
    return

  img_counter = 0
  print("=" * 50)
  print("СКАНЕР СТАРЫХ ФОТОГРАФИЙ ЗАПУЩЕН")
  print("• Положите фотографию на контрастный стол.")
  print("• Зеленый контур покажет найденную фотографию.")
  print("• Нажмите ПРОБЕЛ (SPACE) для захвата и сохранения в JPG.")
  print("• Нажмите ESC для выхода.")
  print("=" * 50)

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
      if len(approx) == 4 and cv2.contourArea(c) > 10000:
        screen_cnt = approx
        break

    display_frame = frame.copy()
    if screen_cnt is not None:
      cv2.drawContours(display_frame, [screen_cnt], -1, (0, 255, 0), 3)
      cv2.putText(
          display_frame,
          "Photo Detected (Press SPACE)",
          (30, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 255, 0),
          2,
      )
    else:
      cv2.putText(
          display_frame,
          "Looking for photo...",
          (30, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 0, 255),
          2,
      )

    cv2.imshow("Scanner Frame", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
      break
    elif key == 32:
      if screen_cnt is not None:
        warped = four_point_transform(frame, screen_cnt.reshape(4, 2))
        filename = os.path.join(output_dir, f"photo_{img_counter}.jpg")
        cv2.imwrite(filename, warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"[УСПЕХ] Сохранено: {filename}")
        img_counter += 1
      else:
        print("[ВНИМАНИЕ] Фотография не найдена. Видны ли все 4 угла?")

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
import cv2
import numpy as np

# Загружаем изображение
img = cv2.imread('neutral.png')

# Конвертируем изображение в RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Конвертируем изображение в GRAYSCALE
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Создаем маску для белого фона
_, binary_mask = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)

# Преобразуем маску в трехканальное изображение
binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

# Используем битовую маску, чтобы удалить белый фон
img_rgb_no_bg = cv2.bitwise_and(img_rgb, binary_mask)

# Добавляем альфа-канал
alpha_channel = np.ones(binary_mask.shape, dtype=np.uint8) * 255
alpha_channel = cv2.cvtColor(alpha_channel, cv2.COLOR_BGR2GRAY)
_, alpha_channel = cv2.threshold(alpha_channel, 240, 255, cv2.THRESH_BINARY)
b, g, r = cv2.split(img_rgb_no_bg)
img_rgba = cv2.merge([b, g, r, alpha_channel])

# Сохраняем полученное изображение
cv2.imwrite('neutral1.png', img_rgba)

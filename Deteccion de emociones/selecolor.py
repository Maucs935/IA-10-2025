import cv2 as cv
import numpy as np

img = cv.imread(r"C:\Users\mauri\Downloads\figura.png", 1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Rangos para rojo
lower_red1 = np.array([0, 40, 40])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 40, 40])
upper_red2 = np.array([180, 255, 255])

# Rangos para verde
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Rangos para azul
lower_blue = np.array([100, 40, 40])
upper_blue = np.array([140, 255, 255])

# Rangos para amarillo
lower_yellow = np.array([20, 40, 40])
upper_yellow = np.array([35, 255, 255])

# Máscaras
mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2

mask_green = cv.inRange(hsv, lower_green, upper_green)
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)

# Función para dibujar el centro de cada figura
def draw_centers(mask, draw_color):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(img, (cx, cy), 5, draw_color, -1)

# Dibuja centros para cada color
draw_centers(mask_red, (0,0,255))      # Rojo
draw_centers(mask_green, (0,255,0))    # Verde
draw_centers(mask_blue, (255,0,0))     # Azul
draw_centers(mask_yellow, (0,255,255)) # Amarillo

res_red = cv.bitwise_and(img, img, mask=mask_red)
res_green = cv.bitwise_and(img, img, mask=mask_green)
res_blue = cv.bitwise_and(img, img, mask=mask_blue)
res_yellow = cv.bitwise_and(img, img, mask=mask_yellow)

cv.imshow('Original con centros', img)
cv.imshow('Rojo', res_red)
cv.imshow('Verde', res_green)
cv.imshow('Azul', res_blue)
cv.imshow('Amarillo', res_yellow)
cv.waitKey(0)
cv.destroyAllWindows()
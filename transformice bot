import cv2
import numpy as np
import pyautogui
import time
from PIL import ImageGrab

WINDOW_REGION = (0, 0, 2084, 950)
MATCH_THRESHOLD = 0.01 
MOVE_DELAY = 0
JUMP_HEIGHT_DIFF = 15
DEBUG = True

mouse_template_1 = cv2.imread(r"C:\Users\your_pc_name\Desktop\New folder (2)\player.png", 0)
mouse_template_2 = cv2.imread(r"C:\Users\your_pc_name\Desktop\New folder (2)\player2.png", 0)
cheese_template = cv2.imread(r"C:\Users\your_pc_name\Desktop\New folder (2)\cheese.png", 0)

def capture_screen():
    img = ImageGrab.grab(bbox=WINDOW_REGION)
    frame_rgb = np.array(img)
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    return frame_gray, frame_rgb

def match_template(frame_gray, template, label):
    result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val >= MATCH_THRESHOLD:
        h, w = template.shape
        center = (max_loc[0] + w // 2, max_loc[1] + h // 2)
        if DEBUG:
            print(f"[+] Found {label} at {center} (confidence: {max_val:.2f})")
        return center
    else:
        if DEBUG:
            print(f"[-] {label} not found.")
        return None

def move_left():
    pyautogui.keyDown('left')
    time.sleep(MOVE_DELAY)
    pyautogui.keyUp('left')

def move_right():
    pyautogui.keyDown('right')
    time.sleep(MOVE_DELAY)
    pyautogui.keyUp('right')

def jump():
    pyautogui.press('up')

def move_toward(mouse, cheese):
    dx = cheese[0] - mouse[0]
    dy = cheese[1] - mouse[1]

    if abs(dx) > 10:
        if dx > 0:
            move_right()
        else:
            move_left()

    if dy < -JUMP_HEIGHT_DIFF:
        jump()

def draw_debug(frame, mouse, cheese):
    if mouse:
        cv2.circle(frame, mouse, 10, (0, 255, 0), 2)
    if cheese:
        cv2.circle(frame, cheese, 10, (0, 255, 255), 2)
    if mouse and cheese:
        cv2.line(frame, mouse, cheese, (0, 255, 255), 2)
    cv2.imshow("Debug View", frame)
    cv2.waitKey(1) 

def main():
    print("[*] Starting Transformice bot...")
    while True:
        gray, rgb = capture_screen()

        mouse_pos = match_template(gray, mouse_template_1, "Mouse (player.png)")

        if not mouse_pos:
            mouse_pos = match_template(gray, mouse_template_2, "Mouse (player2.png)")

        cheese_pos = match_template(gray, cheese_template, "Cheese")

        if mouse_pos and cheese_pos:
            move_toward(mouse_pos, cheese_pos)

        if DEBUG:
            draw_debug(rgb, mouse_pos, cheese_pos)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[*] Bot stopped by user.")
        cv2.destroyAllWindows()

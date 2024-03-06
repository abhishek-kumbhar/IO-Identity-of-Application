import pyautogui as p
import time
import random

lst = ['I Love You PILU 😍😘❤️ ', 'Chutki I Love You !!! ', '🥰❤️😍 ', 'Baby 🤩 ', 'Mummy 🫣🤭 ']
# lst = ['A', 'B', 'C', 'D', 'E']

time.sleep(5)

for i in range(10):
    n = random.randint(0, 4)
    p.typewrite(lst[n])
    p.press('enter')
import cv2
import sys
import tty
import termios

def wait_for_keypress(target_key, exit_key, yes_to_all_key):
    print(f"Press '{target_key}' to continue or '{exit_key}' to exit... ('{yes_to_all_key}') if YES to all...")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    yes_to_all_result = False
    
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            char = sys.stdin.read(1)
            if char == target_key:
                break
            elif char == yes_to_all_key:
                print("Yes to all...")
                yes_to_all_result = True
                break
            elif char == exit_key:
                print("Exiting...")
                sys.exit(0)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return yes_to_all_result

class WindowMessage:
    def __init__(self, txt1=None, col1=None, pos1=None, txt2=None, col2=None, pos2=None, txt3=None, col3=None, pos3=None):

        # Initialize messages with default or provided values
        self.msg = []

        texts = [txt1, txt2, txt3]
        colors = [col1, col2, col3]
        positions = [pos1, pos2, pos3]

        for txt, col, pos in zip(texts, colors, positions):
            self.msg.append({
                'text': txt if txt is not None else '',
                'color': col if col is not None else (255, 0, 0),
                'position': pos if pos is not None else (0, 0)
            })

    def ShowWindowMessages(self, image):
        for i in range(3):
            cv2.putText(
                image, 
                self.msg[i]['text'],
                org=self.msg[i]['position'],
                fontFace=2, fontScale=0.75, color=self.msg[i]['color'])
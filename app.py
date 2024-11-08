from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
from keras.models import model_from_json
import operator
import time
import sys
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
from string import ascii_uppercase

class Application:
    def __init__(self):
        self.directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        self.spell = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        # Load the main model
        try:
            # Ensure model directory exists
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
                raise FileNotFoundError(f"Model directory created at: {self.directory}\nPlease place model files there.")
            
            # Define model file paths
            model_json_path = os.path.join(self.directory, "model-bw.json")
            weights_path = os.path.join(self.directory, "model-bw.weights.h5")
            
            # Verify files exist
            if not os.path.exists(model_json_path):
                raise FileNotFoundError(f"Model architecture file not found at: {model_json_path}")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights file not found at: {weights_path}")
            
            # Load model architecture
            with open(model_json_path, "r") as json_file:
                model_json = json_file.read()
                if not model_json.strip():
                    raise ValueError("Model architecture file is empty")
                self.loaded_model = model_from_json(model_json)
            
            # Load weights
            self.loaded_model.load_weights(weights_path)
            print("Successfully loaded main model")
            
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            print("\nMake sure you have the following files in your 'model' directory:")
            print("1. model-bw.json - The model architecture file")
            print("2. model-bw.weights.h5 - The model weights file")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error loading model: {str(e)}")
            sys.exit(1)

        # Initialize character counters
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x1100")
        
        # Create GUI elements
        self.panel = tk.Label(self.root)
        self.panel.place(x=135, y=10, width=640, height=640)
        
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=460, y=95, width=310, height=310)
        
        self.T = tk.Label(self.root)
        self.T.place(x=31, y=17)
        self.T.config(text="Sign Language to Text", font=("courier", 40, "bold"))
        
        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=640)
        
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=640)
        self.T1.config(text="Character :", font=("Courier", 40, "bold"))
        
        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=700)
        
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=700)
        self.T2.config(text="Word :", font=("Courier", 40, "bold"))
        
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=760)
        
        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=760)
        self.T3.config(text="Sentence :", font=("Courier", 40, "bold"))
        
        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=820)
        self.T4.config(text="Suggestions", fg="red", font=("Courier", 40, "bold"))
        
        # Create buttons
        self.btcall = tk.Button(self.root, command=self.action_call, height=0, width=0)
        self.btcall.config(text="About", font=("Courier", 14))
        self.btcall.place(x=825, y=0)
        
        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=890)
        
        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=890)
        
        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=890)
        
        self.bt4 = tk.Button(self.root, command=self.action4, height=0, width=0)
        self.bt4.place(x=125, y=950)
        
        self.bt5 = tk.Button(self.root, command=self.action5, height=0, width=0)
        self.bt5.place(x=425, y=950)
        
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()
        
    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if ok:
            # Flip and process main video frame
                cv2image = cv2.flip(frame, 1)
                x1 = int(0.5 * frame.shape[1])
                y1 = 10
                x2 = frame.shape[1] - 10
                y2 = int(0.5 * frame.shape[1])
                cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            
            # Convert and display main video frame
                cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
                self.current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=self.current_image)
                self.panel.imgtk = imgtk
                self.panel.config(image=imgtk)
            
            # Process ROI
                cv2image = cv2image[y1:y2, x1:x2]
                gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 2)
                th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Make prediction and update ROI display
                self.predict(res)
                self.current_image2 = Image.fromarray(res)
                imgtk = ImageTk.PhotoImage(image=self.current_image2)
                self.panel2.imgtk = imgtk
                self.panel2.config(image=imgtk)
            
            # Update text displays
                self.panel3.config(text=self.current_symbol, font=("Courier", 50))
                self.panel4.config(text=self.word, font=("Courier", 40))
                self.panel5.config(text=self.str, font=("Courier", 40))
            
            # Handle spell checking
            try:
                if self.word and len(self.word.strip()) > 0:
                    predicts = list(self.spell.candidates(self.word))
                else:
                    predicts = []
            except Exception as e:
                print(f"Spell check error: {str(e)}")
                predicts = []
            
            # Update suggestion buttons
            button_configs = [
                (self.bt1, 0), (self.bt2, 1), 
                (self.bt3, 2), (self.bt4, 3), 
                (self.bt5, 4)
            ]
            
            for button, index in button_configs:
                if len(predicts) > index:
                    button.config(text=predicts[index], font=("Courier", 20))
                else:
                    button.config(text="")
                    
        except Exception as e:
            print(f"Error in video loop: {str(e)}")
        
        finally:
        # Schedule next frame capture
            self.root.after(30, self.video_loop)

    # Rest of your methods remain the same
    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        
        prediction = {}
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        
        if (self.current_symbol == 'blank'):
            for i in ascii_uppercase:
                self.ct[i] = 0
        
        self.ct[self.current_symbol] += 1
        if (self.ct[self.current_symbol] > 60):
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    return
            
            self.ct['blank'] = 0
            if (self.current_symbol == 'blank'):
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if (len(self.word) == 0):
                    self.blank_flag = 0
                self.word += self.current_symbol

    # Include your action methods here
    def action1(self):
        predicts = list(self.spell.candidates(self.word))
        if len(predicts) > 0:
            self.word = predicts[0]

    def action2(self):
        predicts = list(self.spell.candidates(self.word))
        if len(predicts) > 1:
            self.word = predicts[1]

    def action3(self):
        predicts = list(self.spell.candidates(self.word))
        if len(predicts) > 2:
            self.word = predicts[2]

    def action4(self):
        predicts = list(self.spell.candidates(self.word))
        if len(predicts) > 3:
            self.word = predicts[3]

    def action5(self):
        predicts = list(self.spell.candidates(self.word))
        if len(predicts) > 4:
            self.word = predicts[4]

    def action_call(self):
        pass  # Implement if needed

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

print("Starting Application...")
pba = Application()
pba.root.mainloop()
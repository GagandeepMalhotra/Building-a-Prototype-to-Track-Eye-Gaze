import eye_tracker
import numpy as np
import tkinter as tk
import cv2
import time
from PIL import Image, ImageTk
import pyautogui
pyautogui.FAILSAFE = False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import keyboard

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.geometry("1000x750")
        #self.master.resizable(False, False)
        self.master.title("Eye Gaze Program")
        self.pack()
        self.create_widgets()
        frame, self.look_vector, left_eye_region, right_eye_region, reRatio, leRatio = eye_tracker.main([0,0,0], 1)

        self.screen_width, self.screen_height = pyautogui.size()
        self.look_vector_list = []
        self.window_coordinate_list = []
        self.gaze_history = []
        self.spacing = 10
        self.num_windows = 0 
        self.start = None
        self.move_mouse = False
        keyboard.add_hotkey('shift+f5', self.disable_mouse)
        self.show_frame()

    def create_widgets(self):
        self.title_label = tk.Label(self, text="Eye Gaze Predictor", font=("Helvetica", 24, "bold"), pady=20)
        self.title_label.pack(pady=0)

        #create a label for the video feed
        self.video_label = tk.Label(self, bg="black", highlightthickness=1, highlightbackground="black")
        self.video_label.pack(pady=0)

        video_labels_frame = tk.Frame(self)
        video_labels_frame.pack(side='top', padx=0, pady=5)

        self.left_eye_video_label = tk.Label(video_labels_frame, bg="black", highlightthickness=1, highlightbackground="black")
        self.left_eye_video_label.pack(side='left', padx=(0, 5), pady=0)

        self.right_eye_video_label = tk.Label(video_labels_frame, bg="black", highlightthickness=1, highlightbackground="black")
        self.right_eye_video_label.pack(side='left', padx=(5, 0), pady=0)

        container_frame = tk.Frame(self)
        container_frame.pack(side=tk.TOP, padx=5, pady=5)

        plot_checkbox_frame = tk.Frame(container_frame, bd=1, relief=tk.SOLID)
        #Add the checkboxes to the Frame
        self.selected_display = tk.IntVar(value=1)
        self.plot_eyes_radiobutton = tk.Radiobutton(plot_checkbox_frame, text="Eyes Gaze", font=("Helvetica", 10), bd=0, variable=self.selected_display, value=1)
        self.plot_eyes_radiobutton.pack(side=tk.LEFT, pady=5, padx=5)

        self.plot_face_radiobutton = tk.Radiobutton(plot_checkbox_frame, text="Face Gaze", font=("Helvetica", 10), bd=0, variable=self.selected_display, value=2)
        self.plot_face_radiobutton.pack(side=tk.LEFT, pady=5, padx=5)

        self.plot_both_radiobutton = tk.Radiobutton(plot_checkbox_frame, text="Display Both", font=("Helvetica", 10), bd=0, variable=self.selected_display, value=3)
        self.plot_both_radiobutton.pack(side=tk.LEFT, pady=5, padx=5)

        #Pack the Frame with checkboxes and set a border
        plot_checkbox_frame.pack(side=tk.LEFT, padx=5, pady=5)
        plot_checkbox_frame.configure(borderwidth=2, relief=tk.SOLID)

        info_checkbox_frame = tk.Frame(container_frame, bd=1, relief=tk.SOLID)
        #Create checkboxes to toggle gaze visualization
        self.look_vector_checked = tk.BooleanVar(value=False)
        self.look_vector_checkbox = tk.Checkbutton(info_checkbox_frame, text="Print Current Look Vector [x, y, z]", font=("Helvetica", 10), bd=3, variable=self.look_vector_checked)
        self.look_vector_checkbox.pack(side=tk.LEFT, pady=0)

        self.mouse_coord_checked = tk.BooleanVar(value=False)
        self.mouse_coord_checkbox = tk.Checkbutton(info_checkbox_frame, text="Print Predicted Mouse Coordinate [x, y]", font=("Helvetica", 10), bd=3, variable=self.mouse_coord_checked)
        self.mouse_coord_checkbox.pack(side=tk.LEFT, pady=0, padx=10)

        info_checkbox_frame.pack(side=tk.LEFT, padx=5, pady=5)
        info_checkbox_frame.configure(borderwidth=2, relief=tk.SOLID)

        #Create a frame for the Instructions and Information widgets
        frame1 = tk.Frame(self)
        frame1.pack(side=tk.TOP)

        #Create a frame for the instructions label and text widget
        instructions_frame = tk.Frame(frame1)
        instructions_frame.pack(side=tk.LEFT)

        #Create a label widget to display the title "Instructions"
        instructions_lbl = tk.Label(instructions_frame, text="Instructions:", font=("Helvetica", 14, "bold"))
        instructions_lbl.pack(side=tk.TOP, pady=10)

        #Create a text widget to display instructions
        self.instructions = tk.Text(instructions_frame, font=("Helvetica", 10, "bold"), height=10, width=45, state="disabled", highlightthickness=1, highlightbackground="black", wrap="word")
        self.instructions.pack(side=tk.TOP, padx=10, pady=0)
        self.print_to_text(self.instructions, "Welcome!\nPress Calibrate then Start to begin the program:\n\nFor best results, keep your head in clear view; remain still while calibrating.", "")

        #Create a label widget to display the title "Information"
        information_lbl = tk.Label(frame1, text="Information:", font=("Helvetica", 14, "bold"))
        information_lbl.pack(side=tk.TOP, pady=10)

        #Create a text widget to display information
        self.information = tk.Text(frame1, font=("Consolas", 10), height=10, width=90, state="disabled", highlightthickness=1, highlightbackground="black")
        self.information.pack(side=tk.LEFT, padx=10)

        #Create a frame for the buttons
        frame2 = tk.Frame(self)
        frame2.pack(side=tk.TOP, pady=10)

        self.calibrate_button = tk.Button(frame2, text="Calibrate", font=("Helvetica", 14), command=self.display_squares, width=20, height=2, bd=3, disabledforeground="#BDBDBD", relief="raised")
        self.calibrate_button.pack(side=tk.LEFT, padx=10, pady=10)

        #Create a button for calibration
        self.ex_calibrate_button = tk.Button(frame2, text="Calibrate (Experimental)", font=("Helvetica", 14), command=self.set_window_automatic, width=20, height=2, bd=3, disabledforeground="#BDBDBD", relief="raised")
        self.ex_calibrate_button.pack(side=tk.LEFT, padx=10, pady=10)

        #Create a button for starting the program
        self.start_button = tk.Button(frame2, text="Start Controlling Mouse", font=("Helvetica", 14), command=self.start_mouse, width=20, height=2, bd=3, disabledforeground="#BDBDBD", relief="raised", state="disabled")
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

    def disable_mouse(self):
        if self.move_mouse == True:
            self.move_mouse = False
            self.start_button.config(text="Start", state="normal")
            self.show_message('Mouse disabled')

    def show_message(self, text):
        #Create a new window
        message_window = tk.Toplevel()
        
        #Remove window title and frame
        message_window.overrideredirect(True)
        
        #Add message to the window
        message = tk.Label(message_window, text=text, font=('Helvetica', 16, 'bold'))
        message.pack()
        
        #Resize window to fit the text
        message_window.geometry('{}x{}+0+0'.format(message.winfo_reqwidth() + 10, message.winfo_reqheight() + 10))
        message_window.attributes("-topmost", True)  #makes the window stay on top of all other windows
        message_window.attributes("-toolwindow", True)  #removes the window from the taskbar
        message_window.lift()

        self.print_to_text(self.instructions, text, "")
        
        #Destroy window after 5 seconds
        message_window.after(5000, message_window.destroy)
        

    def display_squares(self):
        self.look_vector_list = []
        self.window_coordinate_list = []

        #self.set_windows()
        self.set_windows()

        self.num_windows = 25
        self.calibrate_button.config(text="Calibrating...", state="disabled")
        self.ex_calibrate_button.config(text="Calibrating...", state="disabled") #disable Calibrate button and change text
        self.start_button.config(text="Calibrating...", state="disabled")

    def set_windows(self):
        num_windows = 25
        rows = 5  #number of rows
        cols = 5  #number of columns

        x_start = 0  #starting x-coordinate for first column
        y_start = 0  #starting y-coordinate for first row

        x_step = (self.screen_width - self.spacing * (cols - 1)) / cols  #distance between columns
        y_step = (self.screen_height - self.spacing * (rows - 1)) / rows  #distance between rows

        for i in range(num_windows):
            row = int(i / cols)  #calculate which row the window should be in
            col = i % cols  #calculate which column the window should be in

            x1 = x_start + (col * (x_step + self.spacing))
            y1 = y_start + (row * (y_step + self.spacing))
            x2 = x1 + x_step
            y2 = y1 + y_step

            self.create_square_window("Window {}".format(i+1), x1, y1, x2, y2)
        self.show_message("Keep your head still\nLook at the RED CIRCLE in a green square\nPoint and click at that green square\nRepeat until complete")

    def create_square_window(self, name, x1, y1, x2, y2):
        #Adjust window position if it exceeds maximum coordinates
        if x2 > self.screen_width:
            x1 -= (x2 - self.screen_width)
            x2 = self.screen_width
        if y2 > self.screen_height:
            y1 -= (y2 - self.screen_height)
            y2 = self.screen_height

        window = tk.Toplevel()
        window.title(name)
        window.overrideredirect(True)  #removes the window decorations (i.e. frame, title bar)
        window.geometry("{}x{}+{}+{}".format(int(x2 - x1), int(y2 - y1), int(x1), int(y1)))
        window.attributes("-topmost", True)  #makes the window stay on top of all other windows
        window.attributes("-toolwindow", True)  #removes the window from the taskbar
        window.lift()  #brings the window to the front
        square = tk.Canvas(window, width=int(x2 - x1), height=int(y2 - y1), bg="lime", highlightthickness=0)
        square.pack(fill="both", expand=True)
        # calculate the center of the canvas
        center_x, center_y = (x2 - x1) / 2, (y2 - y1) / 2

        # draw a red circle in the center of the canvas
        radius = 20
        square.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, fill="red")
        #window.after(1000, self.on_square_destroyed(name, window))
        square.bind("<Button-1>", lambda event, lv=self.look_vector: self.on_square_destroyed(name, window))  #destroys the window on left mouse click and prints the mouse position
        return window
    
    def start_mouse(self):
        self.ex_calibrate_button.config(text="Calibrate (Experimental)", state="normal")
        self.calibrate_button.config(text="Calibrate", state="normal")  #enable Calibrate button and change text
        self.start_button.config(text="Running...", state="disabled")
        self.show_message("Shift+F5 to STOP controlling cursor\nLeft Eye Blink to Left-Click\nRight Eye Blink to Right-Click")
        self.move_mouse = True


    def print_to_text(self, textbox, text, value):
        textbox.configure(state="normal")  #enable editing
        #Get the current number of lines in the Text widget
        num_lines = int(textbox.index('end-1c').split('.')[0])
        
        #If the number of lines is greater than or equal to 100, delete the first line
        while num_lines > 50:
            textbox.delete('1.0', '2.0')
            num_lines = int(textbox.index('end-1c').split('.')[0])
        textbox.insert(tk.END, str(text) + str(value) + '\n\n')
        textbox.yview(tk.END)
        textbox.configure(state="disabled")  #disable editing

    def show_frame(self):
        old_look_vector = self.look_vector
        frame, self.look_vector, left_crop_eye_region, right_crop_eye_region, reRatio, leRatio = eye_tracker.main(self.look_vector, self.selected_display.get())
        if self.move_mouse == True:
            if old_look_vector != self.look_vector:
                if reRatio > 6.5 and leRatio < 6.5:
                    pyautogui.rightClick()
                    self.show_message("Right Click")
                    #print("RIGHT EYE BLINKING")
                elif leRatio > 6.5 and reRatio < 6.5:
                    pyautogui.leftClick()
                    self.show_message("Left Click")
                    #print("LEFT EYE BLINKING")
                
                prediction = self.predict_screen_point()
                self.move_mouse_to_point(prediction)
                if self.mouse_coord_checked.get() == True:
                    self.print_to_text(self.information, "Predicted Mouse Coord: ", prediction)

        if self.look_vector_checked.get() == True and old_look_vector != self.look_vector:
            self.print_to_text(self.information, "Predicted look Vector:", self.look_vector)
        
        #convert the OpenCV frame to a PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))      
        #convert the PIL image to a tkinter PhotoImage object
        frame_photo = ImageTk.PhotoImage(image)
        #update the video label with the new photo
        self.video_label.configure(image=frame_photo)
        self.video_label.image = frame_photo  #avoid garbage collection

        if left_crop_eye_region is not None and right_crop_eye_region is not None:
            image = Image.fromarray(cv2.cvtColor(left_crop_eye_region, cv2.COLOR_BGR2RGB))
            left_eye_region_photo = ImageTk.PhotoImage(image)
            self.left_eye_video_label.configure(image=left_eye_region_photo)
            self.left_eye_video_label.image = left_eye_region_photo

            image = Image.fromarray(cv2.cvtColor(right_crop_eye_region, cv2.COLOR_BGR2RGB))
            right_eye_region_photo = ImageTk.PhotoImage(image)
            self.right_eye_video_label.configure(image=right_eye_region_photo)
            self.right_eye_video_label.image = right_eye_region_photo

        #call this method again after 10 milliseconds to show the next frame
        self.after(33, self.show_frame)
        
    def set_window_automatic(self):
        self.look_vector_list = []
        self.window_coordinate_list = []
        self.ex_calibrate_button.config(text="Calibrating...", state="disabled") #disable Calibrate button and change text
        self.calibrate_button.config(text="Calibrating...", state="disabled")
        self.start_button.config(text="Calibrating...", state="disabled")
        
        #Create the window
        window = tk.Tk()
        window.overrideredirect(True)
        window.configure(bg='lime')
        window.geometry("50x50")

        #Get the screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        window.geometry("+{}+{}".format(1, 0))
        window.wm_attributes("-topmost", True)

        self.show_message("Keep your head still\n       Remain looking at the green square as it moves\nClick on the square to begin")

        def on_window_click(event):
            x_direction = 50
            #Start moving the window
            move_window(x_direction)
            
            #Unbind the left mouse button click event to prevent further clicks
            window.unbind("<Button-1>")
            
        window.bind("<Button-1>", on_window_click)
            

        def move_window(x_direction):
            calibration_complete = False

            x, y = window.winfo_x(), window.winfo_y()
            window_width, window_height = window.winfo_width(), window.winfo_height()

            if x + window_width >= screen_width:
                x = screen_width - window_width
                y += window_height

                if (y + window_height >= screen_height and x == screen_width - window_width) or (y + window_height >= screen_height and x == screen_width + window_width):
                    calibration_complete = True
                    window.destroy()
                    self.define_model()
                    self.start_button.config(text="Start", state="normal")

                x_direction = -50  #Flip the x direction
                
            elif x <= 0:
                x_direction = 50  #Flip the x direction
                y += window_height

            x += 1 * x_direction
            
            if calibration_complete == False:
                self.window_coordinate_list.append((x+25,y+25))
                window.geometry("+{}+{}".format(x, y))
                self.look_vector_list.append(self.look_vector)
                
                #Calculate the time interval since the last call to move_window
                time_interval = time.perf_counter() - move_window.last_call_time if hasattr(move_window, 'last_call_time') else 0
                
                #Adjust the time interval to maintain a consistent speed
                adjusted_time_interval = max(0.01, 0.01 - time_interval)
                
                #Schedule the next call to move_window with the adjusted time interval
                window.after(int(adjusted_time_interval * 1000), move_window, x_direction)
                
                #Update the last_call_time attribute
                move_window.last_call_time = time.perf_counter()

    def on_square_destroyed(self, name, window):
        #Callback function called when a square window is destroyed.
        width = window.winfo_width()
        height = window.winfo_height()
        x, y = map(int, window.geometry().split("+")[1:])

        #Compute the center position of the window
        center_x = x + width/2
        center_y = y + height/2
        window.destroy()
        self.print_to_text(self.information, f"{name} square at {center_x,center_y} destroyed with look vector: ", self.look_vector)

        self.look_vector_list.append(self.look_vector)
        self.window_coordinate_list.append((x,y))

        self.num_windows -= 1  #decrement the counter
        if self.num_windows == 0:
            self.calibrate_button.config(text="Calibrate", state="normal")  #enable Calibrate button and change text
            self.ex_calibrate_button.config(text="Calibrate (Experimental)", state="normal")
            self.start_button.config(text="Start", state="normal")
            self.define_model()
            #self.create_plot()

    def define_model(self):
        #self.create_plot()
        self.look_vector_list = np.array(self.look_vector_list)
        self.window_coordinate_list = np.array(self.window_coordinate_list)
        self.screen_model = self.create_screen_model()
        self.ex_calibrate_button.config(text="Calibrate (Experimental)", state="normal")
        self.calibrate_button.config(text="Calibrate", state="normal")  #enable Calibrate button and change text
        self.start_button.config(text="Start", state="normal")

    def move_mouse_to_point(self, point):
        pyautogui.moveTo(point[0][0], point[0][1], tween=pyautogui.easeInOutQuad)

    def predict_screen_point(self):
        #Make predictions
        look_vector_np = np.array(self.look_vector)
        look_vector_np = look_vector_np.reshape(1, -1)
        predictions = self.screen_model.predict(look_vector_np)
        #print('Predicted screen coordinates:', predictions)
        return predictions
    
    def create_screen_model(self):
        #Split the data
        X_train, X_test, y_train, y_test = train_test_split(self.look_vector_list, self.window_coordinate_list, test_size=0.2)

        #Define the model
        screen_model = LinearRegression()

        #Train the model
        screen_model.fit(X_train, y_train)

        #Evaluate the model
        score = screen_model.score(X_test, y_test)
        self.print_to_text(self.information, 'R-squared score:', score)
        if score < 0.7:
            self.show_message("Recommended R-squared score is above 0.7\nYour R-squared score is: {}\nPlease calibrate again keeping your head still\nOnly move your eyes to the square".format(score))
        else:
            self.show_message("Recommended R-squared score is above 0.7\nYour R-squared score is: {}\nPress Start to begin".format(score))
        return screen_model

root = tk.Tk()
app = Application(master=root)
root.iconbitmap("logo.ico")
app.mainloop()

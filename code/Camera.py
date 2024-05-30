#-------- Import Statements ------------#
import tkinter as tk  # standard GUI kit
from tkinter import messagebox  # to display message boxes
from tkinter import filedialog  # to provide window to select the file
from PIL import Image, ImageTk  # for image processing and displaying in Tkinter
import cv2  # for computer vision tasks, such as reading video files and processing frames
import cvzone  # opencv related task
import math
from deep_sort_realtime.deepsort_tracker import DeepSort # contains deepsort tracker for object tracking
from ultralytics import YOLO  # to use YOLO model for object detection
import threading # to run multiple threads and to control detection running


# ------ Registration Class ------------ #
class RegisterWindow:
    def __init__(self, root, login_window):   # define initializing method for the class
        self.root = root  # assigns the root window to the instance variable self.root
        self.root.title("Register")  # to assign the title to the window
        self.root.configure(bg='#ADD8E6')  # configure background color to Light blue
        screen_width = root.winfo_screenwidth()  # obtain width of the window
        screen_height = root.winfo_screenheight()  # obtain height of the window
        self.root.geometry(f"{screen_width}x{screen_height}")  # Set the window size

        self.login_window = login_window  # assign login_window arguments to the instance variable login_window

        self.username_label = tk.Label(root, text="Username:")  # creates a label widgets for dispalying on window
        self.username_label.pack()  # pack the username_label widget to display it in the root window
        self.username_entry = tk.Entry(root) # creates an entry widget to take input from user
        self.username_entry.pack() # pack the entry widgets to display it on window

        self.password_label = tk.Label(root, text="Password:")
        self.password_label.pack()
        self.password_entry = tk.Entry(root, show="*")
        self.password_entry.pack()

        self.register_button = tk.Button(root, text="Register", command=self.register)    # create a button widgets with text Register that calls register method when clicked
        self.register_button.pack()   # pack the register button to display it on window

    # define register method
    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        # store user data in file
        with open("user_data.txt", "a") as file:
            file.write(f"{username}:{password}\n")

        messagebox.showinfo("Success", "User registered successfully")
        self.root.destroy()   # close the registration window
        self.login_window.root.deiconify()  # makes login window visible after registration a new user


# ----------- Login Class ------------ #
class LoginWindow:
    def __init__(self, root):  # define initializing method for the class
        self.root = root
        self.root.title("Login")
        self.root.configure(bg='#ADD8E6')  # Light blue background color
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")  # Set the window size

        self.username_label = tk.Label(root, text="Username:")
        self.username_label.pack()
        self.username_entry = tk.Entry(root)
        self.username_entry.pack()

        self.password_label = tk.Label(root, text="Password:")
        self.password_label.pack()
        self.password_entry = tk.Entry(root, show="*")
        self.password_entry.pack()

        self.login_button = tk.Button(root, text="Login", command=self.login)
        self.login_button.pack()

        self.register_button = tk.Button(root, text="Register", command=self.register_window)
        self.register_button.pack()


    # define login method
    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()


        # open file and read user login information and cjeck the correct login details
        with open("user_data.txt", "r") as file:
            for line in file:
                stored_username, stored_password = line.strip().split(":")  # strip new line character from the line and splits it into username and password
                if username == stored_username and password == stored_password:
                    self.root.destroy()  # close the login window if login is successful
                    main_window = tk.Tk()  # create a new window for Tkinter main application
                    app = ObjectDetectionGUI(main_window)   # creates an instance of the ObjectDetectionGUI class and pass created new window as argument
                    main_window.mainloop()  # start main event loop for the main application window
                    break
            else:
                messagebox.showerror("Error", "Invalid username or password")


    # def register_window method so that user can open registration window from login window for new user
    def register_window(self):
        self.root.withdraw()  # hides the login window
        register_window = tk.Tk()  # creates a new window for registration window
        register_app = RegisterWindow(register_window, self)  # create an instance of the RegisterWindow and passing registration window and self(current login window) as argument
        register_window.mainloop()  # start the main event loop for the registration window



# ---------- Class for object detction and tracking -------------#
class ObjectDetectionGUI:
    def __init__(self, root):  # initialize the object detection gui
        self.root = root
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.title("Object Detection and Tracking GUI")

        self.video_path = ""   # initialize the video_path variable to an empty string to stored the path of selected video
        self.detection_started = False  # initializes detection_started to false (used to identify whether the object detection and tracking currently running)
        self.stop_event = threading.Event()   # initialize the threading event object named stop_event (used as a signal the detection loop ton stop)

        self.model = YOLO('../weights/yolov8n.pt')   # initializes the model variable with YOLO model

        self.create_widgets()  # calls the create_widgets() method which is define below


    # define create_widgets() method to create and display GUI
    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=800, height=600)   # create a canvas widget to display the video frames
        self.canvas.pack()  # pack the camvas widgets to display it in root window

        select_video_button = tk.Button(self.root, text="Select Video",
                                        command=self.select_video)  # create a button widget with the text Select Video that calls select_video() method
        select_video_button.pack()  # pack the select_video_button to display it on the root window

        self.start_detection_button = tk.Button(self.root, text="Start Detection and Tracking",
                                                command=self.start_detection)
        self.start_detection_button.pack()

        self.start_detection_camera = tk.Button(self.root, text="Open Camera and start detction",
                                                command=self.start_detection_camera)
        self.start_detection_camera.pack()

        self.stop_detection_button = tk.Button(self.root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_detection_button.pack()

    def select_video(self):
        # opens a file dialog to select video file and assigns the path to the video_path variable
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    def start_detection(self):
        if not self.video_path:
            return

        self.stop_detection_button.config(state=tk.NORMAL)   # enables to stop button
        self.detection_started = True  # sets the detection_started to true to indicates that detection is running
        self.stop_event.clear()    # clear the stop_event flag to signal that detection is rinning

        #----- Initializes the tracker object for object tracking -----#
        tracker = DeepSort(max_age=5,   # maximum number of frames track can be kept without detection
                           n_init=2,   # initaialze track after 2 frames after confirmation
                           max_iou_distance = 0.7,  # it controls the maximum IoU
                           nms_max_overlap=1.0,  # it controls the NMS
                           max_cosine_distance=0.3, # helps in associating detection (max cosine distance between apperance of two track)
                           embedder="mobilenet",  # type of feature extractor to use for the apperance of the feature
                           half=True,  # it used to minimise the load on memory and GPU
                           bgr=True,  # to convert i/p in Blue Green Red format
                           embedder_gpu=True,  # to use GPU
                           polygon=False,  # which bounded box is used i.e. polygon or rectangle
                           )

        cap = cv2.VideoCapture(self.video_path)  # opens the selected video file for reading
        assert cap.isOpened()   # checks the VideoCapture object cap is successfully open

        while self.detection_started:
            ret, frame = cap.read()  # reads a frame and reads() method return two values
            if not ret:
                break

            results = self.model(frame, stream=True)   # performs the object detection using YOLO model
            detections = []  # initializes a list to store the detected object

            for r in results:
                boxes = r.boxes  # gets the bounded box of the detected object
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]  # gets the coordinates of the bounded box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2-x1, y2-y1

                    cls = int(box.cls[0])  # gets the class label for detected objects
                    current_class = self.model.model.names[cls]  # access the list of class name that YOLO model has been trained to detect

                    conf = math.ceil((box.conf[0] * 100)) / 100  # calculates the confidence score and round upto two decimal

                    if conf > 0.5:
                        detections.append((([x1, y1, w, h]), conf, current_class))

            tracks = tracker.update_tracks(detections, frame=frame)   # updates the object tracking with new detections

            for track in tracks:
                if not track.is_confirmed():
                    continue   # skip unconfiremed tracks
                track_id = track.track_id   # gets the ID of the tracked object
                ltrb = track.to_tlbr()   # call track.to_tlbr() method to obtain bbox coordinates in a format that is suitable for drawing bbox and further processing

                bbox = ltrb  # assigns the bounding box coordinates to bbox variables

                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1, y2-y1
                for detection in detections:
                    bbox, conf, current_class = detection
                    cvzone.putTextRect(frame,f'ID: {track_id}, Class: {current_class}, Confidence:{conf}',(x1,y1), scale=1, thickness=1, colorR=(0,0,255))
                    cvzone.cornerRect(frame, (x1,y1,w,h), l=9, rt=1, colorR=(255,0,255))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ImageTk.PhotoImage(Image.fromarray(frame))

            self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)  # this line create_image item on the canavs at coordinate 0,0
            self.canvas.image = frame  # this line stores a reference to the image(frame) in current object

            self.root.update()  # update the GUI to display the new frame

            if self.stop_event.is_set():  # check the stop event has been set
                break  # exit the loop if stop event has been set

        cap.release()  # release the video capture object


    # define a method for object detction and tracking
    def start_detection_camera(self):
        self.stop_detection_button.config(state=tk.NORMAL)   # enables to stop button
        self.detection_started = True  # sets the detection_started to true to indicates that detection is running
        self.stop_event.clear()    # clear the stop_event flag to signal that detection is rinning

        #----- Initializes the tracker object for object tracking -----#
        tracker = DeepSort(max_age=5,   # maximum number of frames track can be kept without detection
                           n_init=2,   # initaialze track after 2 frames after confirmation
                           max_iou_distance = 0.7,  # it controls the maximum IoU
                           nms_max_overlap=1.0,  # it controls the NMS
                           max_cosine_distance=0.3, # helps in associating detection (max cosine distance between apperance of two track)
                           embedder="mobilenet",  # type of feature extractor to use for the apperance of the feature
                           half=True,  # it used to minimise the load on memory and GPU
                           bgr=True,  # to convert i/p in Blue Green Red format
                           embedder_gpu=True,  # to use GPU
                           polygon=False,  # which bounded box is used i.e. polygon or rectangle
                           )

        cap = cv2.VideoCapture(0)  # opens the selected video file for reading
        assert cap.isOpened()   # checks the VideoCapture object cap is successfully open

        while self.detection_started:
            ret, frame = cap.read()  # reads a frame and reads() method return two values
            if not ret:
                break

            results = self.model(frame, stream=True)   # performs the object detection using YOLO model
            detections = []  # initializes a list to store the detected object

            for r in results:
                boxes = r.boxes  # gets the bounded box of the detected object
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]  # gets the coordinates of the bounded box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2-x1, y2-y1

                    cls = int(box.cls[0])  # gets the class label for detected objects
                    current_class = self.model.model.names[cls]  # access the list of class name that YOLO model has been trained to detect

                    conf = math.ceil((box.conf[0] * 100)) / 100  # calculates the confidence score and round upto two decimal

                    if conf > 0.5:
                        detections.append((([x1, y1, w, h]), conf, current_class))

            tracks = tracker.update_tracks(detections, frame=frame)   # updates the object tracking with new detections

            for track in tracks:
                if not track.is_confirmed():
                    continue   # skip unconfiremed tracks
                track_id = track.track_id   # gets the ID of the tracked object
                ltrb = track.to_tlbr()   # call track.to_tlbr() method to obtain bbox coordinates in a format that is suitable for drawing bbox and further processing

                bbox = ltrb  # assigns the bounding box coordinates to bbox variables

                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2-x1, y2-y1
                for detection in detections:
                    bbox, conf, current_class = detection
                    cvzone.putTextRect(frame,f'ID: {track_id}, Class: {current_class}, Confidence:{conf}',(x1,y1), scale=1, thickness=1, colorR=(0,0,255))
                    cvzone.cornerRect(frame, (x1,y1,w,h), l=9, rt=1, colorR=(255,0,255))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = ImageTk.PhotoImage(Image.fromarray(frame))

            self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)  # this line create_image item on the canavs at coordinate 0,0
            self.canvas.image = frame  # this line stores a reference to the image(frame) in current object

            self.root.update()  # update the GUI to display the new frame

            if self.stop_event.is_set():  # check the stop event has been set
                break  # exit the loop if stop event has been set

        cap.release()  # release the video capture object


    # define stop_detection method
    def stop_detection(self):
        self.detection_started = False   # sets the detection_started variable to false to stop the detection loop
        self.stop_event.set()   # sets the stop event to signal the detection loop to stop
        self.stop_detection_button.config(state=tk.DISABLED)  # disable the stop detection button

if __name__ == "__main__":
    root = tk.Tk()
    login_window = LoginWindow(root)
    root.mainloop()

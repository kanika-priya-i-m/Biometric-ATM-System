import socket
import tqdm
import os
import time
import cv2
import face_recognition
import numpy as np
from skimage import io
from sklearn import svm
from joblib import dump, load
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report,accuracy_score
import mysql.connector
from datetime import datetime
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="root",
  database="medicine_box"
)

# device's IP address
SERVER_HOST = "192.168.1.114"   # server ip
SERVER_PORT = 5001
# receive 4096 bytes each time
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"
# create the server socket
# TCP socket
s = socket.socket()
# bind the socket to our local address
s.bind((SERVER_HOST, SERVER_PORT))
# enabling our server to accept connections
# 5 here is the number of unaccepted connections that
# the system will allow before refusing new connections
s.listen(5)
print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
while(True):
    
    # accept connection if there is any
    client_socket, address = s.accept() 
    # if below code is executed, that means the sender is connected
    print(f"[+] {address} is connected.")
    # receive the file infos
    # receive using client socket, not server socket
    received = client_socket.recv(BUFFER_SIZE).decode('utf-8')
    filename, filesize = received.split(SEPARATOR)
    # remove absolute path if there is
    filename = os.path.basename(filename)
    # convert to integer
    filesize = int(filesize)
    # start receiving the file from the socket
    # and writing to the file stream
    progress = tqdm.tqdm(range(filesize), f"Receiving {filename}", unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        for _ in progress:
            # read 1024 bytes from the socket (receive)
            bytes_read = client_socket.recv(BUFFER_SIZE)
            if not bytes_read:    
                # nothing is received
                # file transmitting is done
                break
            # write to the file the bytes we just received
            f.write(bytes_read)
            # update the progress bar
            progress.update(len(bytes_read))

    # close the client socket
    client_socket.close()
    # close the server socket
    #s.close()
    path = 'C:/Users/PRACHI/Desktop/Avijit/project 2020/Medicine_Box(FR)/Dataset/'
    folders = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for folder in d:
            folders.append(folder)
    known_face_encodings=[]
    known_face_names=[]
    for f in folders:
        ifile=f'C:/Users/PRACHI/Desktop/Avijit/project 2020/Medicine_Box(FR)/Dataset/{f}/{f}_0.jpg'
        ru_image = face_recognition.load_image_file(ifile)
        ru_face_encoding = face_recognition.face_encodings(ru_image)[0]
        known_face_encodings.append(ru_face_encoding)
        known_face_names.append(f)
    clf = svm.SVC(gamma='scale')
    clf.fit(known_face_encodings,known_face_names)
    dump(clf,'SVM.Model')

    TestData="C:/Users/PRACHI/Desktop/Avijit/project 2020/Medicine_Box(FR)"
    #ru_image = face_recognition.load_image_file(f'C:/Users/PRACHI/Desktop/Avijit/project 2020/BankLocker(FR)/Dataset/{data}/{data}.jpg')
    #ru_face_encoding = face_recognition.face_encodings(ru_image)[0]
    #known_face_encodings.append(ru_face_encoding)
    #known_face_names.append(data)
    try:
        frame = (TestData+'//a.jpg')
        unknown_image = face_recognition.load_image_file(frame)
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        pil_image = Image.fromarray(unknown_image)
        draw = ImageDraw.Draw(pil_image)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown Person"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            text_width, text_height = draw.textsize(name)
            print("The Person is:",name)
            f = open("output.txt","w")
            print((name),file = f)
            f.close()
            det=""
            if name == "Unknown Person":
                path = (TestData+'//a.jpg')
                un_image = cv2.imread(path)
                cv2.imwrite("Unknown.jpg",un_image)
                #pil_image.save("Unknown.jpg")
                f=open("result.txt","w")
                print(("unknown"),file = f)
                f.close()
                det="unknown"
                
            else:
                f=open("result.txt","w")
                str1=f'face_matched:{name}'
                print(str1,file = f)
                f.close()
                pid=name
                mycursor = mydb.cursor()

                mycursor.execute("SELECT * FROM patient_details where pid=%s",(pid,))

                myresult = mycursor.fetchall()
                now = datetime.now()
                hour = now.strftime("%H")
                print("Current Hour =", hour)
                

                for x in myresult:
                    if(hour=='8'):
                        det=x[2]
                        print(x[2])
                    elif(hour=='13'):
                        det=x[3]
                        print(x[3])
                    elif(hour=='19'):
                        det=x[4]
                        print(x[4])
                    else:
                        det="no"
                        print(det)
                if(det==""):
                    det="no"
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("192.168.1.103", 8080)) #rpi ip
            client.send(det.encode())
            client.close()
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        del draw
        pil_image.show()
        pil_image.save("recognition/output.jpg")
        os.remove(TestData+'//a.jpg')
        time.sleep(5)
        #f=open('readdata.txt','w')
        #f.write('') 
        #f.close()
    except:
        print('could not read')
        f=open('readdata.txt','w')
        f.write('')
        f.close()
    
    

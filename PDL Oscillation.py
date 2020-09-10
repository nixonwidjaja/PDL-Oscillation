#Created by: Nixon

import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import math

#READ VIDEO AND EXTRACT FRAMES
print("Place your video file in the same folder as this py. file")
video = input("Please type in your video file name: ")
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
count = 1
for i in [1]:
    ret, frame = cap.read()
    cv2.imwrite("frame%d.jpg" % count, frame)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    
#PICK POINTS FOR INSPECTION LINE
%matplotlib qt

from PIL import Image
img = Image.open("frame1.jpg")
imagesize = img.size
xy2imgxy = lambda x,y: (imagesize[0] * x / np.max(ticklx),\
                        imagesize[1] * (np.max(tickly) - y) / np.max(tickly))
ticklx = np.linspace(0,imagesize[0],4)
tickly = np.linspace(0,imagesize[1],4)
tickpx,tickpy = xy2imgxy(ticklx,tickly)
fig,ax = plt.subplots()
ax.imshow(img)
ax.set_xticks(tickpx)
ax.set_yticks(tickpy)
plt.ylabel('Y-AXIS PIXEL', fontsize=14)
plt.xlabel('X-AXIS PIXEL', fontsize=14)
ax.set_xticklabels(ticklx.astype('int'))
ax.set_yticklabels(tickly.astype('int'))
print()
tellme("Click the image first, then pick 2 points for the inspection line")
print()
plt.show()
plt.waitforbuttonpress()

while True:
    pts = []
    while len(pts) < 2:
        plt.title("Pick 1 more point", fontsize=16)
        pts = np.asarray(plt.ginput(2, timeout=-1))
        if len(pts) < 2:
            tellme('Not enough points, starting over')
            time.sleep(1)  # Wait a second
    tellme("Click enter to continue, or click with mouse to redefine points")

    if plt.waitforbuttonpress():
        break
cv2.destroyAllWindows()

coefficients = np.polyfit(pts[:,0], pts[:,1], 1)
aaa = np.array([[0,round(coefficients[1])]])
for i in range(1,imagesize[0]):
    x = i
    y = coefficients[0]*i + coefficients[1]
    aaa = np.append(aaa, [[round(x), round(y)]], axis = 0)

#PICK POINTS FOR CALIBRATION LINE
%matplotlib qt

from PIL import Image
img = Image.open("frame1.jpg")
imagesize = img.size
xy2imgxy = lambda x,y: (imagesize[0] * x / np.max(ticklx),\
                        imagesize[1] * (np.max(tickly) - y) / np.max(tickly))
ticklx = np.linspace(0,imagesize[0],4)
tickly = np.linspace(0,imagesize[1],4)
tickpx,tickpy = xy2imgxy(ticklx,tickly)
fig,ax = plt.subplots()
ax.imshow(img)
ax.set_xticks(tickpx)
ax.set_yticks(tickpy)
plt.ylabel('Y-AXIS PIXEL', fontsize=14)
plt.xlabel('X-AXIS PIXEL', fontsize=14)
ax.set_xticklabels(ticklx.astype('int'))
ax.set_yticklabels(tickly.astype('int'))
plt.plot(aaa[:,0],aaa[:,1],'r')
print()
tellme("Click the image first, then pick 2 points for the calibration line")
print()
plt.show()
plt.waitforbuttonpress()

while True:
    pointss = []
    while len(pointss) < 2:
        plt.title("Pick 1 more point", fontsize=16)
        pointss = np.asarray(plt.ginput(2, timeout=-1))
    tellme("Click enter to continue")
    if plt.waitforbuttonpress():
        break
cv2.destroyAllWindows()

actuallength = input("Please enter the actual length between the 2 points: ")
actuallength = float(actuallength)

def distance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  
imagelength = distance(pointss[0][0], pointss[0][1], pointss[1][0], pointss[1][1]) 

#PLOT LINE SCAN
intensity = []
sumbux = []
for i in range(0, (imagesize[0]-1)):
    sumbux.append(i)
    intensity.append(gray[int(aaa[i][1])][i])

#BOUNDARY & THRESHOLD
%matplotlib qt

plt.figure()
plt.plot(sumbux, intensity)
print()
tellme("Click the image first, then pick 1 point at half intensity approximately near the edge of graphite's movement")
print()
plt.show()
plt.ylabel('PIXEL INTENSITY', fontsize=14)
plt.xlabel('PIXEL ALONG INSPECTION LINE', fontsize=14)
plt.waitforbuttonpress()
titik = []
titik = np.asarray(plt.ginput(1, timeout=-1))
cv2.destroyAllWindows()

property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
totalframe = int(cv2.VideoCapture.get(cap, property_id))

posisiy = round(titik[0][1])
posisix = round(titik[0][0])
plt.axhline(y=posisiy, color='r', linestyle='-')
x = posisix
diff = intensity[int(x)] - posisiy
before = diff
coor = []

#ALL FRAMES ANALYSIS
if posisix < (imagesize[0]/2):
    for j in range(1, totalframe):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensity = []
        for i in range(0, (imagesize[0]-1)):
            intensity.append(gray[int(aaa[i][1])][i])
        x = int(posisix) 
        while x < imagesize[0]:
            diff = intensity[x] - posisiy
            if diff*before <= 0:
                coor.append(int(x))
                break
            before = diff
            x = x+1
else:
    for j in range(1, totalframe):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensity = []
        for i in range(0, (imagesize[0]-1)):
            intensity.append(gray[int(aaa[i][1])][i])
        x = int(posisix)
        while x > 0:
            diff = intensity[x] - posisiy
            if diff*before <= 0:
                coor.append(int(x))
                break
            before = diff
            x = x-1
        
for i in range(0, len(coor)):
    coor[i] = coor[i]*actuallength*1000/imagelength
    
t = []
for i in range(0, len(coor)):
    t.append(i/fps)
plt.figure()
plt.plot(t,coor)
plt.xlabel('time [s]', fontsize=16)
plt.ylabel('z position [mm]', fontsize=16)

#OSCILLATION ANALYSIS

#FIND EQUILIBRIUM POSITION
maxii = []
minii = []
dcoor = coor[1]-coor[0]
before = dcoor
for i in range(int(len(coor)*0.7), int(len(coor)*0.9)):
    dcoor = coor[i+1]-coor[i]
    if dcoor <= 0 and before > 0:
        maxii.append(coor[i])
    if dcoor >= 0 and before < 0:
        minii.append(coor[i])
    before = dcoor
mean = (sum(maxii)+sum(minii))/(len(maxii)+len(minii))
plt.axhline(y=mean, color='r', linestyle='-')

#FIND MAX AND MIN POINTS
timemax = []
timemin = []
maxx = []
mini = []
dcoor = coor[1]-coor[0]
before = dcoor
for i in range(1, len(coor)-1):
    dcoor = coor[i+1]-coor[i]
    if dcoor <= 0 and before > 0:
        if coor[i] > mean :
            timemax.append(t[i])
            maxx.append(coor[i])
    if dcoor >= 0 and before < 0:
        if coor[i] < mean :
            timemin.append(t[i])
            mini.append(coor[i])
    before = dcoor
for k in range(1,5):
    for i in range(0, len(timemin)-2):
        j = 0
        for j in range(0, len(timemax)):
            if j == len(timemax)-1:
                break
            while timemin[i] <= timemax[j] <= timemin[i+1] and timemin[i] <= timemax[j+1] <= timemin[i+1]:
                if maxx[j] > maxx[j+1]:
                    maxx.remove(maxx[j+1])
                    timemax.remove(timemax[j+1])
                    j = j-3
                else:
                    maxx.remove(maxx[j])
                    timemax.remove(timemax[j])
                    j = j-3
    for i in range(0, len(timemax)-2):
        j = 0
        for j in range(0, len(timemin)-1):
            if j == len(timemin)-1:
                break
            while timemax[i] <= timemin[j] <= timemax[i+1] and timemax[i] <= timemin[j+1] <= timemax[i+1]:
                if mini[j] < mini[j+1]:
                    mini.remove(mini[j+1])
                    timemin.remove(timemin[j+1])
                    j = j-3
                else:
                    mini.remove(mini[j])
                    timemin.remove(timemin[j])
                    j = j-3

#FIND OSCILLATION PERIOD, FREQUENCY, AND DAMPING CONSTANT
tav = []
zdif = []
for i in range(0, int(len(maxx)*0.5)):
    av = (timemax[i]+timemin[i])/2
    tav.append(av)
    dif = maxx[i]-mini[i]
    zdif.append(np.log(dif))
period = (tav[len(tav)-1]-tav[0])/(len(tav)-1)
freq = 1/period

plt.figure()
plt.plot(tav, zdif, ':b')
# y = mx + c
m, c = np.polyfit(tav, zdif, 1)
tau = abs(1/m)
xx = np.linspace(0,tav[len(tav)-1],10)
yy = m*xx+c
plt.plot(xx,yy,'-r')
plt.xlabel('time [s]', fontsize=16)
plt.ylabel('ln(z extrema position [mm])', fontsize=16)

print()
print("Period           =", period)
print("Frequency        =", freq)
print("Damping constant =", tau)
print()
print()

#EXPORT OSCILLATION DATA TO TEXT FILE
f = open("Oscillation Data.txt","w+")
f.write('%-10s\t%-10s\t%-7s\t%-10s\t%-10s\t%-14s\t%-20s\n'%("t","z","fps","period (s)","freq (Hz)","damping (1/s)","video name"))
f.write('%-10s\t%-10s\t%-7s\t%-10s\t%-10s\t%-14s\t%-20s\n'%("s","mm","%.2f" %fps,"%.6f" %period,"%.6f" %freq,"%.6f" %tau,video))
for i in range(0, len(coor)) :
     f.write('%-10s\t%-10s\t\n'%("%.6f" % t[i], "%.6f" % coor[i]))
f.close()





















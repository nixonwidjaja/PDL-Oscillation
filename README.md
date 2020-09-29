# PDL-Oscillation

1. Install Spyder and OpenCV.
Enter in Anaconda Prompt:
pip install opencv-python

2. Run Python program. The program will read the video we had recorded and extract its frames. 

3. The program will ask you to pick points for inspection line. Pick two points to form a line where the graphite will oscillate back and forth.
It will detect the brightness intensity of the pixels in our inspection line. The intensity ranges between 0 and 255. After it detects the intensity, the program can deduce the position of the graphite, and thus be able to plot its position with respect to time.

4. The program will ask you to pick points for calibration line.
Pick two points which you know the actual length between those points in real setup.

5. Enter the actual length.

6. We will observe a curve with intensity maxima. The program will then ask us to pick a point at half intensity approximately near the edge of graphite’s movement. Pick 1 point at half intensity approximately near the edge of graphite's oscillation (near the turnaround point).

7. You will obtain the oscillation graph, period, frequency, damping constant, and data exported 
in text file.

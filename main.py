import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')
left_top_x = 0
left_bottom_x = 0
right_bottom_x = 0
right_top_x = 0

left_top_x_previous = 0
left_bottom_x_previous = 0
right_top_x_previous = 0
right_bottom_x_previous = 0

while True:
    ret, frame = cam.read()

    if not ret:
        break


    height, width, _ = frame.shape
    desired_width = np.int(1920/4 - 10)
    desired_height = np.int(1080/3 - 60)
    frame = cv2.resize(frame, (desired_width, desired_height))
    cv2.imshow('Frame', frame)


    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame_gray", frame_gray)

    #coordonate trapez
    upper_right = (0.55 * desired_width, 0.75 * desired_height)
    upper_left = (0.45 * desired_width, 0.75 * desired_height)
    lower_left = (0, desired_height)
    lower_right = (desired_width, desired_height)

    trapezoid_points = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)  #numpy array cu coordonatele trapezului
    trapezoid_frame = np.zeros((desired_height, desired_width), dtype=np.uint8)  #am creat un empty frame

    #cv2.fillConvexPoly(frame_in_which_to_draw, points_of_a_polygon, color_to_draw_with)
    cv2.fillConvexPoly(trapezoid_frame, trapezoid_points, 1) #am desenat trapezul peste empty frame
    cv2.imshow("frame_trapezoid", trapezoid_frame*255)
    frame_road = frame_gray * trapezoid_frame
    cv2.imshow("frame_road", frame_road)

    trapezoid_points = np.float32(trapezoid_points) #convert din int in float coordonatele trapezului

    #coordonatele ecranului
    ul = (0, 0)
    ur = (desired_width, 0)
    ll = (0, desired_height)
    lr = (desired_width, desired_height)

    trapezoid_bounds = np.array([ur, ul, ll, lr], dtype=np.float32) #array-ul cu coordonatele ecranului

    #cv2.getPerspectiveTransform(bounds_of_current_area, bounds_of_area_you_want_to_stretch_to)

    frame_magic = cv2.getPerspectiveTransform(trapezoid_points, trapezoid_bounds)
    frame_stretch2 = cv2.warpPerspective(frame_road, frame_magic, (desired_width, desired_height));
    cv2.imshow("frame_stretch", frame_stretch2)


    frame_blur = cv2.blur(frame_stretch2, ksize=(9, 9))
    cv2.imshow("frame_blur", frame_blur)


    sobel_vertical = np.float32([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
    sobel_horizontal = np.transpose(sobel_vertical)

    frame_edge_detV = cv2.filter2D(frame_blur, -1, sobel_vertical)
    frame_edge_detH = cv2.filter2D(frame_blur, -1, sobel_horizontal)
    frame_edge_detV = np.float32(frame_edge_detV)
    frame_edge_detH = np.float32(frame_edge_detH)
    frame_edge = np.sqrt((frame_edge_detV * frame_edge_detV) + (frame_edge_detH * frame_edge_detH))
    frame_final_edge = cv2.convertScaleAbs(frame_edge)
    cv2.imshow("frame_final_edge", frame_final_edge)


    ret, frame_binarize = cv2.threshold(frame_final_edge, 40, 255, cv2.THRESH_BINARY)
    cv2.imshow("frame_binarize", frame_binarize)

    frame_binarize_copy = frame_binarize.copy()

    left_bound = int(0.05 * desired_width)
    right_bound = int(0.95 * desired_width)

    frame_binarize_copy[:, :left_bound] = 0
    frame_binarize_copy[:, right_bound:] = 0
    cv2.imshow("Frame with Black Edges", frame_binarize_copy)



    half_width = desired_width // 2
    frame_left = frame_binarize_copy[:, :half_width]
    frame_right = frame_binarize_copy[:,half_width:]
    left_coordinates = np.argwhere(frame_left == 255)
    right_coordinates = np.argwhere(frame_right == 255)

    left_xs = left_coordinates[:, 1]
    left_ys = left_coordinates[:, 0]
    right_xs = right_coordinates[:, 1] + half_width
    right_ys = right_coordinates[:, 0]

    cv2.imshow("frame_left", frame_left)
    cv2.imshow(" frame_right", frame_right)

    array1 = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg = 1) #linia de gradul 1 al regresiei liniare
    array2 = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg = 1)

    #left_top_y = 0 si left_bottom_y = desired_height
    #right_top_y = 0 si right_bottom_y = desired_height
    #x=y-b/a

     # Verificare coordonate x și folosirea valorilor de la frame-ul anterior în caz contrar
    if all(-10**8 <= x <= 10**8 for x in [left_top_x, left_bottom_x, right_top_x, right_bottom_x]):
        left_top_x = (0 - array1[0]) // array1[1]
        left_bottom_x = (desired_height - array1[0]) // array1[1]
        right_top_x = (0 - array2[0]) // array2[1]
        right_bottom_x = (desired_height - array2[0]) // array2[1]
        # Actualizare variabile pentru frame-ul următor
        left_top_x_previous = left_top_x
        left_bottom_x_previous = left_bottom_x
        right_top_x_previous = right_top_x
        right_bottom_x_previous = right_bottom_x
    else:
        # Folosire coordonatele de la frame-ul anterior
        left_top_x = left_top_x_previous
        left_bottom_x = left_bottom_x_previous
        right_top_x = right_top_x_previous
        right_bottom_x = right_bottom_x_previous



    left_top = int(left_top_x), int(0)
    left_bottom = int(left_bottom_x), int(desired_height)
    right_top = int(right_top_x), int(0)
    right_bottom =  int(right_bottom_x), int(desired_height)

    cv2.line(frame_binarize_copy,left_top, left_bottom, (200, 0, 0), 5 )
    cv2.line(frame_binarize_copy,right_top, right_bottom, (100, 0, 0), 5 )
    cv2.imshow("Frame line", frame_binarize_copy)

    # frame_stretch = cv2.warpPerspective(frame_road, frame_magic, (desired_width, desired_height));

    #a.
    frame_final_left = np.zeros((frame_gray.shape), dtype= np.uint8)
    frame_final_right = np.zeros((frame_gray.shape), dtype = np.uint8)

    #b.
    cv2.line(frame_final_left, left_top, left_bottom, (255, 0, 0), 100 )
    cv2.line(frame_final_right, right_top, right_bottom, (255, 0, 0), 5 )

    #c.
    magic_matrix = cv2.getPerspectiveTransform(trapezoid_bounds , trapezoid_points)

    #d.
    frame_stretch_left = cv2.warpPerspective(frame_final_left, magic_matrix, (desired_width, desired_height))
    frame_stretch_right = cv2.warpPerspective(frame_final_right, magic_matrix, (desired_width, desired_height))

    #cv2.imshow("Frame left",frame_final_left)
    #cv2.imshow("Frame right",frame_final_right)

    # Assuming frame_left is the binary image containing the left lane line
    left_coordinates = np.argwhere(frame_stretch_left == 255)

    # Extract x and y coordinates separately
    left_xs = left_coordinates[:, 1]
    left_ys = left_coordinates[:, 0]

    # Assuming frame_right is the binary image containing the right lane line
    right_coordinates = np.argwhere(frame_stretch_right == 255)

    # Extract x and y coordinates separately
    right_xs = right_coordinates[:, 1]
    right_ys = right_coordinates[:, 0]

    # Now, right_xs and right_ys contain the x and y coordinates of the white pixels in the right lane line

    frame_copy = frame.copy()

    frame_copy[left_ys, left_xs] = [50, 50, 250]
    frame_copy[right_ys, right_xs] = [50, 250, 50]

    cv2.imshow('final frame', frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()

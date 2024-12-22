from ultralytics import YOLO
import cv2
import ultralytics

ultralytics.checks()
# Load model
model = YOLO('ppe.pt')  # Đường dẫn đến file model


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()


    # Sử dụng model để detect
    results = model(frame)
    result = results[0]
    cv2.imshow("Frame", result.plot())

    # # Hiển thị kết quả
    # result.show()  # Hiển thị kết quả trên cửa sổ hình ảnh
    # # Lưu kết quả
    # result.save()  # Lưu kết quả vào thư mục output

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

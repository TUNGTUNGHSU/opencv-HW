import cv2 as cv
import numpy as np
import dlib
from PIL import Image, ImageDraw, ImageFont


# 加載Dlib的人臉檢測器
detector = dlib.get_frontal_face_detector()

input_mp4 = './Alec_Baldwin.mp4'
output_mp4 = './Alec_Baldwin_OPENCV_HW.mp4'

cap = cv.VideoCapture(input_mp4)

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_mp4, fourcc, fps, (width, height))

effect_name = ""
effect_frames = 140 # 每個效果持續的幀數
frame_count = 0
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)
def putTextChinese(img, text, position, font_path, font_size, color, thickness):
    img_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    img = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
    return img

    
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    effect = frame_count // effect_frames

    if effect == 0:  # 正常效果
        effect_name = "無特效"
    elif effect == 1:  # 灰階效果
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        effect_name = "特效1:灰階"
    elif effect == 2:  # 高斯模糊效果
        frame = cv.GaussianBlur(frame, (15, 15), 0)
        effect_name = "特效2:高斯模糊"
    elif effect == 3:  # Canny邊緣檢測效果
        frame = cv.Canny(frame, 100, 200)
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        effect_name = "特效3:Canny邊緣檢測"
    elif effect == 4:  # 反色效果
        frame = 255 - frame
        effect_name = "特效4:反色"
    elif effect == 5:  # 復古風效果
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = frame.astype('float32')
        frame = frame * (0.272, 0.534, 0.131)
        frame = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX)
        frame = frame.astype('uint8')
        effect_name = "特效5:復古風"
    elif effect == 6:  # 對比度效果
        frame = cv.convertScaleAbs(frame, alpha=2, beta=0)
        effect_name = "特效6:對比度"
    elif effect == 7:  # 旋轉270度效果
        M = cv.getRotationMatrix2D((width // 2, height // 2), 270, 1.0)
        frame = cv.warpAffine(frame, M, (width, height))
        effect_name = "特效7:旋轉270度"
    elif effect == 8: #人臉辨識
        effect_name = "特效8:人臉辨識"
        # 將圖像從BGR轉換為RGB(因為Dlib使用的是RGB)
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # 使用 Dlib 检测器检测图像中的人脸
        faces = detector(image_rgb)
        # 在檢測到的人臉上繪製矩形框
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif effect == 9:  # 卡通化
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.medianBlur(gray_frame, 5)
        edges_frame = cv.adaptiveThreshold(gray_frame, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)
        color_frame = cv.bilateralFilter(frame, 9, 250, 250)
        frame = cv.bitwise_and(color_frame, color_frame, mask=edges_frame)
        effect_name = "特效9:卡通"
    elif effect == 10:  # 素描
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.GaussianBlur(gray_frame, (13, 13), 0)
        edges_frame = cv.Laplacian(gray_frame, cv.CV_8U, ksize=5)
        _, threshold_frame = cv.threshold(edges_frame, 100, 255, cv.THRESH_BINARY_INV)
        frame = cv.cvtColor(threshold_frame, cv.COLOR_GRAY2BGR)
        effect_name = "特效10:素描"
    elif effect == 11:  # 中值模糊
        effect_name = '特效11:中值模糊'
        frame = cv.medianBlur(frame, ksize=7)  # ksize必须是奇数
    elif effect == 12: #雙邊濾波
        effect_name = '特效12:雙邊濾波'
        frame = cv.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)   
    elif effect == 13: #膨脹
        effect_name = '特效13:膨脹'
        kernel = np.ones((5, 5), np.uint8)
        frame = cv.dilate(frame, kernel, iterations=1)
    elif effect == 14: #腐蝕
        effect_name = '特效14:腐蝕'
        kernel = np.ones((5, 5), np.uint8)
        frame = cv.erode(frame, kernel, iterations=1)       
    elif effect == 15:  # 水平翻轉效果
        frame = cv.flip(frame, 1)
        effect_name = "特效15:水平翻轉"
    elif effect == 16:  # 垂直翻轉效果
        frame = cv.flip(frame, 0)
        effect_name = "特效16:垂直翻轉"
    
    elif effect == 17: #4 in 1 + Gamma
        effect_name = '特效17:4合1+伽瑪校正'
        gamma_values = [0.5, 1.5, 2.5, 3.5]
        h, w = frame.shape[:2]
        half_h, half_w = h // 2, w // 2

        # 處理影片的四個區域
        frame[:half_h, :half_w] = adjust_gamma(frame[:half_h, :half_w], gamma_values[0])
        frame[:half_h, half_w:] = adjust_gamma(frame[:half_h, half_w:], gamma_values[1])
        frame[half_h:, :half_w] = adjust_gamma(frame[half_h:, :half_w], gamma_values[2])
        frame[half_h:, half_w:] = adjust_gamma(frame[half_h:, half_w:], gamma_values[3])

        # 在每個區域顯示對應的Gamma值
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2

        for i, gamma_value in enumerate(gamma_values):
            text = f'Gamma: {gamma_value}'
            x_offset = half_w * (i % 2)
            y_offset = half_h * (i // 2)
            x, y = 30 + x_offset, 50 + y_offset
            cv.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    elif effect == 18:  # 波形扭曲
        effect_name = "特效18:波形扭曲"
        rows, cols, _ = frame.shape
        frame_output = np.zeros_like(frame)

        for y in range(rows):
            for x in range(cols):
                offset_x = int(25.0 * np.sin(2 * 3.14 * y / 180))
                offset_y = int(25.0 * np.sin(2 * 3.14 * x / 180))
                new_x = x + offset_x
                new_y = y + offset_y

                if 0 <= new_x < cols and 0 <= new_y < rows:
                    frame_output[y, x] = frame[new_y, new_x]
                else:
                    frame_output[y, x] = 0

        frame = frame_output
    # 在左上角加效果名稱
    fontpath = "NotoSansTC-Black.otf"  # 中文字體檔案路徑
    font_size = 32
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)
    thickness = 2
    text_size, _ = cv.getTextSize(effect_name, font, font_scale, thickness)
    text_width, text_height = text_size
    text_width, text_height = cv.getTextSize(effect_name, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv.rectangle(frame, (0, 0), (text_width + 10, text_height + 10), (0, 0, 0), -1)
    frame = putTextChinese(frame, effect_name, (5, text_height + 5), fontpath, font_size, color, thickness)
    text_position = (5, text_height + 5)
    
    # 將處理過的效果寫入輸出視頻
    out.write(frame)
    frame_count += 1


# 在特效迴圈結束後加"THE END"字卡
end_frames_count = int(fps * 3)  # 在影片結束時顯示"THE END"3秒
end_text = "THE END"
end_font = cv.FONT_HERSHEY_SIMPLEX
end_font_scale = 2
end_color = (255, 0, 0)
end_thickness = 3

text_size, _ = cv.getTextSize(end_text, end_font, end_font_scale, end_thickness)
text_width, text_height = text_size
text_x = (frame_width - text_width) // 2
text_y = (frame_height - text_height) // 2

for _ in range(end_frames_count):
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv.putText(black_frame, end_text, (text_x, text_y), end_font, end_font_scale, end_color, end_thickness, cv.LINE_AA)
    out.write(black_frame)

# 釋放資源
cap.release()
out.release()
cv.destroyAllWindows()
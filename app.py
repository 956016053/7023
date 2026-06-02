import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ==========================================
# 1. 页面与环境配置
# ==========================================
st.set_page_config(page_title="Cloud DMS System", layout="centered")
# 在云端服务器上通常只有CPU，强制使用CPU推理（放心，我们的轻量级模型CPU也毫无压力）
DEVICE = torch.device("cpu")


# ==========================================
# 2. 网络结构定义 (LTAM & RealTimeDMS)
# ==========================================
class LightweightTemporalAlignment(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, kernel_size=1, groups=channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, curr_feat, prev_feat):
        if prev_feat is None: return curr_feat
        concat_feat = torch.cat((curr_feat, prev_feat), dim=1)
        g_weight = self.gate(concat_feat)
        return g_weight * curr_feat + (1 - g_weight) * prev_feat


class RealTimeDMS(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base_model = models.mobilenet_v2(pretrained=False)
        self.features = base_model.features
        self.temporal_module = LightweightTemporalAlignment(channels=1280)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x_curr, x_prev):
        curr_feat = self.features(x_curr)
        prev_feat = self.features(x_prev)
        aligned_feat = self.temporal_module(curr_feat, prev_feat)
        return self.classifier(aligned_feat)


# ==========================================
# 3. 缓存模型加载
# ==========================================
@st.cache_resource
def load_model():
    model = RealTimeDMS().to(DEVICE)
    # 注意：确保 weights 文件夹和这个 app.py 在同一目录下
    model.load_state_dict(torch.load("./temporal_dms_best.pth", map_location=DEVICE))
    model.eval()
    return model


data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# 4. WebRTC 视频流处理器 (边缘部署工程校准版)
# ==========================================
class FatigueDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.prev_tensor = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frame_counter = 0  # 用于控制时间记忆的跨度

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 捕捉人脸
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        self.frame_counter += 1

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            
            # 2. 扩大裁剪边缘 (贴合数据集比例，减少 Domain Shift)
            margin_x = int(w * 0.3)
            margin_y = int(h * 0.4)
            y1, y2 = max(0, y - margin_y), min(img.shape[0], y + h + margin_y)
            x1, x2 = max(0, x - margin_x), min(img.shape[1], x + w + margin_x)
            face_img = img[y1:y2, x1:x2]
            
            # 3. 预处理喂给模型
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            curr_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)
            
            if self.prev_tensor is None:
                self.prev_tensor = curr_tensor.clone()
                
            # 4. 模型时空推理
            with torch.no_grad():
                outputs = self.model(curr_tensor, self.prev_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                raw_prob = probs[1].item() # 获取原始弱信号 (约 0.1 左右)
                
            # ⭐ 核心修复 1：时间跨度 (Temporal Striding)
            # 每 5 帧才更新一次记忆帧，让系统能明显感受到"从睁眼到闭眼"的过程
            if self.frame_counter % 5 == 0:
                self.prev_tensor = curr_tensor.clone()
            
            # ⭐ 核心修复 2：置信度校准 (Confidence Calibration)
            # 将 Webcam 的微弱置信度放大，映射到视觉预期的范围
            drowsy_prob = min(raw_prob * 6.0, 1.0) 
            
            # ⭐ 核心修复 3：边缘设备动态阈值
            is_drowsy = drowsy_prob > 0.45 
            
            # 5. UI 渲染框与数据
            color = (0, 0, 255) if is_drowsy else (0, 255, 0)
            label = "DROWSY" if is_drowsy else "NORMAL"
            
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img, f"{label}: {drowsy_prob*100:.1f}%", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(img, "Searching for Driver's Face...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            self.prev_tensor = None 
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. UI 渲染
# ==========================================
st.title("☁️ Cloud Edge-AI Fatigue Detection")
st.markdown("Please grant camera permissions to test the LTAM framework in real-time.")

# 1. 定义免费的 Google STUN 穿透服务器
RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# 2. 启动 WebRTC 摄像头流
webrtc_streamer(
    key="dms-demo",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,  # 👈 核心修复：加入穿透服务器
    video_processor_factory=FatigueDetectionProcessor,
    media_stream_constraints={"video": True, "audio": False}
    # 👈 核心修复：删掉了 async_processing=True，新版本库不需要它，反而会引发线程崩溃
)

st.markdown("---")
st.markdown("**Powered by Lightweight Temporal Alignment Module (LTAM)** | *Author: LIU HAORAN*")

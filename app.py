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
# 4. WebRTC 视频流处理器
# ==========================================
class FatigueDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.prev_tensor = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 获取前端传来的画面 (BGR格式)
        img = frame.to_ndarray(format="bgr24")

        # 预处理
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        curr_tensor = data_transform(pil_img).unsqueeze(0).to(DEVICE)

        if self.prev_tensor is None:
            self.prev_tensor = curr_tensor.clone()

        # 模型推理
        with torch.no_grad():
            outputs = self.model(curr_tensor, self.prev_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            drowsy_prob = probs[1].item()

        # 缓存当前帧给下一帧使用 (Feature Caching)
        self.prev_tensor = curr_tensor.clone()

        # 画面渲染 (OpenCV 用的是 BGR 色彩空间，红是(0,0,255)，绿是(0,255,0))
        is_drowsy = drowsy_prob > 0.5
        color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        label = "DROWSY" if is_drowsy else "NORMAL"

        # 在画面上打上标签
        cv2.putText(img, f"{label}: {drowsy_prob * 100:.1f}%", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

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

# import cv2
# import numpy as np

# def show_image(image, title):
#     display = (image * 255).astype(np.uint8)
#     cv2.imshow(title, display)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def sharpen_image(src):
#     kernel = np.array([[ 0, -0.0625,  0 ],
#                        [-0.0625,  1.05, -0.0625],
#                        [ 0, -0.0625,  0 ]], dtype=np.float32)
#     result = cv2.filter2D(src, -1, kernel)
#     return result

# def adjust_brightness_contrast(src, alpha, beta):
#     result = src * alpha + beta
#     result = np.clip(result, 0, 1)
#     return result

# def adjust_saturation(src, saturation_factor):
#     result = src.copy()
#     for y in range(src.shape[0]):
#         for x in range(src.shape[1]):
#             pixel = src[y, x]
#             max_val = max(pixel)
#             result[y, x] = pixel + (max_val - pixel) * saturation_factor
#     result = np.clip(result, 0, 1)
#     return result

# def upscale_image(src, scale_factor):
#     """Upscale the image using interpolation."""
#     new_width = int(src.shape[1] * scale_factor)
#     new_height = int(src.shape[0] * scale_factor)
#     result = cv2.resize(src, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
#     return result

# def main():
#     image_path = "C:/Git/CPE 462 Project/tiger.jpg"

#     original = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if original is None:
#         print("Failed to load image!")
#         return

#     # Convert to float32 and normalize to [0, 1]
#     original = original.astype(np.float32) / 255.0
#     show_image(original, "Original Image")

#     # Step 1: Adjust brightness and contrast
#     alpha = 1.05  # Contrast control
#     beta = 0.1    # Brightness control
#     brightness_contrast_adjusted = adjust_brightness_contrast(original, alpha, beta)
#     show_image(brightness_contrast_adjusted, "Brightness & Contrast Adjusted")

#     # Step 2: Adjust saturation
#     saturation_factor = -0.5  # Decrease saturation by 50%
#     saturated = adjust_saturation(brightness_contrast_adjusted, saturation_factor)
#     show_image(saturated, "Saturation Adjusted")

#     # Step 3: Sharpen the image
#     sharpened = sharpen_image(saturated)
#     show_image(sharpened, "Sharpened Image")

#     # Step 4: Upscale the image
#     scale_factor = 2.0  # Upscale by 2x
#     upscaled = upscale_image(sharpened, scale_factor)
#     show_image(upscaled, "Upscaled Image")

#     # Final enhanced image
#     enhanced = upscaled.copy()
#     show_image(enhanced, "Final Enhanced Image")

# if __name__ == "__main__":
#     main()













import cv2
import numpy as np
import torch
from realesrgan import *
from PIL import Image

def show_image(image, title):
    display = (image * 255).astype(np.uint8)
    cv2.imshow(title, display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sharpen_image(src):
    kernel = np.array([[ 0, -0.0625,  0 ],
                       [-0.0625,  1.05, -0.0625],
                       [ 0, -0.0625,  0 ]], dtype=np.float32)
    result = cv2.filter2D(src, -1, kernel)
    return result

def adjust_brightness_contrast(src, alpha, beta):
    result = src * alpha + beta
    result = np.clip(result, 0, 1)
    return result

def adjust_saturation(src, saturation_factor):
    result = src.copy()
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            pixel = src[y, x]
            max_val = max(pixel)
            result[y, x] = pixel + (max_val - pixel) * saturation_factor
    result = np.clip(result, 0, 1)
    return result

def upscale_image_with_realesrgan(src, model_path="C:\\github\\REAL-ESRGAN\\ESRGAN-master\\models\\RealESRGAN_x4plus.pth"):
    import warnings
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray((src_rgb * 255).astype(np.uint8))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        loadnet = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: The model file was not found at {model_path}. Please check the path and try again.")
        return None
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

    if "params_ema" not in loadnet:
        print(f"Error: 'params_ema' key not found in the model file. Available keys: {list(loadnet.keys())}")
        return None

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=2, act_type='prelu')
    model.load_state_dict(loadnet["params_ema"], strict=False)
    model = model.to(device).eval()

    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    img = np.array(pil_image).astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).squeeze(0).cpu().clamp_(0, 1).numpy()

    output = np.transpose(output, (1, 2, 0))

    return output

def main():
    image_path = "C:/Git/CPE 462 Project/tiger.jpg"
    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original is None:
        print("Failed to load image!")
        return

    original = original.astype(np.float32) / 255.0
    show_image(original, "Original Image")

    upscaled = upscale_image_with_realesrgan(original)
    show_image(upscaled, "Upscaled Image")

    upscaled = np.clip(upscaled, 0.0, 1.0)

    alpha = 1.05
    beta = 0.1
    brightness_contrast_adjusted = adjust_brightness_contrast(upscaled, alpha, beta)
    show_image(brightness_contrast_adjusted, "Brightness & Contrast Adjusted")

    saturation_factor = -0.5
    saturated = adjust_saturation(brightness_contrast_adjusted, saturation_factor)
    show_image(saturated, "Saturation Adjusted")

    sharpened = sharpen_image(saturated)
    show_image(sharpened, "Sharpened Image")

    enhanced = sharpened.copy()
    show_image(enhanced, "Final Enhanced Image")

if __name__ == "__main__":
    main()

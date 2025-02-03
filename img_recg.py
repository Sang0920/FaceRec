from face_alignment import align
from inference import load_pretrained_model, to_input
import os
from PIL import Image
from PIL import Image, ImageDraw
from collections import defaultdict
import time
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")
model = load_pretrained_model('ir_50').to(device)

def upscale_image(img, min_size=336):
    """Upscale image to minimum size while maintaining aspect ratio
    Args:
        img: PIL Image object
        min_size: Minimum size for either width or height
    
    Returns:
        PIL Image object
    """
    width, height = img.size
    scale = max(min_size / width, min_size / height)
    if scale > 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img

def get_gallery_info(gallery_dir):
    gallery_info = {}
    for root, _, files in os.walk(gallery_dir):
        identity = os.path.basename(root)  # Use folder name as identity
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                # store full path and modification time
                gallery_info[img_path] = {
                    'mtime': os.path.getmtime(img_path),
                    'identity': identity
                }
    return gallery_info

def load_gallery_faces(gallery_dir, cache_file='gallery_cache.pt'):
    current_info = get_gallery_info(gallery_dir)
    # check if cache exists and is valid
    if os.path.exists(cache_file):
        cache = torch.load(cache_file, weights_only=True)
        if 'files_info' in cache:
            cached_info = cache['files_info']
            # compare current and cached files
            if (set(current_info.keys()) == set(cached_info.keys()) and
                all(current_info[p]['mtime'] == cached_info[p]['mtime'] 
                    for p in current_info.keys())):
                print("Loading cached features...")
                return cache['features'], cache['identities']
                
    print("Generating new features...")
    gallery_features = []
    gallery_paths = []  # Store full paths for cache validation
    identities = []  # Store folder names as identities
    # group images by identity for batch processing
    identity_batches = defaultdict(list)
    for img_path, info in current_info.items():
        identity_batches[info['identity']].append(img_path)
    # process each identity's images
    batch_size = 32
    for identity, paths in identity_batches.items():
        batch_images = []
        batch_paths = []
        for img_path in paths:
            try:
                aligned_rgb_img = align.get_aligned_face(img_path)
                if aligned_rgb_img is None:
                    print(f"Could not align face in {img_path}")
                    continue
                batch_images.append(aligned_rgb_img)
                batch_paths.append(img_path)                
                if len(batch_images) == batch_size:
                    features = process_batch(batch_images)
                    gallery_features.extend(features)
                    gallery_paths.extend(batch_paths)
                    identities.extend([identity] * len(batch_paths))
                    batch_images = []
                    batch_paths = []                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        # process remaining images
        if batch_images:
            features = process_batch(batch_images)
            gallery_features.extend(features)
            gallery_paths.extend(batch_paths)
            identities.extend([identity] * len(batch_paths))

    features = torch.cat(gallery_features)
    # cache results with file info
    torch.save({
        'features': features,
        'identities': identities,
        'files_info': current_info,
        'paths': gallery_paths
    }, cache_file)    
    return features, identities

def process_batch(images):
    features = []
    with torch.no_grad():
        for img in images:
            bgr_tensor = to_input(img).to(device)
            feature, _ = model(bgr_tensor)
            features.append(feature.cpu())
    return features

def recognize_image(image_path, gallery_features, gallery_names, threshold=.35):
    try:
        orig_img = Image.open(image_path).convert('RGB')
        orig_img = upscale_image(orig_img)
        bboxes, faces = align.mtcnn_model.align_multi(orig_img)    
        if not faces:
            return [], [], [], orig_img    
        # process all faces in one batch
        features = process_batch(faces)
        features = torch.cat(features)
        # calculate similarity for all faces at once
        similarity = torch.matmul(features, gallery_features.T)
        max_sim, max_idx = torch.max(similarity, dim=1)
        names = []
        confidences = []
        for sim, idx in zip(max_sim, max_idx):
            if sim > threshold:
                names.append(gallery_names[idx])
            else:
                names.append("Unknown")
            confidences.append(float(sim))        
        return names, confidences, bboxes, orig_img
            
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return [], [], [], None

def draw_box_and_label(image, bbox, name, confidence):
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [int(b) for b in bbox[:4]]  # MTCNN returns [x1, y1, x2, y2, confidence] only take first 4 values
    draw.rectangle([(x1,y1), (x2,y2)], outline='green', width=2)
    label = f"{name} ({confidence:.2f})"
    label_bbox = draw.textbbox((x1, y1-20), label)
    draw.rectangle([(x1, label_bbox[1]), (label_bbox[2], label_bbox[3])], 
                  fill='green')
    draw.text((x1, y1-20), label, fill='white')
    return image

def process_single_image(image_path, gallery_features, gallery_names, threshold=0.25):
    names, confidences, bboxes, orig_img = recognize_image(
        image_path, gallery_features, gallery_names, threshold)
    if len(bboxes) > 0 and orig_img is not None:
        result_img = orig_img
        for name, confidence, bbox in zip(names, confidences, bboxes):
            result_img = draw_box_and_label(result_img, bbox, name, confidence)
        result_img.save('output.png')
        print(f"Processed image saved as output.png")
        for name, confidence in zip(names, confidences):
            print(f"Recognized face as: {name} (Confidence: {confidence})")
    else:
        print("No faces detected")

if __name__ == '__main__':
    start = time.time()
    gallery_features, gallery_names = load_gallery_faces("faces")
    process_single_image("./profiles/2025-02-03/track_2_17-04-25-632251_0.804.png", gallery_features, gallery_names)
    print("Time elapsed:", time.time() - start)
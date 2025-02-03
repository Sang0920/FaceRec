import net
import torch
import os
from face_alignment import align
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
    'ir_101':"pretrained/adaface_ir101_webface12m.ckpt",
    'ir_18':"pretrained/adaface_ir18_casia.ckpt",
}

# def load_pretrained_model(architecture='ir_50'):
#     # load model and pretrained statedict
#     assert architecture in adaface_models.keys()
#     model = net.build_model(architecture)
#     statedict = torch.load(adaface_models[architecture])['state_dict']
#     model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
#     model.load_state_dict(model_statedict)
#     model.eval()
#     return model

# def to_input(pil_rgb_image):
#     np_img = np.array(pil_rgb_image)
#     brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
#     tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
#     return tensor

def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    # Add weights_only=True to fix warning
    statedict = torch.load(adaface_models[architecture], weights_only=False, map_location=device)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    # Convert to numpy array first then to tensor
    brg_array = np.array([brg_img.transpose(2,0,1)])
    tensor = torch.from_numpy(brg_array).float()
    return tensor

if __name__ == '__main__':

    model = load_pretrained_model('ir_50')
    print(model.eval())
    # feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = 'face_alignment/test_images'
    features = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path)
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        features.append(feature)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(f"Similarity score:\n{similarity_scores}")
    
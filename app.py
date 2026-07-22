import gc

def process_images_sequentially(raw_bytes_list: list):
    """Processes images one by one to save RAM. Only generates 1 heatmap."""
    color_imgs = []
    heatmaps = []
    feats_np = []
    
    for i, raw in enumerate(raw_bytes_list):
        nparr = np.frombuffer(raw, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        color_img, ml_img = preprocess_and_crop(img_array)
        color_imgs.append(color_img)
        
        pil_img = Image.fromarray(ml_img)
        x = transform(pil_img).unsqueeze(0).to(device)
        del pil_img, ml_img, img_array, nparr
        
        # 1. Fast forward pass (NO gradients) to get features
        with torch.no_grad():
            feat = densenet(x)
        feats_np.append(feat.cpu().numpy()[0])
        del feat

        # 2. Saliency map (WITH gradients) - ONLY for the first image (CT_A)
        if i == 0:
            x.requires_grad_()
            with torch.enable_grad():
                feat_grad = densenet(x)
                feat_grad.sum().backward()
                saliency, _ = torch.max(x.grad.data.abs(), dim=1)
                saliency = saliency.squeeze().cpu().numpy()
            
            # Process heatmap
            sal = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            sal = cv2.GaussianBlur(sal, (21, 21), 0)
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            
            mask = sal > 0.5
            sal_uint8 = (sal * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            blended = color_img.copy()
            blended[mask] = cv2.addWeighted(color_img, 0.3, heatmap, 0.7, 0)[mask]
            heatmaps.append(Image.fromarray(blended))
            
            # Aggressive memory cleanup
            del x, feat_grad, saliency, sal, mask, sal_uint8, heatmap, blended
            gc.collect()
        else:
            # For the other 6 images, we don't generate a heatmap to save RAM
            heatmaps.append(None)
            del x
            gc.collect()
        
    return np.array(feats_np), color_imgs, heatmaps
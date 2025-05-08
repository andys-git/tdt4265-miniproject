import os

def detect_field_objects(model, frames, folder, conf_thresh):
    frame_dict = {}

    for i, f in enumerate(frames):
        print(f"(object) Processing frame {i + 1} out of {len(frames)} total")
        # Run the model on the image with zero confidence filtering (we'll apply our own thresholds)
        res = model.predict(source=os.path.join(folder, f), conf=0.0, verbose=False)[0]
        # Extract boxes and scores
        boxes   = res.boxes.xyxy.cpu().numpy()    # [[x1, y1, x2, y2], ...]
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs   = res.boxes.conf.cpu().numpy()

        objects = []
        for (x1, y1, x2, y2), cls, cf in zip(boxes, classes, confs):
            label = res.names[cls]
            # Only keep if above the specified threshold
            if cf >= conf_thresh.get(label, 1.0):
                cx = float((x1 + x2) / 2.0)
                cy = float(y2)
                objects.append((label, float(cf), cx, cy))

        frame_dict[f] = {'objects': objects}

    return frame_dict
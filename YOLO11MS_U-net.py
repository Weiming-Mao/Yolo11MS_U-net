import os
import sys
import cv2
import numpy as np
import tempfile
import logging
from collections import defaultdict
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image, ImageDraw, ImageFont

# Import the U-Net class from your provided library
from unet import Unet

# -------------------- Logging Setup --------------------
logging.basicConfig(filename='error_log.txt', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(message)s')

if sys.stdout is None:
    class DummyWriter:
        def write(self, *args, **kwargs):
            pass
    sys.stdout = DummyWriter()
    sys.stderr = DummyWriter()

# -------------------- Model Initialization --------------------
# Model for Stage 1: Graphene grid area extraction
yolo_area = YOLO("area_extract.pt", task="segment")

# Model for Stage 2: Defect detection using U-Net
try:
    unet_defect_detector = Unet()
    # FIX: Manually add the name_classes attribute to the instance
    # This list must match the training order: 0=background, 1=damage, 2=stain
    unet_defect_detector.name_classes = ["background", "damage", "stain"]
except Exception as e:
    logging.error(f"Failed to initialize U-Net model: {e}", exc_info=True)
    print(f"Error: Could not initialize the U-Net model. Please check 'unet.py' and model paths. Error: {e}")
    sys.exit(1)

# -------------------- Image Detection Functions --------------------
def extract_mask_regions(image_path, max_masks=8):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        original_shape = image.shape
        results = yolo_area(source=image, save=False, show_labels=False, show_conf=False)
        masks = results[0].masks.xy
        boxes = results[0].boxes.xyxy.cpu().numpy()
        mask_areas = []
        for i, mask in enumerate(masks):
            m = np.array(mask, dtype=np.int32)
            area = cv2.contourArea(m)
            mask_areas.append((i, area, m))
        mask_areas.sort(key=lambda x: x[1], reverse=True)
        mask_areas = mask_areas[:max_masks]
        total_area = sum(area for (_, area, _) in mask_areas)

        mask_regions, mask_bboxes, mask_masks = {}, {}, {}
        for idx, (i, _, m) in enumerate(mask_areas):
            mask_image = np.full_like(image, 255, dtype=np.uint8)
            cv2.fillPoly(mask_image, [m], (0, 0, 0))
            mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY_INV)[1]
            masked_region = cv2.bitwise_and(image, image, mask=mask_binary)
            white_bg = np.full_like(image, 255, dtype=np.uint8)
            final = np.where(mask_binary[..., None] > 0, masked_region, white_bg)
            x1, y1, x2, y2 = boxes[i].astype(int)
            mask_regions[idx] = final[y1:y2, x1:x2]
            mask_masks[idx] = mask_binary[y1:y2, x1:x2]
            mask_bboxes[idx] = (x1, y1, x2, y2)

        return mask_regions, mask_bboxes, mask_masks, original_shape, total_area
    except Exception:
        logging.error("Error in extract_mask_regions", exc_info=True)
        raise

def cover_mask_regions_with_unet(mask_regions):
    """
    Detects defects in cropped regions using a U-Net model, returning color-coded
    images and statistics on defect count and area.
    """
    try:
        output_regions = {}
        total_mask_counts = {"damage": 0, "stain": 0}
        total_mask_areas = {"damage": 0.0, "stain": 0.0}
        per_image_mask_counts = {}
        per_image_mask_areas = {}

        class_names = unet_defect_detector.name_classes
        colors = unet_defect_detector.colors
        
        damage_idx = class_names.index('damage') if 'damage' in class_names else -1
        stain_idx = class_names.index('stain') if 'stain' in class_names else -1

        original_mix_type = unet_defect_detector.mix_type
        
        for idx, region_bgr in mask_regions.items():
            mask_counts = {"damage": 0, "stain": 0}
            mask_areas = {"damage": 0.0, "stain": 0.0}
            
            if region_bgr.size == 0: continue

            region_pil = Image.fromarray(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB))
            
            # Get the blended image for final visual output by setting mix_type=0
            unet_defect_detector.mix_type = 0
            blended_pil = unet_defect_detector.detect_image(region_pil)
            output_regions[idx] = cv2.cvtColor(np.array(blended_pil), cv2.COLOR_RGB2BGR)
            
            # Get the pure segmentation color map for calculations by setting mix_type=1
            unet_defect_detector.mix_type = 1
            seg_map_pil = unet_defect_detector.detect_image(region_pil)
            seg_map_rgb = np.array(seg_map_pil)

            # Process "damage" defects
            if damage_idx != -1:
                damage_color = colors[damage_idx]
                damage_mask = np.all(seg_map_rgb == damage_color, axis=-1).astype(np.uint8)
                
                area = np.sum(damage_mask)
                if area > 0:
                    mask_areas["damage"] = float(area)
                    num_labels, _, _, _ = cv2.connectedComponentsWithStats(damage_mask, 4, cv2.CV_32S)
                    mask_counts["damage"] = max(0, num_labels - 1)

            # Process "stain" defects
            if stain_idx != -1:
                stain_color = colors[stain_idx]
                stain_mask = np.all(seg_map_rgb == stain_color, axis=-1).astype(np.uint8)

                area = np.sum(stain_mask)
                if area > 0:
                    mask_areas["stain"] = float(area)
                    num_labels, _, _, _ = cv2.connectedComponentsWithStats(stain_mask, 4, cv2.CV_32S)
                    mask_counts["stain"] = max(0, num_labels - 1)

            per_image_mask_counts[idx] = mask_counts
            per_image_mask_areas[idx] = mask_areas
            total_mask_counts["damage"] += mask_counts["damage"]
            total_mask_counts["stain"] += mask_counts["stain"]
            total_mask_areas["damage"] += mask_areas["damage"]
            total_mask_areas["stain"] += mask_areas["stain"]

        # Restore original setting
        unet_defect_detector.mix_type = original_mix_type
        return output_regions, total_mask_counts, per_image_mask_counts, total_mask_areas, per_image_mask_areas
    except Exception:
        logging.error("Error in cover_mask_regions_with_unet", exc_info=True)
        raise


# -------------------- SAHI Related --------------------
yolo11n_model_path = 'coverage_detect.pt'
detection_model = None
def initialize_detection_model():
    global detection_model
    if detection_model is None:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=yolo11n_model_path,
            confidence_threshold=0.5,
            device="cuda:0",
        )
def has_intersection(b1, b2):
    return (b1.minx < b2.maxx and b1.maxx > b2.minx and
            b1.miny < b2.maxy and b1.maxy > b2.miny)
def process_mask_regions(mask_regions):
    try:
        initialize_detection_model()
        out_regs = {}
        total_bbox = defaultdict(int)
        per_bbox = {}
        for idx, reg in mask_regions.items():
            bbox_count = defaultdict(int)
            result = get_sliced_prediction(
                reg, detection_model,
                slice_height=256, slice_width=256,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                postprocess_type='NMS', postprocess_match_threshold=0.1
            )
            preds = result.object_prediction_list
            finals = []
            for p in preds:
                inter = False
                for q in finals:
                    if has_intersection(p.bbox, q.bbox):
                        inter = True
                        if p.score.value > q.score.value:
                            finals.remove(q)
                        break
                if not inter:
                    finals.append(p)
            img = reg.copy()
            for p in finals:
                x1,y1,x2,y2 = map(int,(p.bbox.minx,p.bbox.miny,p.bbox.maxx,p.bbox.maxy))
                if p.category.name=='edge': c=(0,255,0)
                elif p.category.name=='covered': c=(255,0,0)
                else: c=(0,0,255)
                cv2.rectangle(img,(x1,y1),(x2,y2),c,2)
                bbox_count[p.category.name]+=1
                total_bbox[p.category.name]+=1
            out_regs[idx]=img
            per_bbox[idx]=dict(bbox_count)
        return out_regs, dict(total_bbox), per_bbox
    except Exception:
        logging.error("Error in process_mask_regions", exc_info=True)
        raise
def reconstruct_output_image(output_regions, boxes, orig_shape, masks):
    canvas = np.full(orig_shape,255,dtype=np.uint8)
    for idx, reg in output_regions.items():
        if idx not in boxes or idx not in masks: continue
        x1,y1,x2,y2 = boxes[idx]
        th,tw = y2-y1, x2-x1
        rh,rw = reg.shape[:2]
        if (rh,rw)!=(th,tw):
            reg = cv2.resize(reg,(tw,th))
            m = cv2.resize(masks[idx],(tw,th),interpolation=cv2.INTER_NEAREST)
        else:
            m = masks[idx]
        _, m = cv2.threshold(m,1,255,cv2.THRESH_BINARY)
        area = canvas[y1:y2,x1:x2]
        area[m.astype(bool)] = reg[m.astype(bool)]
        canvas[y1:y2,x1:x2] = area
    return canvas

# -------------------- Main Processing Flow (Normal) --------------------
def process_image(image_path):
    try:
        regs, boxes, masks, orig_shape, total_area = extract_mask_regions(image_path)
        dregs, dcounts, _, tmask_areas, _ = cover_mask_regions_with_unet(regs)
        oregs, cover_counts, _ = process_mask_regions(dregs)
        final = reconstruct_output_image(oregs, boxes, orig_shape, masks)

        unc = cover_counts.get('uncovered', 0)
        cov = cover_counts.get('covered', 0)
        edge = cover_counts.get('edge', 0)
        dta = sum(tmask_areas.values())
        hd = (unc + cov + edge) / (total_area - dta) if total_area != dta else 0
        dd = int(dta * hd)
        coverage = (100 * cov) / (cov + unc + dd) if cov + unc + dd > 0 else 0
        defect_pct = (dta / total_area) * 100 if total_area > 0 else 0

        defect_text = (
            f"Damage: {dcounts.get('damage', 0)}\n"
            f"Contamination: {dcounts.get('stain', 0)}\n"
            f"Defective area: {defect_pct:.3f}%"
        )
        cover_text = (
            f"Covered: {cov}\nUncovered: {unc}\nEdge: {edge}\nCoverage: {coverage:.3f}%"
        )

        orig = cv2.imread(image_path)
        return cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), \
               cv2.cvtColor(final, cv2.COLOR_BGR2RGB), defect_text, cover_text
    except Exception:
        logging.error("Error in process_image", exc_info=True)
        raise

# -------------------- Result Combination & File Prep --------------------
def prepare_image_file(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext in ['.png','.tif','.tiff']:
        img = cv2.imread(fp)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg",delete=False)
        cv2.imwrite(tmp.name,img)
        return tmp.name, True
    return fp, False
def create_result_image(orig_img, proc_img, info):
    pil_o = Image.fromarray(orig_img)
    pil_p = Image.fromarray(proc_img)
    th = min(pil_o.height,pil_p.height)
    pil_o = pil_o.resize((int(pil_o.width*th/pil_o.height),th))
    pil_p = pil_p.resize((int(pil_p.width*th/pil_p.height),th))
    bw = pil_o.width + pil_p.width
    bottom = Image.new("RGB",(bw,th),(255,255,255))
    bottom.paste(pil_o,(0,0)); bottom.paste(pil_p,(pil_o.width,0))
    ih = int(th*0.2)
    info_img = Image.new("RGB",(bw,ih),(255,255,255))
    dr = ImageDraw.Draw(info_img)
    row_h = ih/2
    try:
        fsz = int(row_h*0.5)
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",size=fsz)
    except:
        font = ImageFont.load_default()
    cw = bw/3
    row1 = [("Filename", "Filename"), ("Coverage (%)", "Coverage"), ("Uncovered", "Uncovered")]
    row2 = [("Covered", "Covered"), ("Damage", "Damage"), ("Contamination", "Contamination")]
    tb = dr.textbbox((0,0),"A",font=font)
    tht = tb[3]-tb[1]
    y1 = (row_h-tht)/2
    for i,(d,k) in enumerate(row1):
        txt = f"{d}: {info.get(k,'--')}"
        bb = dr.textbbox((0,0),txt,font=font); tw = bb[2]-bb[0]
        x = cw*i + (cw-tw)/2
        clr = (139,0,0) if k == "Coverage" else (0,0,0)
        dr.text((x,y1),txt,font=font,fill=clr)
    y2 = row_h + (row_h-tht)/2
    for i,(d,k) in enumerate(row2):
        txt = f"{d}: {info.get(k,'--')}"
        bb = dr.textbbox((0,0),txt,font=font); tw = bb[2]-bb[0]
        x = cw*i + (cw-tw)/2
        dr.text((x,y2),txt,font=font,fill=(0,0,0))
    final = Image.new("RGB",(bw,ih+th),(255,255,255))
    final.paste(info_img,(0,0)); final.paste(bottom,(0,ih))
    return np.array(final)

# -------------------- PyQt5 GUI --------------------
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QGroupBox, QInputDialog
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QSize
class DefectDetectionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphene Grid Defect Detection & Coverage Quantification")
        self.setWindowIcon(QIcon('zjuico_256x256.ico'))
        self.resize(1200,800)
        self.folder_images = []
        self.current_index = 0
        self.folder_results = []
        self.setup_ui()
    def setup_ui(self):
        self.title_label = QLabel("Graphene Grid Defect Detection & Coverage Quantification")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFont(QFont("Arial",20))
        self.info_group = QGroupBox()
        self.info_group.setStyleSheet("QGroupBox{border:2px solid gray;border-radius:5px;}")
        self.info_labels = {}
        titles = ["Filename","Coverage","Uncovered","Covered","Damage","Contamination"]
        lay = QGridLayout()
        for i,t in enumerate(titles):
            lt = QLabel(f"{t}:"); lt.setFont(QFont("Arial",14))
            lt.setStyleSheet("color:darkred;" if t=="Coverage" else "color:black;")
            lv = QLabel("--"); lv.setFont(QFont("Arial",14))
            lv.setStyleSheet("color:darkred;" if t=="Coverage" else "color:black;")
            self.info_labels[t] = lv
            lay.addWidget(lt,0,i,alignment=Qt.AlignCenter)
            lay.addWidget(lv,1,i,alignment=Qt.AlignCenter)
        self.info_group.setLayout(lay)
        self.orig_lbl = QLabel("Original Image"); self.orig_lbl.setFont(QFont("Arial",12))
        self.orig_lbl.setAlignment(Qt.AlignCenter); self.orig_lbl.setStyleSheet("border:2px solid black;")
        self.orig_lbl.setFixedSize(640,480)
        self.proc_lbl = QLabel("Detection Result"); self.proc_lbl.setFont(QFont("Arial",12))
        self.proc_lbl.setAlignment(Qt.AlignCenter); self.proc_lbl.setStyleSheet("border:2px solid black;")
        self.proc_lbl.setFixedSize(640,480)
        img_lay = QHBoxLayout(); img_lay.addStretch()
        img_lay.addWidget(self.orig_lbl); img_lay.addSpacing(20)
        img_lay.addWidget(self.proc_lbl); img_lay.addStretch()
        self.btn_single = QPushButton("Load Single Image"); self.btn_single.setFont(QFont("Arial",16))
        self.btn_single.setMinimumHeight(60); self.btn_single.clicked.connect(self.load_single_image)
        self.btn_folder = QPushButton("Load Image Folder"); self.btn_folder.setFont(QFont("Arial",16))
        self.btn_folder.setMinimumHeight(60); self.btn_folder.clicked.connect(self.load_image_folder)
        self.btn_special = QPushButton("Special Image Processing"); self.btn_special.setFont(QFont("Arial",16))
        self.btn_special.setMinimumHeight(60); self.btn_special.clicked.connect(self.load_special_folder)
        btn_lay = QHBoxLayout(); btn_lay.addStretch()
        btn_lay.addWidget(self.btn_single); btn_lay.addSpacing(30)
        btn_lay.addWidget(self.btn_folder); btn_lay.addSpacing(30)
        btn_lay.addWidget(self.btn_special); btn_lay.addStretch()
        main = QVBoxLayout(); main.setContentsMargins(20,10,20,10); main.setSpacing(15)
        main.addWidget(self.title_label); main.addWidget(self.info_group)
        main.addLayout(img_lay); main.addLayout(btn_lay)
        self.setLayout(main)
    def load_single_image(self):
        fp,_ = QFileDialog.getOpenFileName(self,"Select Image File","","Image Files (*.png;*.jpg;*.jpeg;*.tif;*.tiff)")
        if fp:
            self.process_and_show(fp)
    def load_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self,"Select Image Folder")
        if not folder: return
        self.folder_images = [
            os.path.join(folder,f) for f in os.listdir(folder)
            if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))
        ]
        if not self.folder_images:
            QMessageBox.information(self,"Info","No image files found in this folder.")
            return
        self.current_index = 0; self.folder_results = []; self.folder_dir = folder
        self.process_folder_image()
    def process_folder_image(self):
        if self.current_index >= len(self.folder_images):
            self.save_results_txt(self.folder_dir); return
        fp = self.folder_images[self.current_index]
        nf, tmp = prepare_image_file(fp)
        try:
            orig, proc, d_txt, c_txt = process_image(nf)
        except Exception as e:
            QMessageBox.critical(self,"Processing Error",str(e)); return
        if tmp: os.remove(nf)
        self._update_info_and_images(fp, d_txt, c_txt, orig, proc)
        self.folder_results.append(self._gather_info_dict(fp))
        info = self.folder_results[-1]
        combo = create_result_image(orig, proc, info)
        rd = os.path.join(self.folder_dir,"results"); os.makedirs(rd,exist_ok=True)
        bn = os.path.splitext(os.path.basename(fp))[0]
        out = os.path.join(rd,f"{bn}_result.jpg")
        cv2.imwrite(out, cv2.cvtColor(combo,cv2.COLOR_RGB2BGR))
        self.current_index += 1
        QTimer.singleShot(500, self.process_folder_image)
    def load_special_folder(self):
        folder = QFileDialog.getExistingDirectory(self,"Select Special Image Folder")
        if not folder: return
        self.folder_images = [
            os.path.join(folder,f) for f in os.listdir(folder)
            if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))
        ]
        if not self.folder_images:
            QMessageBox.information(self,"Info","No image files found in this folder.")
            return
        self.current_index = 0; self.folder_results = []; self.folder_dir = folder
        self.process_special_folder_image()
    def process_special_folder_image(self):
        if self.current_index >= len(self.folder_images):
            self.save_results_txt(self.folder_dir); return
        fp = self.folder_images[self.current_index]
        n, ok = QInputDialog.getInt(
            self, "Select Number of Curled Regions",
            f"File: {os.path.basename(fp)}\nPlease enter n (1-7):",
            min=1, max=7, step=1
        )
        if not ok:
            self.current_index += 1
            QTimer.singleShot(100, self.process_special_folder_image)
            return
        nf, tmp = prepare_image_file(fp)
        try:
            orig, proc, d_txt, s_txt = self.process_special_single(nf, n)
        except Exception as e:
            QMessageBox.critical(self,"Processing Error",str(e)); return
        if tmp: os.remove(nf)
        self._update_info_and_images(fp, d_txt, s_txt, orig, proc)
        rec = self._gather_info_dict(fp)
        self.folder_results.append(rec)
        combo = create_result_image(orig, proc, rec)
        rd = os.path.join(self.folder_dir, "results"); os.makedirs(rd, exist_ok=True)
        bn = os.path.splitext(os.path.basename(fp))[0]
        cv2.imwrite(os.path.join(rd, f"{bn}_result.jpg"),
                      cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
        self.current_index += 1
        QTimer.singleShot(500, self.process_special_folder_image)

    def process_special_single(self, image_path, n):
        regs, boxes, masks, osh, ta = extract_mask_regions(image_path)
        dregs, dcounts, _, tmask_areas, _ = cover_mask_regions_with_unet(regs)
        oregs, ccnts, per_cnts = process_mask_regions(dregs)
        final_bgr = reconstruct_output_image(oregs, boxes, osh, masks)
        proc_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
        orig_bgr = cv2.imread(image_path); orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        densities = []
        for idx, counts in per_cnts.items():
            holes = sum(counts.values())
            area_px = cv2.countNonZero(masks[idx])
            d = holes / area_px if area_px > 0 else 0
            densities.append((idx, d, holes, area_px, counts))
        densities.sort(key=lambda x: x[1])
        low = densities[:n]; high = densities[n:]
        avg_d = sum(x[1] for x in high) / len(high) if high else 0
        total_holes = 0.0; cov_cnt = ccnts.get("covered", 0)
        for item in densities:
            idx, d, h, ap, _ = item
            if item in high:
                total_holes += h
            else:
                total_holes += avg_d * ap
        coverage_special = (cov_cnt / total_holes * 100) if total_holes > 0 else 0.0
        defect_pct = (sum(tmask_areas.values()) / ta * 100) if ta > 0 else 0.0
        d_txt = (
            f"Damage: {dcounts.get('damage',0)}\n"
            f"Contamination: {dcounts.get('stain',0)}\n"
            f"Defective area: {defect_pct:.3f}%"
        )
        s_txt = (
            f"Covered: {ccnts.get('covered',0)}\n"
            f"Uncovered: {ccnts.get('uncovered',0)}\n"
            f"Edge: {ccnts.get('edge',0)}\n"
            f"Coverage: {coverage_special:.3f}%"
        )
        return orig_rgb, proc_rgb, d_txt, s_txt

    def _update_info_and_images(self, fp, d_txt, c_txt, ori, proc):
        self.info_labels["Filename"].setText(os.path.basename(fp))
        def pv(txt, key):
            for ln in txt.splitlines():
                if key in ln:
                    return ln.split(":")[1].strip()
            return "--"
        self.info_labels["Coverage"].setText(pv(c_txt, "Coverage"))
        self.info_labels["Uncovered"].setText(pv(c_txt, "Uncovered"))
        self.info_labels["Covered"].setText(pv(c_txt, "Covered"))
        self.info_labels["Damage"].setText(pv(d_txt, "Damage"))
        self.info_labels["Contamination"].setText(pv(d_txt, "Contamination"))
        self.show_image(self.orig_lbl, ori)
        self.show_image(self.proc_lbl, proc)
    def _gather_info_dict(self, fp):
        return {
            "Filename": os.path.basename(fp),
            "Coverage": self.info_labels["Coverage"].text(),
            "Uncovered": self.info_labels["Uncovered"].text(),
            "Covered": self.info_labels["Covered"].text(),
            "Damage": self.info_labels["Damage"].text(),
            "Contamination": self.info_labels["Contamination"].text(),
        }
    def save_results_txt(self, folder_path):
        rd = os.path.join(folder_path, "results"); os.makedirs(rd, exist_ok=True)
        txt_path = os.path.join(rd, "detection_results.txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("Filename,Coverage,Uncovered,Covered,Damage,Contamination\n")
                for r in self.folder_results:
                    f.write(
                        f"{r['Filename']},{r['Coverage']},{r['Uncovered']},"
                        f"{r['Covered']},{r['Damage']},{r['Contamination']}\n"
                    )
            QMessageBox.information(self, "Complete",
                                    f"Processed {len(self.folder_results)} images. Results saved to:\n{txt_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))
    def process_and_show(self, fp):
        data = np.fromfile(fp, dtype=np.uint8)
        if cv2.imdecode(data, cv2.IMREAD_COLOR) is None:
            QMessageBox.critical(self, "Error", f"Failed to load image: {fp}")
            return
        try:
            ori, proc, d_txt, c_txt = process_image(fp)
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", str(e))
            return
        self._update_info_and_images(fp, d_txt, c_txt, ori, proc)
    def show_image(self, label, img):
        h, w, _ = img.shape
        bpm = 3 * w
        q = QImage(img.data, w, h, bpm, QImage.Format_RGB888)
        pm = QPixmap.fromImage(q).scaled(
            label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pm)
    def resizeEvent(self, event):
        self.adjust_layout(event.size())
        super().resizeEvent(event)
    def adjust_layout(self, size: QSize):
        bw, bh = 1280, 800
        s = min(size.width() / bw, size.height() / bh)
        self.title_label.setFont(QFont("Arial", int(20 * s)))
        f = QFont("Arial", int(14 * s))
        for lbl in self.info_labels.values():
            lbl.setFont(f)
        bf = QFont("Arial", int(16 * s))
        for btn in (self.btn_single, self.btn_folder, self.btn_special):
            btn.setFont(bf)
            btn.setFixedHeight(int(60 * s))
        nw = int(640 * s)
        nh = int(nw * 3 / 4)
        self.orig_lbl.setFixedSize(nw, nh)
        self.proc_lbl.setFixedSize(nw, nh)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DefectDetectionGUI()
    win.show()
    sys.exit(app.exec_())
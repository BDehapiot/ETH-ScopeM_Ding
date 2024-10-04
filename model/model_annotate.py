#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# Napari
import napari
from napari.layers.labels.labels import Labels

# Qt
from qtpy.QtGui import QFont
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QPushButton, QLineEdit, QRadioButton, QGroupBox, 
    QVBoxLayout, QHBoxLayout, QWidget, QLabel
    )

# Skimage
from skimage.measure import label
from skimage.segmentation import find_boundaries, expand_labels, flood_fill

#%% Comments ------------------------------------------------------------------

'''
Todo
- Something wrong with font size in new Napari version
- Reset view on first image?
- RGB image support
- Parameter handling (default, autosaved etc)
'''

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd(), "train")

# Parameters
edit = True
randomize = True
np.random.seed(42)
brush_size = 10

#%% Class : Painter() ---------------------------------------------------------

class Painter:
    
    def __init__(self, train_path, edit=True, randomize=True):
        self.train_path = train_path
        self.edit = edit
        self.randomize = randomize
        self.idx = 0
        self.init_paths()
        self.init_images()
        self.init_viewer()
        self.open_image()
        
        # Timers
        self.next_brush_size_timer = QTimer()
        self.next_brush_size_timer.timeout.connect(self.next_brush_size)
        self.prev_brush_size_timer = QTimer()
        self.prev_brush_size_timer.timeout.connect(self.prev_brush_size)
        
#%% Initialize ----------------------------------------------------------------
        
    def init_paths(self):
        self.img_paths, self.msk_paths = [], []
        for img_path in self.train_path.iterdir():
            if "mask" not in img_path.name:
                self.img_paths.append(img_path)
                self.msk_paths.append(
                    Path(str(img_path).replace(".tif", "_mask.tif")))
        if self.randomize:
            permutation = np.random.permutation(len(self.img_paths))
            self.img_paths = [self.img_paths[i] for i in permutation]
            self.msk_paths = [self.msk_paths[i] for i in permutation]
            
    def update_msk_suffix(self):
        self.msk_suffix = self.line_msk_suffix.text() 
        
    def update_msk_paths(self):
        for i, msk_path in enumerate(self.msk_paths):
            self.msk_paths[i] = Path(str(msk_path).replace(
                ".tif", f"{self.msk_suffix}.tif"))
            
    def init_images(self):
        self.imgs, self.msks = [], []
        for img_path, msk_path in zip(self.img_paths, self.msk_paths):
            img = io.imread(img_path)
            if msk_path.exists() and self.edit:   
                msk = io.imread(msk_path)
            elif not msk_path.exists():
                msk = np.zeros_like(img, dtype="uint8")
            self.imgs.append(img)
            self.msks.append(msk)

    def init_viewer(self):
               
        # Setup viewer
        self.viewer = napari.Viewer()
        self.viewer.add_image(self.imgs[0], name="image")
        self.viewer.add_labels(self.msks[0], name="mask")
        self.viewer.layers["mask"].brush_size = brush_size
        self.viewer.layers["mask"].mode = 'paint'
        
        # Contrast limits
        val = np.hstack([img.ravel() for img in self.imgs])
        self.contrast_limits = (
            np.quantile(val, 0.001), np.quantile(val, 0.999))
        self.viewer.layers["image"].contrast_limits = self.contrast_limits
        self.viewer.layers["image"].gamma = 0.66

        # Create "Actions" menu
        self.act_group_box = QGroupBox("Actions")
        act_group_layout = QVBoxLayout()
        self.btn_next_image = QPushButton("Next Image")
        self.btn_prev_image = QPushButton("Previous Image")
        self.btn_save_mask = QPushButton("Save Mask")
        act_group_layout.addWidget(self.btn_next_image)
        act_group_layout.addWidget(self.btn_prev_image)
        act_group_layout.addWidget(self.btn_save_mask)
        self.act_group_box.setLayout(act_group_layout)
        self.btn_next_image.clicked.connect(self.next_image)
        self.btn_prev_image.clicked.connect(self.prev_image)
        self.btn_save_mask.clicked.connect(self.save_mask)
        
        # Create "Segmentation" menu
        self.seg_group_box = QGroupBox("Segmentation")
        seg_group_layout = QHBoxLayout()
        self.rad_semantic = QRadioButton("Semantic")
        self.rad_instance = QRadioButton("Instance")
        self.rad_instance.setChecked(True)
        seg_group_layout.addWidget(self.rad_semantic)
        seg_group_layout.addWidget(self.rad_instance)
        self.seg_group_box.setLayout(seg_group_layout)
        
        # Create "Options" menu
        self.opt_group_box = QGroupBox("Options")
        opt_group_layout = QVBoxLayout()
        self.line_msk_suffix_label = QLabel("Mask Suffix :")
        self.line_msk_suffix = QLineEdit("")
        self.msk_suffix = self.line_msk_suffix.text() 
        opt_group_layout.addWidget(self.line_msk_suffix_label)
        opt_group_layout.addWidget(self.line_msk_suffix)
        self.opt_group_box.setLayout(opt_group_layout)
        self.line_msk_suffix.textChanged.connect(self.update_msk_suffix)

        # Create texts
        self.info_image = QLabel()
        self.info_image.setFont(QFont("Consolas"))
        self.info_stats = QLabel()
        self.info_stats.setFont(QFont("Consolas"))
        self.info_short = QLabel()
        self.info_short.setFont(QFont("Consolas"))
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.act_group_box)
        self.layout.addWidget(self.seg_group_box)
        self.layout.addWidget(self.opt_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_image)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_stats)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_short)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 
        
#%% Shortcuts -----------------------------------------------------------------
        
        # Viewer

        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_image()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_image()
            
        @Labels.bind_key("Enter", overwrite=True)
        def save_mask_key(viewer):
            self.save_mask() 

        # @self.viewer.bind_key("Enter", overwrite=True)
        # def save_mask_key(viewer):
        #     self.save_mask() 

        @self.viewer.bind_key("0", overwrite=True)
        def pan_switch_key0(viewer):
            self.pan()
            yield
            self.paint()
            
        @self.viewer.bind_key("Space", overwrite=True)
        def pan_switch_key1(viewer):
            self.pan()
            yield
            self.paint()
            
        @self.viewer.bind_key("End", overwrite=True)
        def hide_labels_key(viewer):
            self.hide_labels()
            yield
            self.show_labels()
            
        @self.viewer.bind_key("Backspace", overwrite=True)
        def reset_view_key(viewer):
            self.reset_view()
            
        # Paint
            
        @self.viewer.bind_key("Down", overwrite=True)
        def prev_label_key(viewer):
            self.prev_label()
            
        @self.viewer.bind_key("Up", overwrite=True)
        def next_label_key(viewer):
            self.next_label()
            
        @self.viewer.bind_key("Right", overwrite=True)
        def next_brush_size_key(viewer):
            self.next_brush_size() 
            # time.sleep(125 / 1000) 
            self.next_brush_size_timer.start(30) 
            yield
            self.next_brush_size_timer.stop()
            
        @self.viewer.bind_key("Left", overwrite=True)
        def prev_brush_size_key(viewer):
            self.prev_brush_size() 
            # time.sleep(125 / 1000) 
            self.prev_brush_size_timer.start(30) 
            yield
            self.prev_brush_size_timer.stop()
                       
        @self.viewer.mouse_drag_callbacks.append
        def mouse_actions(viewer, event):
            
            if "Control" in event.modifiers:
                if event.button == 1:
                    self.fill()
                    yield
                    self.paint()
                elif event.button == 2:
                    position = event.position
                    self.erase()
                    self.erase_label(position)
                    yield
                    self.paint()
            
            if "Shift" in event.modifiers:      
                self.pick()
                yield
                self.paint()
            else:
                if event.button == 2:
                    self.erase()
                    yield
                    self.paint()

#%% Function(s) shortcuts -----------------------------------------------------
                
    # Viewer    

    def prev_image(self):
        if self.idx > 0:
            self.idx -= 1
            self.open_image()
        
    def next_image(self):
        if self.idx < len(self.imgs) - 1:
            self.idx += 1
            self.open_image()
        
    def pan(self):
        self.viewer.layers["mask"].mode = "pan_zoom"
        
    def show_labels(self):
        self.viewer.layers["mask"].visible = True
    
    def hide_labels(self):
        self.viewer.layers["mask"].visible = False  
        
    def reset_view(self):
        self.viewer.reset_view()

    # Paint
    
    def prev_label(self):
        if self.viewer.layers["mask"].selected_label > 1:
            self.viewer.layers["mask"].selected_label -= 1 
            
    def next_label(self):
        self.viewer.layers["mask"].selected_label += 1 
               
    def prev_brush_size(self):
        if self.viewer.layers["mask"].brush_size > 1:
            self.viewer.layers["mask"].brush_size -= 1
        
    def next_brush_size(self): 
        self.viewer.layers["mask"].brush_size += 1

    def paint(self):
        self.viewer.layers["mask"].mode = "paint"
        
    def fill(self):
        self.viewer.layers["mask"].mode = "fill"
            
    def erase(self):
        self.viewer.layers["mask"].mode = "erase"
        
    def pick(self):
        self.viewer.layers["mask"].mode = "pick"

    def erase_label(self, position):
        position = tuple((int(position[0]), int(position[1])))
        self.viewer.layers["mask"].data = flood_fill(
            self.viewer.layers["mask"].data, position, 0)
        
#%% Function(s) main ----------------------------------------------------------
        
    def next_free_label(self):
        self.viewer.layers["mask"].selected_label = np.max(
            self.viewer.layers["mask"].data + 1)

    def open_image(self):
        self.viewer.layers["image"].data = self.imgs[self.idx].copy()
        self.viewer.layers["mask"].data = self.msks[self.idx].copy()
        self.next_free_label()
        self.get_info_text()
        self.reset_view()
               
    def solve_labels(self):
        msk = self.viewer.layers["mask"].data
        msk_obj = msk.copy()
        msk_obj[find_boundaries(msk) == 1] = 0
        msk_obj = label(msk_obj, connectivity=1)
        msk_obj = expand_labels(msk_obj)
        msk_obj[msk == 0] = 0
        self.viewer.layers["mask"].data = msk_obj
        
    def save_mask(self):
        if self.rad_instance.isChecked():
            self.solve_labels()
            self.next_free_label()
        msk = self.viewer.layers["mask"].data
        self.update_msk_paths()
        msk_path = self.msk_paths[self.idx]
        self.msks[self.idx] = msk
        io.imsave(msk_path, msk, check_contrast=False) 
        self.get_info_text()

#%% Function(s) info ----------------------------------------------------------
        
    def get_stats(self):
        msk = self.viewer.layers["mask"].data
        msk_obj = label(msk > 0 ^ find_boundaries(msk), connectivity=1)
        self.nObjects = np.maximum(0, len(np.unique(msk_obj)) - 1)
        self.nLabels = np.maximum(0, len(np.unique(msk)) - 1)
        self.minLabel = np.min(msk)
        self.maxLabel = np.max(msk)

    def get_info_text(self):
               
        def shorten_filename(name, max_length=32):
            if len(name) > max_length:
                parts = name.split('_')
                if "mask" not in name:
                    return parts[0] + "..." + parts[-1]
                else:
                    return parts[0] + "..." + parts[-2] + "_" + parts[-1]
            else:
                return name
            
        def set_style(color, size, weight, decoration):
            return (
                " style='"
                f"color: {color};"
                f"font-size: {size}px;"
                f"font-weight: {weight};"
                f"text-decoration: {decoration};"
                "'"
                )

        self.get_stats()
        img_path = self.img_paths[self.idx]
        msk_path = self.msk_paths[self.idx]
        img_name = img_path.name    
        if msk_path.exists() and edit:
            msk_name = msk_path.name 
        elif not msk_path.exists():
            msk_name = "None"
        img_name = shorten_filename(img_name, max_length=32)
        msk_name = shorten_filename(msk_name, max_length=32)

        # Set styles (Titles)
        style0 = set_style("White", 10, "normal", "underline")
        # Set styles (Filenames)
        style1 = set_style("Khaki", 10, "normal", "none")
        # Set styles (Legend)
        style2 = set_style("LightGray", 10, "normal", "none")
        # Set styles (Values)
        style3 = set_style("LightSteelBlue", 10, "normal", "none")
        # Set styles (Shortcuts)
        style4 = set_style("BurlyWood", 10, "normal", "none")
        spacer = "&nbsp;"

        self.info_image.setText(
            f"<p{style0}>Image/Mask<br><br>"
            f"<span{style1}>{img_name}</span><br>"
            f"<span{style1}>{msk_name}</span>"
            )
            
        self.info_stats.setText(
            f"<p{style0}>Statistics<br><br>"
            f"<span{style2}>- n of Object(s)  {spacer * 1}:</span>"
            f"<span{style3}> {self.nObjects}</span><br>"
            f"<span{style2}>- n of Label(s)   {spacer * 2}:</span>"
            f"<span{style3}> {self.nLabels}</span><br>"
            f"<span{style2}>- min/max Label   {spacer * 2}:</span>"
            f"<span{style3}> {self.minLabel}/{self.maxLabel}</span>"
            )
        
        self.info_short.setText(
            f"<p{style0}>Shortcuts<br><br>"            
            f"<span{style2}>- Save Mask       {spacer * 6}:</span>"
            f"<span{style4}> Enter</span><br>"            
            f"<span{style2}>- Reset View      {spacer * 5}:</span>"
            f"<span{style4}> Backspace</span><br>"
            f"<span{style2}>- Pan Image       {spacer * 6}:</span>"
            f"<span{style4}> Spacebar</span><br>"            
            f"<span{style2}>- Hide Labels     {spacer * 4}:</span>"
            f"<span{style4}> End</span><br>"            
            f"<span{style2}>- Next/Prev Image {spacer * 0}:</span>"
            f"<span{style4}> Page[Up/Down]</span><br>"
            f"<span{style2}>- Next/Prev Label {spacer * 0}:</span>"
            f"<span{style4}> Arrow[Up/Down]</span><br>"                        
            f"<span{style2}>- Decr/Incr Brush {spacer * 0}:</span>"
            f"<span{style4}> Arrow[Left/Right]</span><br>"                        
            f"<span{style2}>- Paint/Erase     {spacer * 4}:</span>"
            f"<span{style4}> Mouse[left/Right]</span><br>"           
            f"<span{style2}>- Fill Object     {spacer * 4}:</span>"
            f"<span{style4}> Ctrl+Mouse[left]</span><br>"            
            f"<span{style2}>- Delete Object   {spacer * 2}:</span>"
            f"<span{style4}> Ctrl+Mouse[right]</span><br>"          
            f"<span{style2}>- Pick Label      {spacer * 5}:</span>"
            f"<span{style4}> Shift+Mouse[Left]</span><br>"
            )
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    painter = Painter(train_path, edit=edit, randomize=randomize)
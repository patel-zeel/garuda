import os
import cv2
from glob import glob
import numpy as np
from ipyleaflet import GeomanDrawControl
import leafmap
from copy import deepcopy
import geojson
from ipywidgets import Button, Label, HBox, Dropdown, SelectionSlider, RadioButtons
from IPython.display import display
from garuda.base import geo_to_webm_pixel, webm_pixel_to_geo, xywhr2xyxyxyxy
from garuda.box import OBBLabel
from shapely.geometry import Polygon

class AnnotationTool:
    def __init__(self, labels, classes, zoom, cache_dir, clear_cache=False):
        self.original_labels = deepcopy(labels)
        self.labels = deepcopy(labels)
        self.classes = classes
        self.zoom = zoom
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if clear_cache:
            label_files = glob(os.path.join(self.cache_dir, "label_*.geojson"))
            for label_file in label_files:
                os.remove(label_file)
                
        # initialize
        self.index = 0
        
        # Map
        self.m = leafmap.Map(center=(27, 77), zoom=self.zoom)
        # self.m.add_basemap("Esri.WorldImagery")
        self.m.add_tile_layer("https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/wmts/1.0.0/default028mm/mapserver/tile/32553/{z}/{y}/{x}", name="Esri 2024", attribution="Esri")
        self.m.remove_control(self.m.draw_control)
        self.draw_control = GeomanDrawControl(position='topright',  polyline={}, circle={}, circlemarker={}, marker={}, polygon={}, cut=False)
        
        def on_draw(*args, **kwargs):
            self.status_label.value = "Submit the label to update it."
            self.disable_buttons()
            
        self.draw_control.on_draw(on_draw)
        
        self.draw_control.rectangle = {
        "pathOptions": {
             "fillColor": "#fca45d",
             "color": "#fca45d",
             "fillOpacity": 0.0
         }
         }
        
        self.m.add_control(self.draw_control)
        
        # Interface elements
        # current label
        self.show_label = Label(f"Label {self.index+1}/{len(labels)}")
        
        # status label
        self.status_label = Label("")

        # next_button
        self.next_button = Button(description="next")
        self.next_button.on_click(self.next_button_clicked)

        # previous_button
        self.previous_button = Button(description="previous")
        self.previous_button.on_click(self.previous_button_clicked)
        
        # submit button
        self.submit_button = Button(description="submit")
        self.submit_button.on_click(self.submit_button_clicked)
        
        # reset button
        self.reset_button = Button(description="reset_current_label")
        self.reset_button.on_click(self.reset_button_clicked)
        
        # classes dropdown
        self.classes_dropdown = Dropdown(options=self.classes)
        self.classes_dropdown.on_trait_change(self.on_dropdown_change, 'value')
        
        display(self.show_label)
        display(self.status_label)
        display(HBox([self.submit_button, self.previous_button, self.next_button, self.reset_button, self.classes_dropdown]))
        display(self.m)
        
        # initialize
        loaded_from_cache = self.show_current_label()
        while loaded_from_cache and self.index < len(self.labels) - 1:
            self.next_button_clicked()
            loaded_from_cache = self.show_current_label()
        
    def show_current_label(self):
        self.disable_buttons()
        loaded_from_cache = False
        if os.path.exists(f"{self.cache_dir}/label_{self.index}.geojson"):
            with open(f"{self.cache_dir}/label_{self.index}.geojson", "r") as f:
                data = f.read().strip()
            if data == "Empty_label":
                self.labels[self.index] = None
            else:
                feature = geojson.loads(data)['features'][0]
                self.labels[self.index] = OBBLabel.from_geojson(feature)
            self.enable_buttons()  # allow to move around if label is already present
            loaded_from_cache = True

        label = self.labels[self.index]
        if label is None:
            self.draw_control.data = []
            self.status_label.value = "No label available."
            original_label = self.original_labels[self.index]
            self.m.set_center(original_label.properties['center_lon'], original_label.properties['center_lat'], zoom=self.zoom)
            self.enable_buttons()  # allow to move around if label is empty
        else:
            if loaded_from_cache:
                self.status_label.value = "Label loaded from cache. Submit only to make changes."
            else:
                self.status_label.value = "Label is a valid polygon. Make changes if needed and submit."
            # set pov
            self.m.set_center(label.properties['center_lon'], label.properties['center_lat'], zoom=self.zoom)
            
            # show current label
            feature = label.to_geojson(source=None, task_name=None)
            
            self.draw_control.data = [] # first clear the existing data to trigger the changes in GUI
            feature['properties']['style'] = {'color': self.get_color(feature), 'fillColor': self.get_color(feature), 'fillOpacity': 0.0}
            self.draw_control.data = [feature]
            self.classes_dropdown.value = feature['properties']['class_name']
        
        # update label
        self.show_label.value = f"Label {self.index+1}/{len(self.labels)}"
        return loaded_from_cache
        
    def disable_buttons(self):
        self.next_button.disabled = True
        # self.previous_button.disabled = True
    
    def enable_buttons(self):
        self.next_button.disabled = False
        # self.previous_button.disabled = False
        
    @staticmethod
    def get_color(feature):
        class_name = feature['properties']['class_name']
        if class_name == "CFCBK":
            # red
            return "#ff0000"
        elif class_name == "FCBK":
             # orange
            return "#ffa500"
        elif class_name == "Zigzag":
            # green
            return "#00ff00"
        else:
            # blue
            return "#0000ff"
        
    def submit_button_clicked(self, *args, **kwargs):
        if len(self.draw_control.data) == 0:
            self.labels[self.index] = None
            # remove label from cache
            cache_path = f"{self.cache_dir}/label_{self.index}.geojson"
            with open(cache_path, "w") as f:
                f.write("Empty_label")
            self.enable_buttons() # allow to move around if label is empty
            return
        
        feature = self.draw_control.data[-1]
        try:
            assert feature['geometry']['type'] == 'Polygon'
        except AssertionError:
            if feature['geometry']['type'] != 'Polygon':
                self.status_label.value = "Invalid label. Please correct it or delete it."
                return
        
        coords = []
        for lon, lat in feature['geometry']['coordinates'][0]:
            x, y = geo_to_webm_pixel(lat, lon, self.zoom)
            coords.append([x, y])
        coords = np.array(coords, dtype=np.float32)
        
        (x, y), (w, h), r = cv2.minAreaRect(coords)
        r = np.deg2rad(r)
        rect = xywhr2xyxyxyxy(np.array([x, y, w, h, r]))
        
        coords = []
        for pair in rect:
            lat, lon = webm_pixel_to_geo(pair[0], pair[1], self.zoom)
            coords.append([lon, lat])
        poly = Polygon(coords)
        
        feature['geometry']['coordinates'] = [list(poly.exterior.coords)]
        feature['properties']['source'] = 'hand_validated'
        feature['properties']['task_name'] = 'hand_validation'
        feature['properties']['class_name'] = self.classes_dropdown.value
        feature['properties']['style'] = {'color': self.get_color(feature), 'fillColor': self.get_color(feature), 'fillOpacity': 0.0}
        self.labels[self.index] = OBBLabel.from_geojson(feature)
        self.cache_label()
        self.show_current_label()
        self.enable_buttons()
        self.next_button_clicked()
        
    def cache_label(self):
        cache_path = f"{self.cache_dir}/label_{self.index}.geojson"
        feature = self.labels[self.index].to_geojson(source=None, task_name=None)
        collection = geojson.FeatureCollection([feature])
        with open(cache_path, "w") as f:
            geojson.dump(collection, f)
            
    def reset_button_clicked(self, *args, **kwargs):
        original_label = self.original_labels[self.index]
        self.labels[self.index] = original_label
        feature = self.labels[self.index].to_geojson(source=None, task_name=None)
        self.labels[self.index].properties['style'] = {'color': self.get_color(feature), 'fillColor': self.get_color(feature), 'fillOpacity': 0.0}
        self.cache_label()
        self.show_current_label()
    
    def on_dropdown_change(self, old, new):
        if new != self.labels[self.index].properties['class_name']:
            self.labels[self.index].properties['class_name'] = new
            feature = self.labels[self.index].to_geojson(source=None, task_name=None)
            self.labels[self.index].properties['style'] = {'color': self.get_color(feature), 'fillColor': self.get_color(feature), 'fillOpacity': 0.0}
            self.cache_label()
            self.show_current_label()
    
    def next_button_clicked(self, *args, **kwargs):
        # show next label
        if self.index >= (len(self.labels) - 1):
            pass # do nothing
        else:
            self.index += 1
        self.show_current_label()
    
    def previous_button_clicked(self, *args, **kwargs):
        # show next label
        if self.index <= 0:
            pass # do nothing
        else:
            self.index -= 1
        self.show_current_label()
        
    def to_geojson(self):
        features = []
        labels = glob(f"{self.cache_dir}/label_*.geojson")
        for label in labels:
            with open(label, "r") as f:
                data = f.read().strip()
            if data == "Empty_label":
                continue
            feature = geojson.loads(data)['features'][0]
            features.append(feature)
        collection = geojson.FeatureCollection(features)
        return collection
    
    def save_to_geojson(self, save_dir, save_name=None):
        os.makedirs(save_dir, exist_ok=True)
        collection = self.to_geojson()
        if save_name is None:
            save_name = "hand_validated.geojson"
        save_path = os.path.join(save_dir, save_name)
        with open(save_path, "w") as f:
            geojson.dump(collection, f)






class TimeSeriesAnnotationTool:
    def __init__(self, labels, classes, zoom, cache_dir, clear_cache=False):
        self.original_labels = deepcopy(labels)
        self.labels = deepcopy(labels)
        self.classes = classes
        self.zoom = zoom
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if clear_cache:
            label_files = glob(os.path.join(self.cache_dir, "label_*.geojson"))
            for label_file in label_files:
                os.remove(label_file)
                
        # dates repository
        self.dates = {"2024-09-19": 20337, "2023-08-31": 64776, "2019-07-17": 16681}

        # initialize
        self.index = 0

        # Map
        self.m = leafmap.Map(center=(27, 77), zoom=self.zoom)
        self.m.add_basemap("Esri.WorldImagery")
        self.m.remove_control(self.m.draw_control)
        self.draw_control = GeomanDrawControl(position='topright',  polyline={}, circle={}, circlemarker={}, marker={}, polygon={}, cut=False)
        
        def on_draw(*args, **kwargs):
            self.status_label.value = "Submit the label to update it."
            self.disable_buttons()
            self.submit_button.button_style = "danger"
            
        self.draw_control.on_draw(on_draw)
        
        self.draw_control.rectangle = {
        "pathOptions": {
             "fillColor": "#fca45d",
             "color": "#fca45d",
             "fillOpacity": 0.0
         }
         }
        
        self.m.add_control(self.draw_control)
        
        # Interface elements
        # current label
        self.show_label = Label("")
        
        # status label
        self.status_label = Label("")

        # next_button
        self.next_button = Button(description="next")
        self.next_button.on_click(self.next_button_clicked)

        # previous_button
        self.previous_button = Button(description="previous")
        self.previous_button.on_click(self.previous_button_clicked)
        
        # submit button
        self.submit_button = Button(description="submit")
        self.submit_button.on_click(self.submit_button_clicked)
        
        # reset button
        self.reset_button = Button(description="fall back to original")
        self.reset_button.on_click(self.reset_button_clicked)
        
        # classes dropdown
        self.classes_dropdown = Dropdown(options=self.classes)
        self.classes_dropdown.on_trait_change(self.on_dropdown_change, 'value')
        
        # year mode
        self.year_mode = RadioButtons(options=['all dates', 'one date per year'], description='')
        
        # year slider
        # self.year_slider = 
        
        display(self.show_label)
        display(self.status_label)
        display(HBox([self.submit_button, self.previous_button, self.next_button, self.reset_button, self.classes_dropdown]))
        display(self.m)
        
        # initialize
        loaded_from_cache = self.show_current_label()
        while loaded_from_cache and self.index < len(self.labels) - 1:
            self.next_button_clicked()
            loaded_from_cache = self.show_current_label()
        
    def show_current_label(self):
        self.disable_buttons()
        loaded_from_cache = False
        if os.path.exists(f"{self.cache_dir}/label_{self.index}.geojson"):
            with open(f"{self.cache_dir}/label_{self.index}.geojson", "r") as f:
                data = f.read().strip()
            if data == "Empty_label":
                self.labels[self.index] = None
            else:
                feature = geojson.loads(data)['features'][0]
                self.labels[self.index] = OBBLabel.from_geojson(feature)
            self.enable_buttons()  # allow to move around if label is already present
            loaded_from_cache = True

        label = self.labels[self.index]
        if label is None:
            self.draw_control.data = []
            self.status_label.value = "No label available."
            original_label = self.original_labels[self.index]
            self.m.set_center(original_label.properties['center_lon'], original_label.properties['center_lat'], zoom=self.zoom)
            self.enable_buttons()  # allow to move around if label is empty
        else:
            if loaded_from_cache:
                self.status_label.value = "Label loaded from cache. Submit only to make changes."
            else:
                self.status_label.value = "Label is a valid polygon. Make changes if needed and submit."
            # set pov
            self.m.set_center(label.properties['center_lon'], label.properties['center_lat'], zoom=self.zoom)
            
            # show current label
            feature = label.to_geojson(source=None, task_name=None)
            
            self.draw_control.data = [] # first clear the existing data to trigger the changes in GUI
            feature['properties']['style'] = {'color': self.get_color(feature), 'fillColor': self.get_color(feature), 'fillOpacity': 0.0}
            self.draw_control.data = [feature]
            self.classes_dropdown.value = feature['properties']['class_name']
        
        # update label
        self.show_label.value = f"Label {self.index+1}/{len(self.labels)}"
        return loaded_from_cache
        
    def disable_buttons(self):
        self.next_button.disabled = True
        self.previous_button.disabled = True
    
    def enable_buttons(self):
        self.next_button.disabled = False
        # self.previous_button.disabled = False
        
    @staticmethod
    def get_color(feature):
        class_name = feature['properties']['class_name']
        if class_name == "CFCBK":
            # red
            return "#ff0000"
        elif class_name == "FCBK":
             # orange
            return "#ffa500"
        elif class_name == "Zigzag":
            # green
            return "#00ff00"
        else:
            # blue
            return "#0000ff"
        
    def submit_button_clicked(self, *args, **kwargs):
        if len(self.draw_control.data) == 0:
            self.labels[self.index] = None
            # remove label from cache
            cache_path = f"{self.cache_dir}/label_{self.index}.geojson"
            with open(cache_path, "w") as f:
                f.write("Empty_label")
            self.enable_buttons() # allow to move around if label is empty
            return
        
        feature = self.draw_control.data[-1]
        try:
            assert feature['geometry']['type'] == 'Polygon'
        except AssertionError:
            if feature['geometry']['type'] != 'Polygon':
                self.status_label.value = "Invalid label. Please correct it or delete it."
                return
        
        coords = []
        for lon, lat in feature['geometry']['coordinates'][0]:
            x, y = geo_to_webm_pixel(lat, lon, self.zoom)
            coords.append([x, y])
        coords = np.array(coords, dtype=np.float32)
        
        (x, y), (w, h), r = cv2.minAreaRect(coords)
        r = np.deg2rad(r)
        rect = xywhr2xyxyxyxy(np.array([x, y, w, h, r]))
        
        coords = []
        for pair in rect:
            lat, lon = webm_pixel_to_geo(pair[0], pair[1], self.zoom)
            coords.append([lon, lat])
        poly = Polygon(coords)
        
        feature['geometry']['coordinates'] = [list(poly.exterior.coords)]
        feature['properties']['source'] = 'hand_validated'
        feature['properties']['task_name'] = 'hand_validation'
        feature['properties']['class_name'] = self.classes_dropdown.value
        feature['properties']['style'] = {'color': self.get_color(feature), 'fillColor': self.get_color(feature), 'fillOpacity': 0.0}
        self.labels[self.index] = OBBLabel.from_geojson(feature)
        self.cache_label()
        self.show_current_label()
        self.enable_buttons()
        
    def cache_label(self):
        cache_path = f"{self.cache_dir}/label_{self.index}.geojson"
        feature = self.labels[self.index].to_geojson(source=None, task_name=None)
        collection = geojson.FeatureCollection([feature])
        with open(cache_path, "w") as f:
            geojson.dump(collection, f)
            
    def reset_button_clicked(self, *args, **kwargs):
        original_label = self.original_labels[self.index]
        self.labels[self.index] = original_label
        feature = self.labels[self.index].to_geojson(source=None, task_name=None)
        self.labels[self.index].properties['style'] = {'color': self.get_color(feature), 'fillColor': self.get_color(feature), 'fillOpacity': 0.0}
        self.cache_label()
        self.show_current_label()
    
    def on_dropdown_change(self, old, new):
        if new != self.labels[self.index].properties['class_name']:
            self.labels[self.index].properties['class_name'] = new
            feature = self.labels[self.index].to_geojson(source=None, task_name=None)
            self.labels[self.index].properties['style'] = {'color': self.get_color(feature), 'fillColor': self.get_color(feature), 'fillOpacity': 0.0}
            self.cache_label()
            self.show_current_label()
    
    def next_button_clicked(self, *args, **kwargs):
        # show next label
        if self.index >= (len(self.labels) - 1):
            pass # do nothing
        else:
            self.index += 1
        self.show_current_label()
    
    def previous_button_clicked(self, *args, **kwargs):
        # show next label
        if self.index <= 0:
            pass # do nothing
        else:
            self.index -= 1
        self.show_current_label()
        
    def to_geojson(self):
        features = []
        labels = glob(f"{self.cache_dir}/label_*.geojson")
        for label in labels:
            with open(label, "r") as f:
                data = f.read().strip()
            if data == "Empty_label":
                continue
            feature = geojson.loads(data)['features'][0]
            features.append(feature)
        collection = geojson.FeatureCollection(features)
        return collection
    
    def save_to_geojson(self, save_dir, save_name=None):
        os.makedirs(save_dir, exist_ok=True)
        collection = self.to_geojson()
        if save_name is None:
            save_name = "hand_validated.geojson"
        save_path = os.path.join(save_dir, save_name)
        with open(save_path, "w") as f:
            geojson.dump(collection, f)
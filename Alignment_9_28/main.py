from Functions.gui_setup import GUISetup
from Functions.image_processing import ImageProcessing
from Functions.event_handlers import EventHandlers
from Functions.transformation import Transformation
from Functions.auto_alignment import AutoAlignment
from Functions.machine_learning import MachineLearning
class BrainAlignmentGUI(GUISetup, ImageProcessing, EventHandlers, Transformation, AutoAlignment, MachineLearning):
    def __init__(self, tif_image_path,npy_image_path,tif_scale_factor=[0.02,0.2],npy_scale_factor=[0.3,0.8],spatial_dataset=True):
        
        self.tif_scale_factor = tif_scale_factor
        self.npy_scale_factor = npy_scale_factor
        self.x0=0
        self.y0=0
        self.tif_points_dict = {}
        self.npy_points_dict = {}
        self.vector_points = []
        self.toggle_grid = 0
        self.ax_history = []
        self.minor_grid = []
        self.tif_zoom_mode = False
        self.npy_zoom_mode = False
        self.GPU_MODE = True
        self.sync_mode=False
        self.celltypist_mode=False
        self.previous_scatter_npy = None
        self.previous_scatter_tif = None
        self.spatial_dataset=spatial_dataset
        self.tif_thumbnail, self.tif_high_thumbnail, self.npy_thumbnail, self.npy_high_thumbnail, self.x_factor,self.adata = self.input_image(tif_image_path,npy_image_path,tif_scale_factor,npy_scale_factor,spatial_dataset)
        self.tif_tile = self.tif_thumbnail
        self.npy_tile = self.npy_thumbnail

        self.setup_gui()
    

    # Add all the other methods (onclick, on_select, on_release, del_last_point, etc.) here
    # Remember to change all references to global variables to use self.variable_name
    
    

    

    


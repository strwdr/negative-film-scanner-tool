"""
Negative Film Scanner Tool
Copyright (c) 2025 under MIT License
See LICENSE file for details.
"""

import glob
import os
import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk, StringVar, messagebox
from PIL import Image, ImageTk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ImageProcessor:
    """Class for handling image processing operations"""
    
    @staticmethod
    def make_mono_from_BGR(image):
        """Convert BGR image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def reverse_rgb(image):
        """Invert image colors"""
        return 255 - image
    
    @staticmethod
    def equalize_histogram(image):
        """Apply standard histogram equalization"""
        return cv2.equalizeHist(image)
    
    @staticmethod
    def equalize_adaptive_histogram(image, clip_limit=2.0, tile_grid_size=8):
        """Apply adaptive histogram equalization"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(image)
    
    @staticmethod
    def sharpen(weight, image):
        """Sharpen image using Laplacian filter"""
        # Create a sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * weight
        
        # Apply the kernel
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Ensure the output is within valid range
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def denoise(image, strength=10):
        """Apply non-local means denoising"""
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    @staticmethod
    def adjust_contrast(image, alpha=1.0, beta=0):
        """Adjust contrast and brightness"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    @staticmethod
    def auto_white_balance(image):
        """Automatic white balance"""
        if len(image.shape) == 2:
            return image
            
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            mean = np.mean(image[:,:,i])
            result[:,:,i] = image[:,:,i] * (128 / mean) if mean > 0 else image[:,:,i]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def border_mask(image, threshold=70):
        """Create a mask for the image borders"""
        image_copy = copy.deepcopy(image)
        mask = image_copy > threshold
        image_copy[mask] = 255
        return image_copy
    
    @staticmethod
    def apply_curve(image, curve_points, curve_type='linear'):
        """Apply a custom curve to the image
        
        Args:
            image: Input image
            curve_points: List of (x, y) points defining the curve, normalized to 0-1
            curve_type: Type of interpolation ('linear', 'spline', 'bezier')
        
        Returns:
            Processed image with curve applied
        """
        curve_points = sorted(curve_points, key=lambda p: p[0])
        x_points = np.array([p[0] for p in curve_points]) * 255
        y_points = np.array([p[1] for p in curve_points]) * 255
        
        # Different interpolation methods
        if curve_type == 'spline' and len(curve_points) >= 4:
            # Use cubic spline interpolation for smoother curves
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(x_points, y_points)
            lut = np.clip(cs(np.arange(256)), 0, 255).astype(np.uint8)
        elif curve_type == 'bezier' and len(curve_points) >= 3:
            # Use Bezier curve interpolation
            from scipy.special import comb
            
            def bernstein_poly(i, n, t):
                return comb(n, i) * (t**(i)) * ((1-t)**(n-i))
            
            def bezier_curve(points, num=256):
                n = len(points) - 1
                t = np.linspace(0, 1, num)
                curve = np.zeros((num, 2))
                
                for i, point in enumerate(points):
                    curve += np.outer(bernstein_poly(i, n, t), point)
                
                return curve
            
            # Convert to numpy array for bezier calculation
            points_array = np.array(list(zip(x_points/255, y_points/255)))
            curve = bezier_curve(points_array)
            
            # Map x values to indices and get corresponding y values
            x_indices = np.linspace(0, 1, 256)
            y_values = np.interp(x_indices, curve[:, 0], curve[:, 1])
            
            lut = np.clip(y_values * 255, 0, 255).astype(np.uint8)
        else:
            # Default to linear interpolation
            lut = np.interp(np.arange(256), x_points, y_points).astype(np.uint8)
        
        if len(image.shape) == 2:
            return cv2.LUT(image, lut)
        else:
            result = image.copy()
            for i in range(3):
                result[:, :, i] = cv2.LUT(image[:, :, i], lut)
            return result
    
    @staticmethod
    def apply_levels(image, black_point=0, white_point=255, gamma=1.0):
        """Apply levels adjustment to the image
        
        Args:
            image: Input grayscale image
            black_point: Black point (0-255)
            white_point: White point (0-255)
            gamma: Gamma correction value
        
        Returns:
            Processed image with levels applied
        """
        # Clip the image to the new black and white points
        image_float = image.astype(np.float32)
        
        # Scale to 0-1
        image_float = (image_float - black_point) / (white_point - black_point)
        image_float = np.clip(image_float, 0, 1)
        
        # Apply gamma correction
        if gamma != 1.0:
            image_float = np.power(image_float, 1.0 / gamma)
        
        # Scale back to 0-255
        return (image_float * 255).astype(np.uint8)
    
    @staticmethod
    def apply_unsharp_mask(image, radius=5, amount=1.0, threshold=0):
        """Apply unsharp mask for more controlled sharpening
        
        Args:
            image: Input grayscale image
            radius: Blur radius
            amount: Sharpening strength
            threshold: Threshold for applying sharpening
        
        Returns:
            Sharpened image
        """
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
            
        return sharpened
    
    @staticmethod
    def apply_high_pass(image, radius=10):
        """Apply high pass filter to enhance details
        
        Args:
            image: Input grayscale image
            radius: Filter radius
            
        Returns:
            Filtered image
        """
        # Create a high pass filter
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        high_pass = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return high_pass


class FilmScannerApp:
    """Main application class for the film scanner tool"""
    
    def __init__(self):
        self.processor = ImageProcessor()
        self.original_image = None
        self.tmp_image = None
        self.out_image = None
        self.current_file = None
        self.file_list = []
        self.current_file_idx = 0
        self.window_name = "Negative Film Scanner Tool"
        self.indir = './input'
        self.outdir = './output'
        self.showing_original = False
        
        self.curve_points = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)]
        self.selected_point = None
        self.curve_enabled = False
        
        self.black_point = 0
        self.white_point = 255
        self.gamma = 1.0
        self.levels_enabled = False
        
        self.unsharp_radius = 5
        self.unsharp_amount = 1.0
        self.unsharp_threshold = 0
        self.unsharp_enabled = False
        
        self.high_pass_radius = 10
        self.high_pass_enabled = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the user interface with Tkinter"""
        # Create Tkinter root window
        self.root = tk.Tk()
        self.root.title("Negative Film Scanner Tool")
        self.root.geometry("1400x900")
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        self.root.minsize(1000, 700)
        
        # Create main frame with a paned window for resizable sections
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left frame for image display
        self.left_frame = ttk.Frame(main_pane)
        main_pane.add(self.left_frame, weight=2)
        
        # Create image display canvas
        self.image_canvas = tk.Canvas(self.left_frame, bg="light gray", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create right frame for controls
        right_frame = ttk.Frame(main_pane, width=500)
        main_pane.add(right_frame, weight=1)
        
        # Create buttons frame
        buttons_frame = ttk.LabelFrame(right_frame, text="Navigation")
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create navigation buttons
        nav_btn_frame = ttk.Frame(buttons_frame)
        nav_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_btn_frame, text="Previous", command=self.prev_file).pack(side=tk.LEFT, padx=2, pady=5, expand=True, fill=tk.X)
        ttk.Button(nav_btn_frame, text="Next", command=lambda: self.next_file(save=False)).pack(side=tk.LEFT, padx=2, pady=5, expand=True, fill=tk.X)
        ttk.Button(nav_btn_frame, text="Save & Next", command=lambda: self.next_file(save=True)).pack(side=tk.LEFT, padx=2, pady=5, expand=True, fill=tk.X)
        
        # Create directory frame
        dir_frame = ttk.LabelFrame(right_frame, text="Directories")
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input directory
        ttk.Label(dir_frame, text="Input Directory:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        input_dir_frame = ttk.Frame(dir_frame)
        input_dir_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.input_dir_var = StringVar(value=self.indir)
        ttk.Entry(input_dir_frame, textvariable=self.input_dir_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(input_dir_frame, text="Browse", command=self.select_input_directory).pack(side=tk.RIGHT)
        
        # Output directory
        ttk.Label(dir_frame, text="Output Directory:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        output_dir_frame = ttk.Frame(dir_frame)
        output_dir_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.output_dir_var = StringVar(value=self.outdir)
        ttk.Entry(output_dir_frame, textvariable=self.output_dir_var, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(output_dir_frame, text="Browse", command=self.select_output_directory).pack(side=tk.RIGHT)
        
        # Create action buttons frame
        action_frame = ttk.LabelFrame(right_frame, text="Actions")
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        action_btn_frame = ttk.Frame(action_frame)
        action_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # First row of buttons
        btn_row1 = ttk.Frame(action_btn_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_row1, text="Reset Settings", command=self.reset_settings).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        self.toggle_original_btn = ttk.Button(btn_row1, text="Show Original", command=self.toggle_original_view)
        self.toggle_original_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Second row of buttons
        btn_row2 = ttk.Frame(action_btn_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_row2, text="Histogram", command=self.show_histogram).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Third row of buttons
        btn_row3 = ttk.Frame(action_btn_frame)
        btn_row3.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_row3, text="Batch Process", command=self.batch_process).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        ttk.Button(btn_row3, text="Exit", command=self.exit_app).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Create file info frame
        info_frame = ttk.LabelFrame(right_frame, text="File Information")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_info_var = StringVar(value="No file loaded")
        ttk.Label(info_frame, textvariable=self.file_info_var, wraplength=380).pack(padx=5, pady=5, anchor=tk.W)
        
        # Create controls frame
        controls_frame = ttk.LabelFrame(right_frame, text="Image Controls")
        controls_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for controls
        canvas = tk.Canvas(controls_frame)
        scrollbar = ttk.Scrollbar(controls_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Bind mouse wheel to scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Create window in canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=canvas.winfo_width())
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Make sure the controls frame expands with the window
        controls_frame.pack_propagate(False)
        
        # Configure canvas to resize the scrollable frame when canvas size changes
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)
            
        canvas.bind("<Configure>", configure_scroll_region)
        
        # Create trackbars with Tkinter sliders
        # Use a two-column grid layout for controls
        # Cropping controls
        crop_frame = ttk.LabelFrame(scrollable_frame, text="Cropping")
        crop_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.create_slider(crop_frame, "Rotation ±50°", -50, 50, 0, 1, height=40)
        
        # Create a frame for width/height controls side by side
        crop_size_frame = ttk.Frame(crop_frame)
        crop_size_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Left column - Width
        left_col = ttk.Frame(crop_size_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.create_slider(left_col, "Crop Width", 0, 200, 200, 1, height=40)
        
        # Right column - Height
        right_col = ttk.Frame(crop_size_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        self.create_slider(right_col, "Crop Height", 0, 200, 200, 1, height=40)
        
        # Create a frame for position controls side by side
        crop_pos_frame = ttk.Frame(crop_frame)
        crop_pos_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Left column - X Position
        left_col = ttk.Frame(crop_pos_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.create_slider(left_col, "Crop X Position", 0, 200, 0, 1, height=40)
        
        # Right column - Y Position
        right_col = ttk.Frame(crop_pos_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        self.create_slider(right_col, "Crop Y Position", 0, 200, 0, 1, height=40)
        
        # Enhancement controls
        enhance_frame = ttk.LabelFrame(scrollable_frame, text="Enhancement")
        enhance_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a frame for sharpness/denoise controls side by side
        enhance_sd_frame = ttk.Frame(enhance_frame)
        enhance_sd_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Left column - Sharpness
        left_col = ttk.Frame(enhance_sd_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.create_slider(left_col, "Sharpness", 0, 3.0, 0.0, 0.05, height=40)
        
        # Right column - Denoise
        right_col = ttk.Frame(enhance_sd_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        self.create_slider(right_col, "Denoise", 0, 30, 0, 1, height=40)
        
        # Create a frame for contrast/brightness controls side by side
        enhance_cb_frame = ttk.Frame(enhance_frame)
        enhance_cb_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Left column - Contrast
        left_col = ttk.Frame(enhance_cb_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.create_slider(left_col, "Contrast", 0.1, 4, 1.0, 0.1, height=40, logarithmic=True)
        
        # Right column - Brightness
        right_col = ttk.Frame(enhance_cb_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        self.create_slider(right_col, "Brightness", -50, 50, 0, 1, height=40)
        
        # CLAHE parameters
        clahe_frame = ttk.LabelFrame(scrollable_frame, text="CLAHE Parameters")
        clahe_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a frame for CLAHE controls side by side
        clahe_params_frame = ttk.Frame(clahe_frame)
        clahe_params_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Left column - Clip Limit
        left_col = ttk.Frame(clahe_params_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.create_slider(left_col, "CLAHE Clip Limit", 0, 10, 2.0, 0.1, height=40)
        
        # Right column - Grid Size
        right_col = ttk.Frame(clahe_params_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        self.create_slider(right_col, "CLAHE Grid Size", 1, 21, 8, 1, height=40)
        
        # Toggle options
        toggle_frame = ttk.LabelFrame(scrollable_frame, text="Basic Options")
        toggle_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create checkbuttons and dropdown for options
        self.invert_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="Invert Colors (Negative to Positive)", variable=self.invert_var, 
                        command=self.on_option_change).pack(anchor=tk.W, padx=5, pady=5)
        
        self.sharpen_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggle_frame, text="Apply Sharpening", variable=self.sharpen_var, 
                        command=self.on_option_change).pack(anchor=tk.W, padx=5, pady=5)
        
        self.auto_contrast_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toggle_frame, text="Manual Contrast & Brightness Control", variable=self.auto_contrast_var, 
                        command=self.on_option_change).pack(anchor=tk.W, padx=5, pady=5)
        
        ttk.Label(toggle_frame, text="Histogram Equalization:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.equalize_var = tk.StringVar(value="CLAHE")
        equalize_combo = ttk.Combobox(toggle_frame, textvariable=self.equalize_var, 
                                      values=["None", "CLAHE", "Standard"])
        equalize_combo.pack(fill=tk.X, padx=5, pady=(0, 5))
        equalize_combo.bind("<<ComboboxSelected>>", lambda e: self.on_option_change())
        
        # Advanced options frame
        advanced_frame = ttk.LabelFrame(scrollable_frame, text="Advanced Options")
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Curve adjustment toggle and button
        curve_frame = ttk.Frame(advanced_frame)
        curve_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.curve_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(curve_frame, text="Enable Curve Adjustment", variable=self.curve_var, 
                        command=self.on_option_change).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(curve_frame, text="Curve Editor", command=self.show_curve_editor).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Levels adjustment toggle and controls
        levels_frame = ttk.Frame(advanced_frame)
        levels_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.levels_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(levels_frame, text="Enable Levels Adjustment", variable=self.levels_var, 
                        command=self.on_option_change).pack(anchor=tk.W, pady=2)
        
        # Black point, white point, and gamma sliders
        self.create_slider(levels_frame, "Black Point", 0, 100, 0, 1, height=40)
        self.create_slider(levels_frame, "White Point", 155, 255, 255, 1, height=40)
        self.create_slider(levels_frame, "Gamma", 0.1, 3.0, 1.0, 0.1, height=40)
        
        # Unsharp mask toggle and controls
        unsharp_frame = ttk.Frame(advanced_frame)
        unsharp_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.unsharp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(unsharp_frame, text="Enable Unsharp Mask", variable=self.unsharp_var, 
                        command=self.on_option_change).pack(anchor=tk.W, pady=2)
        
        # Unsharp mask sliders
        self.create_slider(unsharp_frame, "Radius", 0.1, 20.0, 5.0, 0.1, height=40)
        self.create_slider(unsharp_frame, "Amount", 0.1, 5.0, 1.0, 0.1, height=40)
        self.create_slider(unsharp_frame, "Threshold", 0, 50, 0, 1, height=40)
        
        # High pass filter toggle and controls
        highpass_frame = ttk.Frame(advanced_frame)
        highpass_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.highpass_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(highpass_frame, text="Enable High Pass Filter", variable=self.highpass_var, 
                        command=self.on_option_change).pack(anchor=tk.W, pady=2)
        
        # High pass radius slider
        self.create_slider(highpass_frame, "HP Radius", 1, 50, 10, 1, height=40)
        
        # Initialize the image display
        self.display_placeholder()
    
    def display_placeholder(self):
        """Display a placeholder image when no image is loaded"""
        placeholder = np.ones((400, 600, 3), dtype=np.uint8) * 200
        cv2.putText(placeholder, "No image loaded", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self.display_image(placeholder)
    
    def create_slider(self, parent, name, min_val, max_val, default_val, resolution=1, height=30, logarithmic=False):
        """Create a labeled slider (scale) widget with value display"""
        frame = ttk.Frame(parent, height=height)
        frame.pack(fill=tk.X, padx=5, pady=5)
        frame.pack_propagate(False)
        
        # Label on top
        label_frame = ttk.Frame(frame)
        label_frame.pack(fill=tk.X)
        
        ttk.Label(label_frame, text=name).pack(side=tk.LEFT)
        
        # Create variable and value display
        var = tk.DoubleVar(value=default_val)
        value_label = ttk.Label(label_frame, text=f"{default_val:.1f}")
        value_label.pack(side=tk.RIGHT)
        
        # Create slider
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, 
                          orient=tk.HORIZONTAL, 
                          command=lambda val: self.on_slider_change(name, value_label, logarithmic))
        slider.pack(fill=tk.X, expand=True, pady=(2, 0))
        
        # Store the variable in a dictionary for later access
        if not hasattr(self, 'slider_vars'):
            self.slider_vars = {}
        self.slider_vars[name] = var
        
        # Store whether this slider is logarithmic
        if not hasattr(self, 'logarithmic_sliders'):
            self.logarithmic_sliders = set()
        if logarithmic:
            self.logarithmic_sliders.add(name)
    
    def update_file_info(self):
        """Update the file information display"""
        if not self.file_list:
            self.file_info_var.set("No files loaded")
            return
            
        if self.current_file:
            filename = os.path.basename(self.current_file)
            self.file_info_var.set(f"File: {self.current_file_idx+1}/{len(self.file_list)}\n{filename}")
    
    def on_slider_change(self, name=None, label=None, logarithmic=False):
        """Handle slider changes and update the preview"""
        if self.tmp_image is None:
            return
        
        # Update the value label if provided
        if name and label and name in self.slider_vars:
            value = self.slider_vars[name].get()
            label.config(text=f"{value:.1f}")
            
        self.process_image()
    
    def on_option_change(self):
        """Handle option changes and update the preview"""
        if self.tmp_image is None:
            return
            
        self.process_image()
    
    def process_image(self):
        """Process the image with current settings"""
        if self.showing_original:
            self.toggle_original_view()
            return
            
        if self.tmp_image is None:
            return
            
        # Get all slider values
        rotate_deg = self.slider_vars["Rotation ±50°"].get()
        crop_width = self.slider_vars["Crop Width"].get()
        crop_height = self.slider_vars["Crop Height"].get()
        crop_x = self.slider_vars["Crop X Position"].get()
        crop_y = self.slider_vars["Crop Y Position"].get()
        
        sharpness = self.slider_vars["Sharpness"].get()  # Now directly use the value (0-1 range)
        contrast = self.slider_vars["Contrast"].get()
        brightness = self.slider_vars["Brightness"].get()
        denoise_strength = self.slider_vars["Denoise"].get()
        
        clip_limit = self.slider_vars["CLAHE Clip Limit"].get()
        tile_grid_size = int(self.slider_vars["CLAHE Grid Size"].get())
        
        # Get toggle options
        invert_colors = self.invert_var.get()
        sharpen_option = self.sharpen_var.get()
        manual_contrast = self.auto_contrast_var.get()
        equalize_option = self.equalize_var.get()
        
        # Get advanced options
        curve_enabled = self.curve_var.get()
        levels_enabled = self.levels_var.get()
        unsharp_enabled = self.unsharp_var.get()
        highpass_enabled = self.highpass_var.get()
        
        # Get levels values if enabled
        if levels_enabled:
            black_point = int(self.slider_vars["Black Point"].get())
            white_point = int(self.slider_vars["White Point"].get())
            gamma = self.slider_vars["Gamma"].get()
        
        # Get unsharp mask values if enabled
        if unsharp_enabled:
            unsharp_radius = self.slider_vars["Radius"].get()
            unsharp_amount = self.slider_vars["Amount"].get()
            unsharp_threshold = int(self.slider_vars["Threshold"].get())
            
        # Get high pass filter radius if enabled
        if highpass_enabled:
            highpass_radius = self.slider_vars["HP Radius"].get()
        
        # Calculate crop dimensions
        rows, cols = self.tmp_image.shape
        y = int(crop_y * (cols / 200))
        x = int(crop_x * (rows / 200))
        w = int(crop_height * (rows / 200))
        h = int(crop_width * (cols / 200))
        
        # Apply rotation
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), rotate_deg, 1)
        output = cv2.warpAffine(self.tmp_image, M, (cols, rows))
        
        # Apply cropping
        if x < rows and y < cols and x + w <= rows and y + h <= cols and w > 0 and h > 0:
            output = output[x:x + w, y:y + h]
        else:
            # Fallback to safe values if crop is out of bounds
            safe_w = min(w, rows - x) if x < rows else 0
            safe_h = min(h, cols - y) if y < cols else 0
            if x < rows and y < cols and safe_w > 0 and safe_h > 0:
                output = output[x:x + safe_w, y:y + safe_h]
        
        # Apply image processing operations
        if invert_colors:
            output = self.processor.reverse_rgb(output)
            
        if denoise_strength > 0:
            output = self.processor.denoise(output, denoise_strength)
            
        if equalize_option == "CLAHE":
            output = self.processor.equalize_adaptive_histogram(output, clip_limit, tile_grid_size)
        elif equalize_option == "Standard":
            output = self.processor.equalize_histogram(output)
        
        # Apply advanced processing options
        if highpass_enabled:
            output = self.processor.apply_high_pass(output, highpass_radius)
            
        if unsharp_enabled:
            output = self.processor.apply_unsharp_mask(output, unsharp_radius, unsharp_amount, unsharp_threshold)
        elif sharpen_option and sharpness > 0:
            # Apply sharpening with the weight parameter
            output = self.processor.sharpen(sharpness, output)
            
        if levels_enabled:
            output = self.processor.apply_levels(output, black_point, white_point, gamma)
            
        if curve_enabled:
            # Get curve type if available
            curve_type = self.curve_type_var.get() if hasattr(self, 'curve_type_var') else 'linear'
            output = self.processor.apply_curve(output, self.curve_points, curve_type)
            
        if manual_contrast:
            # Apply logarithmic contrast adjustment
            output = self.processor.adjust_contrast(output, contrast, brightness)
        
        # Store the processed image
        self.out_image = copy.deepcopy(output)
        
        # Update the display
        self.update_display()
    
    def update_display(self):
        """Update the image display with the current image"""
        if self.showing_original and self.original_image is not None:
            # Show the original image
            if len(self.original_image.shape) == 3:
                display_img = self.original_image.copy()
            else:
                display_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        elif self.out_image is not None:
            # Show the processed image
            if len(self.out_image.shape) == 2:
                display_img = cv2.cvtColor(self.out_image, cv2.COLOR_GRAY2BGR)
            else:
                display_img = self.out_image.copy()
        else:
            return
            
        # Display the image in the Tkinter canvas
        self.display_image(display_img)
    
    def display_image(self, image):
        """Display an image in the Tkinter canvas"""
        # Convert from BGR to RGB for PIL
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Get canvas dimensions
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 1:  # Canvas not yet realized
            canvas_width = 800
            canvas_height = 600
            
        # Resize image to fit canvas while maintaining aspect ratio
        img_height, img_width = image_rgb.shape[:2]
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(resized_image)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        
        # Clear previous image and display new one
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            canvas_width//2, canvas_height//2,
            image=self.tk_image, anchor=tk.CENTER
        )
    
    def create_preview(self, image, target_height=600):
        """Create a preview of the image with a fixed height"""
        if image is None:
            return np.ones((target_height, int(target_height * 1.5)), dtype=np.uint8) * 128
            
        tmp = copy.deepcopy(image)
        aspect_ratio = tmp.shape[1] / tmp.shape[0]
        target_width = int(target_height * aspect_ratio)
        
        # Resize to target height while maintaining aspect ratio
        return cv2.resize(tmp, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    def reset_settings(self):
        """Reset all sliders to default values"""
        # Reset basic sliders
        self.slider_vars["Rotation ±50°"].set(0)
        self.slider_vars["Crop Width"].set(200)
        self.slider_vars["Crop Height"].set(200)
        self.slider_vars["Crop X Position"].set(0)
        self.slider_vars["Crop Y Position"].set(0)
        self.slider_vars["Sharpness"].set(0.0)  # Updated to match new range
        self.slider_vars["Contrast"].set(1.0)
        self.slider_vars["Brightness"].set(0)
        self.slider_vars["Denoise"].set(0)
        self.slider_vars["CLAHE Clip Limit"].set(2.0)
        self.slider_vars["CLAHE Grid Size"].set(8)
        
        # Reset advanced sliders if they exist
        if "Black Point" in self.slider_vars:
            self.slider_vars["Black Point"].set(0)
        if "White Point" in self.slider_vars:
            self.slider_vars["White Point"].set(255)
        if "Gamma" in self.slider_vars:
            self.slider_vars["Gamma"].set(1.0)
        if "Radius" in self.slider_vars:
            self.slider_vars["Radius"].set(5.0)
        if "Amount" in self.slider_vars:
            self.slider_vars["Amount"].set(1.0)
        if "Threshold" in self.slider_vars:
            self.slider_vars["Threshold"].set(0)
        if "HP Radius" in self.slider_vars:
            self.slider_vars["HP Radius"].set(10)
        
        # Reset basic toggles
        self.invert_var.set(True)
        self.sharpen_var.set(True)
        self.auto_contrast_var.set(False)
        self.equalize_var.set("CLAHE")
        
        # Reset advanced toggles
        self.curve_var.set(False)
        self.curve_enabled = False  # Reset the instance variable
        self.levels_var.set(False)
        self.unsharp_var.set(False)
        self.highpass_var.set(False)
        
        # Reset curve points
        self.curve_points = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)]
        
        # Update the image
        self.process_image()
    
    def toggle_original_view(self):
        """Toggle between showing the original and processed image"""
        if self.original_image is None:
            return
            
        self.showing_original = not self.showing_original
        
        # Update button text
        if self.showing_original:
            self.toggle_original_btn.config(text="Show Processed")
        else:
            self.toggle_original_btn.config(text="Show Original")
            
        self.update_display()
    
    def select_input_directory(self):
        """Open a dialog to select input directory"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        
        if directory:
            self.indir = directory
            self.input_dir_var.set(directory)
            self.outdir = os.path.join(os.path.dirname(directory), 'output')
            self.output_dir_var.set(self.outdir)
            self.ensure_dir(self.outdir)
            self.load_files()
    
    def select_output_directory(self):
        """Open a dialog to select output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        
        if directory:
            self.outdir = directory
            self.output_dir_var.set(directory)
            self.ensure_dir(self.outdir)
    
    def ensure_dir(self, directory):
        """Create directory if it doesn't exist"""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def load_files(self):
        """Load all image files from the input directory"""
        self.ensure_dir(self.indir)
        self.ensure_dir(self.outdir)
        
        self.file_list = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.file_list.extend(glob.glob(os.path.join(self.indir, ext)))
        
        self.file_list.sort()
        self.current_file_idx = 0
        
        if self.file_list:
            self.load_current_file()
        else:
            print(f"No image files found in {self.indir}")
            self.tmp_image = None
            self.out_image = None
            self.update_file_info()
            self.display_placeholder()
    
    def load_current_file(self):
        """Load the current file from the file list"""
        if not self.file_list or self.current_file_idx >= len(self.file_list):
            return
            
        self.current_file = self.file_list[self.current_file_idx]
        print(f"Loading: {self.current_file}")
        
        # Load and convert to grayscale
        self.original_image = cv2.imread(self.current_file, cv2.IMREAD_UNCHANGED)
        if self.original_image is None:
            print(f"Failed to load {self.current_file}")
            return
            
        self.tmp_image = self.processor.make_mono_from_BGR(self.original_image)
        self.update_file_info()
        self.showing_original = False
        self.toggle_original_btn.config(text="Show Original")
        self.process_image()  # Process with current settings
    
    def next_file(self, save=True):
        """Move to the next file, optionally saving the current one"""
        if not self.file_list:
            return
            
        if save and self.out_image is not None:
            self.save_current_file()
            
        self.current_file_idx = (self.current_file_idx + 1) % len(self.file_list)
        self.load_current_file()
    
    def prev_file(self):
        """Move to the previous file"""
        if not self.file_list:
            return
            
        self.current_file_idx = (self.current_file_idx - 1) % len(self.file_list)
        self.load_current_file()
    
    def save_current_file(self):
        """Save the current processed image"""
        if self.out_image is None or self.current_file is None:
            return
            
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
        output_path = os.path.join(self.outdir, f"processed_{base_name}.jpg")
        
        # Save the image
        success = cv2.imwrite(output_path, self.out_image)
        if success:
            print(f"Saved to: {output_path}")
            # Show a temporary success message
            self.show_save_status(f"Saved: {os.path.basename(output_path)}", success=True)
        else:
            print(f"Failed to save to: {output_path}")
            self.show_save_status("Save failed!", success=False)
    
    def show_save_status(self, message, success=True):
        """Show a temporary status message"""
        color = "green" if success else "red"
        status_label = ttk.Label(self.left_frame, text=message, foreground=color, background="white")
        status_label.place(relx=0.5, rely=0.95, anchor=tk.CENTER)
        
        # Schedule the label to be removed after 2 seconds
        self.root.after(2000, lambda: status_label.destroy())
    
    def exit_app(self):
        """Exit the application"""
        if hasattr(self, 'root') and self.root:
            self.root.destroy()
        exit(0)
    
    def batch_process(self):
        """Process all images in the input directory with current settings"""
        if not self.file_list:
            messagebox.showinfo("Batch Process", "No images to process. Please select an input directory first.")
            return
            
        # Ask for confirmation
        if not messagebox.askyesno("Batch Process", 
                                  f"This will process all {len(self.file_list)} images with current settings.\n\nContinue?"):
            return
            
        # Save current file index to restore later
        current_idx = self.current_file_idx
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Processing")
        progress_window.geometry("400x150")
        progress_window.resizable(False, False)
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - progress_window.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - progress_window.winfo_height()) // 2
        progress_window.geometry(f"+{x}+{y}")
        
        # Add progress information
        ttk.Label(progress_window, text="Processing images...").pack(pady=(15, 5))
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(self.file_list))
        progress_bar.pack(fill=tk.X, padx=20, pady=5)
        
        status_var = tk.StringVar(value="Starting...")
        status_label = ttk.Label(progress_window, textvariable=status_var)
        status_label.pack(pady=5)
        
        # Cancel button
        self.batch_cancel = False
        ttk.Button(progress_window, text="Cancel", 
                  command=lambda: setattr(self, 'batch_cancel', True)).pack(pady=10)
        
        # Process files in a separate thread to keep UI responsive
        def process_thread():
            processed = 0
            skipped = 0
            
            try:
                for i, file_path in enumerate(self.file_list):
                    if self.batch_cancel:
                        status_var.set("Cancelled")
                        break
                        
                    # Update progress
                    progress_var.set(i)
                    status_var.set(f"Processing {i+1}/{len(self.file_list)}: {os.path.basename(file_path)}")
                    
                    # Load the image
                    self.current_file = file_path
                    self.current_file_idx = i
                    
                    try:
                        # Load and process image
                        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                        if img is None:
                            skipped += 1
                            continue
                            
                        self.original_image = img
                        self.tmp_image = self.processor.make_mono_from_BGR(img)
                        
                        # Process with current settings (without updating display)
                        self.process_image_batch()
                        
                        # Save the processed image
                        if self.out_image is not None:
                            base_name = os.path.splitext(os.path.basename(file_path))[0]
                            output_path = os.path.join(self.outdir, f"processed_{base_name}.jpg")
                            cv2.imwrite(output_path, self.out_image)
                            processed += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        skipped += 1
                
                # Update final status
                if self.batch_cancel:
                    status_var.set(f"Cancelled. Processed: {processed}, Skipped: {skipped}")
                else:
                    status_var.set(f"DONE! Processed: {processed}, Skipped: {skipped}")
                    # Set progress bar to 100% when complete
                    progress_var.set(len(self.file_list))
            finally:
                # Use after() to safely update UI from a thread
                self.root.after(100, lambda: self._finish_batch_process(progress_window, current_idx))
        
        # Start processing thread
        threading.Thread(target=process_thread, daemon=True).start()
    
    def _finish_batch_process(self, progress_window, current_idx):
        """Safely finish the batch processing from the main thread"""
        # Replace the cancel button with a close button
        for widget in progress_window.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.destroy()
                
        ttk.Button(progress_window, text="Close", 
                  command=progress_window.destroy).pack(pady=10)
        
        # Restore the original image
        self.current_file_idx = current_idx
        if self.file_list:
            self.load_current_file()
    
    def process_image_batch(self):
        """Process the image with current settings (for batch processing)"""
        if self.tmp_image is None:
            return
            
        # Get all slider values
        rotate_deg = self.slider_vars["Rotation ±50°"].get()
        crop_width = self.slider_vars["Crop Width"].get()
        crop_height = self.slider_vars["Crop Height"].get()
        crop_x = self.slider_vars["Crop X Position"].get()
        crop_y = self.slider_vars["Crop Y Position"].get()
        
        sharpness = self.slider_vars["Sharpness"].get()
        contrast = self.slider_vars["Contrast"].get()
        brightness = self.slider_vars["Brightness"].get()
        denoise_strength = self.slider_vars["Denoise"].get()
        
        clip_limit = self.slider_vars["CLAHE Clip Limit"].get()
        tile_grid_size = int(self.slider_vars["CLAHE Grid Size"].get())
        
        # Get toggle options
        invert_colors = self.invert_var.get()
        sharpen_option = self.sharpen_var.get()
        manual_contrast = self.auto_contrast_var.get()
        equalize_option = self.equalize_var.get()
        
        # Get advanced options
        curve_enabled = self.curve_var.get()
        levels_enabled = self.levels_var.get()
        unsharp_enabled = self.unsharp_var.get()
        highpass_enabled = self.highpass_var.get()
        
        # Get levels values if enabled
        if levels_enabled:
            black_point = int(self.slider_vars["Black Point"].get())
            white_point = int(self.slider_vars["White Point"].get())
            gamma = self.slider_vars["Gamma"].get()
        
        # Get unsharp mask values if enabled
        if unsharp_enabled:
            unsharp_radius = self.slider_vars["Radius"].get()
            unsharp_amount = self.slider_vars["Amount"].get()
            unsharp_threshold = int(self.slider_vars["Threshold"].get())
            
        # Get high pass filter radius if enabled
        if highpass_enabled:
            highpass_radius = self.slider_vars["HP Radius"].get()
        
        # Calculate crop dimensions
        rows, cols = self.tmp_image.shape
        y = int(crop_y * (cols / 200))
        x = int(crop_x * (rows / 200))
        w = int(crop_height * (rows / 200))
        h = int(crop_width * (cols / 200))
        
        # Apply rotation
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), rotate_deg, 1)
        output = cv2.warpAffine(self.tmp_image, M, (cols, rows))
        
        # Apply cropping
        if x < rows and y < cols and x + w <= rows and y + h <= cols and w > 0 and h > 0:
            output = output[x:x + w, y:y + h]
        else:
            # Fallback to safe values if crop is out of bounds
            safe_w = min(w, rows - x) if x < rows else 0
            safe_h = min(h, cols - y) if y < cols else 0
            if x < rows and y < cols and safe_w > 0 and safe_h > 0:
                output = output[x:x + safe_w, y:y + safe_h]
        
        # Apply image processing operations
        if invert_colors:
            output = self.processor.reverse_rgb(output)
            
        if denoise_strength > 0:
            output = self.processor.denoise(output, denoise_strength)
            
        if equalize_option == "CLAHE":
            output = self.processor.equalize_adaptive_histogram(output, clip_limit, tile_grid_size)
        elif equalize_option == "Standard":
            output = self.processor.equalize_histogram(output)
        
        # Apply advanced processing options
        if highpass_enabled:
            output = self.processor.apply_high_pass(output, highpass_radius)
            
        if unsharp_enabled:
            output = self.processor.apply_unsharp_mask(output, unsharp_radius, unsharp_amount, unsharp_threshold)
        elif sharpen_option and sharpness > 0:
            # Apply sharpening with the weight parameter
            output = self.processor.sharpen(sharpness, output)
            
        if levels_enabled:
            output = self.processor.apply_levels(output, black_point, white_point, gamma)
            
        if curve_enabled:
            # Get curve type if available
            curve_type = self.curve_type_var.get() if hasattr(self, 'curve_type_var') else 'linear'
            output = self.processor.apply_curve(output, self.curve_points, curve_type)
            
        if manual_contrast:
            output = self.processor.adjust_contrast(output, contrast, brightness)
        
        # Store the processed image
        self.out_image = copy.deepcopy(output)
    
    def show_histogram(self):
        """Show histogram of the current image"""
        if self.out_image is None:
            messagebox.showinfo("Histogram", "No image loaded")
            return
            
        # Create a new window for the histogram
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Image Histogram")
        hist_window.geometry("600x400")
        hist_window.transient(self.root)
        
        # Create a figure for the histogram
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Calculate histogram
        hist = cv2.calcHist([self.out_image], [0], None, [256], [0, 256])
        
        # Plot histogram
        ax.plot(hist, color='black')
        ax.set_xlim([0, 256])
        ax.set_title('Grayscale Histogram')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add the plot to the window
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_curve_editor(self):
        """Show the curve editor window"""
        if self.out_image is None:
            messagebox.showinfo("Curve Editor", "No image loaded")
            return
            
        self.curve_window = tk.Toplevel(self.root)
        self.curve_window.title("Curve Editor")
        self.curve_window.geometry("800x700")
        self.curve_window.transient(self.root)
        self.curve_window.protocol("WM_DELETE_WINDOW", self.on_curve_editor_close)
        
        main_pane = ttk.PanedWindow(self.curve_window, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        curve_frame = ttk.Frame(main_pane)
        main_pane.add(curve_frame, weight=3)
        
        preview_frame = ttk.LabelFrame(main_pane, text="Preview")
        main_pane.add(preview_frame, weight=2)
        
        fig = Figure(figsize=(6, 6), dpi=100)
        self.curve_ax = fig.add_subplot(111)
        
        self.curve_ax.set_xlim(0, 1)
        self.curve_ax.set_ylim(0, 1)
        self.curve_ax.set_title('Tone Curve Editor')
        self.curve_ax.set_xlabel('Input')
        self.curve_ax.set_ylabel('Output')
        self.curve_ax.grid(True, alpha=0.3)
        
        x_points = [p[0] for p in self.curve_points]
        y_points = [p[1] for p in self.curve_points]
        
        x_interp = np.linspace(0, 1, 100)
        y_interp = np.interp(x_interp, x_points, y_points)
        self.curve_line, = self.curve_ax.plot(x_interp, y_interp, 'b-')
        
        self.curve_points_plot, = self.curve_ax.plot(x_points, y_points, 'ro', picker=5)
        
        self.curve_canvas = FigureCanvasTkAgg(fig, master=curve_frame)
        self.curve_canvas.draw()
        self.curve_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        button_frame = ttk.Frame(self.curve_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Reset Curve", command=self.reset_curve).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Add Point", command=self.add_curve_point).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove Point", command=self.remove_curve_point).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Preview", command=self.preview_curve).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply", command=self.apply_curve).pack(side=tk.RIGHT, padx=5)
        
        # Add number of points and curve type controls
        controls_frame = ttk.Frame(button_frame)
        controls_frame.pack(side=tk.LEFT, padx=20)
        
        # Points control
        points_frame = ttk.Frame(controls_frame)
        points_frame.pack(fill=tk.X, pady=2)
        ttk.Label(points_frame, text="Points:").pack(side=tk.LEFT, padx=5)
        self.points_var = tk.StringVar(value=str(len(self.curve_points)))
        points_combo = ttk.Combobox(points_frame, textvariable=self.points_var, 
                                   values=["3", "5", "7", "9", "11", "15", "21", "31", "51"], width=3)
        points_combo.pack(side=tk.LEFT)
        points_combo.bind("<<ComboboxSelected>>", self.on_points_changed)
        
        # Curve type control
        curve_type_frame = ttk.Frame(controls_frame)
        curve_type_frame.pack(fill=tk.X, pady=2)
        ttk.Label(curve_type_frame, text="Type:").pack(side=tk.LEFT, padx=5)
        self.curve_type_var = tk.StringVar(value="linear")
        curve_type_combo = ttk.Combobox(curve_type_frame, textvariable=self.curve_type_var,
                                       values=["linear", "spline", "bezier"], width=8)
        curve_type_combo.pack(side=tk.LEFT)
        curve_type_combo.bind("<<ComboboxSelected>>", lambda e: self.preview_curve())
        
        # Create a canvas for the preview image
        self.preview_canvas = tk.Canvas(preview_frame, bg="light gray", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Connect events
        self.curve_canvas.mpl_connect('button_press_event', self.on_curve_click)
        self.curve_canvas.mpl_connect('button_release_event', self.on_curve_release)
        self.curve_canvas.mpl_connect('motion_notify_event', self.on_curve_motion)
        self.curve_canvas.mpl_connect('pick_event', self.on_curve_pick)
    
    def reset_curve(self):
        """Reset the curve to linear"""
        num_points = 5
        try:
            num_points = int(self.points_var.get())
            if num_points < 3:
                num_points = 3
        except (ValueError, AttributeError):
            pass
            
        self.curve_points = []
        for i in range(num_points):
            x = i / (num_points - 1)
            self.curve_points.append((x, x))
            
        if hasattr(self, 'points_var'):
            self.points_var.set(str(num_points))
            
        self.update_curve_plot()
    
    def add_curve_point(self):
        """Add a new point to the curve"""
        max_dist = 0
        insert_idx = 1
        
        for i in range(len(self.curve_points) - 1):
            x1, y1 = self.curve_points[i]
            x2, y2 = self.curve_points[i + 1]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if dist > max_dist:
                max_dist = dist
                insert_idx = i + 1
                
        x1, y1 = self.curve_points[insert_idx - 1]
        x2, y2 = self.curve_points[insert_idx]
        new_point = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        self.curve_points.insert(insert_idx, new_point)
        self.update_curve_plot()
    
    def remove_curve_point(self):
        """Remove the selected point from the curve"""
        if self.selected_point is None:
            messagebox.showinfo("Curve Editor", "Select a point to remove")
            return
            
        if self.selected_point == 0 or self.selected_point == len(self.curve_points) - 1:
            messagebox.showinfo("Curve Editor", "Cannot remove endpoint")
            return
            
        self.curve_points.pop(self.selected_point)
        self.selected_point = None
        
        if hasattr(self, 'points_var'):
            self.points_var.set(str(len(self.curve_points)))
            
        self.update_curve_plot()
    
    def apply_curve(self):
        """Apply the current curve to the image and close the editor"""
        # Set the curve toggle to enabled
        self.curve_var.set(True)
        self.curve_enabled = True
        
        # Close the curve editor window
        self.curve_window.destroy()
        
        # Process the image with the current curve
        self.process_image()
    
    def preview_curve(self):
        """Preview the curve effect on the current image"""
        if self.out_image is None:
            return
            
        preview_img = self.out_image.copy()
        curve_type = self.curve_type_var.get() if hasattr(self, 'curve_type_var') else 'linear'
        preview_img = self.processor.apply_curve(preview_img, self.curve_points, curve_type)
        self.display_preview(preview_img)
    
    def display_preview(self, image):
        """Display an image in the preview canvas"""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 400
            canvas_height = 300
            
        img_height, img_width = image_rgb.shape[:2]
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        pil_image = Image.fromarray(resized_image)
        self.preview_tk_image = ImageTk.PhotoImage(image=pil_image)
        
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            canvas_width//2, canvas_height//2,
            image=self.preview_tk_image, anchor=tk.CENTER
        )
    
    def update_curve_plot(self):
        """Update the curve plot with current points"""
        self.curve_points = sorted(self.curve_points, key=lambda p: p[0])
        
        x_points = [p[0] for p in self.curve_points]
        y_points = [p[1] for p in self.curve_points]
        
        x_interp = np.linspace(0, 1, 100)
        y_interp = np.interp(x_interp, x_points, y_points)
        self.curve_line.set_data(x_interp, y_interp)
        
        self.curve_points_plot.set_data(x_points, y_points)
        
        self.curve_canvas.draw()
        
        if hasattr(self, 'out_image') and self.out_image is not None:
            self.preview_curve()
    
    def on_curve_pick(self, event):
        """Handle picking a control point"""
        if event.artist == self.curve_points_plot:
            self.selected_point = event.ind[0]
    
    def on_curve_click(self, event):
        """Handle mouse click on the curve plot"""
        if event.inaxes != self.curve_ax:
            return
            
        x_points = [p[0] for p in self.curve_points]
        y_points = [p[1] for p in self.curve_points]
        
        for i, (x, y) in enumerate(zip(x_points, y_points)):
            if abs(event.xdata - x) < 0.05 and abs(event.ydata - y) < 0.05:
                self.selected_point = i
                break
    
    def on_curve_release(self, event):
        """Handle mouse release on the curve plot"""
        self.selected_point = None
    
    def on_curve_motion(self, event):
        """Handle mouse motion on the curve plot"""
        if event.inaxes != self.curve_ax or self.selected_point is None:
            return
            
        if self.selected_point == 0 or self.selected_point == len(self.curve_points) - 1:
            x = self.curve_points[self.selected_point][0]
            y = np.clip(event.ydata, 0, 1)
            self.curve_points[self.selected_point] = (x, y)
        else:
            x = np.clip(event.xdata, 0, 1)
            y = np.clip(event.ydata, 0, 1)
            self.curve_points[self.selected_point] = (x, y)
        
        self.update_curve_plot()
    
    def on_points_changed(self, event):
        """Handle change in number of curve points"""
        try:
            num_points = int(self.points_var.get())
            if num_points < 3:
                num_points = 3
            
            new_points = []
            new_points.append((0.0, 0.0))  # Keep start point
            
            for i in range(1, num_points - 1):
                x = i / (num_points - 1)
                
                # Interpolate y value from current curve
                x_points = [p[0] for p in self.curve_points]
                y_points = [p[1] for p in self.curve_points]
                y = np.interp(x, x_points, y_points)
                
                new_points.append((x, y))
                
            new_points.append((1.0, 1.0))  # Keep end point
            
            self.curve_points = new_points
            self.update_curve_plot()
        except ValueError:
            pass
    
    def on_curve_editor_close(self):
        """Handle the curve editor window being closed with the X button"""
        self.curve_var.set(True)
        self.curve_enabled = True
        
        # Use a reference to the window before destroying it
        window = self.curve_window
        
        # Schedule the window destruction and image processing on the main thread
        self.root.after(10, lambda: self._finish_curve_editor_close(window))
    
    def _finish_curve_editor_close(self, window):
        """Safely finish closing the curve editor from the main thread"""
        if window.winfo_exists():
            window.destroy()
        self.process_image()
    
    def run(self):
        """Main application loop"""
        self.load_files()
        
        # Start the Tkinter main loop
        self.root.mainloop()


def run_film_scanner():
    """Run the film scanner application"""
    app = FilmScannerApp()
    app.run()


if __name__ == '__main__':
    run_film_scanner()

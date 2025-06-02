import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QSlider, 
                           QGridLayout, QVBoxLayout, QHBoxLayout, QFrame, 
                           QShortcut)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QKeySequence

from function import loadDatamatrix

# compute friction force using lugre from vector of velocities
def compute_lugre_friction(time, rel_speed, sigma_0, sigma_1, sigma_2, F_c, F_s, v_s):
    n = len(rel_speed)
    z = np.zeros(n)
    friction_force = np.zeros(n)
    
    # z and friction force for each time step
    for i in range(1, n):
        
        dt = (time[i] - time[i-1])*0.001*0.001
        
        # g(v)
        g_v_rel = (F_c + (F_s - F_c) * np.exp(-(rel_speed[i] / v_s) ** 2))/sigma_0
        
        # update internal state z with forward euler
        z[i] = z[i-1] + dt * (rel_speed[i] - z[i-1] * abs(rel_speed[i]) / g_v_rel)
        
        # dzdt
        dzdt = (z[i] - z[i-1]) / dt
        
        # friction
        friction_force[i] = sigma_0 * z[i] + sigma_1 * dzdt + sigma_2 * rel_speed[i]
    
    return friction_force

class ParameterAdjuster(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # initialize parameters
        self.current_param = 'sigma_0'
        self.step_sizes = {
            'sigma_0': 10,
            'sigma_1': 0.01,
            'sigma_2': 0.0001,
            'F_c': 0.0001,
            'F_s': 0.0001,
            'v_s': 0.000001
        }
        
        self.params = {
            'sigma_0': 4339.44,
            'sigma_1': 0.55,
            'sigma_2': 0.0698,
            'F_c': 0.1930,
            'F_s': 0.0566,
            'v_s': 0.167935
        }
        
        self.param_ranges = {
            'sigma_0': (1E-5, 10000),
            'sigma_1': (1E-5, 400),
            'sigma_2': (1E-5, 5),
            'F_c': (1E-5, 1),
            'F_s': (1E-5, 1),
            'v_s': (1E-5, 1)
        }

        # prevent overflow
        self.scale_factors = {
            'sigma_0': 100,         # 10000 * 100 = 1,000,000
            'sigma_1': 10000,       # 400 * 10000 = 4,000,000
            'sigma_2': 100000,      # 5 * 100000 = 500,000
            'F_c': 1000000,         # 1 * 1000000 = 1,000,000 
            'F_s': 1000000,         # 1 * 1000000 = 1,000,000 
            'v_s': 1000000          # 1 * 1000000 = 1,000,000 
        }
        
        self.is_fullscreen = False
        
        self.last_resize_time = 0
        self.resize_delay = 250  # ms
        
        # set up the UI
        self.setup_ui()
        self.load_data()
        self.update_plot()
    
    def setup_ui(self):
        self.setWindowTitle("LuGre Model Parameter Adjuster")
        self.setGeometry(50, 50, 1920, 1080)  # resolution
        self.setMinimumSize(1400, 900)  # base size
        
        # main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QGridLayout(main_widget)
        
        # parameter panel (left side)
        param_panel = QFrame()
        param_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        param_panel.setMinimumWidth(600)  # enough width for the panel
        param_layout = QVBoxLayout(param_panel)
        param_layout.setContentsMargins(20, 20, 20, 20)
        
        # sliders and labels
        self.sliders = {}
        self.value_labels = {}
        self.param_labels = {}
        
        for param, value in self.params.items():
            # frame for this parameter row
            param_row = QFrame()
            row_layout = QHBoxLayout(param_row)
            row_layout.setSpacing(20)  # spacing
            
            # parameter label
            label = QLabel(param)
            label.setFont(QFont("Arial", 18))
            label.setMinimumWidth(100)  # fixed width for parameter names
            self.param_labels[param] = label
            row_layout.addWidget(label)
            
            # slider
            slider = QSlider(Qt.Horizontal)
            min_val = self.param_ranges[param][0]
            max_val = self.param_ranges[param][1]
            
            # scale factor
            scale_factor = self.scale_factors[param]
            
            slider.setMinimum(int(min_val * scale_factor))
            slider.setMaximum(int(max_val * scale_factor))
            slider.setValue(int(value * scale_factor))
            slider.setFixedWidth(350)
            
            # connect using a lambda that captures param
            slider.valueChanged.connect(lambda val, p=param: self.on_slider_change(p, val))
            self.sliders[param] = slider
            row_layout.addWidget(slider)
            
            # spacing before the value label
            row_layout.addSpacing(10)
            
            # value label
            value_label = QLabel(f"{value:.6f}")
            value_label.setFont(QFont("Arial", 18))
            value_label.setMinimumWidth(150)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ddd;")
            self.value_labels[param] = value_label
            row_layout.addWidget(value_label)
            
            # push everything to the left
            row_layout.addStretch()
            
            param_layout.addWidget(param_row)
        
        param_layout.addStretch()
        
        # plot panel (right side)
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        
        # a matplotlib figure
        plt.rcParams.update({
            'font.size': 20,
            'axes.titlesize': 22,
            'axes.labelsize': 20,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18
        })
        
        # widgets to main layout
        main_layout.addWidget(param_panel, 0, 0, 1, 1)
        main_layout.addWidget(self.plot_widget, 0, 1, 1, 3)
        main_layout.setColumnStretch(0, 0)  # no stretch parameter panel
        main_layout.setColumnStretch(1, 1)  # stretch plot area
        
        # set up keyboard shortcuts
        self.setup_shortcuts()
        
        # highlight initially selected parameter
        self.highlight_selected_parameter()
        
        self.create_figure()
    
    def create_figure(self):
        # clear plot if it exists
        if hasattr(self, 'canvas'):
            layout = self.plot_layout
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        
        # get current size of the plot widget
        self.plot_widget.update()
        width = self.plot_widget.width() / 100.0
        height = self.plot_widget.height() / 100.0
        
        # minimum size
        width = max(width, 8)
        height = max(height, 6)
        
        # figure and axes
        self.fig = plt.figure(figsize=(width, height), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # canvas
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.plot_layout.addWidget(self.canvas)
        
        # navigation toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self.plot_widget)
        self.plot_layout.addWidget(self.toolbar)
    
    # keyboard shortcuts for finetuning parameters
    def setup_shortcuts(self):
        # left arrow - previous parameter
        self.left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.left_shortcut.activated.connect(self.prev_parameter)
        
        # right arrow - next parameter
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.right_shortcut.activated.connect(self.next_parameter)
        
        # up arrow - increase value
        self.up_shortcut = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.up_shortcut.activated.connect(self.increase_value)
        
        # down arrow - decrease value
        self.down_shortcut = QShortcut(QKeySequence(Qt.Key_Down), self)
        self.down_shortcut.activated.connect(self.decrease_value)
        
        # F11 - toggle fullscreen
        self.f11_shortcut = QShortcut(QKeySequence(Qt.Key_F11), self)
        self.f11_shortcut.activated.connect(self.toggle_fullscreen)
        
        # Escape - exit fullscreen
        self.esc_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.esc_shortcut.activated.connect(self.end_fullscreen)
        
        # keyboard shortcuts info
        self.info_label = QLabel("Use ←/→ to select parameter, ↑/↓ to adjust value, F11 for fullscreen")
        self.info_label.setStyleSheet("background-color: #e0e0e0; padding: 5px;")
        self.statusBar().addWidget(self.info_label)
    
    def on_slider_change(self, param, value):
        # convert the integer slider value back to float
        scale_factor = self.scale_factors[param]
        actual_value = value / scale_factor
        self.params[param] = actual_value
        
        # update value label with the actual parameter value
        self.value_labels[param].setText(f"{actual_value:.6f}")
        
        self.update_plot()
    
    # highlight the currently selected parameter
    def highlight_selected_parameter(self):
        for param, label in self.param_labels.items():
            if param == self.current_param:
                label.setStyleSheet("color: blue; font-weight: bold;")
                # highlight the value label
                self.value_labels[param].setStyleSheet(
                    "background-color: #d0e0ff; padding: 5px; border: 2px solid #4080ff; font-weight: bold;"
                )
            else:
                label.setStyleSheet("color: black; font-weight: normal;")
                self.value_labels[param].setStyleSheet(
                    "background-color: #f0f0f0; padding: 5px; border: 1px solid #ddd;"
                )
    
    # data loading and stuff
    def load_data(self):
        self.time = np.load('DATAM/timeM4.npy')
        stage_pos_raw_selected = np.load('DATAM/stage_posM4.npy')
        mobile_speed_raw_selected = np.load('DATAM/mobile_speedM4.npy')
        
        adjust = 0
        window_size = 120
        window_size_pos = 120
        start = 0
        end = 2400
        select_time = False
        Plot_prepare = False
        M = 9.7
        
        self.time, self.rel_speed, stage_df, self.mobile_df = loadDatamatrix(
            stage_pos_raw_selected, 
            mobile_speed_raw_selected, 
            self.time, 
            adjust, 
            window_size, 
            window_size_pos,
            start, 
            end, 
            Plot_prepare, 
            select_time,
            M
        )
    
    # update the plot with current parameters
    def update_plot(self):
        if not hasattr(self, 'ax') or not self.ax:
            return
            
        self.ax.clear()
        
        friction_force = compute_lugre_friction(
            self.time, 
            self.rel_speed, 
            self.params['sigma_0'],
            self.params['sigma_1'],
            self.params['sigma_2'],
            self.params['F_c'],
            self.params['F_s'],
            self.params['v_s']
        )
        
        # set marker size based on fullscreen state
        marker_size = 14 if self.is_fullscreen else 12
        
        # plot
        self.ax.plot(self.rel_speed, self.mobile_df['friction_force_cut']/10.0, 'r.', 
                     label='Measured', alpha=0.6, markersize=marker_size)
        self.ax.plot(self.rel_speed, friction_force, 'b.', 
                     label='Model', alpha=0.6, markersize=marker_size)
        
        label_fontsize = 22 if self.is_fullscreen else 20
        title_fontsize = 24 if self.is_fullscreen else 22
        
        self.ax.set_xlabel('Relative Velocity (m/s)', fontsize=label_fontsize)
        self.ax.set_ylabel('Friction Force (N)', fontsize=label_fontsize)
        self.ax.set_title('Friction Force vs Relative Velocity', fontsize=title_fontsize, fontweight='bold')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend(fontsize=label_fontsize)
        
        self.fig.tight_layout(pad=2.0)  # padding for better spacing
        self.canvas.draw()
    
    # select the next or previous parameter
    def next_parameter(self):
        params_list = list(self.params.keys())
        current_index = params_list.index(self.current_param)
        self.current_param = params_list[(current_index + 1) % len(params_list)]
        self.highlight_selected_parameter()
    
    def prev_parameter(self):
        params_list = list(self.params.keys())
        current_index = params_list.index(self.current_param)
        self.current_param = params_list[(current_index - 1) % len(params_list)]
        self.highlight_selected_parameter()
    
    # increase or decrease the value of the currently selected parameter
    def increase_value(self):
        current_value = self.params[self.current_param]
        new_value = current_value + self.step_sizes[self.current_param]
        max_value = self.param_ranges[self.current_param][1]
        new_value = min(new_value, max_value)
        
        self.params[self.current_param] = new_value
        
        # update slider position
        scale_factor = self.scale_factors[self.current_param]
        self.sliders[self.current_param].setValue(int(new_value * scale_factor))
        
        # update value label
        self.value_labels[self.current_param].setText(f"{new_value:.6f}")
        
        self.update_plot()
    
    def decrease_value(self):
        current_value = self.params[self.current_param]
        new_value = current_value - self.step_sizes[self.current_param]
        min_value = self.param_ranges[self.current_param][0]
        new_value = max(new_value, min_value)
        
        self.params[self.current_param] = new_value
        
        # update slider position
        scale_factor = self.scale_factors[self.current_param]
        self.sliders[self.current_param].setValue(int(new_value * scale_factor))
        
        # update value label
        self.value_labels[self.current_param].setText(f"{new_value:.6f}")
        
        self.update_plot()
    
    # fullscreen
    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        
        if self.is_fullscreen:
            self.showFullScreen()
            
            # update font sizes for fullscreen
            for param, label in self.param_labels.items():
                font = QFont("Arial", 22)
                if param == self.current_param:
                    font.setBold(True)
                label.setFont(font)
                
            for param, label in self.value_labels.items():
                label.setFont(QFont("Arial", 22))
        else:
            self.showNormal()
            
            # reset fonts
            for param, label in self.param_labels.items():
                font = QFont("Arial", 18)
                if param == self.current_param:
                    font.setBold(True)
                label.setFont(font)
                
            for param, label in self.value_labels.items():
                label.setFont(QFont("Arial", 18))
        
        self.update_plot()
    
    def end_fullscreen(self):
        if self.is_fullscreen:
            self.toggle_fullscreen()
    
    # handle resize events
    def resizeEvent(self, event):
        current_time = int(time.time() * 1000)
        
        # only if enough time since last resize
        if current_time - self.last_resize_time > self.resize_delay:
            self.last_resize_time = current_time
            
            # QTimer to defer the resize handling
            QTimer.singleShot(100, self.handle_resize)
        
        # resize event handler
        super().resizeEvent(event)
    
    def handle_resize(self):
        self.create_figure()
        self.update_plot()

# main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ParameterAdjuster()
    window.show()
    sys.exit(app.exec_())
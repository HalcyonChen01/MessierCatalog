#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import math

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.facecolor'] = 'black'


class MessierDataProcessor:
    def scrape_messier_data(self):
        csv_complete_path = os.path.join(os.path.dirname(__file__), '..', 'messier_catalog_complete.csv')
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'messier_catalog.csv')
        
        if os.path.exists(csv_complete_path):
            df = pd.read_csv(csv_complete_path)
            print(f"✅ 加载完整Messier数据: {len(df)} 个天体")
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✅ 加载Messier数据: {len(df)} 个天体")
        else:
            data = [
                ['M1', 'Crab Nebula', 83.6, 22.0, 8.4, 'Supernova remnant', 'winter'],
                ['M31', 'Andromeda', 10.7, 41.3, 3.4, 'Galaxy', 'autumn'],
                ['M42', 'Orion Nebula', 83.8, -5.4, 4.0, 'Nebula', 'winter'],
            ]
            df = pd.DataFrame(data, columns=['Messier', 'Common_Name', 'RA_Decimal', 'Dec_Decimal', 'Magnitude', 'Object_Type', 'Best_Season'])
            print("⚠️ 使用备用数据")

        if 'Object_Type' not in df.columns and 'Type' in df.columns:
            df['Object_Type'] = df['Type']
        
        if 'RA_Decimal' not in df.columns:
            if 'RA' in df.columns:
                df['RA_Decimal'] = df['RA'] * 15
            else:
                df['RA_Decimal'] = 0
        
        if 'Dec_Decimal' not in df.columns:
            df['Dec_Decimal'] = df['Dec'] if 'Dec' in df.columns else 0
        
        df['Name'] = df.apply(lambda r: r.get('Common_Name', '') if pd.notna(r.get('Common_Name')) else r.get('Messier', ''), axis=1)

        def ra_to_best_month(ra_decimal):
            """根据赤经计算最佳观测月份 (0-11对应1-12月)"""
            if pd.isna(ra_decimal):
                return 5
            month_index = int((ra_decimal / 30 + 8) % 12)
            return month_index
        
        if 'RA_Decimal' in df.columns:
            df['Month_Index'] = df['RA_Decimal'].apply(ra_to_best_month)
        else:
            df['Month_Index'] = 5
        
        df['Month_Index'] = df['Month_Index'].astype(int)

        df = df[df['Month_Index'].isin([0, 1, 2, 3, 4, 5])].copy()

        df['X_coord'] = df['RA_Decimal']
        df['Y_coord'] = df['Dec_Decimal']
        df['Z_coord'] = df['Month_Index']
        
        print(f"Data columns: {list(df.columns)}")
        print(f"Object types: {df['Object_Type'].value_counts().to_dict()}")
        print(f"Monthly distribution: {df['Month_Index'].value_counts().sort_index().to_dict()}")
        return df


class CelestialBackground:
    def __init__(self):
        self.stars = self._generate_background_stars()
        self.star_positions = np.array([[s['ra'], s['dec']] for s in self.stars])
        self.star_sizes = np.array([s['size'] for s in self.stars])
        self.star_phases = np.array([s['phase'] for s in self.stars])

    def _generate_background_stars(self, n=100):
        np.random.seed(42)
        stars = []
        for _ in range(n):
            ra = np.random.uniform(0, 360)
            dec = np.random.normal(0, 25)
            dec = max(-80, min(80, dec))
            stars.append({'ra': ra, 'dec': dec, 'size': np.random.uniform(0.5, 3.0), 'phase': np.random.uniform(0, 2*np.pi)})
        return stars

    def draw_background_stars(self, ax, frame=0):
        for i, s in enumerate(self.stars):
            tw = 0.4 + 0.6 * abs(np.sin(s['phase'] + frame * 0.15))
            ax.scatter(s['ra'], s['dec'], c='white', s=s['size'] * (0.8 + tw), 
                      alpha=0.3 * tw, marker='.', edgecolors='none')


class MessierAnimator:
    def __init__(self, df):
        self.df = df
        print(f"Loaded {len(self.df)} objects for animation")
        
        self.bg = CelestialBackground()
        self.object_type_styles = self._setup_object_type_styles()
        self.monthly_data_cache = {}
        for month_idx in range(6):
            month_data = self.df[self.df['Month_Index'] == month_idx]
            self.monthly_data_cache[month_idx] = month_data
        
        self.object_states = {}
        self.trail_positions = defaultdict(list)
        self.fade_states = {}
        self._initialize_object_states()
        
        self.is_paused = False
        
        self.clicked_objects = set()
        self.label_texts = {}
        self.mouse_press_time = None
        self.is_long_press = False
        self.objects_data = {}
        self.paused_frame = 0
        
        self.current_objects_positions = {}
        
        self.month_names = ['January', 'February', 'March', 'April', 'May', 'June']
        
        self.x_ticks = np.arange(0, 361, 60)
        self.y_ticks = np.arange(-45, 81, 25)
        self.z_ticks = np.arange(0, 6)
        self.x_labels = [f'{x}°' for x in self.x_ticks]
        self.y_labels = [f'{y}°' for y in self.y_ticks]
        self.z_labels = [str(i+1) for i in range(6)]
        
        self.frames_per_month = 72
        self.total_frames = 432
        self.rotation_speed = 360 / self.total_frames
        
        self.x_grid_lines = [(x, [-45, 80], [0, 0]) for x in np.arange(0, 361, 90)]
        self.y_grid_lines = [([0, 360], y, [0, 0]) for y in np.arange(-45, 81, 25)]
        
        self.month_labels = {}
        for i in range(6):
            month_name = self.month_names[i][:3]
            x_pos = 0.92
            y_pos = 0.85 - i * 0.12
            self.month_labels[i] = {
                'name': month_name,
                'full_name': self.month_names[i],
                'position': (x_pos, y_pos),
                'clickable_area': {'x': [x_pos-0.03, x_pos+0.03], 'y': [y_pos-0.02, y_pos+0.02]}
            }
        
    def _initialize_object_states(self):
        for idx, obj in self.df.iterrows():
            messier_id = obj.get('Messier', f'obj_{idx}')
            obj_type = obj.get('Object_Type', 'Unknown')
            
            if obj_type in ['Open cluster', 'Cluster & nebula']:
                float_range = 0.3
                pulse_freq = 0.015
            elif obj_type == 'Galaxy':
                float_range = 0.25
                pulse_freq = 0.01
            elif obj_type == 'Globular cluster':
                float_range = 0.2
                pulse_freq = 0.012
            elif obj_type in ['Nebula', 'Planetary nebula']:
                float_range = 0.35
                pulse_freq = 0.008
            else:
                float_range = 0.3
                pulse_freq = 0.01
            
            fade_in_duration = np.random.uniform(12, 48)
            fade_out_duration = np.random.uniform(12, 48)
            fade_delay = np.random.uniform(0, 24)
            
            self.object_states[messier_id] = {
                'original_pos': [obj['X_coord'], obj['Y_coord'], obj['Z_coord']],
                'float_offset': [0.0, 0.0, 0.0],
                'float_phase': [np.random.uniform(0, 2*np.pi) for _ in range(3)],
                'float_range': float_range,
                'pulse_phase': np.random.uniform(0, 2*np.pi),
                'pulse_freq': pulse_freq,
                'last_active_frame': -1,
                'fade_alpha': 0.0,
                'fade_in_duration': fade_in_duration,
                'fade_out_duration': fade_out_duration,
                'fade_delay': fade_delay,
                'fade_start_frame': -1
            }
            
    def _update_object_effects(self, frame, messier_id, is_active):
        if messier_id not in self.object_states:
            return
            
        state = self.object_states[messier_id]
        
        for i in range(3):
            state['float_phase'][i] += 0.02 + i * 0.005
            amplitude = state['float_range'] * (1.2 if is_active else 0.6)
            state['float_offset'][i] = amplitude * np.sin(state['float_phase'][i])
        
        if is_active:
            if state['last_active_frame'] < frame - 72:
                state['fade_start_frame'] = frame + state['fade_delay']
            state['last_active_frame'] = frame
            
            if frame >= state['fade_start_frame'] and state['fade_start_frame'] > 0:
                fade_progress = (frame - state['fade_start_frame']) / state['fade_in_duration']
                state['fade_alpha'] = min(1.0, fade_progress)
            elif state['fade_start_frame'] > 0:
                state['fade_alpha'] = 0.0
            else:
                state['fade_alpha'] = 1.0
        else:
            pulse_amplitude = 0.3
            pulse_base = 0.7
            state['fade_alpha'] = pulse_base + pulse_amplitude * np.sin(frame * 0.01 + state['pulse_phase'])
        
        if is_active:
            current_pos = [
                state['original_pos'][0] + state['float_offset'][0],
                state['original_pos'][1] + state['float_offset'][1],
                state['original_pos'][2] + state['float_offset'][2]
            ]
            self.trail_positions[messier_id].append(current_pos)
            if len(self.trail_positions[messier_id]) > 8:
                self.trail_positions[messier_id].pop(0)
        
        return state
        
    def _get_pulsating_size(self, base_size, frame, pulse_freq, pulse_phase):
        pulse_factor = 0.7 + 0.3 * np.sin(frame * pulse_freq + pulse_phase)
        return base_size * pulse_factor
        
    def _draw_trail_effect(self, ax, messier_id, color):
        if messier_id not in self.trail_positions or len(self.trail_positions[messier_id]) < 2:
            return
            
        positions = self.trail_positions[messier_id]
        for i, pos in enumerate(positions[:-1]):
            alpha = (i + 1) / len(positions) * 0.3
            size = 10 + i * 2
            ax.scatter(pos[0], pos[1], pos[2], 
                      color=color, s=size, alpha=alpha, marker='o', edgecolors='none')
    
    def _draw_radial_glow(self, ax, x, y, z, base_color, size, alpha, style):
        if isinstance(base_color, tuple) and len(base_color) == 3:
            r, g, b = base_color
        else:
            r, g, b = 1.0, 1.0, 1.0
        
        base_radius = np.sqrt(size / np.pi)
        
        glow_layers = 2
        
        for layer in range(glow_layers, 0, -1):
            layer_ratio = layer / glow_layers
            
            glow_radius = base_radius * (1.0 + layer_ratio * 1.2)
            layer_size = np.pi * (glow_radius ** 2)
            
            layer_alpha = alpha * (1.0 - layer_ratio * 0.6)
            
            brightness_factor = 1.0 + (1.0 - layer_ratio) * 0.2
            layer_color = (
                min(1.0, r * brightness_factor),
                min(1.0, g * brightness_factor), 
                min(1.0, b * brightness_factor)
            )
            
            ax.scatter(x, y, z,
                      color=layer_color,
                      s=layer_size,
                      alpha=layer_alpha,
                      marker='o',
                      edgecolors='none')
        
        core_brightness = 1.2
        core_color = (
            min(1.0, r * core_brightness),
            min(1.0, g * core_brightness),
            min(1.0, b * core_brightness)
        )
        
        marker = style.get('marker', 'o')
        edgecolor = style.get('edgecolor', core_color)
        linewidth = style.get('linewidth', 1)
        facecolor = style.get('facecolor', core_color)
        
        if facecolor == 'none':
            scatter_obj = ax.scatter(x, y, z,
                      facecolors='none',
                      edgecolors=edgecolor,
                      s=size * 0.8,
                      alpha=min(1.0, alpha * 1.1),
                      marker=marker,
                      linewidth=linewidth,
                      picker=5)
        else:
            scatter_obj = ax.scatter(x, y, z,
                      color=core_color,
                      s=size * 0.6,
                      alpha=min(1.0, alpha * 1.1),
                      marker=marker,
                      edgecolors=edgecolor,
                      linewidth=linewidth,
                      picker=5)
        
        return scatter_obj
        
        self._precompute_grid_data()
        
        self.month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
    
    def _precompute_grid_data(self):
        self.central_grid_h = []
        for dec in np.arange(-45, 81, 25):
            self.central_grid_h.append({
                'x': np.array([180, 180]),
                'y': np.array([dec, dec]),
                'z': np.array([0, 11])
            })
        
        self.central_grid_v = []
        for month in np.arange(0, 12, 2):
            self.central_grid_v.append({
                'x': np.array([180, 180]),
                'y': np.array([-45, 80]),
                'z': np.array([month, month])
            })
        
        self.bottom_grid_x = []
        for x_pos in np.arange(0, 361, 90):
            self.bottom_grid_x.append({
                'x': np.array([x_pos, x_pos]),
                'y': np.array([-45, 80]),
                'z': np.array([0, 0])
            })
        
        self.bottom_grid_y = []
        for y_pos in np.arange(-45, 81, 25):
            self.bottom_grid_y.append({
                'x': np.array([0, 360]),
                'y': np.array([y_pos, y_pos]),
                'z': np.array([0, 0])
            })
    
    def _setup_object_type_styles(self):
        return {
            'Open cluster': {'marker': 'P', 'linewidth': 2},
            'Cluster & nebula': {'marker': 'P', 'linewidth': 2},
            'Galaxy': {'marker': 'o', 'linewidth': 1},
            'Planetary nebula': {'marker': 'D', 'linewidth': 1},
            'Supernova remnant': {'marker': '^', 'linewidth': 1},
            'Nebula': {'marker': 'o', 'linewidth': 1},
            'Dark nebula': {'marker': 'o', 'linewidth': 1},
            'Globular cluster': {'marker': 'o', 'linewidth': 3, 'facecolor': 'none'},
            'Double star': {'marker': '*', 'linewidth': 1},
        }

    def _get_magnitude_based_color(self, obj_type, magnitude):
        if magnitude <= 5:
            saturation_level = 'high'
        elif magnitude <= 10:
            saturation_level = 'medium'
        else:
            saturation_level = 'low'
        
        color_configs = {
            'Open cluster': {
                'low': (1.0, 0.937, 0.502),
                'medium': (1.0, 0.875, 0.2),
                'high': (1.0, 0.8, 0.0)
            },
            'Cluster & nebula': {
                'low': (1.0, 0.937, 0.502),
                'medium': (1.0, 0.875, 0.2),
                'high': (1.0, 0.8, 0.0)
            },
            'Galaxy': {
                'low': (1.0, 0.6, 0.502),
                'medium': (1.0, 0.333, 0.2),
                'high': (1.0, 0.133, 0.0)
            },
            'Globular cluster': {
                'low': (0.933, 0.667, 0.933),
                'medium': (0.8, 0.333, 0.8),
                'high': (0.667, 0.0, 0.667)
            },
            'Planetary nebula': {
                'low': (0.6, 0.933, 0.933),
                'medium': (0.0, 0.8, 0.8),
                'high': (0.0, 0.6, 0.667)
            },
            'Nebula': {
                'low': (1.0, 0.733, 0.867),   # RGB(255, 187, 221) → #FFBBDD
                'medium': (1.0, 0.4, 0.733),  # RGB(255, 102, 187) → #FF66BB
                'high': (0.867, 0.0, 0.467)   # RGB(221, 0, 119) → #DD0077
            },
            'Supernova remnant': {
                'low': (0.733, 1.0, 0.733),   # RGB(187, 255, 187) → #BBFFBB
                'medium': (0.333, 0.933, 0.533), # RGB(85, 238, 136) → #55EE88
                'high': (0.0, 0.8, 0.333)     # RGB(0, 204, 85) → #00CC55
            },
            'Dark nebula': {
                'low': (0.4, 0.3, 0.2),       # 保持棕色系
                'medium': (0.5, 0.35, 0.25),
                'high': (0.6, 0.4, 0.3)
            },
            'Double star': {
                'low': (0.7, 0.7, 0.7),       # 保持白色系，但有饱和度变化
                'medium': (0.85, 0.85, 0.85),
                'high': (1.0, 1.0, 1.0)
            }
        }
        
        # 获取对应颜色，如果类型不存在则使用默认白色
        colors = color_configs.get(obj_type, {
            'low': (0.5, 0.5, 0.5),
            'medium': (0.75, 0.75, 0.75), 
            'high': (1.0, 1.0, 1.0)
        })
        
        return colors[saturation_level]

    def _plot_object_with_classification(self, ax, obj_data, size=60, alpha=0.9, highlight=False, frame=0):
        """根据天体类型绘制分类标记，包含径向光晕效果和基于星等的动态配色"""
        obj_type = obj_data.get('Object_Type', 'Unknown')
        messier_id = obj_data.get('Messier', f'obj_{obj_data.name}')
        magnitude = obj_data.get('Magnitude', 10.0)  # 获取星等，默认10.0
        
        # 获取基础样式设置
        style = self.object_type_styles.get(obj_type, {'marker': 'o', 'linewidth': 1})
        
        # 根据星等计算动态颜色
        dynamic_color = self._get_magnitude_based_color(obj_type, magnitude)
        
        # 更新样式，使用动态颜色
        style['color'] = dynamic_color
        style['edgecolor'] = dynamic_color
        
        # 更新天体的视觉效果状态（为所有天体提供视觉效果，不分静态/动态）
        state = self._update_object_effects(frame, messier_id, True)  # 强制为所有天体创建效果状态
        if state is None:
            # 如果没有状态信息，使用原始坐标
            x, y, z = obj_data['X_coord'], obj_data['Y_coord'], obj_data['Z_coord']
            final_alpha = alpha
            final_size = size
        else:
            # 应用漂浮效果（所有天体都有轻微漂浮）
            x = obj_data['X_coord'] + state['float_offset'][0]
            y = obj_data['Y_coord'] + state['float_offset'][1]
            z = obj_data['Z_coord'] + state['float_offset'][2]
            
            # 对静态天体应用轻微脉动效果，对动态天体应用渐入渐出效果
            if highlight:
                # 动态天体：应用渐入渐出效果
                final_alpha = alpha * state['fade_alpha']
            else:
                # 静态天体：保持稳定透明度
                final_alpha = alpha
            
            # 应用脉动效果（所有天体都有脉动）
            final_size = self._get_pulsating_size(size, frame, state['pulse_freq'], state['pulse_phase'])
            
            # 绘制尾迹效果（仅对高亮天体）
            if highlight:
                self._draw_trail_effect(ax, messier_id, style['color'])
        
        # 如果是高亮显示，增加大小但保持透明度控制
        if highlight:
            final_size = max(final_size, 80)
            final_alpha = min(final_alpha, 0.8)  # 动态天体最大透明度限制为80%
        
        # 绘制径向光晕效果
        scatter_obj = self._draw_radial_glow(ax, x, y, z, style['color'], final_size, final_alpha, style)
        
        # 只为可点击的天体（高亮天体）存储信息和设置picker
        if highlight and scatter_obj is not None:
            # 存储天体信息用于点击检测
            object_info = {
                'messier_id': messier_id,
                'name': obj_data.get('Name', messier_id),
                'position': (x, y, z),
                'obj_data': obj_data
            }
            
            # 如果name为空或与messier_id相同，使用messier_id
            if pd.isna(object_info['name']) or object_info['name'] == '' or object_info['name'] == object_info['messier_id']:
                object_info['name'] = messier_id
                
            # 将信息绑定到scatter对象上，避免频繁清空字典
            setattr(scatter_obj, 'object_info', object_info)
            
        return scatter_obj
        
        return None

    def toggle_pause(self):
        """切换暂停/播放状态"""
        self.is_paused = not self.is_paused

    def jump_to_month(self, month_number):
        """跳转到指定月份 - 只支持1-6月"""
        if 1 <= month_number <= 6:  # 限制为1-6月
            # 计算目标帧数：每月72帧，月份从0开始索引
            target_frame = (month_number - 1) * 72
            self.paused_frame = target_frame
            month_name = self.month_names[month_number - 1]
            print(f"Jumped to {month_name} (Month {month_number})")
            # 暂停动画并强制更新到目标帧
            self.is_paused = True
            # 立即更新到目标帧并重绘
            if hasattr(self, 'animate_func'):
                self.animate_func(target_frame)  # 直接调用animate函数更新显示
            if hasattr(self, 'fig'):
                self.fig.canvas.draw()  # 强制重绘
        else:
            print(f"Month {month_number} not available. Only January-June (1-6) are supported.")

    def handle_month_click(self, event):
        """处理月份标签点击事件（2D figure坐标系）"""
        if not hasattr(event, 'xdata') and not hasattr(event, 'x'):
            return False
            
        # 获取点击位置在figure坐标系中的位置 (0-1范围)
        if hasattr(event, 'inaxes') and event.inaxes is None:
            # 如果点击在轴外，使用figure坐标
            if hasattr(event, 'x') and hasattr(event, 'y'):
                # 使用matplotlib的内置坐标转换
                click_x_fig = event.x / self.fig.bbox.width
                click_y_fig = 1.0 - (event.y / self.fig.bbox.height)  # matplotlib的Y坐标是从下往上
                print(f"Click detected at figure coords: ({click_x_fig:.3f}, {click_y_fig:.3f})")  # 调试信息
                
                # 检查是否点击了月份标签
                for month_idx, label_info in self.month_labels.items():
                    area = label_info['clickable_area']
                    label_pos = label_info['position']
                    
                    if (area['x'][0] <= click_x_fig <= area['x'][1] and 
                        area['y'][0] <= click_y_fig <= area['y'][1]):
                        print(f"✓ Clicked on {label_info['name']} (Month {month_idx+1})")
                        self.jump_to_month(month_idx + 1)
                        return True
        
        return False

    def update_objects_positions(self, frame):
        """更新当前帧所有天体的位置信息"""
        self.current_objects_positions.clear()
        
        # 当前月份的天体
        current_month_idx = (frame // 72) % 12
        monthly_data = self.monthly_data_cache[current_month_idx]
        current_month_messier_ids = set(monthly_data['Messier'].values) if len(monthly_data) > 0 else set()
        
        # 添加静态天体位置
        for _, obj in self.df.iterrows():
            if obj['Messier'] not in current_month_messier_ids:
                messier_id = obj.get('Messier', f'obj_{obj.name}')
                if messier_id in self.object_states:
                    state = self.object_states[messier_id]
                    position = (
                        obj['X_coord'] + state['float_offset'][0],
                        obj['Y_coord'] + state['float_offset'][1],
                        obj['Z_coord'] + state['float_offset'][2]
                    )
                else:
                    position = (obj['X_coord'], obj['Y_coord'], obj['Z_coord'])
                
                self.current_objects_positions[messier_id] = {
                    'position': position,
                    'data': obj,
                    'is_dynamic': False
                }
        
        # 添加动态天体位置
        if len(monthly_data) > 0:
            for _, obj in monthly_data.iterrows():
                messier_id = obj.get('Messier', f'obj_{obj.name}')
                if messier_id in self.object_states:
                    state = self.object_states[messier_id]
                    position = (
                        obj['X_coord'] + state['float_offset'][0],
                        obj['Y_coord'] + state['float_offset'][1],
                        obj['Z_coord'] + state['float_offset'][2]
                    )
                else:
                    position = (obj['X_coord'], obj['Y_coord'], obj['Z_coord'])
                
                self.current_objects_positions[messier_id] = {
                    'position': position,
                    'data': obj,
                    'is_dynamic': True
                }
        
        # 添加调试信息
        if frame % 72 == 0:  # 每个月输出一次
            print(f"  Objects positions updated: {len(self.current_objects_positions)} total objects")

    def create_animation(self):
        print('Creating 3D animation window...')
        self.fig = plt.figure(figsize=(12, 9), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='black')
        self.fig.patch.set_facecolor('black')

        # 添加图例
        self._add_legend()

        # 基本轴设置 - 扩展X轴范围以显示月份标签
        self.ax.set_xlim(0, 400)  # 扩展X轴以容纳月份标签
        self.ax.set_ylim(-45, 80)  # 优化Dec范围，聚焦在有数据的区域
        self.ax.set_zlim(0, 11)  # 0-11对应1-12月
        self.ax.set_xlabel('X (Right Ascension)', color='white')
        self.ax.set_ylabel('Y (Declination)', color='white')
        self.ax.set_zlabel('Z (Month)', color='white')  # 显示官方Z轴标签

        self.ax.set_xticks(np.arange(0, 361, 60))
        self.ax.set_yticks(np.arange(-45, 81, 25))  # 调整Y轴刻度间距
        
        # 添加键盘事件处理（仅保留空格键）
        def on_key_press(event):
            if event.key == ' ':  # 空格键暂停/播放
                self.toggle_pause()
                print(f"Animation {'Paused' if self.is_paused else 'Resumed'}")
        
        # 移除鼠标悬浮功能
        def on_mouse_move(event):
            pass
        
        # 添加鼠标点击事件处理
        def on_mouse_click(event):
            if event.button != 1:  # 只处理左键点击
                return
            # 先尝试处理月份标签点击（在figure区域）
            if self.handle_month_click(event):
                return  # 如果是月份标签点击，直接返回
        
        # 添加pick事件处理（用于处理可点击的月份标签）
        def on_pick(event):
            # 检查是否点击了月份标签
            if hasattr(event, 'artist') and hasattr(event.artist, 'get_text'):
                clicked_text = event.artist.get_text()
                # 查找对应的月份
                for month_idx, label_info in self.month_labels.items():
                    if label_info['name'] == clicked_text:
                        print(f"✓ Picked month label: {clicked_text} (Month {month_idx+1})")
                        self.jump_to_month(month_idx + 1)
                        return
        
        # 连接事件
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', on_mouse_click)
        self.fig.canvas.mpl_connect('pick_event', on_pick)
        self.ax.set_zticks(np.arange(0, 12))
        self.ax.set_xticklabels([f'{x}°' for x in np.arange(0, 361, 60)], color='white')
        self.ax.set_yticklabels([f'{y}°' for y in np.arange(-45, 81, 25)], color='white')
        self.ax.set_zticklabels([str(i+1) for i in range(12)], color='white')  # 1-12月份数字

        # 中央参考平面将在天体绘制后绘制，以形成正确的层级关系
        # 此处暂时移除参考平面的绘制

        # 隐藏所有默认网格线
        try:
            self.ax.xaxis._axinfo['grid']['color'] = (0.0, 0.0, 0.0, 0.0)
            self.ax.yaxis._axinfo['grid']['color'] = (0.0, 0.0, 0.0, 0.0)
            self.ax.zaxis._axinfo['grid']['color'] = (0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass
        self.ax.grid(False)  # 完全关闭网格

        # 隐藏或设置面板为黑色，防止默认灰色面板显示
        try:
            for a in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                try:
                    a.pane.set_visible(False)
                except Exception:
                    try:
                        a.pane.set_facecolor((0, 0, 0, 1))
                    except Exception:
                        pass
        except Exception:
            pass

        def animate(frame):
            # 暂停控制逻辑
            if self.is_paused:
                current_frame = self.paused_frame
            else:
                self.paused_frame = frame
                current_frame = frame
            
            # 清除之前在figure上绘制的文本（月份标签）
            for text in self.fig.texts[:]:
                if hasattr(text, '_picker') and text._picker:  # 只清除可点击的文本
                    text.remove()
            
            # 清除之前的标签文本
            for text in self.label_texts.values():
                text.remove()
            self.label_texts.clear()
            
            # 清除内容并快速重置坐标轴
            self.ax.clear()
            
            # 不需要每帧清空objects_data，由绑定机制处理
            self.ax.set_xlim(0, 360)
            self.ax.set_ylim(-45, 80)  # 恢复正常Y轴范围，不需要扩展了
            self.ax.set_zlim(0, 11)

            # 使用预计算的标签和刻度
            self.ax.set_xlabel('X (Right Ascension)', color='white')
            self.ax.set_ylabel('Y (Declination)', color='white')
            self.ax.set_zlabel('Z (Month)', color='white')
            self.ax.set_xticks(self.x_ticks)
            self.ax.set_yticks(self.y_ticks)
            self.ax.set_zticks(self.z_ticks)
            self.ax.set_xticklabels(self.x_labels, color='white')
            self.ax.set_yticklabels(self.y_labels, color='white')
            self.ax.set_zticklabels(self.z_labels, color='white')
            
            # 设置轴颜色
            self.ax.zaxis.label.set_color('white')
            self.ax.tick_params(axis='z', colors='white')

# 暂时移除中央参考平面的绘制，将在天体绘制后重新绘制以形成正确的层级关系

            # 每帧确保面板被隐藏或设为黑色（ax.clear() 会重置 pane）
            try:
                for a in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
                    try:
                        a.pane.set_visible(False)
                    except Exception:
                        try:
                            a.pane.set_facecolor((0, 0, 0, 1))
                        except Exception:
                            pass
            except Exception:
                pass

            # 隐藏所有默认网格线
            try:
                self.ax.xaxis._axinfo['grid']['color'] = (0.0, 0.0, 0.0, 0.0)
                self.ax.yaxis._axinfo['grid']['color'] = (0.0, 0.0, 0.0, 0.0)
                self.ax.zaxis._axinfo['grid']['color'] = (0.0, 0.0, 0.0, 0.0)
            except Exception:
                pass
            self.ax.grid(False)  # 完全关闭网格
            
            # 绘制底部网格线（使用预计算坐标）
            for x, y_range, z_range in self.x_grid_lines:
                self.ax.plot([x, x], y_range, z_range, color='white', alpha=0.4, linewidth=1.0)
            
            for x_range, y, z_range in self.y_grid_lines:
                self.ax.plot(x_range, [y, y], z_range, color='white', alpha=0.4, linewidth=1.0)

            # Z轴标签已通过 ax.set_zticklabels() 设置，无需重复的手动标签

            # 背景恒星（减少更新频率）
            if current_frame % 15 == 0:  
                self.bg.draw_background_stars(self.ax, frame=current_frame)

            # 当前月份计算（使用预计算值）- 限制为前6个月 (January to June)
            current_month_idx = (current_frame // self.frames_per_month) % 6
            monthly_data = self.monthly_data_cache[current_month_idx]
            current_month_messier_ids = set(monthly_data['Messier'].values) if len(monthly_data) > 0 else set()
            
            # 静态天体（跳过当前月份）- 减少一半数量
            static_count = 0
            for _, obj in self.df.iterrows():
                if obj['Messier'] not in current_month_messier_ids:
                    # 只显示一半的静态天体（跳过奇数索引）
                    if static_count % 2 == 0:
                        self._plot_object_with_classification(self.ax, obj, size=2.5, alpha=0.6, highlight=False, frame=current_frame)
                    static_count += 1

            # 动态天体（当前月份）- 增大尺寸使其更显眼
            if len(monthly_data) > 0:
                for _, obj in monthly_data.iterrows():
                    self._plot_object_with_classification(self.ax, obj, size=120, alpha=0.8, highlight=True, frame=current_frame)
                    
                    # 自动为动态天体显示名称
                    messier_id = obj.get('Messier')
                    name = obj.get('Name', messier_id)
                    if pd.isna(name) or name == '':
                        name = messier_id
                    
                    state = self.object_states.get(messier_id)
                    if state:
                        x = obj['X_coord'] + state['float_offset'][0]
                        y = obj['Y_coord'] + state['float_offset'][1]
                        z = obj['Z_coord'] + state['float_offset'][2]
                        
                        # 计算标签位置偏移
                        offset_distance = 25
                        label_x, label_y, label_z = x - offset_distance, y, z
                        
                        # 创建标签文本
                        text = self.ax.text(label_x, label_y, label_z, name,
                                           fontsize=9, color='white', ha='right', va='center', zorder=100,
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                        self.label_texts[messier_id] = text
            else:
                # 如果当前月份没有数据，显示提示
                self.ax.text(180, 0, 5.5, 'No objects for this month', 
                           fontsize=14, color='white', ha='center', va='center')
            
            # 绘制被点击天体的名字标签
            for messier_id in self.clicked_objects:
                # 遍历当前显示的天体寻找对应的messier_id
                found_position = None
                found_name = None
                
                # 在当前月份动态天体中查找
                if len(monthly_data) > 0:
                    matching_obj = monthly_data[monthly_data['Messier'] == messier_id]
                    if not matching_obj.empty:
                        obj_row = matching_obj.iloc[0]
                        found_position = (obj_row['X_coord'], obj_row['Y_coord'], obj_row['Z_coord'])
                        found_name = obj_row.get('Name', messier_id)
                        if pd.isna(found_name) or found_name == '' or found_name == messier_id:
                            found_name = messier_id
                
                if found_position and found_name:
                    x, y, z = found_position
                    # 计算标签位置偏移到左侧
                    offset_distance = 25
                    label_x = x - offset_distance
                    label_y = y
                    label_z = z
                    
                    # 创建标签文本
                    text = self.ax.text(label_x, label_y, label_z, found_name,
                                       fontsize=9, color='white', ha='right', va='center', zorder=100,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                    self.label_texts[messier_id] = text

            # 更新天体位置信息以支持点击
            self.update_objects_positions(current_frame)

            # 绘制月份标签（右侧竖排，独立2D覆盖层，不跟随3D旋转）
            for month_idx, label_info in self.month_labels.items():
                x_fig, y_fig = label_info['position']  # figure坐标系(0-1)
                month_name = label_info['name']
                # 高亮显示当前月份
                if month_idx == current_month_idx:
                    color = 'yellow'
                    fontweight = 'bold'
                    fontsize = 12
                    bbox = dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.8)
                else:
                    color = 'lightgray'
                    fontweight = 'normal'
                    fontsize = 10
                    bbox = dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.6)
                
                # 使用figure坐标系绘制2D文本，不会跟随3D旋转
                self.fig.text(x_fig, y_fig, month_name, 
                           fontsize=fontsize, color=color, fontweight=fontweight,
                           ha='center', va='center', bbox=bbox,
                           picker=True, transform=self.fig.transFigure)

            # 在天体绘制完成后重新绘制中央参考平面，形成正确的层级关系
            # 重绘中央参考平面 - 调整Y范围，让天体能够在前景显示
            yy, zz = np.meshgrid(np.linspace(-45, 80, 2), np.linspace(0, 11, 2))
            xx = np.full_like(yy, 180)  
            self.ax.plot_surface(xx, yy, zz, alpha=0.08, color=(1.0, 1.0, 1.0), linewidth=0)  # 降低透明度让天体更突出
            
            # 在中央参考平面上添加白色网格线（更低透明度）
            # 水平网格线（沿赤纬方向）
            for dec in np.arange(-60, 61, 30):
                self.ax.plot([180, 180], [dec, dec], [0, 11], color='white', alpha=0.3, linewidth=1.0)  # 降低网格线透明度
            
            # 垂直网格线（沿月份方向）
            for month in np.arange(0, 12, 2):
                self.ax.plot([180, 180], [-45, 80], [month, month], color='white', alpha=0.3, linewidth=1.0)  # 降低网格线透明度

            # 标题与视角
            current_month = self.month_names[current_month_idx]
            
            # 统计当前月份的天体类型和数量
            if len(monthly_data) > 0:
                type_counts = monthly_data['Object_Type'].value_counts()
                type_summary = ', '.join([f'{t}({c})' for t, c in type_counts.head(3).items()])
                object_count = len(monthly_data)
                pause_indicator = " [PAUSED - Press SPACE to resume]" if self.is_paused else " [Press SPACE to pause]"
                title = f'Messier Objects - {current_month} Best Viewing ({object_count} objects){pause_indicator}\nMain Types: {type_summary} | Click objects for details | Click month labels to jump'
            else:
                pause_indicator = " [PAUSED - Press SPACE to resume]" if self.is_paused else " [Press SPACE to pause]"
                title = f'Messier Objects - {current_month} Best Viewing (0 objects){pause_indicator}\nNo objects available for this month | Click objects for details | Click month labels to jump'
            
            self.ax.set_title(title, color='white', fontsize=11)
            
            # 摄像机旋转（使用预计算值）
            azim_angle = current_frame * self.rotation_speed
            self.ax.view_init(elev=20, azim=azim_angle)

            object_count = len(monthly_data) if len(monthly_data) > 0 else 0
            current_second = current_frame / 24  # 当前时间（秒）
            # 显示暂停状态
            pause_status = " [PAUSED]" if self.is_paused else ""
            print(f"Frame {current_frame}: {current_month} - {object_count} objects, time: {current_second:.1f}s, angle: {azim_angle:.1f}°{pause_status}")
            return []

        # 存储animate函数的引用，以便在jump_to_month中使用
        self.animate_func = animate

        # 优化的动画参数：降低帧率提高性能
        anim = animation.FuncAnimation(self.fig, animate, frames=self.total_frames, interval=50, blit=False, repeat=True)
        
        # 设置鼠标事件处理
        self.setup_mouse_events()
        
        return anim
    
    def setup_mouse_events(self):
        """设置鼠标事件处理"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
    
    def on_mouse_press(self, event):
        """鼠标按下事件"""
        import time
        self.mouse_press_time = time.time()
        self.is_long_press = False
    
    def on_mouse_release(self, event):
        """鼠标释放事件"""
        if self.mouse_press_time is not None:
            import time
            press_duration = time.time() - self.mouse_press_time
            if press_duration >= 0.5:  # 长按0.5秒以上
                self.is_long_press = True
        self.mouse_press_time = None
    
    def on_pick(self, event):
        """点击天体事件 - 使用matplotlib的picker机制"""
        if hasattr(event.artist, 'object_info'):
            obj_info = event.artist.object_info
            messier_id = obj_info['messier_id']
            name = obj_info['name']
            
            if self.is_long_press:
                # 长按：保持显示
                self.clicked_objects.add(messier_id)
                print(f"长按显示: {name}")
            else:
                # 短按：切换显示状态
                if messier_id in self.clicked_objects:
                    self.clicked_objects.remove(messier_id)
                    print(f"隐藏标签: {name}")
                else:
                    self.clicked_objects.add(messier_id)
                    print(f"显示标签: {name}")
    
    def is_point_near_object(self, mouse_event, obj_x, obj_y, obj_z):
        """已废弃 - 现在使用matplotlib的picker机制"""
        return False

    def _add_legend(self):
        """在左下角添加图例"""
        legend_elements = [
            plt.Line2D([0], [0], marker='P', color='w', label='Open cluster', markerfacecolor=self._get_magnitude_based_color('Open cluster', 3), markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='P', color='w', label='Cluster & nebula', markerfacecolor=self._get_magnitude_based_color('Cluster & nebula', 3), markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', label='Galaxy', markerfacecolor=self._get_magnitude_based_color('Galaxy', 3), markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='D', color='w', label='Planetary nebula', markerfacecolor='purple', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='^', color='w', label='Supernova remnant', markerfacecolor=self._get_magnitude_based_color('Supernova remnant', 3), markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', label='Nebula', markerfacecolor=self._get_magnitude_based_color('Nebula', 3), markersize=8, linestyle='None'),
            plt.Line2D([0], [0], marker='s', color='w', label='Dark nebula', markerfacecolor=self._get_magnitude_based_color('Dark nebula', 12), markersize=8, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', label='Globular cluster', markerfacecolor='none', markeredgecolor=self._get_magnitude_based_color('Globular cluster', 3), markersize=10, linestyle='None', linewidth=3),
            plt.Line2D([0], [0], marker='*', color='w', label='Double star', markerfacecolor=self._get_magnitude_based_color('Double star', 3), markersize=12, linestyle='None')
        ]

        legend = self.fig.legend(handles=legend_elements, loc='lower left',
                                 bbox_to_anchor=(0.02, 0.02),
                                 fontsize=9,
                                 labelcolor='white',
                                 facecolor='black',
                                 edgecolor='gray',
                                 framealpha=0.7)
        legend.set_title("Messier Object Types", prop={'size': 11, 'weight': 'bold'})
        legend.get_title().set_color('white')

def main():
    print('=== Messier Objects Monthly Animation (36s, 24fps) ===')
    processor = MessierDataProcessor()
    df = processor.scrape_messier_data()
    animator = MessierAnimator(df)
    anim = animator.create_animation()
    try:
        # 保存为24fps的GIF，36秒总时长
        anim.save('messier_monthly_24fps.gif', writer='pillow', fps=24)
        print('✓ Animation saved as: messier_monthly_24fps.gif (36 seconds, 24fps)')
    except Exception as e:
        print('Failed to save animation:', e)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()
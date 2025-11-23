#!/usr/bin/env python3
"""
Real-time monitoring dashboard for 100K image generation
Provides live statistics and progress tracking
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

class GenerationMonitor:
    """Real-time monitoring for large-scale generation"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, 'progress.json')
        self.stats_file = os.path.join(output_dir, 'final_stats.json')
        
        # Data storage for plotting
        self.timestamps = deque(maxlen=100)
        self.generation_rates = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.quality_scores = deque(maxlen=100)
        
        # Setup matplotlib for real-time plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('100K Image Generation - Live Monitor', fontsize=16)
        
    def load_progress(self):
        """Load current progress from file"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error loading progress: {e}")
            return None
    
    def calculate_statistics(self, progress_data):
        """Calculate current statistics"""
        if not progress_data:
            return None
        
        current_time = time.time()
        start_time = progress_data.get('start_time', current_time)
        generated_count = progress_data.get('generated_count', 0)
        failed_count = progress_data.get('failed_count', 0)
        
        elapsed_time = current_time - start_time
        elapsed_hours = elapsed_time / 3600
        
        if elapsed_time > 0:
            generation_rate = generated_count / elapsed_hours  # images per hour
        else:
            generation_rate = 0
        
        target_images = progress_data.get('config', {}).get('target_images', 100000)
        progress_percent = (generated_count / target_images) * 100
        
        if generation_rate > 0:
            remaining_images = target_images - generated_count
            eta_hours = remaining_images / generation_rate
            eta_time = datetime.now() + timedelta(hours=eta_hours)
        else:
            eta_hours = 0
            eta_time = None
        
        # Quality statistics
        stats = progress_data.get('stats', {})
        quality_scores = stats.get('quality_scores', [])
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Memory usage
        memory_usage = stats.get('memory_usage', [])
        current_memory = memory_usage[-1] if memory_usage else {}
        
        return {
            'generated_count': generated_count,
            'failed_count': failed_count,
            'target_images': target_images,
            'progress_percent': progress_percent,
            'elapsed_hours': elapsed_hours,
            'generation_rate': generation_rate,
            'eta_hours': eta_hours,
            'eta_time': eta_time,
            'avg_quality': avg_quality,
            'current_memory': current_memory,
            'success_rate': generated_count / (generated_count + failed_count) if (generated_count + failed_count) > 0 else 0
        }
    
    def update_plots(self, stats):
        """Update real-time plots"""
        if not stats:
            return
        
        current_time = datetime.now()
        
        # Store data for plotting
        self.timestamps.append(current_time)
        self.generation_rates.append(stats['generation_rate'])
        self.memory_usage.append(stats['current_memory'].get('gpu_memory_used', 0))
        self.quality_scores.append(stats['avg_quality'])
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Generation Progress
        ax1 = self.axes[0, 0]
        ax1.pie([stats['generated_count'], stats['target_images'] - stats['generated_count']], 
                labels=['Generated', 'Remaining'], 
                autopct='%1.1f%%',
                colors=['green', 'lightgray'])
        ax1.set_title(f"Progress: {stats['generated_count']:,}/{stats['target_images']:,}")
        
        # Plot 2: Generation Rate Over Time
        ax2 = self.axes[0, 1]
        if len(self.timestamps) > 1:
            ax2.plot(list(self.timestamps), list(self.generation_rates), 'b-')
            ax2.set_title('Generation Rate (images/hour)')
            ax2.set_ylabel('Images/Hour')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Memory Usage
        ax3 = self.axes[1, 0]
        if len(self.timestamps) > 1:
            ax3.plot(list(self.timestamps), list(self.memory_usage), 'r-')
            ax3.set_title('GPU Memory Usage (GB)')
            ax3.set_ylabel('Memory (GB)')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Quality Score Distribution
        ax4 = self.axes[1, 1]
        if self.quality_scores:
            ax4.hist(list(self.quality_scores), bins=20, alpha=0.7, color='orange')
            ax4.set_title('Quality Score Distribution')
            ax4.set_xlabel('Quality Score')
            ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
    
    def print_status(self, stats):
        """Print current status to console"""
        if not stats:
            print(" Waiting for generation to start...")
            return
        
        print("\n" + "="*60)
        print(f" 100K IMAGE GENERATION - LIVE STATUS")
        print("="*60)
        
        print(f" PROGRESS:")
        print(f"   Generated: {stats['generated_count']:,} images")
        print(f"   Failed: {stats['failed_count']:,} images")
        print(f"   Target: {stats['target_images']:,} images")
        print(f"   Progress: {stats['progress_percent']:.1f}%")
        print(f"   Success Rate: {stats['success_rate']*100:.1f}%")
        
        print(f"\n TIMING:")
        print(f"   Elapsed: {stats['elapsed_hours']:.1f} hours")
        print(f"   Generation Rate: {stats['generation_rate']:.1f} images/hour")
        if stats['eta_time']:
            print(f"   ETA: {stats['eta_time'].strftime('%Y-%m-%d %H:%M')} ({stats['eta_hours']:.1f}h remaining)")
        
        print(f"\ QUALITY:")
        print(f"   Average Quality Score: {stats['avg_quality']:.3f}")
        
        print(f"\n RESOURCES:")
        memory_info = stats['current_memory']
        if memory_info:
            print(f"   GPU Memory: {memory_info.get('gpu_memory_used', 0):.1f}GB")
            print(f"   System Memory: {memory_info.get('system_memory_percent', 0):.1f}%")
        
        # Progress bar
        progress_width = 50
        filled_width = int(progress_width * stats['progress_percent'] / 100)
        bar = "█" * filled_width + "░" * (progress_width - filled_width)
        print(f"\n [{bar}] {stats['progress_percent']:.1f}%")
    
    def run_console_monitor(self, refresh_interval=30):
        """Run console-only monitoring"""
        print(" Console monitoring started (Ctrl+C to exit)")
        
        try:
            while True:
                progress_data = self.load_progress()
                stats = self.calculate_statistics(progress_data)
                
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                self.print_status(stats)
                
                # Check if generation is complete
                if stats and stats['generated_count'] >= stats['target_images']:
                    print("\n🎉 GENERATION COMPLETE!")
                    break
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n Monitoring stopped by user")
    
    def run_gui_monitor(self, refresh_interval=10):
        """Run GUI monitoring with plots"""
        print(" GUI monitoring started (close window to exit)")
        
        def update_dashboard(frame):
            progress_data = self.load_progress()
            stats = self.calculate_statistics(progress_data)
            
            if stats:
                self.update_plots(stats)
                self.print_status(stats)
                
                # Check if complete
                if stats['generated_count'] >= stats['target_images']:
                    print("\n GENERATION COMPLETE!")
                    plt.close('all')
                    return
        
        # Start animation
        ani = animation.FuncAnimation(self.fig, update_dashboard, 
                                    interval=refresh_interval*1000, 
                                    repeat=True)
        
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\n Monitoring stopped by user")
        finally:
            plt.close('all')

def main():
    """Main monitoring function"""
    
    parser = argparse.ArgumentParser(description='Monitor 100K image generation')
    parser.add_argument('--output-dir', type=str, default='rural_driving_100k',
                       help='Output directory to monitor')
    parser.add_argument('--mode', choices=['console', 'gui'], default='console',
                       help='Monitoring mode')
    parser.add_argument('--refresh', type=int, default=30,
                       help='Refresh interval in seconds')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f" Output directory not found: {args.output_dir}")
        print(" Make sure generation has started and directory exists")
        return 1
    
    monitor = GenerationMonitor(args.output_dir)
    
    if args.mode == 'console':
        monitor.run_console_monitor(args.refresh)
    else:
        try:
            monitor.run_gui_monitor(args.refresh)
        except ImportError:
            print(" GUI mode requires matplotlib")
            print(" Install with: pip install matplotlib")
            print(" Falling back to console mode...")
            monitor.run_console_monitor(args.refresh)
    
    return 0

if __name__ == "__main__":
    exit(main())

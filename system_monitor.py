"""
System resource monitoring for AirTrace RU Backend

Implements CPU, memory, and network usage tracking with performance
baseline establishment and resource utilization monitoring.
"""

import asyncio
import logging
import psutil
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Deque
from threading import Lock
import statistics

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics snapshot"""
    timestamp: datetime
    cpu_usage: float  # Percentage
    memory_usage: int  # Bytes
    memory_percent: float  # Percentage
    disk_usage: int  # Bytes
    disk_percent: float  # Percentage
    network_io_sent: int  # Bytes
    network_io_recv: int  # Bytes
    active_connections: int
    process_count: int
    load_average: Optional[float] = None  # 1-minute load average (Unix only)


@dataclass
class NetworkIO:
    """Network I/O statistics"""
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int


@dataclass
class DiskIO:
    """Disk I/O statistics"""
    read_bytes: int
    write_bytes: int
    read_count: int
    write_count: int
    read_time: int  # milliseconds
    write_time: int  # milliseconds


@dataclass
class ProcessMetrics:
    """Process-specific metrics"""
    pid: int
    name: str
    cpu_percent: float
    memory_rss: int  # Resident Set Size
    memory_vms: int  # Virtual Memory Size
    memory_percent: float
    num_threads: int
    num_fds: int  # File descriptors (Unix only)
    create_time: float
    status: str


@dataclass
class ResourceBaseline:
    """Performance baseline for system resources"""
    cpu_baseline: float
    memory_baseline: int
    disk_baseline: int
    network_baseline_sent: int
    network_baseline_recv: int
    established_at: datetime
    sample_count: int


class SystemResourceMonitor:
    """
    System resource monitoring with baseline establishment.
    
    Monitors:
    - CPU usage and load average
    - Memory usage and availability
    - Disk usage and I/O
    - Network usage and I/O
    - Process metrics
    - Performance baselines
    """
    
    def __init__(self, max_history: int = 1000, baseline_window_minutes: int = 30):
        """
        Initialize system resource monitor.
        
        Args:
            max_history: Maximum number of metrics snapshots to keep
            baseline_window_minutes: Time window for baseline calculation
        """
        self.max_history = max_history
        self.baseline_window = timedelta(minutes=baseline_window_minutes)
        
        # Thread-safe storage
        self._lock = Lock()
        
        # Metrics history
        self.metrics_history: Deque[SystemMetrics] = deque(maxlen=max_history)
        
        # Performance baseline
        self.baseline: Optional[ResourceBaseline] = None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval = 30.0  # seconds
        
        # Alert thresholds (percentages)
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'load_average': 2.0
        }
        
        # Alert callbacks
        self.alert_callbacks: List[callable] = []
        
        # Network I/O baseline for delta calculations
        self._last_network_io: Optional[NetworkIO] = None
        self._last_disk_io: Optional[DiskIO] = None
        
        logger.info(f"System resource monitor initialized with {max_history} history "
                   f"and {baseline_window_minutes} minute baseline window")
    
    async def start_monitoring(self, interval: float = 30.0) -> None:
        """
        Start continuous system monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            logger.warning("System monitoring is already active")
            return
        
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"System monitoring started with {interval}s interval")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous system monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """
        Collect current system resource metrics.
        
        Returns:
            SystemMetrics: Current system metrics snapshot
        """
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1.0)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.used
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = disk.used
            disk_percent = (disk.used / disk.total) * 100
            
            # Network metrics
            network = psutil.net_io_counters()
            network_io_sent = network.bytes_sent
            network_io_recv = network.bytes_recv
            
            # Connection count
            try:
                connections = psutil.net_connections()
                active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                active_connections = 0
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix only)
            load_average = None
            try:
                if hasattr(psutil, 'getloadavg'):
                    load_average = psutil.getloadavg()[0]  # 1-minute load average
            except (AttributeError, OSError):
                pass
            
            metrics = SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_percent=memory_percent,
                disk_usage=disk_usage,
                disk_percent=disk_percent,
                network_io_sent=network_io_sent,
                network_io_recv=network_io_recv,
                active_connections=active_connections,
                process_count=process_count,
                load_average=load_average
            )
            
            # Store metrics
            with self._lock:
                self.metrics_history.append(metrics)
            
            # Check for alerts
            self._check_resource_alerts(metrics)
            
            logger.debug(f"Collected system metrics: CPU={cpu_usage:.1f}% "
                        f"Memory={memory_percent:.1f}% Disk={disk_percent:.1f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return minimal metrics on error
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=0.0,
                memory_usage=0,
                memory_percent=0.0,
                disk_usage=0,
                disk_percent=0.0,
                network_io_sent=0,
                network_io_recv=0,
                active_connections=0,
                process_count=0
            )
    
    def get_network_io_stats(self) -> NetworkIO:
        """Get detailed network I/O statistics"""
        try:
            net_io = psutil.net_io_counters()
            return NetworkIO(
                bytes_sent=net_io.bytes_sent,
                bytes_recv=net_io.bytes_recv,
                packets_sent=net_io.packets_sent,
                packets_recv=net_io.packets_recv,
                errors_in=net_io.errin,
                errors_out=net_io.errout,
                drops_in=net_io.dropin,
                drops_out=net_io.dropout
            )
        except Exception as e:
            logger.error(f"Error getting network I/O stats: {e}")
            return NetworkIO(0, 0, 0, 0, 0, 0, 0, 0)
    
    def get_disk_io_stats(self) -> DiskIO:
        """Get detailed disk I/O statistics"""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                return DiskIO(
                    read_bytes=disk_io.read_bytes,
                    write_bytes=disk_io.write_bytes,
                    read_count=disk_io.read_count,
                    write_count=disk_io.write_count,
                    read_time=disk_io.read_time,
                    write_time=disk_io.write_time
                )
        except Exception as e:
            logger.error(f"Error getting disk I/O stats: {e}")
        
        return DiskIO(0, 0, 0, 0, 0, 0)
    
    def get_process_metrics(self, pid: Optional[int] = None) -> List[ProcessMetrics]:
        """
        Get metrics for specific process or all processes.
        
        Args:
            pid: Process ID (None for all processes)
            
        Returns:
            List of ProcessMetrics
        """
        processes = []
        
        try:
            if pid:
                # Get specific process
                try:
                    proc = psutil.Process(pid)
                    processes.append(self._get_single_process_metrics(proc))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            else:
                # Get all processes (limited to avoid performance issues)
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        if len(processes) >= 50:  # Limit to top 50 processes
                            break
                        processes.append(self._get_single_process_metrics(proc))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
        except Exception as e:
            logger.error(f"Error getting process metrics: {e}")
        
        return processes
    
    def establish_baseline(self, force_recalculate: bool = False) -> ResourceBaseline:
        """
        Establish performance baseline from historical data.
        
        Args:
            force_recalculate: Force recalculation even if baseline exists
            
        Returns:
            ResourceBaseline: Established baseline
        """
        if self.baseline and not force_recalculate:
            return self.baseline
        
        with self._lock:
            if len(self.metrics_history) < 10:
                logger.warning("Insufficient data for baseline establishment")
                # Create minimal baseline
                current_metrics = self.collect_system_metrics()
                self.baseline = ResourceBaseline(
                    cpu_baseline=current_metrics.cpu_usage,
                    memory_baseline=current_metrics.memory_usage,
                    disk_baseline=current_metrics.disk_usage,
                    network_baseline_sent=current_metrics.network_io_sent,
                    network_baseline_recv=current_metrics.network_io_recv,
                    established_at=datetime.now(timezone.utc),
                    sample_count=1
                )
                return self.baseline
            
            # Calculate baseline from recent metrics
            cutoff_time = datetime.now(timezone.utc) - self.baseline_window
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                recent_metrics = list(self.metrics_history)[-10:]  # Use last 10 if no recent data
            
            # Calculate averages
            cpu_values = [m.cpu_usage for m in recent_metrics]
            memory_values = [m.memory_usage for m in recent_metrics]
            disk_values = [m.disk_usage for m in recent_metrics]
            network_sent_values = [m.network_io_sent for m in recent_metrics]
            network_recv_values = [m.network_io_recv for m in recent_metrics]
            
            self.baseline = ResourceBaseline(
                cpu_baseline=statistics.mean(cpu_values),
                memory_baseline=int(statistics.mean(memory_values)),
                disk_baseline=int(statistics.mean(disk_values)),
                network_baseline_sent=int(statistics.mean(network_sent_values)),
                network_baseline_recv=int(statistics.mean(network_recv_values)),
                established_at=datetime.now(timezone.utc),
                sample_count=len(recent_metrics)
            )
        
        logger.info(f"Performance baseline established: CPU={self.baseline.cpu_baseline:.1f}% "
                   f"Memory={self.baseline.memory_baseline/1024/1024:.0f}MB "
                   f"from {self.baseline.sample_count} samples")
        
        return self.baseline
    
    def get_resource_usage_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get resource usage statistics for specified time window.
        
        Args:
            time_window: Time window for statistics (default: 1 hour)
            
        Returns:
            Dict with resource usage statistics
        """
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.now(timezone.utc) - time_window
        
        with self._lock:
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        disk_values = [m.disk_percent for m in recent_metrics]
        
        stats = {
            'sample_count': len(recent_metrics),
            'time_window_minutes': time_window.total_seconds() / 60,
            'cpu_stats': {
                'avg': statistics.mean(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values),
                'median': statistics.median(cpu_values)
            },
            'memory_stats': {
                'avg': statistics.mean(memory_values),
                'min': min(memory_values),
                'max': max(memory_values),
                'median': statistics.median(memory_values)
            },
            'disk_stats': {
                'avg': statistics.mean(disk_values),
                'min': min(disk_values),
                'max': max(disk_values),
                'median': statistics.median(disk_values)
            }
        }
        
        # Add load average stats if available
        load_values = [m.load_average for m in recent_metrics if m.load_average is not None]
        if load_values:
            stats['load_average_stats'] = {
                'avg': statistics.mean(load_values),
                'min': min(load_values),
                'max': max(load_values),
                'median': statistics.median(load_values)
            }
        
        return stats
    
    def set_alert_threshold(self, metric: str, threshold: float) -> None:
        """
        Set alert threshold for a resource metric.
        
        Args:
            metric: Metric name (cpu_usage, memory_usage, disk_usage, load_average)
            threshold: Threshold value
        """
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            logger.info(f"Resource alert threshold updated: {metric} = {threshold}")
        else:
            logger.warning(f"Unknown resource metric for alert threshold: {metric}")
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add callback function for resource alerts"""
        self.alert_callbacks.append(callback)
        logger.info("Resource alert callback added")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        logger.info("System monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Collect metrics
                self.collect_system_metrics()
                
                # Update baseline periodically
                if len(self.metrics_history) % 20 == 0:  # Every 20 samples
                    self.establish_baseline()
                
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
        
        logger.info("System monitoring loop stopped")
    
    def _get_single_process_metrics(self, proc: psutil.Process) -> ProcessMetrics:
        """Get metrics for a single process"""
        with proc.oneshot():
            return ProcessMetrics(
                pid=proc.pid,
                name=proc.name(),
                cpu_percent=proc.cpu_percent(),
                memory_rss=proc.memory_info().rss,
                memory_vms=proc.memory_info().vms,
                memory_percent=proc.memory_percent(),
                num_threads=proc.num_threads(),
                num_fds=proc.num_fds() if hasattr(proc, 'num_fds') else 0,
                create_time=proc.create_time(),
                status=proc.status()
            )
    
    def _check_resource_alerts(self, metrics: SystemMetrics) -> None:
        """Check if resource metrics trigger any alerts"""
        try:
            # CPU usage alert
            if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
                self._trigger_alert('high_cpu_usage', {
                    'cpu_usage': metrics.cpu_usage,
                    'threshold': self.alert_thresholds['cpu_usage']
                })
            
            # Memory usage alert
            if metrics.memory_percent > self.alert_thresholds['memory_usage']:
                self._trigger_alert('high_memory_usage', {
                    'memory_percent': metrics.memory_percent,
                    'threshold': self.alert_thresholds['memory_usage']
                })
            
            # Disk usage alert
            if metrics.disk_percent > self.alert_thresholds['disk_usage']:
                self._trigger_alert('high_disk_usage', {
                    'disk_percent': metrics.disk_percent,
                    'threshold': self.alert_thresholds['disk_usage']
                })
            
            # Load average alert (if available)
            if (metrics.load_average is not None and 
                metrics.load_average > self.alert_thresholds['load_average']):
                self._trigger_alert('high_load_average', {
                    'load_average': metrics.load_average,
                    'threshold': self.alert_thresholds['load_average']
                })
                
        except Exception as e:
            logger.error(f"Error checking resource alerts: {e}")
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger resource alert"""
        alert_data = {
            'type': alert_type,
            'timestamp': datetime.now(timezone.utc),
            'data': data
        }
        
        logger.warning(f"Resource alert triggered: {alert_type} - {data}")
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in resource alert callback: {e}")


# Global system monitor instance
_system_monitor: Optional[SystemResourceMonitor] = None


def get_system_monitor() -> SystemResourceMonitor:
    """Get global system resource monitor instance"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemResourceMonitor()
    return _system_monitor


def setup_system_monitoring(max_history: int = 1000, 
                           baseline_window_minutes: int = 30) -> SystemResourceMonitor:
    """
    Setup global system resource monitoring.
    
    Args:
        max_history: Maximum metrics history to keep
        baseline_window_minutes: Baseline calculation window
        
    Returns:
        SystemResourceMonitor instance
    """
    global _system_monitor
    _system_monitor = SystemResourceMonitor(max_history, baseline_window_minutes)
    logger.info("System resource monitoring setup completed")
    return _system_monitor
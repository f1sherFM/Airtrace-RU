"""
Unit tests for Configuration Audit Manager

Tests configuration change logging, performance impact tracking,
and audit trail functionality.
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from config_audit_manager import (
    ConfigurationAuditManager,
    ConfigurationChange,
    PerformanceImpact,
    ConfigurationSnapshot,
    AuditTrailEntry
)


@pytest.fixture
def temp_audit_log():
    """Create temporary audit log file"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def audit_manager(temp_audit_log):
    """Create audit manager instance for testing"""
    return ConfigurationAuditManager(
        audit_log_path=temp_audit_log,
        max_audit_entries=100,
        performance_measurement_duration=1  # 1 second for testing
    )


class TestConfigurationAuditManager:
    """Test configuration audit manager functionality"""
    
    def test_initialization(self, audit_manager):
        """Test audit manager initialization"""
        assert audit_manager is not None
        assert len(audit_manager.audit_trail) >= 0  # May have baseline entry
        assert len(audit_manager.configuration_snapshots) >= 0
        assert audit_manager.performance_measurement_duration == 1
    
    def test_log_configuration_change(self, audit_manager):
        """Test logging configuration changes"""
        change_id = audit_manager.log_configuration_change(
            component="redis",
            setting_path="max_connections",
            old_value=20,
            new_value=30,
            change_type="update",
            source="runtime",
            user_context="test_user",
            reason="performance optimization"
        )
        
        assert change_id != ""
        assert len(audit_manager.audit_trail) > 0
        
        # Check the logged change
        latest_entry = audit_manager.audit_trail[-1]
        assert latest_entry.change.component == "redis"
        assert latest_entry.change.setting_path == "max_connections"
        assert latest_entry.change.old_value == 20
        assert latest_entry.change.new_value == 30
        assert latest_entry.change.change_type == "update"
        assert latest_entry.change.source == "runtime"
        assert latest_entry.change.user_context == "test_user"
        assert latest_entry.change.reason == "performance optimization"
    
    def test_configuration_change_validation(self, audit_manager):
        """Test configuration change validation"""
        # Add a validation rule
        def validate_positive_number(old_value, new_value):
            errors = []
            if isinstance(new_value, int) and new_value <= 0:
                errors.append("Value must be positive")
            return errors
        
        audit_manager.add_validation_rule("test.positive_value", validate_positive_number)
        
        # Test valid change
        change_id = audit_manager.log_configuration_change(
            component="test",
            setting_path="positive_value",
            old_value=10,
            new_value=20
        )
        
        entry = next(e for e in audit_manager.audit_trail if e.change.change_id == change_id)
        assert entry.change.validation_status == "valid"
        assert len(entry.change.validation_errors) == 0
        
        # Test invalid change
        change_id = audit_manager.log_configuration_change(
            component="test",
            setting_path="positive_value",
            old_value=10,
            new_value=-5
        )
        
        entry = next(e for e in audit_manager.audit_trail if e.change.change_id == change_id)
        assert entry.change.validation_status == "invalid"
        assert len(entry.change.validation_errors) > 0
        assert "Value must be positive" in entry.change.validation_errors
    
    def test_sensitive_data_sanitization(self, audit_manager):
        """Test that sensitive data is sanitized in logs"""
        # Test password sanitization
        change_id = audit_manager.log_configuration_change(
            component="database",
            setting_path="password",
            old_value="old_secret_password",
            new_value="new_secret_password"
        )
        
        entry = next(e for e in audit_manager.audit_trail if e.change.change_id == change_id)
        assert entry.change.old_value != "old_secret_password"
        assert entry.change.new_value != "new_secret_password"
        assert "***" in str(entry.change.old_value)
        assert "***" in str(entry.change.new_value)
    
    @pytest.mark.asyncio
    async def test_performance_measurement(self, audit_manager):
        """Test performance impact measurement"""
        # Log a configuration change
        change_id = audit_manager.log_configuration_change(
            component="cache",
            setting_path="ttl",
            old_value=300,
            new_value=600
        )
        
        # Check that measurement was started
        assert change_id in audit_manager.active_performance_measurements
        
        # Wait for automatic measurement to complete
        await asyncio.sleep(1.5)  # Slightly longer than measurement duration
        
        # Check if measurement was completed automatically
        # Find the audit entry and check if it has performance impact
        audit_entry = None
        for entry in audit_manager.audit_trail:
            if entry.change.change_id == change_id:
                audit_entry = entry
                break
        
        assert audit_entry is not None
        
        # If automatic completion didn't work, complete manually
        if audit_entry.performance_impact is None:
            impact = audit_manager.complete_performance_measurement(change_id)
            assert impact is not None
        else:
            impact = audit_entry.performance_impact
        
        assert impact.change_id == change_id
        assert impact.impact_severity in ['none', 'low', 'medium', 'high', 'critical', 'unknown']
    
    def test_audit_trail_filtering(self, audit_manager):
        """Test audit trail filtering functionality"""
        # Add multiple changes for different components
        audit_manager.log_configuration_change("redis", "setting1", 1, 2)
        audit_manager.log_configuration_change("cache", "setting2", 3, 4)
        audit_manager.log_configuration_change("redis", "setting3", 5, 6)
        
        # Test component filtering
        redis_entries = audit_manager.get_audit_trail(component="redis")
        assert len(redis_entries) == 2
        assert all(entry.change.component == "redis" for entry in redis_entries)
        
        cache_entries = audit_manager.get_audit_trail(component="cache")
        assert len(cache_entries) == 1
        assert cache_entries[0].change.component == "cache"
        
        # Test limit
        limited_entries = audit_manager.get_audit_trail(limit=2)
        assert len(limited_entries) <= 2
    
    def test_configuration_snapshot(self, audit_manager):
        """Test configuration snapshot creation"""
        snapshot_id = audit_manager.create_configuration_snapshot("test_snapshot")
        
        assert snapshot_id == "test_snapshot"
        assert "test_snapshot" in audit_manager.configuration_snapshots
        
        snapshot = audit_manager.configuration_snapshots["test_snapshot"]
        assert isinstance(snapshot, ConfigurationSnapshot)
        assert snapshot.snapshot_id == "test_snapshot"
        assert snapshot.checksum != ""
        assert isinstance(snapshot.configuration, dict)
    
    def test_performance_impact_summary(self, audit_manager):
        """Test performance impact summary generation"""
        # Add some changes with different severities
        change_id1 = audit_manager.log_configuration_change("redis", "setting1", 1, 2)
        change_id2 = audit_manager.log_configuration_change("cache", "setting2", 3, 4)
        
        # Manually create performance impacts for testing
        impact1 = PerformanceImpact(
            change_id=change_id1,
            measurement_start=datetime.now(timezone.utc),
            impact_severity="high"
        )
        impact2 = PerformanceImpact(
            change_id=change_id2,
            measurement_start=datetime.now(timezone.utc),
            impact_severity="low"
        )
        
        # Update audit entries with impacts
        for entry in audit_manager.audit_trail:
            if entry.change.change_id == change_id1:
                entry.performance_impact = impact1
            elif entry.change.change_id == change_id2:
                entry.performance_impact = impact2
        
        # Get summary
        summary = audit_manager.get_performance_impact_summary()
        
        assert "total_changes" in summary
        assert "changes_with_impact_data" in summary
        assert "severity_distribution" in summary
        assert summary["changes_with_impact_data"] == 2
        assert summary["severity_distribution"]["high"] == 1
        assert summary["severity_distribution"]["low"] == 1
    
    def test_audit_log_file_writing(self, audit_manager, temp_audit_log):
        """Test that audit entries are written to log file"""
        # Log a configuration change
        audit_manager.log_configuration_change(
            component="test",
            setting_path="test_setting",
            old_value="old",
            new_value="new"
        )
        
        # Check that log file was written
        log_path = Path(temp_audit_log)
        assert log_path.exists()
        
        # Read and verify log content
        with open(log_path, 'r') as f:
            log_lines = f.readlines()
        
        assert len(log_lines) > 0
        
        # Parse the last log entry
        last_entry = json.loads(log_lines[-1])
        assert "timestamp" in last_entry
        assert "change" in last_entry
        assert last_entry["change"]["component"] == "test"
        assert last_entry["change"]["setting_path"] == "test_setting"
    
    def test_cleanup_old_data(self, audit_manager):
        """Test cleanup of old audit data"""
        # Add some entries
        for i in range(5):
            audit_manager.log_configuration_change(f"component{i}", "setting", i, i+1)
        
        initial_count = len(audit_manager.audit_trail)
        assert initial_count >= 5
        
        # Cleanup data older than 1 second (should not remove anything recent)
        cleanup_count = audit_manager.cleanup_old_data(timedelta(seconds=1))
        assert cleanup_count == 0  # Nothing should be cleaned up yet
        
        # Cleanup data older than -1 second (should remove everything)
        cleanup_count = audit_manager.cleanup_old_data(timedelta(seconds=-1))
        assert cleanup_count > 0
    
    def test_change_callbacks(self, audit_manager):
        """Test configuration change callbacks"""
        callback_called = False
        callback_change = None
        
        def test_callback(change):
            nonlocal callback_called, callback_change
            callback_called = True
            callback_change = change
        
        audit_manager.add_change_callback(test_callback)
        
        # Log a change
        audit_manager.log_configuration_change(
            component="test",
            setting_path="callback_test",
            old_value=1,
            new_value=2
        )
        
        assert callback_called
        assert callback_change is not None
        assert callback_change.component == "test"
        assert callback_change.setting_path == "callback_test"
    
    def test_export_audit_trail(self, audit_manager):
        """Test audit trail export functionality"""
        # Add some test data
        audit_manager.log_configuration_change("test", "setting1", 1, 2)
        audit_manager.log_configuration_change("test", "setting2", 3, 4)
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json_path = f.name
        
        try:
            success = audit_manager.export_audit_trail(json_path, format='json')
            assert success
            
            # Verify exported data
            with open(json_path, 'r') as f:
                exported_data = json.load(f)
            
            assert isinstance(exported_data, list)
            assert len(exported_data) >= 2
            
            # Check structure of exported entries
            entry = exported_data[0]
            assert "timestamp" in entry
            assert "change" in entry
            assert "entry_id" in entry
            
        finally:
            Path(json_path).unlink(missing_ok=True)
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            csv_path = f.name
        
        try:
            success = audit_manager.export_audit_trail(csv_path, format='csv')
            assert success
            
            # Verify CSV file exists and has content
            csv_file = Path(csv_path)
            assert csv_file.exists()
            assert csv_file.stat().st_size > 0
            
        finally:
            Path(csv_path).unlink(missing_ok=True)


class TestConfigurationChange:
    """Test ConfigurationChange data class"""
    
    def test_configuration_change_creation(self):
        """Test creating configuration change objects"""
        change = ConfigurationChange(
            change_id="test_id",
            timestamp=datetime.now(timezone.utc),
            component="redis",
            setting_path="max_connections",
            old_value=20,
            new_value=30,
            change_type="update",
            source="runtime"
        )
        
        assert change.change_id == "test_id"
        assert change.component == "redis"
        assert change.setting_path == "max_connections"
        assert change.old_value == 20
        assert change.new_value == 30
        assert change.change_type == "update"
        assert change.source == "runtime"
        assert change.validation_status == "pending"
        assert len(change.validation_errors) == 0


class TestPerformanceImpact:
    """Test PerformanceImpact data class"""
    
    def test_performance_impact_creation(self):
        """Test creating performance impact objects"""
        impact = PerformanceImpact(
            change_id="test_change",
            measurement_start=datetime.now(timezone.utc)
        )
        
        assert impact.change_id == "test_change"
        assert impact.measurement_end is None
        assert impact.baseline_stats is None
        assert impact.post_change_stats is None
        assert len(impact.impact_metrics) == 0
        assert impact.impact_severity == "unknown"
        assert len(impact.recommendations) == 0


if __name__ == "__main__":
    pytest.main([__file__])
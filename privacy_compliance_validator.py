"""
Privacy Compliance Validator for AirTrace RU Backend

Validates that all performance optimizations maintain privacy guarantees by ensuring:
- No coordinate logging in performance monitoring
- Metrics anonymization for privacy protection  
- Cache keys don't contain PII
- All data handling complies with privacy requirements
"""

import logging
import re
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PrivacyViolationType(Enum):
    """Types of privacy violations"""
    COORDINATE_LOGGING = "coordinate_logging"
    PII_IN_CACHE_KEY = "pii_in_cache_key"
    UNMASKED_METRICS = "unmasked_metrics"
    COORDINATE_IN_LOGS = "coordinate_in_logs"
    USER_TRACKING = "user_tracking"
    EXTERNAL_DATA_LEAK = "external_data_leak"


@dataclass
class PrivacyViolation:
    """Privacy violation record"""
    violation_type: PrivacyViolationType
    description: str
    location: str
    severity: str  # "low", "medium", "high", "critical"
    data_involved: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class PrivacyComplianceReport:
    """Privacy compliance validation report"""
    is_compliant: bool
    violations: List[PrivacyViolation]
    warnings: List[str]
    recommendations: List[str]
    validation_timestamp: datetime
    total_checks: int
    passed_checks: int
    
    @property
    def compliance_score(self) -> float:
        """Calculate compliance score as percentage"""
        if self.total_checks == 0:
            return 100.0
        return (self.passed_checks / self.total_checks) * 100.0


class PrivacyComplianceValidator:
    """
    Validates privacy compliance across all system components.
    
    Ensures that performance optimizations don't compromise user privacy
    by checking for coordinate logging, PII in cache keys, and proper
    metrics anonymization.
    """
    
    def __init__(self):
        self.violations: List[PrivacyViolation] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
        
        # Patterns for detecting coordinates and PII
        self.coordinate_patterns = [
            r'\b-?\d{1,3}\.\d{1,10}\b',  # Decimal coordinates
            r'\b-?\d{1,3}°\d{1,2}\'[\d\.]*"?\b',  # DMS coordinates
            r'lat(?:itude)?[:\s=]+[-\d\.]+',  # Latitude labels
            r'lon(?:gitude)?[:\s=]+[-\d\.]+',  # Longitude labels
            r'coord(?:inate)?s?[:\s=]+[-\d\.,\s]+',  # Coordinate labels
        ]
        
        self.pii_patterns = [
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b\+?[\d\s\-\(\)]{10,}\b',  # Phone numbers
        ]
        
        # Allowed domains for coordinate transmission
        self.allowed_coordinate_domains = {
            'air-quality-api.open-meteo.com',
            'api.weatherapi.com'
        }
    
    def validate_cache_key_privacy(self, cache_key: str, context: str = "") -> bool:
        """
        Validate that cache keys don't contain PII or raw coordinates.
        
        Returns True if compliant, False if violations found.
        """
        violations_found = False
        
        # Check for coordinate patterns in cache key
        for pattern in self.coordinate_patterns:
            if re.search(pattern, cache_key, re.IGNORECASE):
                self.violations.append(PrivacyViolation(
                    violation_type=PrivacyViolationType.PII_IN_CACHE_KEY,
                    description=f"Cache key contains potential coordinate data: {pattern}",
                    location=f"Cache key: {cache_key[:50]}... in {context}",
                    severity="high",
                    data_involved=cache_key
                ))
                violations_found = True
        
        # Check for other PII patterns
        for pattern in self.pii_patterns:
            if re.search(pattern, cache_key):
                self.violations.append(PrivacyViolation(
                    violation_type=PrivacyViolationType.PII_IN_CACHE_KEY,
                    description=f"Cache key contains potential PII: {pattern}",
                    location=f"Cache key: {cache_key[:50]}... in {context}",
                    severity="critical",
                    data_involved=cache_key
                ))
                violations_found = True
        
        # Check if cache key is properly hashed/anonymized
        if not self._is_properly_anonymized(cache_key):
            self.warnings.append(
                f"Cache key may not be properly anonymized: {cache_key[:20]}... in {context}"
            )
        
        return not violations_found
    
    def validate_metrics_anonymization(self, metrics_data: Dict[str, Any], context: str = "") -> bool:
        """
        Validate that metrics data is properly anonymized.
        
        Returns True if compliant, False if violations found.
        """
        violations_found = False
        
        # Convert metrics to string for pattern matching
        metrics_str = str(metrics_data)
        
        # Check for coordinate patterns in metrics
        for pattern in self.coordinate_patterns:
            matches = re.findall(pattern, metrics_str, re.IGNORECASE)
            if matches:
                self.violations.append(PrivacyViolation(
                    violation_type=PrivacyViolationType.UNMASKED_METRICS,
                    description=f"Metrics contain potential coordinate data: {matches[:3]}",
                    location=f"Metrics in {context}",
                    severity="high",
                    data_involved=str(matches)
                ))
                violations_found = True
        
        # Check for PII patterns in metrics
        for pattern in self.pii_patterns:
            matches = re.findall(pattern, metrics_str)
            if matches:
                self.violations.append(PrivacyViolation(
                    violation_type=PrivacyViolationType.UNMASKED_METRICS,
                    description=f"Metrics contain potential PII: {matches[:3]}",
                    location=f"Metrics in {context}",
                    severity="critical",
                    data_involved=str(matches)
                ))
                violations_found = True
        
        # Check for specific sensitive keys
        sensitive_keys = ['latitude', 'longitude', 'lat', 'lon', 'coordinates', 'location']
        for key in sensitive_keys:
            if self._contains_sensitive_key(metrics_data, key):
                self.violations.append(PrivacyViolation(
                    violation_type=PrivacyViolationType.UNMASKED_METRICS,
                    description=f"Metrics contain sensitive key: {key}",
                    location=f"Metrics key '{key}' in {context}",
                    severity="medium",
                    data_involved=key
                ))
                violations_found = True
        
        return not violations_found
    
    def validate_log_message_privacy(self, log_message: str, context: str = "") -> bool:
        """
        Validate that log messages don't contain coordinates or PII.
        
        Returns True if compliant, False if violations found.
        """
        violations_found = False
        
        # Check for coordinate patterns in log message
        for pattern in self.coordinate_patterns:
            matches = re.findall(pattern, log_message, re.IGNORECASE)
            if matches:
                # Allow coordinates in specific contexts (like filtered logs)
                if not self._is_allowed_coordinate_context(log_message):
                    self.violations.append(PrivacyViolation(
                        violation_type=PrivacyViolationType.COORDINATE_IN_LOGS,
                        description=f"Log message contains potential coordinates: {matches[:3]}",
                        location=f"Log message in {context}",
                        severity="high",
                        data_involved=log_message[:100]
                    ))
                    violations_found = True
        
        # Check for PII patterns in log message
        for pattern in self.pii_patterns:
            matches = re.findall(pattern, log_message)
            if matches:
                self.violations.append(PrivacyViolation(
                    violation_type=PrivacyViolationType.COORDINATE_IN_LOGS,
                    description=f"Log message contains potential PII: {matches[:3]}",
                    location=f"Log message in {context}",
                    severity="critical",
                    data_involved=log_message[:100]
                ))
                violations_found = True
        
        return not violations_found
    
    def validate_external_request_privacy(self, url: str, params: Dict[str, Any], context: str = "") -> bool:
        """
        Validate that external requests only send coordinates to allowed domains.
        
        Returns True if compliant, False if violations found.
        """
        violations_found = False
        
        # Extract domain from URL
        domain = self._extract_domain(url)
        
        # Check if coordinates are being sent
        has_coordinates = self._contains_coordinates(params)
        
        if has_coordinates and domain not in self.allowed_coordinate_domains:
            self.violations.append(PrivacyViolation(
                violation_type=PrivacyViolationType.EXTERNAL_DATA_LEAK,
                description=f"Coordinates sent to unauthorized domain: {domain}",
                location=f"External request in {context}",
                severity="critical",
                data_involved=f"Domain: {domain}, URL: {url[:100]}"
            ))
            violations_found = True
        
        # Check for other PII in parameters
        params_str = str(params)
        for pattern in self.pii_patterns:
            matches = re.findall(pattern, params_str)
            if matches:
                self.violations.append(PrivacyViolation(
                    violation_type=PrivacyViolationType.EXTERNAL_DATA_LEAK,
                    description=f"PII sent in external request: {matches[:3]}",
                    location=f"External request to {domain} in {context}",
                    severity="critical",
                    data_involved=f"PII: {matches}"
                ))
                violations_found = True
        
        return not violations_found
    
    def validate_performance_monitoring_privacy(self, monitoring_data: Dict[str, Any], context: str = "") -> bool:
        """
        Validate that performance monitoring doesn't log coordinates.
        
        Returns True if compliant, False if violations found.
        """
        violations_found = False
        
        # Check monitoring data for coordinates
        if not self.validate_metrics_anonymization(monitoring_data, f"Performance monitoring in {context}"):
            violations_found = True
        
        # Check for specific performance monitoring violations
        if 'request_details' in monitoring_data:
            request_details = monitoring_data['request_details']
            if isinstance(request_details, dict):
                # Check for coordinate parameters in request details
                for key, value in request_details.items():
                    if key.lower() in ['lat', 'lon', 'latitude', 'longitude'] and isinstance(value, (int, float)):
                        self.violations.append(PrivacyViolation(
                            violation_type=PrivacyViolationType.COORDINATE_LOGGING,
                            description=f"Performance monitoring logs coordinate parameter: {key}={value}",
                            location=f"Performance monitoring request details in {context}",
                            severity="high",
                            data_involved=f"{key}={value}"
                        ))
                        violations_found = True
        
        return not violations_found
    
    def generate_compliance_report(self) -> PrivacyComplianceReport:
        """Generate comprehensive privacy compliance report"""
        total_checks = len(self.violations) + len(self.warnings)
        passed_checks = len(self.warnings)  # Warnings are not failures
        
        is_compliant = len(self.violations) == 0
        
        # Generate recommendations based on violations
        self._generate_recommendations()
        
        return PrivacyComplianceReport(
            is_compliant=is_compliant,
            violations=self.violations.copy(),
            warnings=self.warnings.copy(),
            recommendations=self.recommendations.copy(),
            validation_timestamp=datetime.now(timezone.utc),
            total_checks=max(total_checks, 1),  # Avoid division by zero
            passed_checks=passed_checks
        )
    
    def _is_properly_anonymized(self, cache_key: str) -> bool:
        """Check if cache key appears to be properly anonymized"""
        hash_coordinates_enabled, coordinate_precision = self._get_cache_privacy_settings()

        # Check if it's a hash (hex string of reasonable length)
        if re.match(r'^[a-f0-9]{16,}$', cache_key.lower()):
            return True
        
        # Check if it uses coordinate precision rounding
        if hash_coordinates_enabled:
            return True
        
        # Check if coordinates are rounded to low precision
        coordinate_matches = re.findall(r'-?\d+\.\d+', cache_key)
        for match in coordinate_matches:
            decimal_places = len(match.split('.')[1])
            if decimal_places > coordinate_precision:
                return False
        
        return True

    def _get_cache_privacy_settings(self) -> Tuple[bool, int]:
        """
        Lazily resolve cache privacy settings to avoid circular imports during startup.
        Falls back to safe defaults if config is not yet available.
        """
        try:
            from config import config as runtime_config

            hash_coordinates = bool(getattr(runtime_config.cache, "hash_coordinates", True))
            precision = int(getattr(runtime_config.cache, "coordinate_precision", 3))
            return hash_coordinates, max(1, precision)
        except Exception:
            return True, 3
    
    def _contains_sensitive_key(self, data: Any, key: str) -> bool:
        """Recursively check if data contains sensitive key"""
        if isinstance(data, dict):
            if key in data:
                return True
            for value in data.values():
                if self._contains_sensitive_key(value, key):
                    return True
        elif isinstance(data, (list, tuple)):
            for item in data:
                if self._contains_sensitive_key(item, key):
                    return True
        return False
    
    def _is_allowed_coordinate_context(self, log_message: str) -> bool:
        """Check if coordinate in log message is in allowed context"""
        allowed_contexts = [
            'COORDINATE_FILTERED',
            'FILTERED_FOR_PRIVACY',
            'coordinates filtered',
            'privacy filtered',
            'anonymized',
            'hashed'
        ]
        
        message_lower = log_message.lower()
        return any(context.lower() in message_lower for context in allowed_contexts)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return url.lower()
    
    def _contains_coordinates(self, params: Dict[str, Any]) -> bool:
        """Check if parameters contain coordinate data"""
        coordinate_keys = ['lat', 'latitude', 'lon', 'longitude', 'coords', 'coordinates']
        
        for key, value in params.items():
            if key.lower() in coordinate_keys:
                return True
            
            # Check if value looks like a coordinate
            if isinstance(value, (int, float)) and -180 <= value <= 180:
                return True
        
        return False
    
    def _generate_recommendations(self):
        """Generate privacy compliance recommendations based on violations"""
        violation_types = {v.violation_type for v in self.violations}
        
        if PrivacyViolationType.COORDINATE_LOGGING in violation_types:
            self.recommendations.append(
                "Enable coordinate hashing in cache configuration to prevent coordinate logging"
            )
            self.recommendations.append(
                "Review performance monitoring to ensure coordinates are not logged"
            )
        
        if PrivacyViolationType.PII_IN_CACHE_KEY in violation_types:
            self.recommendations.append(
                "Implement proper cache key anonymization using hashing or coordinate rounding"
            )
        
        if PrivacyViolationType.UNMASKED_METRICS in violation_types:
            self.recommendations.append(
                "Add metrics anonymization layer to filter sensitive data from performance metrics"
            )
        
        if PrivacyViolationType.EXTERNAL_DATA_LEAK in violation_types:
            self.recommendations.append(
                "Review external API integrations to ensure only authorized domains receive coordinate data"
            )
            self.recommendations.append(
                "Implement request validation to prevent PII transmission to external services"
            )
        
        if PrivacyViolationType.COORDINATE_IN_LOGS in violation_types:
            self.recommendations.append(
                "Implement log message filtering to remove coordinates and PII from log output"
            )
            self.recommendations.append(
                "Use privacy-aware logging middleware for all coordinate-related operations"
            )
    
    def reset(self):
        """Reset validator state for new validation run"""
        self.violations.clear()
        self.warnings.clear()
        self.recommendations.clear()


# Global privacy compliance validator instance
privacy_validator = PrivacyComplianceValidator()


def validate_cache_key_privacy(cache_key: str, context: str = "") -> bool:
    """Convenience function for cache key privacy validation"""
    return privacy_validator.validate_cache_key_privacy(cache_key, context)


def validate_metrics_privacy(metrics_data: Dict[str, Any], context: str = "") -> bool:
    """Convenience function for metrics privacy validation"""
    return privacy_validator.validate_metrics_anonymization(metrics_data, context)


def validate_log_privacy(log_message: str, context: str = "") -> bool:
    """Convenience function for log message privacy validation"""
    return privacy_validator.validate_log_message_privacy(log_message, context)


def get_privacy_compliance_report() -> PrivacyComplianceReport:
    """Get current privacy compliance report"""
    return privacy_validator.generate_compliance_report()

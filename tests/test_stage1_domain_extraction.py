from domain.aqi.calculator import AQICalculator as DomainAQICalculator
from domain.nmu.detector import NMUDetector as DomainNMUDetector
from domain.pollutants.aggregator import get_pollutant_name_russian as domain_pollutant_name
from utils import AQICalculator, NMUDetector, get_pollutant_name_russian


def test_stage1_utils_facade_reexports_domain_classes():
    assert AQICalculator is DomainAQICalculator
    assert NMUDetector is DomainNMUDetector


def test_stage1_pollutant_name_helper_matches_domain_module():
    assert get_pollutant_name_russian("pm2_5") == domain_pollutant_name("pm2_5")
    assert get_pollutant_name_russian("pm10") == "Взвешенные частицы PM10"

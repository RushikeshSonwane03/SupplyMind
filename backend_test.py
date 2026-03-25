#!/usr/bin/env python3
"""
Backend API Testing for SupplyMind - Agentic AI Operating System for Supply Chain Intelligence
Tests all backend endpoints and core functionality
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Configuration
BASE_URL = "https://agentic-supply.preview.emergentagent.com/api"
TIMEOUT = 30

class SupplyMindAPITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SupplyMind-API-Tester/1.0'
        })
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []
        self.results = {}

    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name} - {details}")
            self.failed_tests.append(f"{test_name}: {details}")
        
        self.results[test_name] = {
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }

    def test_health_endpoint(self) -> bool:
        """Test /health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            
            if response.status_code != 200:
                self.log_result("Health Check", False, f"Status {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ["status", "ollama_connected", "model", "timestamp"]
            
            for field in required_fields:
                if field not in data:
                    self.log_result("Health Check", False, f"Missing field: {field}")
                    return False
            
            if data["status"] != "healthy":
                self.log_result("Health Check", False, f"Status not healthy: {data['status']}")
                return False
            
            self.log_result("Health Check", True, f"Ollama: {data['ollama_connected']}, Model: {data['model']}")
            return True
            
        except Exception as e:
            self.log_result("Health Check", False, str(e))
            return False

    def test_workflows_endpoint(self) -> bool:
        """Test /workflows endpoint - should return 10 workflows"""
        try:
            response = self.session.get(f"{self.base_url}/workflows", timeout=TIMEOUT)
            
            if response.status_code != 200:
                self.log_result("Workflows API", False, f"Status {response.status_code}")
                return False
            
            data = response.json()
            
            if "workflows" not in data or "count" not in data:
                self.log_result("Workflows API", False, "Missing workflows or count field")
                return False
            
            workflows = data["workflows"]
            
            if data["count"] != 10:
                self.log_result("Workflows API", False, f"Expected 10 workflows, got {data['count']}")
                return False
            
            if len(workflows) != 10:
                self.log_result("Workflows API", False, f"Count mismatch: declared {data['count']}, actual {len(workflows)}")
                return False
            
            # Validate workflow structure
            required_fields = ["id", "name", "description", "category", "workflow_type", "agents_involved"]
            for i, workflow in enumerate(workflows):
                for field in required_fields:
                    if field not in workflow:
                        self.log_result("Workflows API", False, f"Workflow {i} missing field: {field}")
                        return False
            
            self.log_result("Workflows API", True, f"Retrieved {len(workflows)} workflows")
            return True
            
        except Exception as e:
            self.log_result("Workflows API", False, str(e))
            return False

    def test_individual_workflow(self) -> bool:
        """Test /workflows/{id} endpoint"""
        try:
            # First get workflows list to get an ID
            workflows_response = self.session.get(f"{self.base_url}/workflows", timeout=TIMEOUT)
            if workflows_response.status_code != 200:
                self.log_result("Individual Workflow", False, "Cannot get workflows list")
                return False
            
            workflows_data = workflows_response.json()
            if not workflows_data.get("workflows"):
                self.log_result("Individual Workflow", False, "No workflows available")
                return False
            
            workflow_id = workflows_data["workflows"][0]["id"]
            
            response = self.session.get(f"{self.base_url}/workflows/{workflow_id}", timeout=TIMEOUT)
            
            if response.status_code != 200:
                self.log_result("Individual Workflow", False, f"Status {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ["id", "name", "description", "category", "workflow_type", "agents_involved"]
            
            for field in required_fields:
                if field not in data:
                    self.log_result("Individual Workflow", False, f"Missing field: {field}")
                    return False
            
            self.log_result("Individual Workflow", True, f"Retrieved workflow: {data['name']}")
            return True
            
        except Exception as e:
            self.log_result("Individual Workflow", False, str(e))
            return False

    def test_agent_states_endpoint(self) -> bool:
        """Test /agents/states endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/agents/states", timeout=TIMEOUT)
            
            if response.status_code != 200:
                self.log_result("Agent States", False, f"Status {response.status_code}")
                return False
            
            data = response.json()
            
            if "agents" not in data:
                self.log_result("Agent States", False, "Missing agents field")
                return False
            
            agents = data["agents"]
            expected_agents = ["orchestrator", "demand", "inventory", "supplier", "action"]
            
            for agent in expected_agents:
                if agent not in agents:
                    self.log_result("Agent States", False, f"Missing agent: {agent}")
                    return False
                
                agent_data = agents[agent]
                required_fields = ["agent", "status"]
                
                for field in required_fields:
                    if field not in agent_data:
                        self.log_result("Agent States", False, f"Agent {agent} missing field: {field}")
                        return False
            
            self.log_result("Agent States", True, f"Retrieved states for {len(agents)} agents")
            return True
            
        except Exception as e:
            self.log_result("Agent States", False, str(e))
            return False

    def test_supplier_risk_analysis(self) -> bool:
        """Test /supplier-risk/analyze endpoint"""
        try:
            test_data = {
                "supplier_name": "Test Supplier Inc",
                "include_news": True
            }
            
            response = self.session.post(
                f"{self.base_url}/supplier-risk/analyze",
                json=test_data,
                timeout=TIMEOUT * 2  # Give more time for LLM processing
            )
            
            if response.status_code != 200:
                self.log_result("Supplier Risk Analysis", False, f"Status {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ["supplier_name", "risk_level", "risk_score", "risk_factors", "llm_reasoning", "recommendation"]
            
            for field in required_fields:
                if field not in data:
                    self.log_result("Supplier Risk Analysis", False, f"Missing field: {field}")
                    return False
            
            # Validate risk_level values
            valid_risk_levels = ["low", "medium", "high", "critical"]
            if data["risk_level"] not in valid_risk_levels:
                self.log_result("Supplier Risk Analysis", False, f"Invalid risk_level: {data['risk_level']}")
                return False
            
            # Validate risk_score is numeric and in valid range
            try:
                risk_score = float(data["risk_score"])
                if not (0 <= risk_score <= 100):
                    self.log_result("Supplier Risk Analysis", False, f"Risk score out of range: {risk_score}")
                    return False
            except (ValueError, TypeError):
                self.log_result("Supplier Risk Analysis", False, f"Invalid risk score: {data['risk_score']}")
                return False
            
            self.log_result("Supplier Risk Analysis", True, f"Risk: {data['risk_level']} ({data['risk_score']}%)")
            return True
            
        except Exception as e:
            self.log_result("Supplier Risk Analysis", False, str(e))
            return False

    def test_reports_master(self) -> bool:
        """Test /reports/master endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/reports/master", timeout=TIMEOUT)
            
            if response.status_code != 200:
                self.log_result("Master Report", False, f"Status {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ["title", "sections", "generated_at"]
            
            for field in required_fields:
                if field not in data:
                    self.log_result("Master Report", False, f"Missing field: {field}")
                    return False
            
            sections = data["sections"]
            if not isinstance(sections, list) or len(sections) == 0:
                self.log_result("Master Report", False, "No sections found")
                return False
            
            # Validate section structure
            for i, section in enumerate(sections):
                section_fields = ["id", "title", "content"]
                for field in section_fields:
                    if field not in section:
                        self.log_result("Master Report", False, f"Section {i} missing field: {field}")
                        return False
            
            self.log_result("Master Report", True, f"Retrieved report with {len(sections)} sections")
            return True
            
        except Exception as e:
            self.log_result("Master Report", False, str(e))
            return False

    def test_reports_slides(self) -> bool:
        """Test /reports/slides endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/reports/slides", timeout=TIMEOUT)
            
            if response.status_code != 200:
                self.log_result("Slides Report", False, f"Status {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ["title", "slides", "generated_at"]
            
            for field in required_fields:
                if field not in data:
                    self.log_result("Slides Report", False, f"Missing field: {field}")
                    return False
            
            slides = data["slides"]
            if not isinstance(slides, list) or len(slides) == 0:
                self.log_result("Slides Report", False, "No slides found")
                return False
            
            # Validate slide structure
            for i, slide in enumerate(slides):
                slide_fields = ["number", "title"]
                for field in slide_fields:
                    if field not in slide:
                        self.log_result("Slides Report", False, f"Slide {i} missing field: {field}")
                        return False
            
            self.log_result("Slides Report", True, f"Retrieved {len(slides)} slides")
            return True
            
        except Exception as e:
            self.log_result("Slides Report", False, str(e))
            return False

    def test_metrics_endpoints(self) -> bool:
        """Test metrics endpoints"""
        success_count = 0
        
        # Test system metrics
        try:
            response = self.session.get(f"{self.base_url}/metrics/system", timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                required_fields = ["total_workflows", "active_agents", "decisions_today"]
                if all(field in data for field in required_fields):
                    success_count += 1
                    print("✅ System Metrics")
                else:
                    print("❌ System Metrics - Missing required fields")
            else:
                print(f"❌ System Metrics - Status {response.status_code}")
        except Exception as e:
            print(f"❌ System Metrics - {str(e)}")
        
        # Test KPIs
        try:
            response = self.session.get(f"{self.base_url}/metrics/kpis", timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                if "kpis" in data and isinstance(data["kpis"], list):
                    success_count += 1
                    print("✅ KPI Metrics")
                else:
                    print("❌ KPI Metrics - Invalid structure")
            else:
                print(f"❌ KPI Metrics - Status {response.status_code}")
        except Exception as e:
            print(f"❌ KPI Metrics - {str(e)}")
        
        self.tests_run += 2
        self.tests_passed += success_count
        
        return success_count == 2

    def test_analytics_endpoints(self) -> bool:
        """Test analytics endpoints"""
        endpoints = [
            "demand-trend",
            "inventory-levels", 
            "supplier-performance",
            "cost-breakdown"
        ]
        
        success_count = 0
        
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}/analytics/{endpoint}", timeout=TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and isinstance(data["data"], list):
                        success_count += 1
                        print(f"✅ Analytics - {endpoint}")
                    else:
                        print(f"❌ Analytics - {endpoint} - Invalid structure")
                else:
                    print(f"❌ Analytics - {endpoint} - Status {response.status_code}")
            except Exception as e:
                print(f"❌ Analytics - {endpoint} - {str(e)}")
        
        self.tests_run += len(endpoints)
        self.tests_passed += success_count
        
        return success_count == len(endpoints)

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all backend tests"""
        print(f"🔍 Testing SupplyMind Backend API: {self.base_url}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Core functionality tests
        print("\n📊 Core API Tests:")
        self.test_health_endpoint()
        self.test_workflows_endpoint()
        self.test_individual_workflow()
        self.test_agent_states_endpoint()
        
        print("\n🤖 AI/LLM Tests:")
        self.test_supplier_risk_analysis()
        
        print("\n📄 Reports Tests:")
        self.test_reports_master()
        self.test_reports_slides()
        
        print("\n📈 Data/Analytics Tests:")
        self.test_metrics_endpoints()
        self.test_analytics_endpoints()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary:")
        print(f"   Tests Run: {self.tests_run}")
        print(f"   Passed: {self.tests_passed}")
        print(f"   Failed: {self.tests_run - self.tests_passed}")
        print(f"   Success Rate: {(self.tests_passed / self.tests_run * 100):.1f}%")
        print(f"   Execution Time: {execution_time:.2f}s")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for failed_test in self.failed_tests:
                print(f"   - {failed_test}")
        
        return {
            "total_tests": self.tests_run,
            "passed_tests": self.tests_passed,
            "failed_tests": self.tests_run - self.tests_passed,
            "success_rate": self.tests_passed / self.tests_run * 100,
            "execution_time": execution_time,
            "failed_test_details": self.failed_tests,
            "all_results": self.results
        }

def main():
    """Main test execution"""
    print("🚀 SupplyMind Backend API Testing Suite")
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = SupplyMindAPITester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results["failed_tests"] == 0:
        print("\n🎉 All tests passed! Backend is functioning correctly.")
        return 0
    else:
        print(f"\n⚠️  {results['failed_tests']} test(s) failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
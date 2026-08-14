"""
Quick test script to verify the environment system is working.
Run this after starting env_server.py
"""

import uuid
import requests
import time


def test_server_health():
    """Test if server is running."""
    try:
        response = requests.get("http://0.0.0.0:8001/env/available", timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
            print(f"  Available environments: {response.json()['available_environments']}")
            return True
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Is it running?")
        print("  Start with: python env_server.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_environment(env_name: str):
    """Test a specific environment."""
    print(f"\n{'='*60}")
    print(f"Testing {env_name} environment")
    print('='*60)
    
    task_id = str(uuid.uuid4())
    base_url = "http://0.0.0.0:8001"
    
    try:
        # 1. Initialize
        print(f"1. Initializing environment...")
        response = requests.post(
            f"{base_url}/env/initialize",
            json={
                "task_id": task_id,
                "env_name": env_name,
                "env_config": {},
            },
            timeout=10,
        )
        assert response.status_code == 200, f"Initialize failed: {response.text}"
        print(f"   ✓ Environment initialized")
        
        # 2. Reset
        print(f"2. Resetting environment...")
        response = requests.post(
            f"{base_url}/env/reset",
            json={"task_id": task_id},
            timeout=10,
        )
        assert response.status_code == 200, f"Reset failed: {response.text}"
        obs = response.json()["observation"]
        print(f"   ✓ Reset successful")
        print(f"   Initial observation: {str(obs)[:100]}...")
        
        # 3. Step
        print(f"3. Taking a step...")
        action = "search[test]" if env_name == "webshop" else "goto[https://example.com]" if env_name == "browsecomp-plus" else "search_flights[NYC, LA, 2025-03-01]"
        response = requests.post(
            f"{base_url}/env/step",
            json={"task_id": task_id, "action": action},
            timeout=10,
        )
        assert response.status_code == 200, f"Step failed: {response.text}"
        result = response.json()
        print(f"   ✓ Step successful")
        print(f"   Action: {action}")
        print(f"   Reward: {result.get('reward', 0)}")
        print(f"   Done: {result.get('done', False)}")
        
        # 4. Get observation
        print(f"4. Getting current observation...")
        response = requests.post(
            f"{base_url}/env/get_observation",
            json={"task_id": task_id},
            timeout=10,
        )
        assert response.status_code == 200, f"Get observation failed: {response.text}"
        print(f"   ✓ Got observation")
        
        # 5. Close
        print(f"5. Closing environment...")
        response = requests.post(
            f"{base_url}/env/close",
            json={"task_id": task_id},
            timeout=10,
        )
        assert response.status_code == 200, f"Close failed: {response.text}"
        print(f"   ✓ Environment closed")
        
        print(f"\n✓ All tests passed for {env_name}!")
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def test_list_environments():
    """Test listing active environments."""
    print(f"\n{'='*60}")
    print(f"Testing list environments")
    print('='*60)
    
    try:
        response = requests.get("http://0.0.0.0:8001/env/list", timeout=5)
        assert response.status_code == 200
        data = response.json()
        print(f"✓ Active environments: {data['environments']}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ENVIRONMENT SYSTEM TEST SUITE")
    print("="*60)
    
    # Test 1: Server health
    if not test_server_health():
        print("\n❌ Server health check failed. Exiting.")
        return
    
    time.sleep(1)
    
    # Test 2: Test each environment
    environments = ["webshop", "browsecomp-plus", "travel_planner"]
    results = {}
    
    for env_name in environments:
        time.sleep(1)
        results[env_name] = test_environment(env_name)
    
    time.sleep(1)
    
    # Test 3: List environments
    test_list_environments()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for env_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{env_name:20s}: {status}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

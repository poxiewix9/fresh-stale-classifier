"""
Quick test script to verify the API works locally
"""
import requests
import json

# Test with a sample image URL
test_image_url = "https://images.unsplash.com/photo-1610832958506-aa56368176cf?w=400"  # Apple image

print("🧪 Testing Fresh/Stale Classifier API...")
print(f"Image URL: {test_image_url}")
print()

try:
    response = requests.post(
        "http://localhost:8000/classify",
        json={"imageUrl": test_image_url},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Success!")
        print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
except requests.exceptions.ConnectionError:
    print("❌ Could not connect to API")
    print("Make sure the API is running: python app.py")
except Exception as e:
    print(f"❌ Error: {e}")


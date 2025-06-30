# Python Version
import requests
import json

def test_gemini_api(api_key):
    """
    Test Google Gemini API with a simple request
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Hello, please respond with a simple greeting."
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("API Key is working!")
        else:
            print("API Key has issues")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# Usage
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = "AIzaSyC1i7rneq_8021sgC0b9xnwXjQ2Rb-57-8"
    test_gemini_api(API_KEY)
import requests

# Replace this with your actual API endpoint
url = "sk-5dc5bfd245664ec4a0927999e47b6adb"



# Payload (adjust based on the expected format of your API)
payload = {
    "text": "test"
}

# Test the connection and make a request to the API
try:
    response = requests.post(url,  json=payload)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("API connection successful!")
        print("Response:", response.json())
    else:
        print(f"Failed to connect. Status Code: {response.status_code}")
        print("Response:", response.text)
except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")

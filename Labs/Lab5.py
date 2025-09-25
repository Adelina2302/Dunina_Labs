import streamlit as st
import requests
import json
from openai import OpenAI

def get_current_weather(location):
    API_key = st.secrets["OPEN_WEATHER_API"]
    location = location.strip()
    # Always use format City,CountryCode for OpenWeather
    if "," not in location:
        location = location + ",US"
    urlbase = "https://api.openweathermap.org/data/2.5/"
    urlweather = f"weather?q={location}&appid={API_key}"
    url = urlbase + urlweather
    response = requests.get(url)
    data = response.json()
    if data.get("cod") != 200:
        st.error(f"OpenWeather API error: {data.get('message', '')} (code: {data.get('cod')})")
        return {"error": "Could not retrieve weather data."}
    temp = data['main']['temp'] - 273.15
    feels_like = data['main']['feels_like'] - 273.15
    temp_min = data['main']['temp_min'] - 273.15
    temp_max = data['main']['temp_max'] - 273.15
    humidity = data['main']['humidity']
    weather_desc = data['weather'][0]['description'].capitalize()
    return {
        "location": location,
        "temperature": round(temp, 2),
        "feels_like": round(feels_like, 2),
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),
        "humidity": round(humidity, 2),
        "description": weather_desc
    }

def clothing_suggestion(temp_celsius, weather_desc):
    if temp_celsius < 5:
        return "It's very cold. Wear a warm coat, hat, gloves, and scarf."
    elif temp_celsius < 15:
        return "It's cool. Wear a jacket or sweater."
    elif temp_celsius < 22:
        return "It's mild. A long-sleeve shirt or light jacket is enough."
    elif temp_celsius < 28:
        return "It's warm. T-shirt and pants/shorts are fine."
    else:
        return "It's hot. Wear light, breathable clothes and stay hydrated!"

def is_good_for_picnic(temp_celsius, weather_desc):
    bad_conditions = ["rain", "storm", "snow", "thunder", "drizzle"]
    if any(cond in weather_desc.lower() for cond in bad_conditions):
        return "Not a good day for a picnic due to weather."
    if temp_celsius < 10 or temp_celsius > 32:
        return "Not the best temperature for a picnic."
    return "Yes! The weather looks good for a picnic."

def get_openai_suggestion(city):
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)
    weather_function = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. Paris,FR"
                    }
                },
                "required": ["location"]
            }
        }
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant who gives weather and clothing advice. Also say if today is a good day for a picnic."},
        {"role": "user", "content": f"What should I wear today in {city}?"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[weather_function],
        tool_choice="auto"
    )
    msg = response.choices[0].message

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_call = msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        location = args.get("location", "")
        if not location or location.strip() == "":
            location = "Syracuse,US"
        if "," not in location:
            location = location.strip() + ",US"
        weather = get_current_weather(location)
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": json.dumps(weather)
        })
        followup = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        content = followup.choices[0].message.content.strip()
        if not content:
            return "OpenAI returned an empty suggestion. Try a different city or check your API/model access."
        return content
    else:
        content = msg.content.strip()
        if not content:
            return "OpenAI returned an empty suggestion. Try a different city or check your API/model access."
        return content

def get_mistral_suggestion(city, weather):
    mistral_api_key = st.secrets["MISTRAL_API_KEY"]
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {mistral_api_key}",
        "Content-Type": "application/json"
    }
    mistral_tools = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g. Paris,FR"
                    }
                },
                "required": ["location"]
            }
        }
    }]
    messages = [
        {"role": "system", "content": "You are a helpful assistant who gives weather and clothing advice. Also say if today is a good day for a picnic."},
        {"role": "user", "content": f"What should I wear today in {city}?"}
    ]
    data = {
        "model": "mistral-large-latest",
        "messages": messages,
        "tools": mistral_tools,
        "tool_choice": "auto"
    }
    r = requests.post(url, headers=headers, data=json.dumps(data))
    if r.status_code != 200:
        return "Mistral API error: " + r.text
    resp = r.json()
    msg = resp['choices'][0]['message']
    if "tool_calls" in msg and msg["tool_calls"]:
        tool_call = msg["tool_calls"][0]
        tool_call_id = tool_call["id"]
        args = json.loads(tool_call["function"]["arguments"])
        location = args.get("location", "")
        if not location or location.strip() == "":
            location = "Syracuse,US"
        if "," not in location:
            location = location.strip() + ",US"
        weather = get_current_weather(location)
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"]
                }
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_call["function"]["name"],
            "content": json.dumps(weather)
        })
        data2 = {
            "model": "mistral-large-latest",
            "messages": messages,
        }
        r2 = requests.post(url, headers=headers, data=json.dumps(data2))
        if r2.status_code != 200:
            return "Mistral API error: " + r2.text
        resp2 = r2.json()
        content = resp2['choices'][0]['message']['content'].strip()
        if not content:
            return "Mistral returned an empty suggestion. Try a different city or check your API/model access."
        return content
    else:
        content = msg.get('content', '').strip()
        if not content:
            return "Mistral returned an empty suggestion. Try a different city or check your API/model access."
        return content

def get_cohere_suggestion(city, weather):
    cohere_api_key = st.secrets["COHERE_API_KEY"]
    url = "https://api.cohere.ai/v1/chat"
    headers = {
        "Authorization": f"Bearer {cohere_api_key}",
        "Content-Type": "application/json"
    }
    weather_text = (
        f"The current weather in {weather['location']} is: "
        f"{weather['description']}, temperature {weather['temperature']} °C, "
        f"feels like {weather['feels_like']} °C, humidity {weather['humidity']}%. "
    )
    prompt = (
        f"{weather_text} Given this weather, what are the most appropriate clothes to wear today? "
        f"Also say if today is a good day for a picnic. Give a short, practical suggestion."
    )
    data = {
        "message": prompt,
        "model": "command-a-03-2025",
        "temperature": 0.3,
        "max_tokens": 120,
    }
    r = requests.post(url, headers=headers, data=json.dumps(data))
    if r.status_code == 200:
        content = r.json().get("text", "").strip()
        if not content:
            return "Cohere returned an empty suggestion. Try a different city or check your API/model access."
        return content
    else:
        return "Cohere API error: " + r.text

st.title("What to Wear Bot (OpenAI, Mistral, Cohere)")

vendor = st.selectbox(
    "Choose LLM vendor for clothing advice:",
    ("OpenAI", "Mistral", "Cohere")
)

city = st.text_input("Enter a city:", placeholder="e.g., Paris,FR or London,GB. Leave blank for Syracuse,US")
if not city.strip():
    city = "Syracuse,US"

weather = get_current_weather(city)

if st.button("Ask the Bot"):
    if "error" in weather:
        st.error(weather["error"])
    else:
        # Stylish weather info block
        st.markdown(
            f"""
            <div style="background-color:#F3F9FF;padding:18px 24px 18px 24px;border-radius:14px;margin-bottom:15px;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                <h3 style="margin-bottom:4px;">🌤️ <b>Weather in {weather['location']}</b></h3>
                <ul style="list-style:none;padding-left:0;line-height:2;margin-bottom:0;">
                    <li><b>🌡️ Temperature:</b> {weather['temperature']}°C (feels like {weather['feels_like']}°C)</li>
                    <li><b>🔻 Min:</b> {weather['temp_min']}°C &nbsp;&nbsp; <b>🔺 Max:</b> {weather['temp_max']}°C</li>
                    <li><b>💧 Humidity:</b> {weather['humidity']}%</li>
                    <li><b>🌥️ Description:</b> {weather['description']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )
        st.markdown("---")
        st.write("👕 **Rule-based clothing suggestion:**")
        st.info(clothing_suggestion(weather['temperature'], weather['description']))
        st.write("🥪 **Rule-based picnic advice:**")
        st.info(is_good_for_picnic(weather['temperature'], weather['description']))
        st.markdown("---")
        st.write(f"🤖 {vendor} clothing suggestion:")
        if vendor == "OpenAI":
            suggestion = get_openai_suggestion(city)
        elif vendor == "Mistral":
            suggestion = get_mistral_suggestion(city, weather)
        elif vendor == "Cohere":
            suggestion = get_cohere_suggestion(city, weather)
        else:
            suggestion = "Unknown vendor."
        if not suggestion:
            st.warning(f"{vendor} returned an empty suggestion. Try a different city or check your API/model access.")
        else:
            st.success(suggestion)
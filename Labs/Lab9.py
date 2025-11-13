#I chose to complete the AI Travel Planner lab instead of the other option because 
#it looked engaging and practical. 
#I liked that this project uses multiple AI agents that communicate and make decisions using both real data and reasoning from an LLM. 
#It felt like a realistic example of how AI systems can actually be applied in the real world.
#Another reason is that the second lab was created by my project team, so I wanted to avoid repeating that work. 


import streamlit as st
import requests
import json
import re
from datetime import datetime, timedelta

MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
OWM_KEY = st.secrets["OPEN_WEATHER_API"]

def llm_call(prompt, temperature=0.3):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-small",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 512
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

def get_weather_data(city, date):
    today = datetime.utcnow()
    trip_day = datetime.strptime(date, "%Y-%m-%d")
    delta_days = (trip_day - today).days
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OWM_KEY}&units=metric&lang=en"
    r = requests.get(url)
    data = r.json()
    if "list" not in data:
        return f"Error: city '{city}' not found."
    if delta_days <= 4:
        nearest = min(
            data["list"],
            key=lambda x: abs(datetime.strptime(x["dt_txt"], "%Y-%m-%d %H:%M:%S") - trip_day)
        )
        return {
            "date": nearest["dt_txt"],
            "temp": nearest["main"]["temp"],
            "desc": nearest["weather"][0]["description"],
        }
    else:
        city_temp = sum(x["main"]["temp"] for x in data["list"]) / len(data["list"])
        city_weather = data["list"][0]["weather"][0]["main"]
        prompt = (
            f"Predict the weather in {city} for {date} based on typical climate: "
            f"Average temperature: {city_temp:.1f}, weather type: {city_weather}. Respond briefly in English."
        )
        desc = llm_call(prompt)
        return {"date": date, "temp": city_temp, "desc": desc}

def calculate_travel_info(origin, destination):
    prompt = (
        f"Estimate the best way to travel from {origin} to {destination}: "
        "car/train/flight, approximately how long it will take, and the distance. "
        "Give 1–2 brief travel tips. "
        "Result in JSON: mode, duration_hours, distance_km, tips. "
        "Don't advise driving internationally."
    )
    result = llm_call(prompt)
    clean = re.sub(r"```json|```", "", result).strip()
    try:
        plan = json.loads(clean)
    except Exception:
        plan = {"mode": "unknown", "tips": result}
    return plan

class Agent:
    def __init__(self, name):
        self.name = name
        self.memory = {}
    def perceive(self, env):
        pass
    def reason(self):
        pass
    def act(self, env):
        pass

class WeatherAgent(Agent):
    def __init__(self):
        super().__init__("WeatherAgent")
    def perceive(self, env):
        self.origin = env["origin"]
        self.destination = env["destination"]
        self.date = env["trip_start"]
    def reason(self):
        self.origin_weather = get_weather_data(self.origin, self.date)
        self.dest_weather = get_weather_data(self.destination, self.date)
    def act(self, env):
        env["weather"] = {
            self.origin: self.origin_weather,
            self.destination: self.dest_weather
        }

class LogisticsAgent(Agent):
    def __init__(self):
        super().__init__("LogisticsAgent")
    def perceive(self, env):
        self.origin = env["origin"]
        self.destination = env["destination"]
    def reason(self):
        self.plan = calculate_travel_info(self.origin, self.destination)
    def act(self, env):
        env["logistics"] = self.plan

class PackingAgent(Agent):
    def __init__(self):
        super().__init__("PackingAgent")
    def perceive(self, env):
        w = env["weather"]
        self.origin = env["origin"]
        self.destination = env["destination"]
        self.duration = env["duration"]
        self.w1 = w[self.origin]
        self.w2 = w[self.destination]
    def reason(self):
        prompt = (
            f"Trip from {self.origin} to {self.destination} for {self.duration} days. "
            f"Weather at origin: {self.w1['desc']} ({self.w1['temp']}°C), "
            f"at destination: {self.w2['desc']} ({self.w2['temp']}°C). "
            "Suggest clothes, gadgets, and accessories for the trip. Return as a short bullet list."
        )
        self.suggestions = llm_call(prompt)
    def act(self, env):
        env["packing"] = self.suggestions

class ActivityAgent(Agent):
    def __init__(self):
        super().__init__("ActivityAgent")
    def perceive(self, env):
        self.destination = env["destination"]
        self.duration = env["duration"]
    def reason(self):
        prompt = (
            f"Make a sample daily itinerary for {self.duration} days in {self.destination}. "
            "Include local attractions, cafes, museums, and walks. "
            "Keep it concise and in English."
        )
        self.itinerary = llm_call(prompt)
    def act(self, env):
        env["itinerary"] = self.itinerary

class MultiAgentSystem:
    def __init__(self, agents):
        self.agents = agents
    def run_all(self, env):
        for agent in self.agents:
            agent.perceive(env)
            agent.reason()
            agent.act(env)
        return env

st.title("Multi-Agent AI Trip Planner - Lab 9 (Mistral + OWM)")

with st.form("trip_form"):
    col1, col2 = st.columns(2)
    origin = col1.text_input("Origin City")
    destination = col2.text_input("Destination City")
    dep_date = st.date_input("Departure Date", min_value=datetime.today())
    duration = st.number_input("Trip Duration (days)", min_value=1, value=5)
    submitted = st.form_submit_button("Plan Trip")

if submitted:
    trip_start = dep_date.strftime("%Y-%m-%d")
    trip_end = (dep_date + timedelta(days=duration)).strftime("%Y-%m-%d")
    env = {
        "origin": origin,
        "destination": destination,
        "trip_start": trip_start,
        "trip_end": trip_end,
        "duration": duration,
    }
    agents = [WeatherAgent(), LogisticsAgent(), PackingAgent(), ActivityAgent()]
    mas = MultiAgentSystem(agents)
    env = mas.run_all(env)

    st.header("Overview")
    st.write(f"Trip from **{origin}** to **{destination}**, {duration} days ({trip_start} → {trip_end})")
    st.header("Weather")
    w = env["weather"]
    st.write(f"{origin}: {w[origin]['desc']} ({w[origin]['temp']}°C)" if isinstance(w[origin], dict) else w[origin])
    st.write(f"{destination}: {w[destination]['desc']} ({w[destination]['temp']}°C)" if isinstance(w[destination], dict) else w[destination])
    st.header("Logistics")
    st.json(env["logistics"])
    st.write("Tips:", env["logistics"].get("tips", ""))
    st.header("Packing Suggestions")
    st.markdown(env["packing"])
    st.header("Day-by-day Itinerary")
    st.markdown(env["itinerary"])
    

st.caption("If city name is invalid, you'll see a weather error message.")

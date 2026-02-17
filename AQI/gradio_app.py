
import gradio as gr
import requests

# Use API if running; else load model in-process
API_BASE = "http://localhost:8000"


def predict_via_api(features_dict):
    """Call FastAPI /predict/single."""
    try:
        r = requests.post(f"{API_BASE}/predict/single", json={"features": features_dict}, timeout=10)
        if r.status_code == 200:
            p = r.json().get("prediction", {})
            return f"**AQI: {p.get('aqi')}** — {p.get('aqi_label', '')}"
        return f"Error: {r.status_code} {r.text}"
    except Exception as e:
        return f"Error: {e}"


def predict_next_3_days_via_api():
    """Call FastAPI /predict/next_3_days and format as table text."""
    try:
        r = requests.get(f"{API_BASE}/predict/next_3_days", timeout=30)
        if r.status_code != 200:
            return f"API error: {r.status_code}\n{r.text}"
        preds = r.json().get("predictions", [])
        if not preds:
            return "No predictions."
        lines = ["Timestamp (UTC) | AQI | Label"]
        for x in preds[:24]:  # first 24 hours
            lines.append(f"{x.get('timestamp_utc', '')[:16]} | {x.get('aqi')} | {x.get('aqi_label', '')}")
        if len(preds) > 24:
            lines.append(f"... and {len(preds) - 24} more hours")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


# Build minimal inputs for one prediction (key features only; rest can be defaults in backend via imputer)
def build_features(pm25, pm10, no2, temp, humidity, hour, day_week, month):
    import math
    pm_ratio = pm25 / pm10 if pm10 else 0.7
    pm_sum = pm25 + pm10
    return {
        "pm25": float(pm25), "pm10": float(pm10), "pm_ratio": pm_ratio, "pm_sum": pm_sum,
        "no2": float(no2), "so2": 5.0, "co": 200.0, "o3": 80.0,
        "temperature": float(temp), "humidity": float(humidity), "wind_speed": 3.0,
        "hour_sin": math.sin(2 * math.pi * hour / 24), "hour_cos": math.cos(2 * math.pi * hour / 24),
        "day_sin": math.sin(2 * math.pi * day_week / 7), "day_cos": math.cos(2 * math.pi * day_week / 7),
        "month_sin": math.sin(2 * math.pi * month / 12), "month_cos": math.cos(2 * math.pi * month / 12),
    }


with gr.Blocks(title="AQI Prediction", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AQI Prediction (Feature Store)")
    gr.Markdown("Ensure the FastAPI backend is running: `uvicorn api_backend:app --port 8000`")

    with gr.Tab("Single prediction"):
        pm25 = gr.Number(label="PM2.5", value=35)
        pm10 = gr.Number(label="PM10", value=50)
        no2 = gr.Number(label="NO2", value=1.0)
        temp = gr.Number(label="Temperature (°C)", value=25)
        humidity = gr.Number(label="Humidity (%)", value=60)
        hour = gr.Slider(0, 23, step=1, value=12, label="Hour")
        day_week = gr.Slider(0, 6, step=1, value=2, label="Day of week (0=Mon)")
        month = gr.Slider(1, 12, step=1, value=6, label="Month")
        btn = gr.Button("Predict AQI")
        out = gr.Markdown()

        def run_single(pm25, pm10, no2, temp, humidity, hour, day_week, month):
            fe = build_features(pm25, pm10, no2, temp, humidity, int(hour), int(day_week), int(month))
            return predict_via_api(fe)

        btn.click(
            fn=run_single,
            inputs=[pm25, pm10, no2, temp, humidity, hour, day_week, month],
            outputs=out,
        )

    with gr.Tab("Next 3 days forecast"):
        btn3 = gr.Button("Fetch and predict next 3 days")
        out3 = gr.Textbox(label="Hourly predictions", lines=30)
        btn3.click(fn=predict_next_3_days_via_api, inputs=[], outputs=out3)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

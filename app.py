import os
import numpy as np
import gradio as gr
import google.generativeai as genai
import pickle

# Load the Crop Recommender model
with open(r"RandomForest.pkl", 'rb') as f:
    model = pickle.load(f)

# Configure the Gemini AI[API_KEYS not mentioned here]
genai.configure(api_key='')

# Create the Gemini model configuration
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

gemini_model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
)

chat_session = gemini_model.start_chat(history=[])

# Function to predict the crop
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    #print(input_data)
    prediction = model.predict(input_data)
    return prediction[0]

# Function to handle chatbot interaction
def chatbot_response(query,N,P,K,temp,H,ph,mm,c):
    prompt=f'''
    You are an Agribot.Your task is to help farmers solving their queries and recommending them solutions to get more crop yield.
    Remember to encourage Organic Farming.
    Details of Soil:
        Nitrogen (N) Content in Soil : {N}
        Phosphorous (P) Content in Soil : {P}
        Potassium (K) Content in Soil : {K}
        Temperature (°C) : {temp}
        Humidity (%) : {H}
        pH of Soil : {ph}
        Rainfall (mm) : {mm}
    Predicted Crop(99% accurate Model) : {c}
    Query : {query}
    '''
    response = chat_session.send_message(prompt)
    return response.text

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Crop Prediction and Gemini AI Chatbot")
    gr.Markdown("## Enter Soil and Weather Parameters")
    with gr.Row():
        with gr.Row():
           
            N = gr.Number(label="Nitrogen (N) Content in Soil", value=104)
            P = gr.Number(label="Phosphorous (P) Content in Soil", value=18)
            K = gr.Number(label="Potassium (K) Content in Soil", value=30)
            temperature = gr.Number(label="Temperature (°C)", value=23.60)
        with gr.Row():
            humidity = gr.Number(label="Humidity (%)", value=60.3)
            ph = gr.Number(label="pH of Soil", value=6.7)
            rainfall = gr.Number(label="Rainfall (mm)", value=140.91)
            prediction_output = gr.Textbox(label="Predicted Crop")
    predict_btn = gr.Button("Predict Crop")
    gr.Markdown("## Ask the Gemini AI about Crop Recommendations")
    with gr.Column():
            
            query = gr.Textbox(label="Your Query")
            query_btn = gr.Button("Ask Gemini AI")
            chatbot_output = gr.Markdown(label="Gemini AI Response")
    
    predict_btn.click(fn=predict_crop, inputs=[N, P, K, temperature, humidity, ph, rainfall], outputs=prediction_output)
    query_btn.click(fn=chatbot_response, inputs=[query,N, P, K, temperature, humidity, ph, rainfall,prediction_output], outputs=chatbot_output)

# Launch the interface
demo.launch()
